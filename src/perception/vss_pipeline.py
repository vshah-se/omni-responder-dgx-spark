import os
import sys
import re
import json
import time
import base64
import hashlib
import subprocess
import datetime
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Generator, Tuple
from src.config.settings import settings

@dataclass
class VisualContextSummary:
    """Standardized JSON schema emitted purely by Cosmos Reasoner VLM for the Orchestrator."""
    location: str = "Surveillance Scene Location"
    camera_id: str = "CAM-DGX-SPARK-01"
    crisis_type: str = "Roadway Traffic Observation"
    severity: str = "LOW"  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    vehicles_involved: int = 0
    hazard_indicators: List[str] = field(default_factory=list)
    raw_summary: str = ""
    confidence: float = 0.95
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamFrameEvent:
    """Temporal frame event during live continuous video monitoring."""
    timestamp: str
    elapsed_seconds: float
    status: str
    severity: str
    scene_description: str
    visual_summary: Optional[VisualContextSummary] = None

class VSSPerceptionPipeline:
    """Unbiased Vision Pipeline: Pure Pixel Differencing -> Cosmos Reasoner VLM Reasoning."""

    # Tunable Hyperparameters for the persistent-deviation incident scanner
    SCAN_SAMPLE_FPS = 1.0           # thumbnails decoded per second of feed
    SCAN_PIXEL_THRESHOLD = 18       # grey levels of deviation that count as changed
    SCAN_WINDOW_SECONDS = 8.0       # how long a change must hold to count as an incident
    SCAN_PERSISTENCE_RATIO = 0.75   # fraction of the window a pixel must stay deviant
    MAX_VLM_PROBES = 3              # candidate moments we will spend a VLM call on
    VLM_TIMEOUT_SECONDS = 120       # a loaded NIM can exceed 35s on a multi-frame burst

    # The severity rubric deliberately contains NO reusable incident labels. An earlier
    # version listed "Active collision impact" as the CRITICAL example and the model
    # copied that phrase into crisis_type on empty roads, reporting CRITICAL while its
    # own summary said the traffic was normal. Describe first, judge second.
    STAGE2_DEEP_PROMPT = """You are an objective Edge Surveillance Vision Reasoner on NVIDIA DGX Spark.
Analyze this sequence of frames from a roadway surveillance camera.

Report ONLY what is physically visible in these frames. Do not infer an emergency that you cannot see.

Work in this order:
1. First write raw_summary: 2-3 sentences describing exactly what is visible.
2. Then count vehicles_involved: vehicles that are damaged, stopped, or otherwise part of an incident, including any emergency vehicles attending the scene. If traffic is simply flowing, this is 0.
3. Then list hazard_indicators: only things you can SEE (smoke, fire, fluid, debris, wreckage, flashing emergency lights, stopped vehicle). Empty array if none.
4. Then set crisis_type: your own short description of the scene.
5. Finally set severity, consistent with what you just wrote:
   - LOW: traffic flowing normally. No accident, no stopped vehicle, no responders, no hazard. THIS IS THE CORRECT ANSWER FOR ORDINARY TRAFFIC.
   - MEDIUM: a vehicle stopped on the shoulder, hazard lights, a tow operation, emergency responders or flashing emergency lights present, or hazardous weather.
   - HIGH or CRITICAL: you can SEE a collision, wreckage, debris blocking lanes, fire, smoke, or a spill.

Consistency rules (a violation means your answer is wrong):
- If your raw_summary says there is no accident and no hazard and traffic is flowing, severity MUST be LOW.
- If you can see emergency responders, police cars, fire apparatus or flashing emergency beacons, severity is AT LEAST MEDIUM. Never LOW.
- Do not report HIGH or CRITICAL unless you can name something visible in the frames: a collision, wreckage, debris, fire, smoke or a spill.

Output ONLY a STRICT JSON object with these exact keys, in this order:
{
  "raw_summary": "2-3 sentences of exactly what is visible",
  "location": "physical setting observed in the video pixels",
  "camera_id": "string",
  "vehicles_involved": integer,
  "hazard_indicators": ["only hazards visible in frame, empty array if none"],
  "crisis_type": "short objective description of the scene",
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "confidence": float (0.0 to 1.0)
}"""

    def __init__(self, endpoint_url: Optional[str] = None, model_name: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("COSMOS_VLM_URL", "http://localhost:30082/v1")
        self.model_name = model_name or os.getenv("COSMOS_VLM_MODEL")
        # Ranking a feed means decoding it end to end, and one run asks for the same
        # ranking twice: stream_video_feed() scans to announce the anomaly, then
        # analyze_incident() scans again to choose probe points. Same file, same
        # answer, paid for twice. Keyed on identity+size+mtime rather than path alone
        # so a replaced file under a reused name is not served a stale ranking.
        self._candidate_cache: Dict[Tuple[str, int, int], List[Tuple[float, float]]] = {}

    def _get_active_model_name(self) -> str:
        """Queries the NIM container for its registered model name."""
        if self.model_name:
            return self.model_name
        try:
            req = urllib.request.Request(f"{self.endpoint_url}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if models:
                    self.model_name = models[0]["id"]
                    return self.model_name
        except Exception:
            pass
        return "nvidia/cosmos-reason2-8b"

    def get_video_duration(self, video_path: str) -> float:
        """Retrieves exact video duration via ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            out = subprocess.check_output(cmd, timeout=5).decode().strip()
            return max(2.0, float(out))
        except Exception:
            return 12.0

    def extract_single_frame(self, video_path: str, timestamp_sec: float) -> Optional[str]:
        """Extracts a high-definition JPEG frame at a specific timestamp as base64."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp_sec),
                "-i", video_path,
                "-vframes", "1",
                "-vf", "scale=854:-1",
                "-q:v", "2",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = proc.communicate(timeout=8)
            if len(out) > 2000:
                return base64.b64encode(out).decode("utf-8")
        except Exception:
            pass
        return None



    def extract_grayscale_series(self, video_path: str, sample_fps: Optional[float] = None) -> List[Tuple[float, bytes]]:
        """Decodes the entire feed once into 64x64 grayscale thumbnails.

        A single ffmpeg pass replaces one seek per sample: a 6-minute feed yields
        360 samples in ~2s, where the previous seek-per-sample approach managed 24.
        """
        fps = sample_fps or self.SCAN_SAMPLE_FPS
        cmd = [
            "ffmpeg", "-v", "error", "-i", video_path,
            "-vf", f"fps={fps},scale=64:64,format=gray",
            "-f", "rawvideo", "-"
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
            raw = proc.stdout
        except Exception:
            return []
        frame_bytes = 64 * 64
        return [(i / fps, raw[i * frame_bytes:(i + 1) * frame_bytes]) for i in range(len(raw) // frame_bytes)]

    def _exposure_normalized(self, frames: List[Tuple[float, bytes]]) -> List[Tuple[float, List[int]]]:
        """Subtracts each frame's mean brightness.

        Outdoor cameras drift in exposure over minutes. Without this correction the
        drift registers as every pixel changing at once and buries genuine local events:
        on a 6-minute night feed it ranked an empty road above an active incident scene.
        """
        pixels = 64 * 64
        normalized = []
        for t, frame in frames:
            mean = sum(frame) / pixels
            normalized.append((t, [frame[p] - mean for p in range(pixels)]))
        return normalized

    def _median_background(self, normalized: List[Tuple[float, List[int]]]) -> List[float]:
        """Per-pixel median across the feed: the roadway as it normally looks."""
        pixels = 64 * 64
        midpoint = len(normalized) // 2
        return [sorted(f[1][p] for f in normalized)[midpoint] for p in range(pixels)]

    def rank_incident_candidates(self, video_path: str) -> List[Tuple[float, float]]:
        """Ranks moments by how much of the scene has changed AND STAYED changed.

        Raw pixel change is the wrong signal on a traffic camera: it measures traffic
        volume, so busy free-flowing lanes always outrank a stationary crash. An
        incident instead leaves something that arrives and remains - a stopped vehicle,
        wreckage, responders, a queue. This scores sustained deviation from the
        background instead, and returns candidates spread across the feed, best first.

        Memoised per file: the decode pass dominates the cost and both callers in a
        single run want the identical answer. A cache miss behaves exactly as before.
        """
        cache_key = None
        try:
            st = os.stat(video_path)
            cache_key = (os.path.realpath(video_path), st.st_size, int(st.st_mtime))
            if cache_key in self._candidate_cache:
                sys.stderr.write("[Incident Scanner] Reusing ranking for this feed (cached).\n")
                return list(self._candidate_cache[cache_key])
        except OSError:
            pass                                  # unreadable path: fall through uncached

        frames = self.extract_grayscale_series(video_path)
        if len(frames) < 4:
            if cache_key is not None:
                self._candidate_cache[cache_key] = []
            return []

        pixels = 64 * 64
        normalized = self._exposure_normalized(frames)
        background = self._median_background(normalized)
        masks = [
            (t, [1 if abs(f[p] - background[p]) > self.SCAN_PIXEL_THRESHOLD else 0 for p in range(pixels)])
            for t, f in normalized
        ]

        window = max(2, int(self.SCAN_WINDOW_SECONDS * self.SCAN_SAMPLE_FPS))
        required = self.SCAN_PERSISTENCE_RATIO * window
        scored: List[Tuple[float, float]] = []
        for i in range(len(masks) - window + 1):
            chunk = masks[i:i + window]
            persistent = sum(1 for p in range(pixels) if sum(c[1][p] for c in chunk) >= required)
            scored.append((chunk[len(chunk) // 2][0], persistent / pixels * 100.0))

        # Spread candidates out so all probes do not land inside one event
        duration = frames[-1][0] or 1.0
        separation = max(self.SCAN_WINDOW_SECONDS, duration / 8.0)
        picked: List[Tuple[float, float]] = []
        for t, score in sorted(scored, key=lambda x: -x[1]):
            if all(abs(t - pt) >= separation for pt, _ in picked):
                picked.append((t, score))
            if len(picked) >= self.MAX_VLM_PROBES:
                break
        if cache_key is not None:
            self._candidate_cache[cache_key] = list(picked)
        return picked

    # A scene where this share of pixels has persistently changed is worth reporting
    ANOMALY_REPORT_THRESHOLD_PCT = 3.0

    # Phrases meaning the model itself saw no incident, used only to overrule a severity
    # that contradicts the model's own description.
    NO_INCIDENT_PHRASES = (
        "no accident", "no collision", "no incident", "no hazard", "no emergency",
        "no obstruction", "no visible signs of", "no signs of", "no damage",
        "normal traffic", "traffic flows normally", "traffic is flowing normally",
    )
    # Evidence that something IS happening, even with nothing wrecked in frame.
    # A tow truck is deliberately NOT here: one hauling a car along the motorway is
    # ordinary traffic, and listing it turned a clear road into a CODE AMBER callout.
    RESPONDER_PHRASES = (
        "police", "fire truck", "fire engine", "fire apparatus", "ambulance", "paramedic",
        "emergency vehicle", "emergency responder", "emergency personnel", "emergency lights",
        "flashing", "beacon",
    )
    # Words that flip a sentence to describing an ABSENCE
    NEGATION_MARKERS = ("no ", "not ", "without ", "absent", "none ", "n't ")

    # Distinct KINDS of responder, counted separately from RESPONDER_PHRASES above.
    # One kind on a road may be a vehicle passing through; two kinds standing on the
    # same road is a scene being worked by more than one service. Tow trucks stay out
    # for the same reason they are absent from RESPONDER_PHRASES.
    RESPONDER_FAMILIES = {
        "police": ("police", "law enforcement", "patrol car", "squad car", "state trooper"),
        "fire": ("fire truck", "fire engine", "fire apparatus", "firefighter", "ladder truck"),
        "medical": ("ambulance", "paramedic", "medic unit", "ems "),
    }
    # The road itself is stopped, queued or narrowed. Every phrase here describes
    # something the model can see directly, which is the point: it lets severity rise
    # on observation rather than on inferring a wreck nobody photographed.
    TRAFFIC_ARRESTED_PHRASES = (
        "halted", "standstill", "stopped traffic", "traffic is stopped",
        "traffic has stopped", "backed up", "queuing", "queued", "gridlock",
        "road closed", "lane closed", "lanes closed", "closed to traffic",
    )
    LANE_OBSTRUCTION_PHRASES = (
        "obstructing", "obstructed", "blocking", "blocked", "impassable",
    )

    def scan_motion_and_anomaly_points(self, video_path: str) -> Tuple[bool, float, str]:
        """Locates the most incident-like moment in the feed.

        Kept for the dashboard adapter and stream narration; ranking lives in
        rank_incident_candidates().
        """
        sys.stderr.write("[Incident Scanner] Decoding feed and scoring persistent change...\n")
        candidates = self.rank_incident_candidates(video_path)
        if not candidates:
            return False, self.get_video_duration(video_path) * 0.5, "Standard traffic flow"

        best_t, best_score = candidates[0]
        if best_score >= self.ANOMALY_REPORT_THRESHOLD_PCT:
            return True, best_t, f"Sustained scene change detected ({best_score:.1f}% of view held changed)"
        return False, best_t, "Continuous traffic flow"

    def analyze_incident(self, video_path: str, location_hint: Optional[str] = None) -> VisualContextSummary:
        """Probes candidate moments until one shows a real incident.

        A single probe cannot be trusted to land on the event: on a 6-minute feed the
        highest-scoring moment may be heavy traffic while the incident sits elsewhere.
        Quiet footage therefore costs one VLM call, and only ambiguous footage escalates.
        """
        candidates = self.rank_incident_candidates(video_path)
        if not candidates:
            candidates = [(self.get_video_duration(video_path) * 0.5, 0.0)]

        first_summary: Optional[VisualContextSummary] = None
        failed_summary: Optional[VisualContextSummary] = None
        for index, (timestamp, score) in enumerate(candidates[:self.MAX_VLM_PROBES], start=1):
            burst_frames = self.extract_targeted_burst(video_path, timestamp)
            if not burst_frames:
                continue
            sys.stderr.write(
                f"[Incident Scanner] Probe {index}/{min(len(candidates), self.MAX_VLM_PROBES)} "
                f"at T={timestamp:.1f}s (persistence {score:.1f}%)\n"
            )
            summary = self.deep_sequence_diagnosis(burst_frames, location_hint)
            summary.camera_id = self.camera_id_for(video_path)

            # A probe the VLM never answered is not evidence of a quiet road
            if summary.crisis_type == "VLM_CONNECTION_OFFLINE":
                failed_summary = summary
                continue
            if summary.severity != "LOW":
                return summary
            if first_summary is None:
                first_summary = summary

        if first_summary is not None:
            return first_summary
        if failed_summary is not None:
            return failed_summary
        # All frame extractions failed (corrupt file, disk error) — return safely instead of crashing
        return VisualContextSummary(
            location=location_hint or "Unknown",
            camera_id=self.camera_id_for(video_path),
            crisis_type="FRAME_EXTRACTION_FAILED",
            severity="LOW",
            vehicles_involved=0,
            raw_summary=f"ERROR: Could not extract any frames from feed. The file may be corrupt or unreadable.",
            confidence=0.0
        )

    def extract_targeted_burst(self, video_path: str, center_t: float) -> List[Tuple[float, str]]:
        """Extracts high-density burst of frames around the exact anomaly moment."""
        duration = self.get_video_duration(video_path)
        # Reverted back to 4 frames so it safely fits within the 16k context window without connection resets
        offsets = [-1.5, 0.0, 1.5, 3.0]
        timestamps = [max(0.2, min(duration - 0.2, center_t + offset)) for offset in offsets]
        timestamps = sorted(list(set([round(t, 2) for t in timestamps])))

        burst_frames = []
        for t in timestamps:
            b64 = self.extract_single_frame(video_path, t)
            if b64:
                burst_frames.append((t, b64))
        return burst_frames

    def deep_sequence_diagnosis(self, frames_sequence: List[Tuple[float, str]], location_hint: Optional[str] = None) -> VisualContextSummary:
        """Queries Cosmos Reasoner with high-definition targeted impact burst."""
        url = f"{self.endpoint_url}/chat/completions"
        active_model = self._get_active_model_name()

        loc_text = f"Registered Location Metadata: {location_hint}" if location_hint else "Derive physical setting directly from video pixels."
        content_elements = [
            {
                "type": "text",
                "text": f"{self.STAGE2_DEEP_PROMPT}\n\n{loc_text}\nHigh-Definition Frame Sequence:"
            }
        ]

        for t_sec, b64 in frames_sequence:
            content_elements.append({"type": "text", "text": f"[Frame @ T={t_sec:.1f}s]:"})
            content_elements.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        content_elements.append({"type": "text", "text": "Analyze the frames and output the strict JSON object now."})

        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": content_elements}],
            "temperature": 0.1,
            "max_tokens": 2048 # Expanded capacity for deep reasoning
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.VLM_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_text = data["choices"][0]["message"]["content"]
                return self.parse_vlm_json_output(raw_text, location_hint)
        except Exception as e:
            # A socket timeout raises TimeoutError, which is neither URLError nor
            # ConnectionError, so a slow NIM used to crash the whole run mid-demo.
            sys.stderr.write(f"\n\033[1;31m[WARN] Cosmos VLM probe failed ({type(e).__name__}): {e}\033[0m\n")
            sys.stderr.write("\033[1;33mCheck the NIM container is running and the prompt fits NIM_MAX_MODEL_LEN.\033[0m\n")
            return VisualContextSummary(
                location=location_hint or "Unknown",
                camera_id="CAM-DGX-SPARK-01",
                crisis_type="VLM_CONNECTION_OFFLINE",
                severity="LOW",
                vehicles_involved=0,
                raw_summary=f"ERROR: The Vision-Language Model probe failed ({type(e).__name__}): {e}",
                confidence=0.0
            )

    def _responders_present(self, scene_text: str) -> bool:
        """True only where responders are described as PRESENT.

        Scoped per sentence, because a flat substring test reads "there are no
        collisions, stopped vehicles, or emergency responders visible" as responders
        being on scene and escalates a clear road to CODE AMBER.
        """
        for sentence in re.split(r"[.;]", scene_text):
            if any(phrase in sentence for phrase in self.RESPONDER_PHRASES) and \
                    not any(marker in sentence for marker in self.NEGATION_MARKERS):
                return True
        return False

    def _responder_families_present(self, scene_text: str) -> int:
        """How many DISTINCT kinds of responder are described as being on scene.

        Sentence-scoped and negation-aware for the same reason as
        _responders_present(): "no police or fire crews are visible" describes an
        empty road, not two services attending one.
        """
        families = set()
        for sentence in re.split(r"[.;]", scene_text):
            if any(marker in sentence for marker in self.NEGATION_MARKERS):
                continue
            for family, phrases in self.RESPONDER_FAMILIES.items():
                if any(phrase in sentence for phrase in phrases):
                    families.add(family)
        return len(families)

    def _scene_arrested(self, scene_text: str) -> bool:
        """True where the road is described as stopped, queued, or a lane obstructed."""
        for sentence in re.split(r"[.;]", scene_text):
            if any(marker in sentence for marker in self.NEGATION_MARKERS):
                continue
            if any(p in sentence for p in self.TRAFFIC_ARRESTED_PHRASES) or \
                    any(p in sentence for p in self.LANE_OBSTRUCTION_PHRASES):
                return True
        return False

    def camera_id_for(self, video_path: str) -> str:
        """Stable edge camera identity derived from a SHA-1 hash of the first 8KB of video content.

        Using the filename is bogus engineering — the camera's identity must come from
        what it records, not what someone named the file on disk.
        """
        try:
            with open(video_path, "rb") as f:
                digest = hashlib.sha1(f.read(8192)).hexdigest()[:8].upper()
            return f"CAM-EDGE-{digest}"
        except Exception:
            return "CAM-EDGE-UNKNOWN"

    def stream_video_feed(
        self,
        video_path: str,
        location_hint: Optional[str] = None,
        speed_multiplier: float = 1.0
    ) -> Generator[StreamFrameEvent, None, None]:
        """Streams real-time edge monitoring driven purely by pixel analysis and Cosmos Reasoner."""
        duration = self.get_video_duration(video_path)
        has_anomaly, anomaly_t, anomaly_desc = self.scan_motion_and_anomaly_points(video_path)

        camera_id = self.camera_id_for(video_path)

        live_ts = datetime.datetime.now().strftime("%H:%M:%S")
        yield StreamFrameEvent(
            timestamp=live_ts,
            elapsed_seconds=0.0,
            status="NORMAL_MONITORING",
            severity="LOW",
            scene_description=f"Edge Camera ({camera_id}) online ({duration:.1f}s feed). Pixel motion scanner active...",
            visual_summary=None
        )

        time.sleep(0.5 / max(speed_multiplier, 0.1))

        if has_anomaly:
            live_ts = datetime.datetime.now().strftime("%H:%M:%S")
            yield StreamFrameEvent(
                timestamp=live_ts,
                elapsed_seconds=anomaly_t,
                # Neutral status: pixel scanner detected change, VLM has not yet confirmed severity
                status="ANOMALY_DETECTED",
                severity="LOW",
                scene_description=f"[T={anomaly_t:.1f}s] Sustained scene change detected. Dispatching Cosmos VLM probes...",
                visual_summary=None
            )

        summary = self.analyze_incident(video_path, location_hint)
        summary.camera_id = camera_id
        summary.timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Route status dynamically from Cosmos Reasoner's confirmed output only
        if summary.severity in ["CRITICAL", "HIGH", "SEVERE"]:
            final_status = "CRISIS_IMPACT"
        elif summary.severity in ["MEDIUM", "MODERATE"]:
            final_status = "ROADSIDE_ASSISTANCE"
        else:
            final_status = "ROUTINE_ALL_CLEAR"

        yield StreamFrameEvent(
            timestamp=summary.timestamp,
            elapsed_seconds=anomaly_t if has_anomaly else duration * 0.5,
            status=final_status,
            severity=summary.severity,
            scene_description=f"Cosmos Reasoner Diagnosis: {summary.raw_summary}",
            visual_summary=summary
        )

    def process_video_file(self, video_path: str, location_hint: Optional[str] = None) -> VisualContextSummary:
        """Direct end-to-end processing purely on video frames."""
        return self.analyze_incident(video_path, location_hint=location_hint)

    def parse_vlm_json_output(self, raw_response: str, location_hint: Optional[str] = None) -> VisualContextSummary:
        """Parses and preserves exact visual reasoning from Cosmos Reasoner with zero overwrites."""
        cleaned = raw_response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            
            # Extract fields directly from Cosmos VLM response
            vlm_summary = str(data.get("raw_summary", raw_response[:250]))
            vlm_crisis = str(data.get("crisis_type", "Roadway Traffic Observation"))
            vlm_hazards = list(data.get("hazard_indicators", []))
            vlm_location = location_hint or str(data.get("location", "Surveillance Camera Scene"))
            vlm_camera = str(data.get("camera_id", "CAM-DGX-SPARK-01"))
            vlm_confidence = float(data.get("confidence", 0.96))

            # Normalize severity safely without altering Cosmos's intent
            raw_sev = str(data.get("severity", "LOW")).upper()
            if raw_sev in ["SEVERE", "CRITICAL"]:
                severity_str = "CRITICAL"
            elif raw_sev in ["HIGH"]:
                severity_str = "HIGH"
            elif raw_sev in ["MEDIUM", "MODERATE"]:
                severity_str = "MEDIUM"
            else:
                severity_str = "LOW"

            # Parse vehicles count from Cosmos
            raw_vehicles = data.get("vehicles_involved", 1)
            if isinstance(raw_vehicles, list):
                vehicles_count = len(raw_vehicles)
            elif isinstance(raw_vehicles, str) and raw_vehicles.isdigit():
                vehicles_count = int(raw_vehicles)
            elif isinstance(raw_vehicles, (int, float)):
                vehicles_count = int(raw_vehicles)
            else:
                vehicles_count = 1 if severity_str in ["MEDIUM", "HIGH", "CRITICAL"] else 0

            # Safety net for a model echoing a severity it did not observe. Only fires on
            # a genuine self-contradiction: an emergency verdict over a description that
            # states there was no incident, with no vehicles and no hazard to point at.
            # Deliberately narrow - a scene with responders is never flattened to all-clear.
            scene_text = f"{vlm_summary} {vlm_crisis}".lower()
            denies_incident = any(phrase in scene_text for phrase in self.NO_INCIDENT_PHRASES)
            shows_responders = self._responders_present(scene_text)
            # vehicles_count is deliberately NOT part of this test. It used to require
            # vehicles_count == 0, which made the net almost inert: a traffic camera
            # nearly always has vehicles in frame, and the model does not reliably
            # distinguish "vehicles present" from "vehicles involved in an incident"
            # (hence the three parsing fallbacks above). Gating the safety net on the
            # least reliable field in the response defeated the safety net. On aic21_93
            # the model returned "There are no visible signs of a collision" alongside
            # severity CRITICAL, and the net did not fire because it also reported a
            # vehicle. The remaining conditions — an emergency verdict, no hazard to
            # point at, prose that denies an incident, and no responders on scene —
            # are a genuine self-contradiction on their own.
            if (severity_str in ["HIGH", "CRITICAL"]
                    and not vlm_hazards and denies_incident and not shows_responders):
                severity_str = "LOW"

            # Responders on scene mean something happened, whatever the traffic is doing.
            # The rubric asks for MEDIUM here but the model does not reliably comply, so
            # the floor is enforced rather than requested.
            if severity_str == "LOW" and shows_responders:
                severity_str = "MEDIUM"

            # Escalate on the scale of the response, not on seeing the wreck.
            #
            # The rubric caps severity at MEDIUM unless the model can name visible
            # wreckage, fire, debris or a spill. That is right for hallucination
            # control and wrong for attended scenes, because the better attended an
            # incident is the more likely the wreck is screened by the responders, has
            # already been cleared, or is simply invisible at night. In that case the
            # reported severity moves INVERSELY to the real one.
            #
            # On tanker.mp4 the model described a fire truck, police cars, one lane
            # obstructed and traffic halted, then correctly added that it could see no
            # collision — and the pipeline settled on MEDIUM, which the dashboard
            # printed as "minor" over a road full of emergency vehicles.
            #
            # Two DISTINCT responder families plus a stopped or narrowed road is an
            # incident being worked. Every term is something the model said it saw, so
            # nothing here is inferred. The aic21_01 case behind ee2e026 — a tow truck
            # hauling a car along a flowing motorway — fails both halves: tow trucks
            # are not a family, and flowing traffic is not arrested.
            if (severity_str in ["LOW", "MEDIUM"]
                    and self._responder_families_present(scene_text) >= 2
                    and self._scene_arrested(scene_text)):
                severity_str = "HIGH"

            return VisualContextSummary(
                location=vlm_location,
                camera_id=vlm_camera,
                crisis_type=vlm_crisis,
                severity=severity_str,
                vehicles_involved=vehicles_count,
                hazard_indicators=vlm_hazards,
                raw_summary=vlm_summary,
                confidence=vlm_confidence,
                timestamp=datetime.datetime.now().strftime("%H:%M:%S")
            )
        except Exception as e:
            # An unreadable response is a PERCEPTION FAILURE, not an incident. This
            # used to return HIGH with one vehicle, so any malformed reply — including
            # one whose text read "The road is completely clear" — manufactured an
            # emergency and dispatched on it. Degrade the same way a failed VLM probe
            # does (VLM_CONNECTION_OFFLINE above): LOW so nothing escalates, zero
            # confidence so the dashboard can show the run as degraded rather than calm.
            import sys
            sys.stderr.write(
                f"\n\033[1;31m[WARN] Could not parse Cosmos VLM response "
                f"({type(e).__name__}): {e}\033[0m\n"
            )
            return VisualContextSummary(
                location=location_hint or "Roadway Surveillance Scene",
                camera_id="CAM-DGX-SPARK-01",
                crisis_type="VLM_PARSE_ERROR",
                severity="LOW",
                vehicles_involved=0,
                hazard_indicators=[],
                raw_summary=f"ERROR: Cosmos VLM response could not be parsed "
                            f"({type(e).__name__}). Raw: {raw_response[:200]}",
                confidence=0.0,
                timestamp=datetime.datetime.now().strftime("%H:%M:%S")
            )


