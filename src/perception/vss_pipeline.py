import os
import json
import time
import base64
import subprocess
import datetime
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Generator, Tuple
from src.config.settings import settings

@dataclass
class VisualContextSummary:
    """Standardized JSON schema emitted by Cosmos Reasoner VLM for the Orchestrator."""
    location: str
    camera_id: str = "CAM-DGX-SPARK-01"
    crisis_type: str = "Physical Emergency Incident"
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
    """Interfaces directly with NVIDIA Cosmos Reasoner 2 VLM running on Port 30082."""

    SYSTEM_PROMPT = """You are an autonomous Emergency Incident Vision Agent running on NVIDIA DGX Spark.
You are given a chronological sequence of video frames sampled across the ENTIRE duration of a camera feed (from start to finish).
Carefully inspect the full temporal progression across all frames:
1. Did a traffic collision, vehicle impact, chemical spill, fire, or hazard occur at ANY point in the sequence?
2. If an incident occurred, identify all vehicles involved, damage, smoke/flames/fluids, and severity.
3. If the entire sequence shows only normal driving with no accidents, set severity to "LOW", crisis_type to "Normal Traffic Flow", and hazard_indicators to [].

Output a STRICT valid JSON object with these exact keys:
{
  "location": "string",
  "camera_id": "string",
  "crisis_type": "string (e.g. Collision / Chemical Spill / Normal Flow)",
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "vehicles_involved": integer,
  "hazard_indicators": ["list of visible hazards like smoke, fire, fluid leak, wreckage, or empty if none"],
  "raw_summary": "detailed 2-3 sentence visual summary describing the progression and outcome across the frames",
  "confidence": float (0.0 to 1.0)
}
Output ONLY the JSON object without commentary."""

    def __init__(self, endpoint_url: Optional[str] = None, model_name: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("COSMOS_VLM_URL", "http://localhost:30082/v1")
        self.model_name = model_name

    def _get_active_model_name(self) -> str:
        """Dynamically queries the NIM container to get its exact registered model name."""
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
        """Dynamically retrieves the exact duration of any video file in seconds."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            out = subprocess.check_output(cmd, timeout=5).decode().strip()
            return max(1.0, float(out))
        except Exception:
            return 15.0

    def extract_full_video_keyframes(self, video_path: str, num_frames: int = 8) -> List[Tuple[float, str]]:
        """Evenly samples keyframes across the ENTIRE duration of the video file with timestamps."""
        duration = self.get_video_duration(video_path)
        frames_with_ts = []

        # Calculate evenly spaced sample timestamps across duration (e.g. at 10%, 25%, 40%, 60%, 75%, 90%)
        sample_times = [round((i + 0.5) * (duration / num_frames), 2) for i in range(num_frames)]

        for t in sample_times:
            try:
                cmd = [
                    "ffmpeg", "-y", "-ss", str(t),
                    "-i", video_path,
                    "-vframes", "1",
                    "-vf", "scale=640:-1",
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "-"
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                out, _ = proc.communicate(timeout=8)
                if len(out) > 1000:
                    b64 = base64.b64encode(out).decode("utf-8")
                    frames_with_ts.append((t, b64))
            except Exception:
                continue

        return frames_with_ts

    def process_video_file(self, video_path: str, location_hint: str = "5th Ave & Market St Intersection") -> VisualContextSummary:
        """Sends full temporal keyframe sequence to the live Cosmos Reasoner VLM on Port 30082."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        try:
            return self._query_live_cosmos_vlm(video_path, location_hint)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            print(f"\033[1;31m[ERROR] Cosmos VLM returned HTTP {e.code}: {error_body}\033[0m")
            return self._extract_scenario_heuristic(video_path, location_hint)
        except Exception as e:
            print(f"\033[1;33m[WARN] Cosmos VLM connection error: {e}. Using fallback.\033[0m")
            return self._extract_scenario_heuristic(video_path, location_hint)

    def _query_live_cosmos_vlm(self, video_path: str, location_hint: str) -> VisualContextSummary:
        """Calls OpenAI-compatible /v1/chat/completions with full chronological sequence."""
        url = f"{self.endpoint_url}/chat/completions"
        active_model = self._get_active_model_name()
        duration = self.get_video_duration(video_path)
        
        # Sample 6-8 frames across the entire video
        frames_with_ts = self.extract_full_video_keyframes(video_path, num_frames=6)

        prompt_intro = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Location Hint: {location_hint}\n"
            f"Total Video Duration: {duration:.1f} seconds. Below are {len(frames_with_ts)} chronological frames sampled across the full video:"
        )

        content_elements = [{"type": "text", "text": prompt_intro}]

        for t_sec, b64 in frames_with_ts:
            content_elements.append({
                "type": "text",
                "text": f"[Frame Timestamp T={t_sec:.1f}s / {duration:.1f}s]:"
            })
            content_elements.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        content_elements.append({
            "type": "text",
            "text": "Analyze the full temporal progression above. Output the strict JSON object now."
        })

        payload = {
            "model": active_model,
            "messages": [
                {"role": "user", "content": content_elements}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data["choices"][0]["message"]["content"]
            return self.parse_vlm_json_output(raw_text)

    def stream_video_feed(
        self,
        video_path: str,
        location_hint: str = "5th Ave & Market St Intersection",
        speed_multiplier: float = 1.0
    ) -> Generator[StreamFrameEvent, None, None]:
        """Streams real-time edge monitoring with live wall-clock timestamps and real video duration."""
        start_time = time.time()
        video_duration = self.get_video_duration(video_path)

        # 1. Ingest camera stream connection
        live_ts = datetime.datetime.now().strftime("%H:%M:%S")
        yield StreamFrameEvent(
            timestamp=live_ts,
            elapsed_seconds=0.0,
            status="ACTIVE_STREAM_INGESTION",
            severity="NORMAL",
            scene_description=f"Edge Camera active at {location_hint} ({video_duration:.1f}s feed). Sampling temporal frames across video for Cosmos VLM...",
            visual_summary=None
        )

        # 2. Query Cosmos Reasoner across full video duration
        summary = self.process_video_file(video_path, location_hint)
        elapsed = time.time() - start_time
        summary.timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # 3. Determine event status based on severity
        is_crisis = summary.severity in ["CRITICAL", "HIGH", "SEVERE"]
        status_label = "CRISIS_IMPACT_DETECTED" if is_crisis else "SCENE_MONITORED_NORMAL"
        live_ts = datetime.datetime.now().strftime("%H:%M:%S")

        yield StreamFrameEvent(
            timestamp=live_ts,
            elapsed_seconds=round(elapsed, 2),
            status=status_label,
            severity=summary.severity,
            scene_description=f"Cosmos VLM Edge Analysis ({elapsed:.2f}s): {summary.raw_summary}",
            visual_summary=summary
        )

    def parse_vlm_json_output(self, raw_response: str) -> VisualContextSummary:
        """Parses and validates raw VLM text output into the typed VisualContextSummary schema."""
        cleaned = raw_response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            
            raw_vehicles = data.get("vehicles_involved", 1)
            if isinstance(raw_vehicles, list):
                vehicles_count = len(raw_vehicles)
            elif isinstance(raw_vehicles, str) and not raw_vehicles.isdigit():
                vehicles_count = 2
            else:
                vehicles_count = int(raw_vehicles)

            raw_sev = str(data.get("severity", "LOW")).upper()
            raw_summary_text = data.get("raw_summary", raw_response[:250])
            hazard_list = list(data.get("hazard_indicators", []))

            # Smart severity normalization
            if any(phrase in raw_summary_text.lower() for phrase in ["no accidents", "no visible signs of collision", "no immediate hazards", "typical urban traffic", "no signs of collision or damage"]):
                if not hazard_list:
                    raw_sev = "LOW"

            if raw_sev in ["SEVERE", "CRITICAL"]:
                severity_str = "CRITICAL"
            elif raw_sev in ["HIGH"]:
                severity_str = "HIGH"
            elif raw_sev in ["MEDIUM", "MODERATE"]:
                severity_str = "MEDIUM"
            else:
                severity_str = "LOW"

            return VisualContextSummary(
                location=data.get("location", "5th Ave & Market St Intersection"),
                camera_id=str(data.get("camera_id", "CAM-DGX-SPARK-01")),
                crisis_type=data.get("crisis_type", "Traffic Observation"),
                severity=severity_str,
                vehicles_involved=vehicles_count,
                hazard_indicators=hazard_list,
                raw_summary=raw_summary_text,
                confidence=float(data.get("confidence", 0.95)),
                timestamp=datetime.datetime.now().strftime("%H:%M:%S")
            )
        except Exception:
            return VisualContextSummary(
                location="5th Ave & Market St Intersection",
                camera_id="CAM-DGX-SPARK-01",
                crisis_type="Traffic Scene Observation",
                severity="LOW",
                vehicles_involved=1,
                hazard_indicators=[],
                raw_summary=raw_response,
                confidence=0.90,
                timestamp=datetime.datetime.now().strftime("%H:%M:%S")
            )

    def _extract_scenario_heuristic(self, video_path: str, location_hint: str) -> VisualContextSummary:
        """Offline fallback only when Spark NIM is unreachable."""
        filename = os.path.basename(video_path).lower()
        return VisualContextSummary(
            location=location_hint,
            camera_id="CAM-OFFLINE-MOCK",
            crisis_type="Physical traffic collision",
            severity="HIGH",
            vehicles_involved=2,
            hazard_indicators=["vehicle wreckage", "blocked traffic"],
            raw_summary="Traffic collision blocking lanes requiring emergency dispatch.",
            confidence=0.90,
            timestamp=datetime.datetime.now().strftime("%H:%M:%S")
        )
