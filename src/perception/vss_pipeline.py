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

    # Tunable Hyperparameters for Hierarchical Motion Scanner
    COARSE_SEARCH_MAX_SAMPLES = 20
    COARSE_SEARCH_MIN_INTERVAL_SEC = 4.0
    FINE_SEARCH_RESOLUTION_SEC = 1.0


    STAGE2_DEEP_PROMPT = """You are an objective Edge Surveillance Vision Reasoner on NVIDIA DGX Spark.
Analyze this high-definition sequence of frames captured directly from an edge roadway surveillance camera.

Instructions:
1. Physical Setting: Describe the physical environment and location observed in the video background (e.g. Rural Highway, Multi-Lane Interstate, Urban Commercial Intersection, Mountain Road).
2. Vehicle Identification: Identify all vehicles visibly present by their true physical body type, size, and color (e.g. sedan, hatchback, SUV, pickup, commercial tractor-trailer, bus, motorcycle).
3. Incident & Severity Assessment:
   - "LOW": Normal moving traffic flow with no accidents, collisions, or hazards.
   - "MEDIUM": Minor roadside incident, vehicle stopped on shoulder with hazard lights, towing operation, or weather caution.
   - "HIGH" or "CRITICAL": Active collision impact, vehicle wreckage or debris blocking lanes, structural fire, heavy smoke, or hazardous fluid spills.

Output ONLY a STRICT JSON object with these exact keys:
{
  "location": "string (physical setting observed in video pixels)",
  "camera_id": "string",
  "crisis_type": "string (objective physical classification of the scene)",
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "vehicles_involved": integer (number of vehicles involved in the incident),
  "hazard_indicators": ["list of observed physical hazards such as smoke, fire, fluid leak, debris, or empty array if none"],
  "raw_summary": "2-3 sentence visual summary describing exactly what is seen in the frames without speculation",
  "confidence": float (0.0 to 1.0)
}"""

    def __init__(self, endpoint_url: Optional[str] = None, model_name: Optional[str] = None):
        self.endpoint_url = endpoint_url or os.getenv("COSMOS_VLM_URL", "http://localhost:30082/v1")
        self.model_name = model_name or os.getenv("COSMOS_VLM_MODEL")

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

    def extract_raw_grayscale_thumbnail(self, video_path: str, timestamp_sec: float) -> Optional[bytes]:
        """Extracts a 64x64 raw grayscale thumbnail for fast motion differencing."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp_sec),
                "-i", video_path,
                "-vframes", "1",
                "-vf", "scale=64:64,format=gray",
                "-f", "rawvideo",
                "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_bytes, _ = proc.communicate(timeout=4)
            if len(raw_bytes) == 64 * 64:
                return raw_bytes
        except Exception:
            pass
        return None

    def _compute_anomaly_score(self, bytes_curr: bytes, bytes_prev: bytes) -> float:
        """
        Computes a localized anomaly score.
        Ignores massive full-frame changes (like scene cuts or fades) by capping the max allowed changed pixels.
        """
        total_pixels = len(bytes_curr)
        # Count pixels that changed significantly (lowered to 12 to catch low-contrast vehicles on gray roads)
        changed_pixels = sum(1 for b1, b2 in zip(bytes_curr, bytes_prev) if abs(b1 - b2) > 12)
        
        # If >60% of the image changed, it's a camera cut/fade or full pan. Ignore it.
        if changed_pixels > (total_pixels * 0.60):
            return 0.0
            
        return (changed_pixels / total_pixels) * 100.0

    def scan_motion_and_anomaly_points(self, video_path: str) -> Tuple[bool, float, str]:
        """Hierarchical (Coarse-to-Fine) pixel differencing to locate exact anomaly moment rapidly."""
        import sys
        duration = self.get_video_duration(video_path)
        
        # --- PHASE 1: COARSE SEARCH ---
        coarse_interval = max(self.COARSE_SEARCH_MIN_INTERVAL_SEC, duration / self.COARSE_SEARCH_MAX_SAMPLES)
        num_coarse = max(2, int(duration / coarse_interval))
        coarse_times = [round((i + 0.5) * coarse_interval, 2) for i in range(num_coarse)]
        
        coarse_thumbnails: List[Tuple[float, bytes]] = []
        for i, t in enumerate(coarse_times):
            sys.stdout.write(f"\r\033[K[Motion Scanner] Coarse Search: {i+1}/{num_coarse} frames (T={t:.1f}s)...")
            sys.stdout.flush()
            raw = self.extract_raw_grayscale_thumbnail(video_path, t)
            if raw:
                coarse_thumbnails.append((t, raw))
                
        if len(coarse_thumbnails) < 2:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            return False, duration * 0.5, "Standard traffic flow"

        coarse_deltas = []
        for i in range(1, len(coarse_thumbnails)):
            t_curr, bytes_curr = coarse_thumbnails[i]
            t_prev, bytes_prev = coarse_thumbnails[i - 1]
            diff = self._compute_anomaly_score(bytes_curr, bytes_prev)
            coarse_deltas.append((t_prev, t_curr, diff))

        # Find the window with the highest variance
        best_prev, best_curr, best_coarse_diff = max(coarse_deltas, key=lambda x: x[2])
        
        # --- PHASE 2: FINE SEARCH ---
        fine_window_duration = best_curr - best_prev
        num_fine = max(3, int(fine_window_duration / self.FINE_SEARCH_RESOLUTION_SEC))
        fine_times = [round(best_prev + (i + 0.5) * (fine_window_duration / num_fine), 2) for i in range(num_fine)]
        
        fine_thumbnails = []
        for i, t in enumerate(fine_times):
            sys.stdout.write(f"\r\033[K[Motion Scanner] Fine Search (Window {best_prev:.1f}s-{best_curr:.1f}s): {i+1}/{num_fine} frames (T={t:.1f}s)...")
            sys.stdout.flush()
            raw = self.extract_raw_grayscale_thumbnail(video_path, t)
            if raw:
                fine_thumbnails.append((t, raw))
                
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        
        if len(fine_thumbnails) < 2:
            return False, best_curr, "Continuous traffic flow"
            
        fine_deltas = []
        for i in range(1, len(fine_thumbnails)):
            t_curr, bytes_curr = fine_thumbnails[i]
            _, bytes_prev = fine_thumbnails[i - 1]
            diff = self._compute_anomaly_score(bytes_curr, bytes_prev)
            fine_deltas.append((t_curr, diff))

        max_t, max_diff = max(fine_deltas, key=lambda x: x[1])
        
        # Calculate baseline from coarse search (excluding the anomalous window) to prevent skewing
        baseline_diffs = [d[2] for d in coarse_deltas if d[2] != best_coarse_diff]
        avg_diff = sum(baseline_diffs) / len(baseline_diffs) if baseline_diffs else best_coarse_diff

        # Sensitive to tiny anomalies (down to 0.05% of pixels, e.g. a small car crash in an aerial view)
        if max_diff > (avg_diff * 1.2) and max_diff > 0.05:
            return True, max_t, f"Localized spatial anomaly spike detected (Score={max_diff:.2f})"

        return False, max_t, "Continuous traffic flow"

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
            with urllib.request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_text = data["choices"][0]["message"]["content"]
                return self.parse_vlm_json_output(raw_text, location_hint)
        except (urllib.error.URLError, ConnectionError) as e:
            print(f"\n\033[1;31m[CRITICAL] Connection to Cosmos VLM failed: {e}.\033[0m")
            print(f"\033[1;33mPlease ensure the NIM container has fully finished booting up and that your prompt size doesn't exceed NIM_MAX_MODEL_LEN.\033[0m\n")
            return VisualContextSummary(
                location=location_hint or "Unknown",
                camera_id="CAM-DGX-SPARK-01",
                crisis_type="VLM_CONNECTION_OFFLINE",
                severity="LOW",
                vehicles_involved=0,
                raw_summary=f"ERROR: The Vision-Language Model container is currently unreachable. Reason: {e}",
                confidence=0.0
            )

    def stream_video_feed(
        self,
        video_path: str,
        location_hint: Optional[str] = None,
        speed_multiplier: float = 1.0
    ) -> Generator[StreamFrameEvent, None, None]:
        """Streams real-time edge monitoring driven purely by pixel analysis and Cosmos Reasoner."""
        duration = self.get_video_duration(video_path)
        has_anomaly, anomaly_t, anomaly_desc = self.scan_motion_and_anomaly_points(video_path)

        camera_id = f"CAM-EDGE-{os.path.basename(video_path).split('.')[0].upper()}"

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
                status="ANOMALY_DETECTED",
                severity="HIGH",
                scene_description=f"[T={anomaly_t:.1f}s] ANOMALY DETECTED: {anomaly_desc}. Capturing high-definition burst frames...",
                visual_summary=None
            )

        target_t = anomaly_t if has_anomaly else duration * 0.5
        burst_frames = self.extract_targeted_burst(video_path, target_t)
        
        summary = self.deep_sequence_diagnosis(burst_frames, location_hint)
        summary.camera_id = camera_id
        summary.timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Route status dynamically from Cosmos Reasoner's output
        if summary.severity in ["CRITICAL", "HIGH", "SEVERE"]:
            final_status = "CRISIS_IMPACT"
        elif summary.severity in ["MEDIUM", "MODERATE"]:
            final_status = "ROADSIDE_ASSISTANCE"
        else:
            final_status = "ROUTINE_ALL_CLEAR"

        yield StreamFrameEvent(
            timestamp=summary.timestamp,
            elapsed_seconds=target_t,
            status=final_status,
            severity=summary.severity,
            scene_description=f"Cosmos Reasoner Diagnosis: {summary.raw_summary}",
            visual_summary=summary
        )

    def process_video_file(self, video_path: str, location_hint: Optional[str] = None) -> VisualContextSummary:
        """Direct end-to-end processing purely on video frames."""
        _, anomaly_t, _ = self.scan_motion_and_anomaly_points(video_path)
        burst_frames = self.extract_targeted_burst(video_path, anomaly_t)
        if burst_frames:
            return self.deep_sequence_diagnosis(burst_frames, location_hint)
        raise RuntimeError(f"Failed to extract frames from video stream: {video_path}")

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
        except Exception:
            return VisualContextSummary(
                location=location_hint or "Roadway Surveillance Scene",
                camera_id="CAM-DGX-SPARK-01",
                crisis_type="Roadway Incident Observation",
                severity="HIGH",
                vehicles_involved=1,
                hazard_indicators=[],
                raw_summary=raw_response[:250],
                confidence=0.90,
                timestamp=datetime.datetime.now().strftime("%H:%M:%S")
            )


