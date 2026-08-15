import os
import json
import time
import base64
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Generator
from src.config.settings import settings

@dataclass
class VisualContextSummary:
    """Standardized JSON schema emitted by Cosmos Reasoner VLM for the Orchestrator."""
    location: str
    camera_id: str = "CAM-DGX-SPARK-01"
    crisis_type: str = "Physical Emergency Incident"
    severity: str = "HIGH"  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    vehicles_involved: int = 0
    hazard_indicators: List[str] = field(default_factory=list)
    raw_summary: str = ""
    confidence: float = 0.95
    timestamp: str = "00:14"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StreamFrameEvent:
    """Temporal frame event during live continuous video monitoring."""
    timestamp: str
    elapsed_seconds: int
    status: str
    severity: str
    scene_description: str
    visual_summary: Optional[VisualContextSummary] = None

class VSSPerceptionPipeline:
    """Interfaces directly with NVIDIA Cosmos Reasoner 2 VLM running on Port 30082."""

    SYSTEM_PROMPT = """You are an autonomous Emergency Dispatch Vision Agent running locally on NVIDIA DGX Spark.
Carefully examine the provided video frames of the physical incident.
You must output ONLY a valid JSON object with these exact keys:
{
  "location": "string (physical location hint or intersection)",
  "camera_id": "string",
  "crisis_type": "string (what exact physical event occurred)",
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "vehicles_involved": integer,
  "hazard_indicators": ["list of visible hazards like smoke, fire, leaked liquids, wreckage, placards"],
  "raw_summary": "detailed 2-3 sentence visual summary describing exactly what is seen in the frames",
  "confidence": float (0.0 to 1.0)
}
Do not include markdown or conversational commentary outside the JSON object."""

    def __init__(self, endpoint_url: Optional[str] = None, model_name: str = "nvidia/cosmos-reason2-8b"):
        # Port 30082 is the live Cosmos Reasoner NIM container on DGX Spark
        self.endpoint_url = endpoint_url or os.getenv("COSMOS_VLM_URL", "http://localhost:30082/v1")
        self.model_name = model_name

    def extract_keyframes_base64(self, video_path: str, max_frames: int = 4) -> List[str]:
        """Extracts representative JPEG keyframes from video using ffmpeg or direct binary sampling."""
        frames_b64 = []
        
        # Try extracting clean frames via ffmpeg if available on host
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1/{max(1, max_frames)}",
                "-vframes", str(max_frames),
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = proc.communicate(timeout=10)
            
            # Split MJPEG stream by JPEG markers
            raw_jpegs = out.split(b"\xff\xd8")
            for chunk in raw_jpegs[1:]:
                jpeg_bytes = b"\xff\xd8" + chunk
                if len(jpeg_bytes) > 1000:
                    frames_b64.append(base64.b64encode(jpeg_bytes).decode("utf-8"))
        except Exception:
            pass

        return frames_b64

    def process_video_file(self, video_path: str, location_hint: str = "5th Ave & Market St Intersection") -> VisualContextSummary:
        """Sends real video keyframes to the live Cosmos Reasoner VLM on Port 30082."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # 1. Attempt live query to Cosmos Reasoner NIM on Port 30082
        try:
            return self._query_live_cosmos_vlm(video_path, location_hint)
        except Exception as e:
            # If NIM container is temporarily unreachable, use offline fallback
            print(f"\033[1;33m[WARN] Cosmos VLM on {self.endpoint_url} unreachable ({e}). Using heuristic.\033[0m")
            return self._extract_scenario_heuristic(video_path, location_hint)

    def _query_live_cosmos_vlm(self, video_path: str, location_hint: str) -> VisualContextSummary:
        """Calls OpenAI-compatible /v1/chat/completions on Cosmos Reasoner NIM."""
        url = f"{self.endpoint_url}/chat/completions"
        frames_b64 = self.extract_keyframes_base64(video_path, max_frames=3)

        content_elements = [
            {
                "type": "text",
                "text": f"Incident location: {location_hint}. Analyze these video frames and output the emergency dispatch JSON."
            }
        ]

        if frames_b64:
            for b64 in frames_b64:
                content_elements.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        else:
            # If ffmpeg not present, pass raw video container reference
            with open(video_path, "rb") as f:
                sample_b64 = base64.b64encode(f.read(1024 * 1024 * 2)).decode("utf-8")
            content_elements.append({
                "type": "image_url",
                "image_url": {"url": f"data:video/mp4;base64,{sample_b64}"}
            })

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data["choices"][0]["message"]["content"]
            return self.parse_vlm_json_output(raw_text)

    def stream_video_feed(
        self,
        video_path: str,
        location_hint: str = "Highway 101 Exit 5",
        speed_multiplier: float = 1.0
    ) -> Generator[StreamFrameEvent, None, None]:
        """Simulates live continuous video ingestion from pre-incident normal state to crash occurrence."""
        timeline = [
            (0, "00:00", "NORMAL_MONITORING", "LOW", "Camera feed active. Normal traffic flow at 45 mph. Road clear, 0 hazards."),
            (3, "00:03", "NORMAL_MONITORING", "LOW", "Vehicles maintaining safe following distance. Surface dry, visibility 100%."),
            (7, "00:07", "NORMAL_MONITORING", "LOW", "Traffic passing through intersection normally. Zero anomalies."),
            (10, "00:10", "ANOMALY_DETECTED", "MEDIUM", "Sudden heavy braking detected in lane 2. Rapid vehicle deceleration observed."),
            (12, "00:12", "ANOMALY_DETECTED", "MEDIUM", "Erratic trajectory detected. Impact imminent."),
            (14, "00:14", "CRISIS_IMPACT", "CRITICAL", "CRASH IMPACT DETECTED! Analyzing real video frames on DGX Spark...")
        ]

        final_summary = self.process_video_file(video_path, location_hint)

        for elapsed, ts, status, severity, desc in timeline:
            time.sleep(1.0 / max(speed_multiplier, 0.1))
            if status == "CRISIS_IMPACT":
                final_summary.timestamp = ts
                yield StreamFrameEvent(
                    timestamp=ts,
                    elapsed_seconds=elapsed,
                    status=status,
                    severity=severity,
                    scene_description=f"CRASH IMPACT! {final_summary.raw_summary}",
                    visual_summary=final_summary
                )
            else:
                yield StreamFrameEvent(
                    timestamp=ts,
                    elapsed_seconds=elapsed,
                    status=status,
                    severity=severity,
                    scene_description=desc,
                    visual_summary=None
                )

    def parse_vlm_json_output(self, raw_response: str) -> VisualContextSummary:
        """Parses and validates raw VLM text output into the typed VisualContextSummary schema."""
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return VisualContextSummary(
            location=data.get("location", "Unknown Location"),
            camera_id=data.get("camera_id", "CAM-DGX-SPARK-01"),
            crisis_type=data.get("crisis_type", "Emergency Incident"),
            severity=data.get("severity", "HIGH"),
            vehicles_involved=int(data.get("vehicles_involved", 1)),
            hazard_indicators=list(data.get("hazard_indicators", [])),
            raw_summary=data.get("raw_summary", ""),
            confidence=float(data.get("confidence", 0.95))
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
            confidence=0.90
        )
