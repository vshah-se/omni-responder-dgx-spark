import os
import json
from typing import Dict, Any, Optional, Union, Generator
from src.perception.vss_pipeline import VSSPerceptionPipeline, VisualContextSummary, StreamFrameEvent
from src.agents.hazmat_agent import HazmatAgent
from src.agents.traffic_agent import TrafficAgent
from src.agents.comms_agent import CommsAgent
from src.config.settings import settings

class IncidentOrchestrator:
    """Master Multi-Agent Incident Orchestrator powered by NVIDIA Nemotron 9B FP8 on DGX Spark."""

    def __init__(self, nim_endpoint_url: Optional[str] = None):
        self.nim_endpoint_url = nim_endpoint_url or settings.nim_endpoint_url
        self.perception = VSSPerceptionPipeline()
        self.hazmat_agent = HazmatAgent()
        self.traffic_agent = TrafficAgent()
        self.comms_agent = CommsAgent()

    def process_incident(self, video_input: Union[str, VisualContextSummary], location_hint: Optional[str] = None) -> Dict[str, Any]:
        """Main end-to-end incident dispatch loop."""
        # 1. Perception Phase (NVIDIA VSS / Cosmos Reasoner VLM)
        if isinstance(video_input, VisualContextSummary):
            visual_summary = video_input
        else:
            visual_summary = self.perception.process_video_file(video_input, location_hint=location_hint)

        effective_location = visual_summary.location

        # 2. Hazmat Sub-Agent (Local ERG 2024 Knowledge Base Lookup)
        hazmat_result = self.hazmat_agent.analyze_hazard(
            visual_indicators=visual_summary.hazard_indicators,
            severity=visual_summary.severity
        )

        # 3. Traffic Sub-Agent (Digital Message Signs & Perimeter Isolation)
        traffic_result = self.traffic_agent.dispatch_reroute(
            location=effective_location,
            isolation_radius_meters=hazmat_result.get("isolation_radius_meters", 0),
            severity=visual_summary.severity
        )

        # 4. Comms Sub-Agent (911 CAD Dispatch Briefing Synthesis)
        dispatch_report = self.comms_agent.generate_dispatch_brief(
            perception_data=visual_summary.to_dict(),
            hazmat_data=hazmat_result,
            traffic_data=traffic_result
        )

        # 5. Assemble Consolidated Incident Record
        return {
            "incident_id": dispatch_report["cad_id"],
            "timestamp": dispatch_report["timestamp"],
            "platform": {
                "hardware": settings.target_hardware,
                "orchestrator_model": settings.orchestrator_llm,
                "vision_model": settings.vlm_model,
                "privacy_guarantee": "100% On-Premise DGX Spark (0% Cloud Video Egress)"
            },
            "perception": visual_summary.to_dict(),
            "hazmat": hazmat_result,
            "traffic": traffic_result,
            "dispatch_report": dispatch_report
        }

    def stream_incident(
        self,
        video_path: str,
        location_hint: Optional[str] = None,
        speed_multiplier: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams real-time edge monitoring with live wall-clock timestamps and real video duration."""
        for event in self.perception.stream_video_feed(video_path, location_hint=location_hint, speed_multiplier=speed_multiplier):
            if event.status in ["CRISIS_IMPACT", "ROADSIDE_ASSISTANCE", "ROUTINE_ALL_CLEAR"] and event.visual_summary:
                full_dispatch = self.process_incident(event.visual_summary, location_hint=location_hint)
                yield {
                    "event_type": "DISPATCH_SUMMARY_TRIGGERED",
                    "timestamp": event.timestamp,
                    "elapsed_seconds": event.elapsed_seconds,
                    "status": event.status,
                    "severity": event.severity,
                    "scene_description": event.scene_description,
                    "incident_record": full_dispatch
                }
            else:
                yield {
                    "event_type": "EDGE_MONITORING_FRAME",
                    "timestamp": event.timestamp,
                    "elapsed_seconds": event.elapsed_seconds,
                    "status": event.status,
                    "severity": event.severity,
                    "scene_description": event.scene_description,
                    "incident_record": None
                }
