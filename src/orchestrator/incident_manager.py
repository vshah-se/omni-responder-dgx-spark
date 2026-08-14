from typing import Dict, Any
from src.perception.vss_pipeline import VSSPerceptionPipeline
from src.agents.hazmat_agent import HazmatAgent
from src.agents.traffic_agent import TrafficAgent
from src.agents.comms_agent import CommsAgent

class IncidentOrchestrator:
    """Master Orchestrator running locally on DGX Spark (70B LLM / agent controller)."""

    def __init__(self):
        self.perception = VSSPerceptionPipeline()
        self.hazmat_agent = HazmatAgent()
        self.traffic_agent = TrafficAgent()
        self.comms_agent = CommsAgent()

    def process_incident(self, feed_id: str) -> Dict[str, Any]:
        # 1. Perception phase (Local VSS & VLM)
        visual_summary = self.perception.process_feed(feed_id)
        
        # 2. Hazmat sub-agent call
        hazmat_result = self.hazmat_agent.analyze_hazard(visual_summary.hazard_indicators)
        
        # 3. Traffic routing sub-agent call
        traffic_result = self.traffic_agent.dispatch_reroute(
            location=visual_summary.location,
            isolation_radius_meters=hazmat_result.get("isolation_radius_meters", 100)
        )
        
        # 4. First responder communications synthesis
        dispatch_report = self.comms_agent.generate_dispatch_brief(
            perception_data={
                "location": visual_summary.location,
                "crisis_type": visual_summary.crisis_type,
                "vehicles_involved": visual_summary.vehicles_involved,
                "raw_summary": visual_summary.raw_summary
            },
            hazmat_data=hazmat_result,
            traffic_data=traffic_result
        )

        return {
            "perception": visual_summary,
            "hazmat": hazmat_result,
            "traffic": traffic_result,
            "dispatch_report": dispatch_report
        }
