import datetime
from typing import Dict, Any, List

class CommsAgent:
    """Synthesizes high-priority CAD (Computer-Aided Dispatch) reports for first responders."""

    def generate_dispatch_brief(self, perception_data: Dict[str, Any], hazmat_data: Dict[str, Any], traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        is_identified = hazmat_data.get("status") == "IDENTIFIED"
        priority = "CRITICAL - CODE RED" if is_identified else "HIGH - CODE AMBER"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        cad_incident_id = f"CAD-EMG-{datetime.datetime.now().strftime('%H%M%S')}"

        units = [
            "Hazmat Response Unit 7",
            "Fire Battalion Engine 12",
            "Heavy Rescue Squad 3",
            "EMS Paramedic Unit 9",
            "Metro Traffic Police Unit 22"
        ]

        summary_text = (
            f"**[CAD DISPATCH BRIEF - {priority}]**\n"
            f"• Incident ID: {cad_incident_id} | Timestamp: {timestamp}\n"
            f"• Location: {perception_data.get('location')} (Camera ID: {perception_data.get('camera_id', 'N/A')})\n"
            f"• Crisis Type: {perception_data.get('crisis_type')}\n"
            f"• Hazard Identified: {hazmat_data.get('chemical_name')} ({hazmat_data.get('un_number')}) - {hazmat_data.get('hazard_class')}\n"
            f"• Protective Action Zone: {hazmat_data.get('isolation_radius_meters')}m initial isolation | Day: {hazmat_data.get('day_protection_km')}km | Night: {hazmat_data.get('night_protection_km')}km\n"
            f"• Responder PPE Required: {hazmat_data.get('ppe_required')}\n"
            f"• Containment Guidance: {hazmat_data.get('fire_response')}\n"
            f"• Medical First Aid: {hazmat_data.get('first_aid')}\n"
            f"• Traffic Controls: Detour {traffic_data.get('closure_id')} active with {len(traffic_data.get('actions_triggered', []))} measures."
        )

        return {
            "cad_id": cad_incident_id,
            "timestamp": timestamp,
            "dispatch_code": priority,
            "incident_type": perception_data.get("crisis_type"),
            "target_units": units,
            "briefing_summary": summary_text
        }
