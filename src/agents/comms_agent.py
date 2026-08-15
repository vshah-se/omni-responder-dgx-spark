import datetime
from typing import Dict, Any, List

class CommsAgent:
    """Synthesizes high-priority CAD (Computer-Aided Dispatch) reports for emergency services."""

    def generate_dispatch_brief(self, perception_data: Dict[str, Any], hazmat_data: Dict[str, Any], traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        severity = perception_data.get("severity", "LOW").upper()
        hazmat_status = hazmat_data.get("status", "NO_HAZARDS_DETECTED")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        cad_incident_id = f"CAD-EMG-{datetime.datetime.now().strftime('%H%M%S')}"

        if hazmat_status == "IDENTIFIED" or severity in ["CRITICAL", "SEVERE"]:
            priority = "CRITICAL - CODE RED"
            units = ["Hazmat Response Unit 7", "Fire Battalion Engine 12", "Heavy Rescue Squad 3", "EMS Paramedic Unit 9", "Metro Traffic Police Unit 22"]
            status_desc = "EMERGENCY DISPATCH TRIGGERED"
        elif severity in ["HIGH", "MEDIUM"] or hazmat_status == "SUSPICIOUS_HAZARD":
            priority = "HIGH - CODE AMBER"
            units = ["Metro Traffic Police Unit 10", "Roadway Tow Service Squad", "EMS Patrol Unit 4"]
            status_desc = "TRAFFIC & SAFETY RESPONSE ACTIVE"
        else:
            priority = "ROUTINE - CODE GREEN"
            units = ["Traffic Management Center (Automated Monitoring)"]
            status_desc = "NO EMERGENCY UNITS REQUIRED"

        summary_text = (
            f"**[CAD DISPATCH BRIEF - {priority}]**\n"
            f"• Incident ID: {cad_incident_id} | Timestamp: {timestamp}\n"
            f"• Location: {perception_data.get('location')} (Camera ID: {perception_data.get('camera_id', 'N/A')})\n"
            f"• Classification: {perception_data.get('crisis_type')} (Severity: {severity})\n"
            f"• Hazard Status: {hazmat_data.get('chemical_name')} ({hazmat_data.get('un_number')})\n"
            f"• Protective Action Zone: {hazmat_data.get('isolation_radius_meters')}m standoff | PPE: {hazmat_data.get('ppe_required')}\n"
            f"• Containment Guidance: {hazmat_data.get('fire_response')}\n"
            f"• Medical First Aid: {hazmat_data.get('first_aid')}\n"
            f"• Traffic Status: {traffic_data.get('status')} via {traffic_data.get('closure_id')} ({len(traffic_data.get('actions_triggered', []))} active actions)."
        )

        return {
            "cad_id": cad_incident_id,
            "timestamp": timestamp,
            "dispatch_code": priority,
            "status_description": status_desc,
            "incident_type": perception_data.get("crisis_type"),
            "target_units": units,
            "briefing_summary": summary_text
        }
