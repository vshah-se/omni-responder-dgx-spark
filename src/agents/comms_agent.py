import datetime
from typing import Dict, Any, List

class CommsAgent:
    """Synthesizes high-priority CAD (Computer-Aided Dispatch) reports for emergency services."""

    def generate_dispatch_brief(self, perception_data: Dict[str, Any], hazmat_data: Dict[str, Any], traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        severity = str(perception_data.get("severity", "LOW")).upper()
        hazmat_status = hazmat_data.get("status", "NO_HAZARDS_DETECTED")
        crisis_type = perception_data.get("crisis_type", "Normal Traffic Flow")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        cad_incident_id = f"CAD-EMG-{datetime.datetime.now().strftime('%H%M%S')}"

        if (hazmat_status == "IDENTIFIED" or severity in ["CRITICAL", "SEVERE"]) and "All Clear" not in crisis_type and severity != "LOW":
            priority = "CRITICAL - CODE RED"
            units = ["Hazmat Response Unit 7", "Fire Battalion Engine 12", "Heavy Rescue Squad 3", "EMS Paramedic Unit 9", "Metro Traffic Police Unit 22"]
            status_desc = "EMERGENCY UNITS DISPATCHED (CODE RED)"
        elif (severity in ["HIGH", "MEDIUM"] or hazmat_status == "SUSPICIOUS_HAZARD") and severity != "LOW":
            priority = "HIGH - CODE AMBER"
            units = ["Metro Traffic Police Unit 10", "Roadway Service Patrol"]
            status_desc = "SAFETY & TRAFFIC CONTROL ACTIVE"
        else:
            priority = "ROUTINE - CODE GREEN"
            units = ["None Required (Autonomous Edge Monitoring Active)"]
            status_desc = "ALL CLEAR - NO EMERGENCY DISPATCH REQUIRED"

        summary_text = (
            f"**[CAD DISPATCH BRIEF - {priority}]**\n"
            f"• Incident ID: {cad_incident_id} | Timestamp: {timestamp}\n"
            f"• Location: {perception_data.get('location')} (Camera ID: {perception_data.get('camera_id', 'N/A')})\n"
            f"• Scene Assessment: {crisis_type} (Severity: {severity})\n"
            f"• Hazard Status: {hazmat_data.get('chemical_name')} ({hazmat_data.get('un_number')})\n"
            f"• Protective Action Zone: {hazmat_data.get('isolation_radius_meters')}m standoff | PPE: {hazmat_data.get('ppe_required')}\n"
            f"• Traffic Status: {traffic_data.get('status')} via {traffic_data.get('closure_id')}\n"
            f"• Units Status: {status_desc}"
        )

        return {
            "cad_id": cad_incident_id,
            "timestamp": timestamp,
            "dispatch_code": priority,
            "status_description": status_desc,
            "incident_type": crisis_type,
            "target_units": units,
            "briefing_summary": summary_text
        }
