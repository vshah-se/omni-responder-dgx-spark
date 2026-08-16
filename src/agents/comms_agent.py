import datetime
from typing import Dict, Any, List

class CommsAgent:
    """Synthesizes high-priority CAD (Computer-Aided Dispatch) reports for emergency services."""

    def generate_dispatch_brief(self, perception_data: Dict[str, Any], hazmat_data: Dict[str, Any], traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        severity = str(perception_data.get("severity", "LOW")).upper()
        hazmat_status = hazmat_data.get("status", "NO_HAZARDS_DETECTED")
        crisis_type = perception_data.get("crisis_type", "Normal Traffic Flow")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cad_incident_id = f"CAD-EMG-{datetime.datetime.now().strftime('%H%M%S')}"

        is_roadside = any(term in crisis_type.lower() for term in ["roadside", "shoulder", "towing", "disabled"])

        vehicles = int(perception_data.get("vehicles_involved", 0))
        has_confirmed_hazmat = hazmat_status == "IDENTIFIED" and hazmat_data.get("isolation_radius_meters", 0) > 0

        # Tier 1: Confirmed collision/spill with physical evidence — CODE RED
        # CRITICAL/SEVERE is always Tier 1. HIGH only escalates if there are
        # actual vehicles involved or a confirmed hazmat material to point at:
        # a bare "HIGH" from the VLM without corroborating evidence must not
        # auto-dispatch heavy rescue and paramedics.
        is_tier1 = (
            severity in ["CRITICAL", "SEVERE"]
            or (severity == "HIGH" and (vehicles > 0 or has_confirmed_hazmat))
            or (has_confirmed_hazmat and not is_roadside)
        )

        if is_tier1 and not is_roadside:
            priority = "CRITICAL - CODE RED"
            units = ["Hazmat Response Unit 7", "Fire Battalion Engine 12", "Heavy Rescue Squad 3", "EMS Paramedic Unit 9", "Metro Traffic Police Unit 22"]
            status_desc = "EMERGENCY UNITS DISPATCHED (CODE RED)"

        # Tier 2: Roadside assistance / minor congestion / bare HIGH with no evidence — CODE AMBER
        elif is_roadside or severity in ["HIGH", "MEDIUM", "MODERATE"]:
            priority = "HIGH - CODE AMBER"
            units = ["Highway Safety Patrol Unit 4", "Roadway Tow & Recovery Squad"]
            status_desc = "ROADSIDE SAFETY & TOW SUPPORT ACTIVE (CODE AMBER)"

        # Tier 3: All clear / normal flow
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
            f"• Dispatched Response: {status_desc}"
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
