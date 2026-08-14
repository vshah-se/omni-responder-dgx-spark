from typing import Dict, Any

class CommsAgent:
    """Synthesizes high-priority emergency dispatch reports for first responders."""

    def generate_dispatch_brief(self, perception_data: Dict[str, Any], hazmat_data: Dict[str, Any], traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        priority = "CRITICAL - CODE RED" if hazmat_data.get("status") == "IDENTIFIED" else "HIGH - CODE AMBER"
        
        return {
            "dispatch_code": priority,
            "incident_type": perception_data.get("crisis_type"),
            "target_units": ["Hazmat Engine 4", "EMS Battalion 2", "Traffic Police Unit 10"],
            "briefing_summary": (
                f"**INCIDENT AT {perception_data.get('location')}**\n"
                f"- Vehicles: {perception_data.get('vehicles_involved')}\n"
                f"- Hazard: {hazmat_data.get('chemical_name', 'Unknown')} (UN: {hazmat_data.get('un_number', 'N/A')})\n"
                f"- Required PPE: {hazmat_data.get('ppe_required')}\n"
                f"- Action: {hazmat_data.get('recommended_action')}\n"
                f"- Perimeter & Traffic: {len(traffic_data.get('actions_triggered', []))} automated actions active."
            )
        }
