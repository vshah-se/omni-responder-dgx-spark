from typing import Dict, Any, List

class TrafficAgent:
    """Orchestrates mock city traffic routing, digital signs, and signal timing."""

    def __init__(self):
        self.active_closures = []

    def dispatch_reroute(self, location: str, isolation_radius_meters: int) -> Dict[str, Any]:
        closure_id = f"ROUTE-BLOCK-{len(self.active_closures) + 101}"
        actions = [
            f"Set Digital Sign Board #12: 'INCIDENT AT {location.upper()} - DETOUR VIA 8TH AVE'",
            f"Adjust signal timing: Extend East/West Green phase on 4th & 6th Ave corridors",
            f"Broadcast autonomous navigation warning zone ({isolation_radius_meters}m radius perimeter)"
        ]
        self.active_closures.append({"id": closure_id, "location": location, "actions": actions})
        return {
            "status": "EXECUTED",
            "closure_id": closure_id,
            "actions_triggered": actions
        }
