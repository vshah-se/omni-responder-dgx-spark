from typing import Dict, Any, List

class TrafficAgent:
    """Orchestrates mock city traffic routing, digital signs, and signal timing."""

    def __init__(self):
        self.active_closures = []

    def dispatch_reroute(self, location: str, isolation_radius_meters: int) -> Dict[str, Any]:
        closure_id = f"ROUTE-BLOCK-{len(self.active_closures) + 101}"

        actions = [
            f"VMS Board #104: 'INCIDENT AT {location.upper()} - AVOID AREA - USE DETOUR'",
            f"Traffic Signals: Green wave priority corridor enabled on outer arterial routes",
            f"Navigation Advisory: Broadcast geofenced perimeter lock ({isolation_radius_meters}m isolation zone)"
        ]

        closure_record = {
            "id": closure_id,
            "location": location,
            "isolation_radius_meters": isolation_radius_meters,
            "actions": actions
        }
        self.active_closures.append(closure_record)

        return {
            "status": "EXECUTED",
            "closure_id": closure_id,
            "isolation_radius_meters": isolation_radius_meters,
            "actions_triggered": actions
        }
