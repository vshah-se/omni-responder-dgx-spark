from typing import Dict, Any, List

class TrafficAgent:
    """Orchestrates mock city traffic routing, digital signs, and signal timing."""

    def __init__(self):
        self.active_closures = []

    def dispatch_reroute(self, location: str, isolation_radius_meters: int, severity: str = "HIGH") -> Dict[str, Any]:
        if isolation_radius_meters == 0 and severity in ["LOW", "NORMAL"]:
            # Routine traffic management (No emergency roadblock)
            return {
                "status": "MONITORING_ACTIVE",
                "closure_id": "TRAFFIC-FLOW-NORMAL",
                "isolation_radius_meters": 0,
                "actions_triggered": [
                    f"VMS Board: 'TRAFFIC FLOW NORMAL AT {location.upper()}'",
                    "Signal Timing: Adaptive signal cycle active based on live queue volume",
                    "Navigation Advisory: Clear route (0m restriction)"
                ]
            }

        if severity in ["MEDIUM", "CONGESTION"] and isolation_radius_meters <= 50:
            # Congestion management without hard closures
            return {
                "status": "CONGESTION_MITIGATION",
                "closure_id": f"FLOW-OPT-{len(self.active_closures) + 101}",
                "isolation_radius_meters": 0,
                "actions_triggered": [
                    f"VMS Board: 'SLOW TRAFFIC AT {location.upper()} - EXPECT MINOR DELAYS'",
                    "Signal Timing: Extend green phase +15s to clear bottleneck corridor",
                    "Navigation Advisory: Suggest alternate arterials to prevent backup"
                ]
            }

        # Full emergency perimeter lock for severe accidents and hazmat breaches
        closure_id = f"ROUTE-BLOCK-{len(self.active_closures) + 101}"
        actions = [
            f"VMS Board #104: 'INCIDENT AT {location.upper()} - AVOID AREA - USE DETOUR'",
            f"Traffic Signals: Green wave priority corridor enabled for emergency responders",
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
