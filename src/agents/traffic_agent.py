from typing import Dict, Any, List

class TrafficAgent:
    """Orchestrates mock city traffic routing, digital signs, and signal timing."""

    def __init__(self):
        self.active_closures = []

    def dispatch_reroute(self, location: str, isolation_radius_meters: int, severity: str = "LOW") -> Dict[str, Any]:
        severity_upper = severity.upper()

        if isolation_radius_meters == 0 or severity_upper in ["LOW", "NORMAL"]:
            # Routine traffic flow: No closures, no emergency reroutes
            return {
                "status": "ALL_CLEAR_MONITORING",
                "closure_id": "TRAFFIC-FLOW-NORMAL",
                "isolation_radius_meters": 0,
                "actions_triggered": [
                    f"VMS Board: 'ALL CLEAR - NORMAL TRAFFIC FLOW AT {location.upper()}'",
                    "Traffic Signals: Standard timed cycle active",
                    "Navigation Advisory: 0m restriction (Normal operations)"
                ]
            }

        if severity_upper in ["MEDIUM", "CONGESTION"] and isolation_radius_meters <= 50:
            # Minor congestion mitigation: Adjust signals, no roadblocks
            return {
                "status": "CONGESTION_MITIGATION",
                "closure_id": f"FLOW-OPT-{len(self.active_closures) + 101}",
                "isolation_radius_meters": 0,
                "actions_triggered": [
                    f"VMS Board: 'SLOW TRAFFIC AT {location.upper()} - EXPECT MINOR DELAYS'",
                    "Traffic Signals: Adaptive green phase extended +15s",
                    "Navigation Advisory: Suggest alternate arterials to prevent backup"
                ]
            }

        # Emergency road closures for real critical accidents and hazmat breaches
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
            "status": "EMERGENCY_PERIMETER_LOCKED",
            "closure_id": closure_id,
            "isolation_radius_meters": isolation_radius_meters,
            "actions_triggered": actions
        }
