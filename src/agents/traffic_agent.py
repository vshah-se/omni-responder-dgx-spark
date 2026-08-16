from typing import Dict, Any, List

class TrafficAgent:
    """Orchestrates mock city traffic routing, digital signs, and signal timing."""

    def __init__(self):
        self.active_closures = []
        self._MAX_CLOSURE_HISTORY = 20  # prevent unbounded growth across a long session

    def dispatch_reroute(self, location: str, isolation_radius_meters: int, severity: str = "LOW") -> Dict[str, Any]:
        severity_upper = severity.upper()
        sign_location = location.upper()

        if severity_upper in ["LOW", "NORMAL"]:
            # Routine traffic flow: No closures, no emergency reroutes
            return {
                "status": "ALL_CLEAR_MONITORING",
                "closure_id": "TRAFFIC-FLOW-NORMAL",
                "isolation_radius_meters": 0,
                "actions_triggered": [
                    f"Digital VMS: 'ALL CLEAR - NORMAL TRAFFIC FLOW AT {sign_location}'",
                    "Traffic Signals: Standard timed cycle active",
                    "Navigation Advisory: 0m restriction (Normal operations)"
                ]
            }

        if severity_upper in ["MEDIUM", "MODERATE", "CONGESTION"] and isolation_radius_meters <= 50:
            # Roadside breakdown / minor congestion: Adjust signals and post caution, no roadblocks
            return {
                "status": "CONGESTION_MITIGATION",
                "closure_id": f"FLOW-OPT-{len(self.active_closures) + 101}",
                "isolation_radius_meters": 0,
                "actions_triggered": [
                    f"Digital VMS: 'CAUTION: SLOW TRAFFIC / VEHICLE ON SHOULDER AT {sign_location}'",
                    "Traffic Signals: Adaptive green phase extended +15s",
                    "Navigation Advisory: Suggest alternate arterials to prevent backup"
                ]
            }

        # Emergency road closures for real critical accidents and hazmat breaches.
        # A geofenced perimeter is only claimed when there is a hazmat isolation
        # distance to enforce. With no identified material the radius is 0, and
        # announcing a "0m isolation zone" is self-contradictory — responders
        # still get lane closure and priority signals, but no perimeter lock.
        has_isolation_zone = isolation_radius_meters > 0
        closure_id = f"ROUTE-BLOCK-{len(self.active_closures) + 101}"
        actions = [
            f"Digital VMS: 'INCIDENT AT {sign_location} - AVOID AREA - USE DETOUR'",
            f"Traffic Signals: Green wave priority corridor enabled for emergency responders",
            (f"Navigation Advisory: Broadcast geofenced perimeter lock ({isolation_radius_meters}m isolation zone)"
             if has_isolation_zone else
             "Navigation Advisory: Lane closure and responder access only (no hazmat isolation zone)")
        ]

        closure_record = {
            "id": closure_id,
            "location": location,
            "isolation_radius_meters": isolation_radius_meters,
            "actions": actions
        }
        self.active_closures.append(closure_record)
        # Trim the tail so the list never grows beyond MAX_CLOSURE_HISTORY entries
        if len(self.active_closures) > self._MAX_CLOSURE_HISTORY:
            self.active_closures = self.active_closures[-self._MAX_CLOSURE_HISTORY:]

        return {
            "status": "EMERGENCY_PERIMETER_LOCKED" if has_isolation_zone else "EMERGENCY_LANE_CLOSURE",
            "closure_id": closure_id,
            "isolation_radius_meters": isolation_radius_meters,
            "actions_triggered": actions
        }
