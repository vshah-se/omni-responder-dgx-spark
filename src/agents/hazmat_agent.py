import json
import os
from typing import Dict, Any, Optional
from src.config.settings import settings

class HazmatAgent:
    """Identifies chemical threats using local Emergency Response database."""

    def __init__(self, db_path: str = settings.hazmat_db_path):
        self.db_path = db_path
        self.db = self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return []

    def analyze_hazard(self, visual_indicators: list[str]) -> Dict[str, Any]:
        """Cross-references visual clues with local chemical hazard database."""
        for entry in self.db:
            for keyword in entry.get("visual_indicators", []):
                for ind in visual_indicators:
                    if any(word in ind.lower() for word in keyword.lower().split()):
                        return {
                            "status": "IDENTIFIED",
                            "un_number": entry["un_number"],
                            "chemical_name": entry["chemical_name"],
                            "hazard_class": entry["hazard_class"],
                            "isolation_radius_meters": entry["initial_isolation_distance_meters"],
                            "ppe_required": entry["ppe_required"],
                            "recommended_action": entry["fire_response"]
                        }

        return {
            "status": "UNKNOWN_HAZARD",
            "isolation_radius_meters": 100,
            "ppe_required": "Level A (Full encapsulating SCBA precautionary)",
            "recommended_action": "Approach with caution from upwind. Evacuate 100m radius."
        }
