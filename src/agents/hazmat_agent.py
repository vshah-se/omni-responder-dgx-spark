import json
import os
from typing import Dict, Any, List
from src.config.settings import settings

class HazmatAgent:
    """Identifies chemical threats using local Emergency Response Guidebook (ERG) database."""

    def __init__(self, db_path: str = settings.hazmat_db_path):
        self.db_path = db_path
        self.db = self._load_database()

    def _load_database(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return []

    def analyze_hazard(self, visual_indicators: List[str]) -> Dict[str, Any]:
        """Cross-references visual clues with local chemical hazard database."""
        for entry in self.db:
            indicators = entry.get("visual_indicators", [])
            for ind in indicators:
                for observed in visual_indicators:
                    obs_words = set(observed.lower().split())
                    ind_words = set(ind.lower().split())
                    if len(obs_words.intersection(ind_words)) >= 1:
                        return {
                            "status": "IDENTIFIED",
                            "un_number": entry["un_number"],
                            "chemical_name": entry["chemical_name"],
                            "hazard_class": entry["hazard_class"],
                            "isolation_radius_meters": entry["initial_isolation_distance_meters"],
                            "day_protection_km": entry.get("day_protection_distance_km", 0.5),
                            "night_protection_km": entry.get("night_protection_distance_km", 1.0),
                            "ppe_required": entry["ppe_required"],
                            "fire_response": entry["fire_response"],
                            "first_aid": entry.get("first_aid", "Move victims upwind and administer oxygen.")
                        }

        return {
            "status": "UNKNOWN_HAZARD",
            "un_number": "UN-UNKNOWN",
            "chemical_name": "Unidentified Hazardous Substance",
            "hazard_class": "Class 9 (Precautionary Hazard)",
            "isolation_radius_meters": 100,
            "day_protection_km": 0.5,
            "night_protection_km": 1.0,
            "ppe_required": "Level A (Full encapsulating SCBA precautionary)",
            "fire_response": "Approach with caution from upwind. Maintain 100m standoff perimeter.",
            "first_aid": "Evacuate upwind immediately."
        }
