import json
import os
from typing import Dict, Any, List
from src.config.settings import settings

class HazmatAgent:
    """Identifies chemical threats using local Emergency Response Guidebook (ERG) database."""

    # Words that should NEVER trigger a chemical alert by themselves
    STOP_WORDS = {"vehicle", "car", "truck", "road", "highway", "shoulder", "lane", "traffic", "operation", "present", "none", "clear"}

    def __init__(self, db_path: str = settings.hazmat_db_path):
        self.db_path = db_path
        self.db = self._load_database()

    def _load_database(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return []

    def analyze_hazard(self, visual_indicators: List[str], severity: str = "HIGH") -> Dict[str, Any]:
        """Cross-references visual clues with local chemical hazard database with strict phrase matching."""
        # If no indicators are reported or scene is routine/low/roadside without chemical leaks
        if not visual_indicators or severity in ["LOW", "NORMAL"]:
            return {
                "status": "NO_HAZARDS_DETECTED",
                "un_number": "NONE",
                "chemical_name": "None (No Hazardous Materials Observed)",
                "hazard_class": "N/A",
                "isolation_radius_meters": 0,
                "day_protection_km": 0.0,
                "night_protection_km": 0.0,
                "ppe_required": "None Required (Standard Operations)",
                "fire_response": "Standard road operations.",
                "first_aid": "None required."
            }

        # Filter out generic stop words
        cleaned_indicators = []
        for ind in visual_indicators:
            words = [w.lower() for w in ind.split() if w.lower() not in self.STOP_WORDS]
            if words:
                cleaned_indicators.append(" ".join(words))

        if not cleaned_indicators:
            return {
                "status": "NO_HAZARDS_DETECTED",
                "un_number": "NONE",
                "chemical_name": "None (No Hazardous Materials Observed)",
                "hazard_class": "N/A",
                "isolation_radius_meters": 0,
                "day_protection_km": 0.0,
                "night_protection_km": 0.0,
                "ppe_required": "None Required (Standard Operations)",
                "fire_response": "Standard road operations.",
                "first_aid": "None required."
            }

        # Match against specific chemical keywords (e.g. 'fuel spill', 'gasoline leak', 'chlorine', 'ammonia', 'acid')
        for entry in self.db:
            db_indicators = entry.get("visual_indicators", [])
            for db_ind in db_indicators:
                db_ind_lower = db_ind.lower()
                for obs in cleaned_indicators:
                    obs_lower = obs.lower()
                    # Require strong substring or multi-word match
                    if obs_lower in db_ind_lower or db_ind_lower in obs_lower:
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

        # Only if severe collision with verified smoke/flames/unidentified fluids
        if severity in ["CRITICAL", "HIGH", "SEVERE"]:
            return {
                "status": "SUSPICIOUS_HAZARD",
                "un_number": "UN-UNKNOWN",
                "chemical_name": "Unidentified Vapor / Fluid Hazard",
                "hazard_class": "Class 9 (Precautionary Hazard)",
                "isolation_radius_meters": 50,
                "day_protection_km": 0.3,
                "night_protection_km": 0.5,
                "ppe_required": "Level B chemical protective clothing with SCBA",
                "fire_response": "Approach with caution from upwind. Maintain 50m standoff perimeter.",
                "first_aid": "Evacuate upwind if respiratory distress observed."
            }

        return {
            "status": "NO_HAZARDS_DETECTED",
            "un_number": "NONE",
            "chemical_name": "None (No Hazardous Materials Observed)",
            "hazard_class": "N/A",
            "isolation_radius_meters": 0,
            "day_protection_km": 0.0,
            "night_protection_km": 0.0,
            "ppe_required": "None Required (Standard Operations)",
            "fire_response": "Standard road operations.",
            "first_aid": "None required."
        }
