import re
import json
import os
from typing import Dict, Any, List
from src.config.settings import settings

class HazmatAgent:
    """Identifies chemical threats using local Emergency Response Guidebook (ERG) database."""

    # Explicit chemical/fire/hazard indicators that MUST be present to consider a hazmat match
    HAZARD_KEYWORDS = {
        "gasoline", "fuel", "diesel", "chlorine", "acid", "sulfuric", "ammonia",
        "spill", "leak", "plume", "flame", "flames", "fire", "burning", "smoke",
        "chemical", "toxic", "corrosive", "vapor", "vapors", "explosion", "placard",
        "sheen", "fumes", "ignited", "igniting", "hazardous", "gas", "cloud",
        "greenish", "acrid", "pungent", "oily", "corrosion"
    }

    # Words that flip the meaning of the hazard keyword that follows them
    NEGATION_MARKERS = ("no ", "not ", "without ", "absent ", "none ", "n't ", "non-", "clear of ")

    def _hazard_keyword_present(self, text: str) -> bool:
        """Returns True only if a hazard keyword appears in a non-negated sentence.

        A flat set intersection reads 'no fuel leak detected' as a hazmat event.
        Scoping per sentence prevents that: a negation marker must not immediately
        precede the keyword within the same sentence fragment.
        """
        for sentence in re.split(r"[.;,]", text.lower()):
            for kw in self.HAZARD_KEYWORDS:
                if kw in sentence:
                    # Check whether any negation marker appears before the keyword
                    kw_pos = sentence.index(kw)
                    preceding = sentence[:kw_pos]
                    if not any(marker in preceding for marker in self.NEGATION_MARKERS):
                        return True
        return False

    def __init__(self, db_path: str = settings.hazmat_db_path):
        self.db_path = db_path
        self.db = self._load_database()

    def _load_database(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return []

    def analyze_hazard(self, visual_indicators: List[str], severity: str = "HIGH") -> Dict[str, Any]:
        """Cross-references visual clues with local chemical hazard database with strict keyword verification."""
        # 1. If scene is low/normal or no indicators are reported, return all-clear
        if not visual_indicators or severity in ["LOW", "NORMAL"]:
            return self._all_clear_response()

        # 2. Check if ANY actual hazard/chemical keyword is present WITHOUT negation
        raw_text = " ".join(visual_indicators).lower()
        if not self._hazard_keyword_present(raw_text):
            return self._all_clear_response()

        # Also build a word set for the database overlap check below
        words = set(raw_text.replace(",", " ").replace(".", " ").replace("-", " ").split())

        # 3. Match against specific chemical records in ERG database
        for entry in self.db:
            db_indicators = entry.get("visual_indicators", [])
            for db_ind in db_indicators:
                db_words = set(db_ind.lower().replace("-", " ").split())
                # Require at least 1 significant chemical keyword match
                overlap = words.intersection(db_words)
                if len(overlap) >= 1 and any(w in self.HAZARD_KEYWORDS for w in overlap):
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

        # 4. If genuine fire/smoke/spill was detected but chemical is unconfirmed in critical incident
        if severity in ["CRITICAL", "HIGH", "SEVERE"] and any(kw in words for kw in ["smoke", "fire", "flame", "spill", "leak", "vapor", "fumes"]):
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

        return self._all_clear_response()

    def _all_clear_response(self) -> Dict[str, Any]:
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
