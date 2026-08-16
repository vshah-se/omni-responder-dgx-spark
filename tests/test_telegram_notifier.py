import unittest
from src.notifiers.telegram_notifier import TelegramNotifier


def _record(dispatch_code, severity="HIGH", hazmat_status="NO_HAZARDS_DETECTED"):
    return {
        "perception": {
            "location": "Nighttime Urban Highway",
            "camera_id": "CAM-EDGE-TEST",
            "crisis_type": "Vehicle collision blocking active lanes",
            "severity": severity,
            "vehicles_involved": 2,
            "hazard_indicators": ["wreckage", "smoke"],
        },
        "hazmat": {
            "status": hazmat_status,
            "chemical_name": "Gasoline",
            "un_number": "UN1203",
            "isolation_radius_meters": 50,
            "ppe_required": "Level B",
        },
        "traffic": {"status": "EMERGENCY_PERIMETER_LOCKED", "closure_id": "ROUTE-BLOCK-101"},
        "dispatch_report": {
            "cad_id": "CAD-EMG-TEST01",
            "dispatch_code": dispatch_code,
            "target_units": ["Fire Battalion Engine 12", "EMS Paramedic Unit 9"],
        },
    }


class TestTelegramNotifier(unittest.TestCase):
    def setUp(self):
        # Explicit empty credentials: never read the real env/config in unit tests
        self.notifier = TelegramNotifier(bot_token="", chat_id="")
        self.notifier.bot_token = ""
        self.notifier.chat_id = ""

    def test_disabled_without_credentials_never_raises(self):
        result = self.notifier.notify_dispatch(_record("CRITICAL - CODE RED"))
        self.assertEqual(result["status"], "DISABLED")

    def test_code_red_message_includes_units_and_incident(self):
        text = self.notifier.format_message(_record("CRITICAL - CODE RED", hazmat_status="IDENTIFIED"))
        self.assertIn("🚨", text)
        self.assertIn("CAD-EMG-TEST01", text)
        self.assertIn("Simulated call to: Fire Battalion Engine 12, EMS Paramedic Unit 9", text)
        self.assertIn("Gasoline (UN1203)", text)

    def test_code_amber_uses_warning_icon(self):
        text = self.notifier.format_message(_record("HIGH - CODE AMBER"))
        self.assertIn("⚠️", text)
        self.assertNotIn("Hazmat:", text)  # NO_HAZARDS_DETECTED omits the hazmat block

    def test_code_green_is_terse(self):
        text = self.notifier.format_message(_record("ROUTINE - CODE GREEN", severity="LOW"))
        self.assertIn("🟢", text)
        self.assertIn("ALL CLEAR", text)
        self.assertNotIn("Simulated call", text)
        self.assertLess(len(text.splitlines()), 5)


class TestDispatchTag(unittest.TestCase):
    """The tag line is what makes repeated demo runs distinguishable on a phone."""

    def setUp(self):
        self.notifier = TelegramNotifier(bot_token="", chat_id="")

    def test_tag_present_on_every_tier(self):
        for code in ("CRITICAL - CODE RED", "HIGH - CODE AMBER", "ROUTINE - CODE GREEN"):
            rec = _record(code)
            rec["source_video"] = "aic21_80.mp4"
            text = self.notifier.format_message(rec)
            self.assertIn("aic21_80.mp4", text, f"missing video name for {code}")
            self.assertIn(" · ", text, f"missing tag separator for {code}")

    def test_tag_id_is_unique_per_message(self):
        rec = _record("CRITICAL - CODE RED")
        first = self.notifier.format_message(rec).splitlines()[1]
        second = self.notifier.format_message(rec).splitlines()[1]
        self.assertNotEqual(first, second)

    def test_tag_survives_missing_source_video(self):
        text = self.notifier.format_message(_record("HIGH - CODE AMBER"))
        self.assertIn("unknown-source", text)


if __name__ == "__main__":
    unittest.main()
