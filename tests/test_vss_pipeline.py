import os
import re
import json
from src.perception.vss_pipeline import VSSPerceptionPipeline, VisualContextSummary

def test_vlm_json_parser():
    pipeline = VSSPerceptionPipeline()
    sample_vlm_raw = """
    ```json
    {
      "location": "Urban Commercial Intersection",
      "camera_id": "CAM-402",
      "crisis_type": "Vehicle Collision",
      "severity": "CRITICAL",
      "vehicles_involved": 2,
      "hazard_indicators": ["vehicle wreckage", "debris on roadway"],
      "raw_summary": "Two-vehicle collision between yellow sedan and red hatchback resulting in localized wreckage.",
      "confidence": 0.98
    }
    ```
    """
    summary = pipeline.parse_vlm_json_output(sample_vlm_raw)
    assert isinstance(summary, VisualContextSummary)
    assert summary.location == "Urban Commercial Intersection"
    assert summary.severity == "CRITICAL"
    assert summary.vehicles_involved == 2
    assert "vehicle wreckage" in summary.hazard_indicators
    assert summary.confidence == 0.98

def test_pixel_motion_differencing():
    pipeline = VSSPerceptionPipeline()
    # Test on any available test video
    test_video = "data/video_clips/scenario_1.mp4"
    if os.path.exists(test_video):
        has_anomaly, anomaly_t, desc = pipeline.scan_motion_and_anomaly_points(test_video)
        assert isinstance(has_anomaly, bool)
        assert anomaly_t > 0.0
        assert isinstance(desc, str)

if __name__ == "__main__":
    test_vlm_json_parser()
    test_pixel_motion_differencing()
    print("✅ All perception pipeline pixel tests passed successfully!")


def test_camera_id_derived_from_content(tmp_path):
    """Identity comes from what the camera records, not what the file is called.

    Rewritten for 1d52f4c, which moved camera_id_for() from the filename to a SHA-1
    of the first 8KB of content. The test still asserted the filename form and had
    been failing since. What is worth locking in is the property that motivated the
    change: renaming a file must not rename the camera.
    """
    p = VSSPerceptionPipeline()
    real = "data/video_clips/aic21_80.mp4"

    cam = p.camera_id_for(real)
    assert re.fullmatch(r"CAM-EDGE-[0-9A-F]{8}", cam), cam
    assert p.camera_id_for(real) == cam, "identity must be stable across calls"

    # Same bytes under a different name must yield the same camera.
    renamed = tmp_path / "totally_different_name.mp4"
    renamed.write_bytes(open(real, "rb").read(8192))
    assert p.camera_id_for(str(renamed)) == cam

    # An unreadable path degrades to a marker rather than raising mid-pipeline.
    assert p.camera_id_for("/abs/path/does_not_exist.mp4") == "CAM-EDGE-UNKNOWN"


def test_process_video_file_overwrites_generic_vlm_camera_id(monkeypatch):
    """Plain (non-stream) mode previously leaked the VLM's 'edge_surveillance_camera'."""
    p = VSSPerceptionPipeline()
    monkeypatch.setattr(p, "scan_motion_and_anomaly_points", lambda v: (True, 1.0, "x"))
    monkeypatch.setattr(p, "extract_targeted_burst", lambda v, t: [(1.0, "fakeb64")])
    monkeypatch.setattr(
        p, "deep_sequence_diagnosis",
        lambda frames, hint=None: VisualContextSummary(camera_id="edge_surveillance_camera")
    )
    summary = p.process_video_file("data/video_clips/aic21_80.mp4")
    # Asserted against the canonical derivation rather than a literal, so this test
    # survives the next change to how identity is computed. The point of the test is
    # that the VLM's invented name is overwritten, not what it is overwritten with.
    assert summary.camera_id != "edge_surveillance_camera"
    assert summary.camera_id == p.camera_id_for("data/video_clips/aic21_80.mp4")


def test_parser_downgrades_contradictory_severity():
    """A model echoing 'CRITICAL' with no vehicles and no hazards is contradicting itself."""
    p = VSSPerceptionPipeline()
    raw = json.dumps({
        "raw_summary": "Traffic flows normally with no signs of any accident.",
        "location": "Multi-Lane Interstate", "camera_id": "x",
        "vehicles_involved": 0, "hazard_indicators": [],
        "crisis_type": "Active collision impact", "severity": "CRITICAL", "confidence": 0.98,
    })
    assert p.parse_vlm_json_output(raw).severity == "LOW"


def test_parser_keeps_severity_when_evidence_present():
    p = VSSPerceptionPipeline()
    raw = json.dumps({
        "raw_summary": "Two cars collided; smoke is rising from the wreck.",
        "location": "Interstate", "camera_id": "x",
        "vehicles_involved": 2, "hazard_indicators": ["smoke", "wreckage"],
        "crisis_type": "Collision with smoke", "severity": "CRITICAL", "confidence": 0.9,
    })
    assert p.parse_vlm_json_output(raw).severity == "CRITICAL"


def test_prompt_contains_no_copyable_incident_label():
    """The old rubric handed the model the phrase it then pasted into crisis_type."""
    assert "Active collision impact" not in VSSPerceptionPipeline.STAGE2_DEEP_PROMPT


def test_escalation_stops_at_first_real_incident(monkeypatch):
    p = VSSPerceptionPipeline()
    monkeypatch.setattr(p, "rank_incident_candidates", lambda v: [(10.0, 9.0), (50.0, 4.0), (90.0, 2.0)])
    monkeypatch.setattr(p, "extract_targeted_burst", lambda v, t: [(t, "b64")])
    seen = []

    def fake_vlm(frames, hint=None):
        t = frames[0][0]
        seen.append(t)
        sev = "CRITICAL" if t == 50.0 else "LOW"
        return VisualContextSummary(severity=sev, crisis_type=f"probe@{t}")

    monkeypatch.setattr(p, "deep_sequence_diagnosis", fake_vlm)
    summary = p.analyze_incident("data/video_clips/aic21_95.mp4")
    assert summary.severity == "CRITICAL"
    assert seen == [10.0, 50.0], "must stop probing once an incident is found"


def test_escalation_reports_all_clear_when_every_probe_is_quiet(monkeypatch):
    p = VSSPerceptionPipeline()
    monkeypatch.setattr(p, "rank_incident_candidates", lambda v: [(10.0, 1.0), (50.0, 0.5)])
    monkeypatch.setattr(p, "extract_targeted_burst", lambda v, t: [(t, "b64")])
    monkeypatch.setattr(p, "deep_sequence_diagnosis",
                        lambda frames, hint=None: VisualContextSummary(severity="LOW"))
    assert p.analyze_incident("data/video_clips/aic21_80.mp4").severity == "LOW"


def test_guard_never_flattens_a_scene_with_responders():
    """A fire truck on scene must not be downgraded to all-clear by the safety net."""
    p = VSSPerceptionPipeline()
    raw = json.dumps({
        "raw_summary": "Police cars and a fire truck with flashing lights are on the shoulder. "
                       "No damage is visible from this angle.",
        "location": "Highway", "camera_id": "x",
        "vehicles_involved": 0, "hazard_indicators": [],
        "crisis_type": "Emergency response in progress", "severity": "HIGH", "confidence": 0.9,
    })
    assert p.parse_vlm_json_output(raw).severity == "HIGH"


def test_responders_on_scene_raise_low_to_medium():
    """scenario_2: model described a fire truck and still returned LOW."""
    p = VSSPerceptionPipeline()
    raw = json.dumps({
        "raw_summary": "A fire truck is parked at the intersection while other cars pass.",
        "location": "Intersection", "camera_id": "x",
        "vehicles_involved": 0, "hazard_indicators": [],
        "crisis_type": "Fire truck present at an intersection with normal traffic flow",
        "severity": "LOW", "confidence": 0.9,
    })
    assert p.parse_vlm_json_output(raw).severity == "MEDIUM"


def test_absence_of_responders_stays_low():
    """'no emergency vehicles' must not be read as responders being present."""
    p = VSSPerceptionPipeline()
    raw = json.dumps({
        "raw_summary": "Traffic flows normally. There are no emergency vehicles or police present.",
        "location": "Highway", "camera_id": "x",
        "vehicles_involved": 0, "hazard_indicators": [],
        "crisis_type": "Normal traffic flow", "severity": "LOW", "confidence": 0.95,
    })
    assert p.parse_vlm_json_output(raw).severity == "LOW"


def test_vlm_timeout_does_not_crash_the_run(monkeypatch):
    """A socket timeout is not a URLError; catching only URLError crashed the demo."""
    p = VSSPerceptionPipeline()
    monkeypatch.setattr(p, "_get_active_model_name", lambda: "test-model")

    def boom(*a, **kw):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    summary = p.deep_sequence_diagnosis([(1.0, "b64")])
    assert summary.crisis_type == "VLM_CONNECTION_OFFLINE"
    assert summary.confidence == 0.0


def test_failed_probe_is_not_reported_as_all_clear(monkeypatch):
    """A probe the VLM never answered must not masquerade as a quiet road."""
    p = VSSPerceptionPipeline()
    monkeypatch.setattr(p, "rank_incident_candidates", lambda v: [(10.0, 5.0), (60.0, 4.0)])
    monkeypatch.setattr(p, "extract_targeted_burst", lambda v, t: [(t, "b64")])

    def vlm(frames, hint=None):
        if frames[0][0] == 10.0:
            return VisualContextSummary(crisis_type="VLM_CONNECTION_OFFLINE", severity="LOW", confidence=0.0)
        return VisualContextSummary(crisis_type="Collision", severity="CRITICAL",
                                    vehicles_involved=2, hazard_indicators=["smoke"])

    monkeypatch.setattr(p, "deep_sequence_diagnosis", vlm)
    summary = p.analyze_incident("data/video_clips/x.mp4")
    assert summary.severity == "CRITICAL", "must keep probing past a failed VLM call"


def test_passing_tow_truck_is_not_an_emergency():
    """Real aic21_01 output: a tow truck hauling a car became a CODE AMBER callout."""
    p = VSSPerceptionPipeline()
    raw = json.dumps({
        "raw_summary": "The highway is busy with traffic moving smoothly in both directions. "
                       "Vehicles include cars, trucks, and a white tow truck towing a red car. "
                       "There are no collisions, stopped vehicles, or emergency responders visible.",
        "location": "Highway", "camera_id": "x",
        "vehicles_involved": 0, "hazard_indicators": [],
        "crisis_type": "No crisis detected; normal traffic flow on a multi-lane highway",
        "severity": "LOW", "confidence": 0.97,
    })
    assert p.parse_vlm_json_output(raw).severity == "LOW"


def test_negated_responders_in_a_list_stay_low():
    p = VSSPerceptionPipeline()
    assert p._responders_present("there are no collisions, stopped vehicles, or emergency responders visible") is False
    assert p._responders_present("a police car blocks the left lane") is True
    assert p._responders_present("traffic is normal. an ambulance is parked on the shoulder") is True
