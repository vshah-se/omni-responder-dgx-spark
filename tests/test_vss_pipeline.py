import os
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


def test_camera_id_derived_from_filename():
    """The VLM invents a generic camera name; both entry points must overwrite it."""
    p = VSSPerceptionPipeline()
    assert p.camera_id_for("data/video_clips/aic21_80.mp4") == "CAM-EDGE-AIC21_80"
    assert p.camera_id_for("/abs/path/scenario_1.mp4") == "CAM-EDGE-SCENARIO_1"


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
    assert summary.camera_id == "CAM-EDGE-AIC21_80"


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
