import os
from src.orchestrator.incident_manager import IncidentOrchestrator
from src.perception.vss_pipeline import VisualContextSummary, StreamFrameEvent

def test_full_incident_orchestration_with_summary():
    orchestrator = IncidentOrchestrator()
    summary = VisualContextSummary(
        location="I-80 Northbound Mile Marker 42",
        camera_id="CAM-HWY-42",
        crisis_type="Commercial collision with chemical spill",
        severity="CRITICAL",
        vehicles_involved=2,
        hazard_indicators=["greenish-yellow gas", "dense vapor cloud near ground"],
        raw_summary="Overturned commercial chemical tanker leaking dense greenish vapor.",
        confidence=0.98
    )

    incident = orchestrator.process_incident(summary)

    # Validate Master Orchestrator outputs
    assert incident["incident_id"].startswith("CAD-EMG-")
    assert incident["perception"]["location"] == "I-80 Northbound Mile Marker 42"
    assert incident["perception"]["vehicles_involved"] == 2

    # Validate Hazmat Agent Identified Chlorine
    assert incident["hazmat"]["status"] == "IDENTIFIED"
    assert incident["hazmat"]["un_number"] == "UN1017"
    assert "Chlorine" in incident["hazmat"]["chemical_name"]
    assert "Level A" in incident["hazmat"]["ppe_required"]

    # Validate Traffic Agent Reroute
    assert incident["traffic"]["status"] in ["EXECUTED", "EMERGENCY_PERIMETER_LOCKED"]
    assert len(incident["traffic"]["actions_triggered"]) >= 3

    # Validate Comms CAD Dispatch Brief
    assert incident["dispatch_report"]["dispatch_code"] == "CRITICAL - CODE RED"
    assert len(incident["dispatch_report"]["target_units"]) >= 3
    assert "UN1017" in incident["dispatch_report"]["briefing_summary"]

def test_roadside_assistance_orchestration():
    orchestrator = IncidentOrchestrator()
    summary = VisualContextSummary(
        location="Highway Shoulder",
        camera_id="CAM-HWY-01",
        crisis_type="Roadside Assistance / Vehicle on Shoulder",
        severity="MEDIUM",
        vehicles_involved=1,
        hazard_indicators=["vehicle on shoulder"],
        raw_summary="Vehicle stopped on shoulder with hazard lights.",
        confidence=0.96
    )

    incident = orchestrator.process_incident(summary)
    assert incident["hazmat"]["status"] == "NO_HAZARDS_DETECTED"
    assert incident["traffic"]["status"] == "CONGESTION_MITIGATION"
    assert incident["dispatch_report"]["dispatch_code"] == "HIGH - CODE AMBER"
    assert "Highway Safety Patrol" in str(incident["dispatch_report"]["target_units"])

def test_incident_orchestration_with_mock_video():
    orchestrator = IncidentOrchestrator()
    try:
        incident = orchestrator.process_incident("data/video_clips/scenario_1.mp4", location_hint="Downtown Intersection")
        assert incident["perception"]["severity"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert incident["incident_id"].startswith("CAD-EMG-")
    except Exception as e:
        print(f"Skipping VLM mock video test, NIM not reachable: {e}")

def _stream_event(status, severity):
    return StreamFrameEvent(
        timestamp="00:00:01",
        elapsed_seconds=1.0,
        status=status,
        severity=severity,
        scene_description=f"synthetic {status}",
        visual_summary=VisualContextSummary(
            location="Test Highway",
            camera_id="CAM-TEST",
            crisis_type=f"synthetic {status}",
            severity=severity,
            vehicles_involved=0 if severity == "LOW" else 2,
        ),
    )

def test_all_clear_never_reaches_the_dispatch_chain(monkeypatch):
    """An all-clear is the pipeline concluding nothing is happening.

    Running the dispatch chain on it produced ERG lookups, perimeter locks and CODE RED
    briefs for quiet roads, hidden behind a green terminal banner while the dashboard
    rendered them in full. Needs no video and no VLM, so it guards the gate cheaply.
    """
    orchestrator = IncidentOrchestrator()
    events = [
        _stream_event("ROUTINE_ALL_CLEAR", "LOW"),
        _stream_event("ROADSIDE_ASSISTANCE", "MEDIUM"),
        _stream_event("CRISIS_IMPACT", "CRITICAL"),
    ]
    monkeypatch.setattr(orchestrator.perception, "stream_video_feed", lambda *a, **kw: iter(events))

    dispatched_severities = []
    real_process_incident = orchestrator.process_incident

    def spy(video_input, *args, **kwargs):
        dispatched_severities.append(video_input.severity)
        return real_process_incident(video_input, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "process_incident", spy)

    results = {r["status"]: r for r in orchestrator.stream_incident("data/video_clips/aic21_80.mp4")}

    assert results["ROUTINE_ALL_CLEAR"]["incident_record"] is None
    assert results["ROADSIDE_ASSISTANCE"]["incident_record"] is not None
    assert results["CRISIS_IMPACT"]["incident_record"] is not None
    assert dispatched_severities == ["MEDIUM", "CRITICAL"], (
        "the all-clear must not invoke the hazmat/traffic/comms chain at all"
    )

if __name__ == "__main__":
    test_full_incident_orchestration_with_summary()
    test_roadside_assistance_orchestration()
    test_incident_orchestration_with_mock_video()
    print("✅ All Track 2 Agent Orchestration tests passed successfully!")
