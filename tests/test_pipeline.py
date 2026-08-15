import os
from src.orchestrator.incident_manager import IncidentOrchestrator
from src.perception.vss_pipeline import VisualContextSummary

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
    assert incident["traffic"]["status"] == "EXECUTED"
    assert len(incident["traffic"]["actions_triggered"]) >= 3

    # Validate Comms CAD Dispatch Brief
    assert incident["dispatch_report"]["dispatch_code"] == "CRITICAL - CODE RED"
    assert len(incident["dispatch_report"]["target_units"]) >= 3
    assert "UN1017" in incident["dispatch_report"]["briefing_summary"]

def test_incident_orchestration_with_mock_video():
    orchestrator = IncidentOrchestrator()
    incident = orchestrator.process_incident("data/video_clips/crash_scenario_1.mp4", location_hint="Downtown Intersection")
    
    assert incident["perception"]["severity"] == "CRITICAL"
    assert incident["hazmat"]["status"] == "IDENTIFIED"
    assert "Chlorine" in incident["hazmat"]["chemical_name"]
    assert incident["traffic"]["status"] == "EXECUTED"

if __name__ == "__main__":
    test_full_incident_orchestration_with_summary()
    test_incident_orchestration_with_mock_video()
    print("✅ All Track 2 Agent Orchestration tests passed successfully!")
