from src.orchestrator.incident_manager import IncidentOrchestrator

def test_incident_orchestration():
    orchestrator = IncidentOrchestrator()
    res = orchestrator.process_incident("feed_test")
    assert res is not None
    assert res["perception"].vehicles_involved == 2
    assert "dispatch_report" in res
    assert res["traffic"]["status"] == "EXECUTED"
