import os
import json
from src.perception.vss_pipeline import VSSPerceptionPipeline, VisualContextSummary

def test_vlm_json_parser():
    pipeline = VSSPerceptionPipeline()
    sample_vlm_raw = """
    ```json
    {
      "location": "5th Ave & Market St",
      "camera_id": "CAM-402",
      "crisis_type": "Commercial tanker breach",
      "severity": "CRITICAL",
      "vehicles_involved": 2,
      "hazard_indicators": ["green chemical vapor", "liquid leak"],
      "raw_summary": "Two-vehicle collision with green vapor leak.",
      "confidence": 0.98
    }
    ```
    """
    summary = pipeline.parse_vlm_json_output(sample_vlm_raw)
    assert isinstance(summary, VisualContextSummary)
    assert summary.location == "5th Ave & Market St"
    assert summary.severity == "CRITICAL"
    assert summary.vehicles_involved == 2
    assert "green chemical vapor" in summary.hazard_indicators
    assert summary.confidence == 0.98

def test_video_file_processing_scenario_1():
    pipeline = VSSPerceptionPipeline()
    summary = pipeline._extract_scenario_heuristic("data/video_clips/crash_scenario_1.mp4", "Highway 101 Exit 5")
    assert summary.severity == "CRITICAL"
    assert summary.vehicles_involved == 2
    assert "green chemical leak" in summary.hazard_indicators

def test_video_file_processing_scenario_2():
    pipeline = VSSPerceptionPipeline()
    summary = pipeline._extract_scenario_heuristic("data/video_clips/crash_scenario_2.mp4", "Downtown Intermodal Hub")
    assert summary.severity == "HIGH"
    assert summary.vehicles_involved == 3
    assert "clear to amber liquid pool" in summary.hazard_indicators

if __name__ == "__main__":
    test_vlm_json_parser()
    test_video_file_processing_scenario_1()
    test_video_file_processing_scenario_2()
    print("✅ All perception pipeline unit tests passed successfully!")
