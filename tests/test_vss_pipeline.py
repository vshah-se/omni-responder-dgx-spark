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
    test_video = "data/video_clips/crash_scenario_1.mp4"
    if os.path.exists(test_video):
        has_anomaly, anomaly_t, desc = pipeline.scan_motion_and_anomaly_points(test_video)
        assert isinstance(has_anomaly, bool)
        assert anomaly_t > 0.0
        assert isinstance(desc, str)

if __name__ == "__main__":
    test_vlm_json_parser()
    test_pixel_motion_differencing()
    print("✅ All perception pipeline pixel tests passed successfully!")
