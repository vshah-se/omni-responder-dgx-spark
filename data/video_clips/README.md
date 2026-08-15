# Video Clips Directory 🎥

This directory contains simulated emergency and traffic camera `.mp4` feeds for Omni-Responder.

### Standard Demo Clips:
1. `scenario_1.mp4` — Multi-vehicle crash with commercial chemical tanker breach (Chlorine vapor plume).
2. `scenario_2.mp4` — Highway rollover with fuel spill.

### Running Live Ingestion:
```bash
python3 -m src.main --video data/video_clips/scenario_1.mp4 --stream
```
