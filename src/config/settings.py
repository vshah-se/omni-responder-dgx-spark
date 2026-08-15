import os
from dataclasses import dataclass

@dataclass
class SystemSettings:
    app_name: str = "Omni-Responder DGX Spark"
    target_hardware: str = "NVIDIA DGX Spark (Grace Blackwell 128GB Unified Memory)"
    vlm_model: str = "nvidia/cosmos-reason2-8b"
    orchestrator_llm: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"
    nim_endpoint_url: str = os.getenv("NIM_ENDPOINT_URL", "http://localhost:8000/v1")
    vss_endpoint_url: str = os.getenv("VSS_ENDPOINT_URL", "http://localhost:8000/v1")
    hazmat_db_path: str = "data/hazmat_db.json"
    mock_traffic_api_url: str = "http://localhost:9000/api/v1/traffic"

settings = SystemSettings()
