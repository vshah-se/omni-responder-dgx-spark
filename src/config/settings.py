import os
from pydantic import BaseModel, Field

class SystemSettings(BaseModel):
    app_name: str = "Omni-Responder DGX Spark"
    target_hardware: str = "NVIDIA DGX Spark (Grace Blackwell 128GB Unified Memory)"
    vlm_model: str = Field(default="meta/llama-3.2-11b-vision-instruct", description="Local Vision-Language Model")
    orchestrator_llm: str = Field(default="meta/llama-3.3-70b-instruct", description="Local 70B parameter Orchestration LLM")
    nim_endpoint_url: str = Field(default=os.getenv("NIM_ENDPOINT_URL", "http://localhost:8000/v1"))
    hazmat_db_path: str = Field(default="data/hazmat_db.json")
    mock_traffic_api_url: str = Field(default="http://localhost:9000/api/v1/traffic")

settings = SystemSettings()
