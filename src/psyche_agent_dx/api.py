from __future__ import annotations

from fastapi import FastAPI

from psyche_agent_dx.config import get_settings
from psyche_agent_dx.pipeline import build_default_pipeline
from psyche_agent_dx.schemas import DiagnosisRequest, DiagnosisResponse


settings = get_settings()
app = FastAPI(
    title="Psyche Agent Dx",
    version="0.1.0",
    description="Mental health diagnosis support prototype with planner, expert, and retrieval agents.",
)

pipeline = build_default_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "diagnosis_backend": settings.diagnosis_backend,
        "chatglm_model_id": settings.chatglm_model_id,
    }


@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose(request: DiagnosisRequest) -> DiagnosisResponse:
    return pipeline.run(request)
