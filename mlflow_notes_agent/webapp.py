"""Minimal FastAPI app used by the LangGraph server."""

from __future__ import annotations

from fastapi import FastAPI

from mlflow_notes_agent.config import get_settings

app = FastAPI(title="MLflow Mortgage Lending Agent")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "assistant_id": "agent",
        "mlflow_experiment": settings.mlflow_experiment,
        "model": settings.llm_model,
        "prompt": "mortgage-lending-agent@production",
    }
