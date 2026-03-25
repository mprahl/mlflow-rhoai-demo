"""Minimal FastAPI app used by the LangGraph dev server."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from mlflow_notes_agent.config import get_settings
from mlflow_notes_agent.tracing import configure_mlflow


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_mlflow()
    yield


app = FastAPI(title="MLflow Notes Agent", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "assistant_id": "agent",
        "mlflow_experiment": settings.mlflow_experiment,
    }
