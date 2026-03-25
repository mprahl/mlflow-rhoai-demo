"""MLflow tracing helpers."""

from __future__ import annotations

import os
from functools import lru_cache

from mlflow_notes_agent.config import get_settings


@lru_cache(maxsize=1)
def configure_mlflow() -> None:
    settings = get_settings()

    if settings.mlflow_tracking_auth:
        os.environ.setdefault("MLFLOW_TRACKING_AUTH", settings.mlflow_tracking_auth)
    if settings.mlflow_workspace:
        os.environ.setdefault("MLFLOW_WORKSPACE", settings.mlflow_workspace)

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()
