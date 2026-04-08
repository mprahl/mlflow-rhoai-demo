"""Environment-backed settings for the MLflow notes demo."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _clean_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _normalize_gemini_api_key_env() -> None:
    """Mirror the configured Gemini API key across both env var names."""
    google_api_key = _clean_env("GOOGLE_API_KEY")
    gemini_api_key = _clean_env("GEMINI_API_KEY")
    resolved_api_key = google_api_key or gemini_api_key
    if not resolved_api_key:
        return

    if not google_api_key:
        os.environ["GOOGLE_API_KEY"] = resolved_api_key
    if not gemini_api_key:
        os.environ["GEMINI_API_KEY"] = resolved_api_key


_normalize_gemini_api_key_env()


def slugify_summary(value: str, *, max_words: int = 8, max_length: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    words = [word for word in cleaned.split("-") if word][:max_words]
    slug = "-".join(words)[:max_length].strip("-")
    return slug or "note"


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    github_token: str
    github_repo: str
    github_branch: str
    mlflow_tracking_uri: str
    mlflow_tracking_auth: str
    mlflow_workspace: str
    mlflow_experiment: str
    gemini_model: str
    github_api_url: str
    app_name: str


def _require(name: str) -> str:
    value = _clean_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _require_gemini_api_key() -> str:
    _normalize_gemini_api_key_env()
    value = _clean_env("GOOGLE_API_KEY") or _clean_env("GEMINI_API_KEY")
    if not value:
        raise RuntimeError("Missing required environment variable: GOOGLE_API_KEY or GEMINI_API_KEY")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        google_api_key=_require_gemini_api_key(),
        github_token=_require("GITHUB_TOKEN"),
        github_repo=_require("GITHUB_REPO"),
        github_branch=_require("GITHUB_BRANCH"),
        mlflow_tracking_uri=_clean_env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        mlflow_tracking_auth=_clean_env("MLFLOW_TRACKING_AUTH"),
        mlflow_workspace=_clean_env("MLFLOW_WORKSPACE"),
        mlflow_experiment=_clean_env("MLFLOW_EXPERIMENT", "mlflow-demo"),
        gemini_model=_clean_env("GEMINI_MODEL", "gemini-2.5-flash"),
        github_api_url=_clean_env("GITHUB_API_URL", "https://api.github.com"),
        app_name=_clean_env("APP_NAME", "mlflow-notes-agent"),
    )
