"""LangGraph agent definition for the MLflow notes demo."""

from __future__ import annotations

from datetime import date
from functools import cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from mlflow_notes_agent.config import get_settings
from mlflow_notes_agent.tools import NOTES_TOOLS
from mlflow_notes_agent.tracing import configure_mlflow


def build_system_prompt() -> str:
    return f"""\
You are a note-taking assistant for a Red Hat OpenShift AI demo.

Your job is to help the user collect notes, optionally enrich those notes with
DuckDuckGo or Wikipedia research, and store the final note as a markdown file in
GitHub.

Today's date is {date.today().isoformat()}.

Rules:
- Use the available tools instead of making up external facts.
- Prefer Wikipedia for background context and DuckDuckGo for broader or current web context.
- When the user asks what notes already exist, use the note tools.
- When the user asks for today's date or time, answer directly or use `get_today_date`.
- Only call `save_note` when the user clearly asks to save, create, or store a note.
- Only call `delete_note` when the user clearly asks to delete a saved note.
- Before saving, gather enough detail to produce a useful title, summary, and note body.
- Keep saved notes concise, readable, and source-aware.
- Mention which sources were used when you enriched the note with web research.
- If the user only wants brainstorming or research, do not save anything yet.
"""


@cache
def build_agent():
    configure_mlflow()
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.2,
        google_api_key=settings.google_api_key,
    )
    return create_react_agent(
        model=model,
        tools=NOTES_TOOLS,
        prompt=build_system_prompt(),
        name="mlflow_notes_agent",
    )
