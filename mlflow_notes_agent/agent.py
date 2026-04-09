"""LangGraph agent definition for the MLflow notes demo."""

from __future__ import annotations

from datetime import date
from functools import cache
from typing import Any

from langchain_core.messages import AIMessage
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


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, str) or not isinstance(content, list):
        return content

    normalized: list[Any] = []
    pending_text: dict[str, Any] | None = None

    def flush_pending_text() -> None:
        nonlocal pending_text
        if pending_text is None:
            return
        normalized.append(pending_text)
        pending_text = None

    def text_metadata(part: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in part.items() if key not in {"type", "text"}}

    for item in content:
        if isinstance(item, str):
            if pending_text is None:
                pending_text = {"type": "text", "text": item}
            else:
                pending_text["text"] += item
            continue

        if (
            isinstance(item, dict)
            and item.get("type") in {"text", "text_delta"}
            and isinstance(item.get("text"), str)
        ):
            next_text = {**item, "type": "text"}
            if pending_text is None:
                pending_text = next_text
            elif text_metadata(pending_text) == text_metadata(next_text):
                pending_text["text"] += next_text["text"]
            else:
                flush_pending_text()
                pending_text = next_text
            continue

        flush_pending_text()
        normalized.append(item)

    flush_pending_text()

    if (
        len(normalized) == 1
        and isinstance(normalized[0], dict)
        and normalized[0].keys() == {"type", "text"}
        and normalized[0].get("type") == "text"
    ):
        return normalized[0]["text"]
    return normalized


def _normalize_last_ai_message(state: dict[str, Any]) -> dict[str, list[AIMessage]]:
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return {}

    normalized_content = _normalize_message_content(last_message.content)
    if normalized_content == last_message.content:
        return {}

    return {
        "messages": [
            last_message.model_copy(update={"content": normalized_content}),
        ]
    }


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
        post_model_hook=_normalize_last_ai_message,
        name="mlflow_notes_agent",
    )
