"""LangGraph mortgage agent backed by llm-d and an MLflow-registered prompt."""

from __future__ import annotations

from datetime import date
from functools import cache
from typing import Any

import mlflow
from langchain.agents import create_agent
from langchain.agents.middleware import after_model, dynamic_prompt
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from mlflow_notes_agent.config import get_settings
from mlflow_notes_agent.tools import MORTGAGE_TOOLS

DEFAULT_PROMPT_URI = "prompts:/mortgage-lending-agent@production"


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, str) or not isinstance(content, list):
        return content

    normalized: list[Any] = []
    pending_text: dict[str, Any] | None = None

    def flush_pending_text() -> None:
        nonlocal pending_text
        if pending_text is not None:
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
    if not messages or not isinstance(messages[-1], AIMessage):
        return {}

    last_message = messages[-1]
    normalized_content = _normalize_message_content(last_message.content)
    if normalized_content == last_message.content:
        return {}
    return {"messages": [last_message.model_copy(update={"content": normalized_content})]}


@cache
def build_agent(prompt_uri: str = DEFAULT_PROMPT_URI):
    settings = get_settings()
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()
    model = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    @dynamic_prompt
    def prompt(_request):
        return mlflow.genai.load_prompt(prompt_uri, cache_ttl_seconds=5).format(
            current_date=date.today().isoformat(),
        )

    @after_model
    def normalize_message(state, _runtime):
        return _normalize_last_ai_message(state)

    return create_agent(
        model=model,
        tools=MORTGAGE_TOOLS,
        middleware=[prompt, normalize_message],
        name="mortgage_lending_agent",
    )


def build_default_agent():
    """Build the server graph without accepting LangGraph run configuration.

    ``langgraph-api`` interprets every single factory argument as its run config.
    Keep the prompt-selecting builder above for evaluations, while this zero-argument
    factory prevents that config dictionary from being passed as ``prompt_uri``.
    """
    return build_agent()
