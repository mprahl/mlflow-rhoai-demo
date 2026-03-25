"""Simple CLI entrypoint for local smoke testing."""

from __future__ import annotations

from typing import Any

from mlflow_notes_agent.agent import build_agent


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def main() -> None:
    agent = build_agent()
    messages: list[dict[str, str]] = []

    print("MLflow Notes Agent CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        assistant_text = _stringify_message_content(messages[-1].content)
        print(f"\nAssistant> {assistant_text}\n")
