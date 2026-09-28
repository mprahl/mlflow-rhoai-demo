"""LangGraph mortgage agent backed by llm-d and an MLflow-registered prompt."""

from __future__ import annotations

from datetime import date
from functools import cache

import mlflow
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_openai import ChatOpenAI

from mlflow_notes_agent.config import get_settings
from mlflow_notes_agent.tools import MORTGAGE_TOOLS

DEFAULT_PROMPT_URI = "prompts:/mortgage-lending-agent@production"


@cache
def build_agent(prompt_uri: str = DEFAULT_PROMPT_URI):
    settings = get_settings()
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog(run_tracer_inline=True)
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

    return create_agent(
        model=model,
        tools=MORTGAGE_TOOLS,
        middleware=[prompt],
        name="mortgage_lending_agent",
    )


def build_default_agent():
    """Build the server graph without accepting LangGraph run configuration.

    ``langgraph-api`` interprets every single factory argument as its run config.
    Keep the prompt-selecting builder above for evaluations, while this zero-argument
    factory prevents that config dictionary from being passed as ``prompt_uri``.
    """
    return build_agent()
