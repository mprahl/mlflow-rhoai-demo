"""MLflow LLM judges for post-run trace evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from functools import cache
from typing import TYPE_CHECKING

import mlflow
from mlflow.genai.judges import make_judge

from mlflow_notes_agent.tracing import configure_mlflow

if TYPE_CHECKING:
    from mlflow.entities import Trace

JUDGE_MODEL_URI = "gemini:/gemini-2.5-pro"


def _judge_model_uri() -> str:
    return JUDGE_MODEL_URI


@cache
def get_tool_accuracy_judge(model_uri: str | None = None):
    return make_judge(
        name="tool_accuracy_ok",
        description="Judge whether the agent used tools appropriately and accurately.",
        instructions="""
        Analyze the {{ trace }} and determine whether the agent's tool usage was accurate.

        Consider:
        - Did the agent choose the correct tools for the user's request?
        - Were tool arguments reasonable and specific?
        - Did the final answer faithfully reflect the tool outputs?
        - Did the agent avoid unnecessary or misleading tool calls?

        Return true if the tool usage was overall accurate and helpful.
        Return false if the tool usage was incorrect, misleading, or incomplete.
        """,
        model=model_uri or _judge_model_uri(),
        feedback_value_type=bool,
    )


@cache
def get_user_frustration_judge(model_uri: str | None = None):
    return make_judge(
        name="low_user_frustration",
        description="Judge whether the interaction shows low user frustration.",
        instructions="""
        Analyze the {{ trace }} and assess whether the interaction shows low user frustration.

        Consider:
        - Did the user need to repeat or restate requests?
        - Did the agent ignore instructions or fail to complete the task?
        - Did the final outcome appear to resolve the user's need?
        - Did the conversation show signals of confusion, dissatisfaction, or friction?

        Return true if the interaction shows low or no user frustration.
        Return false if the interaction suggests the user was frustrated,
        dissatisfied, or their request was not resolved well.
        """,
        model=model_uri or _judge_model_uri(),
        feedback_value_type=bool,
    )


JUDGE_FACTORIES = {
    "tool_accuracy_ok": get_tool_accuracy_judge,
    "low_user_frustration": get_user_frustration_judge,
}


def _missing_judge_names(trace: Trace) -> list[str]:
    missing: list[str] = []
    for judge_name in JUDGE_FACTORIES:
        if not trace.search_assessments(name=judge_name):
            missing.append(judge_name)
    return missing


def _get_experiment_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")
    return experiment.experiment_id


def find_unscored_traces(
    experiment_name: str,
    *,
    max_traces: int | None = None,
) -> list[tuple[str, list[str]]]:
    configure_mlflow()
    experiment_id = _get_experiment_id(experiment_name)
    traces = mlflow.search_traces(
        locations=[experiment_id],
        max_results=max_traces,
        order_by=["timestamp_ms DESC"],
        return_type="list",
        include_spans=False,
    )

    unscored: list[tuple[str, list[str]]] = []
    for trace in traces:
        missing = _missing_judge_names(trace)
        if missing:
            unscored.append((trace.info.trace_id, missing))
    return unscored


def _get_trace_with_retry(trace_id: str, *, attempts: int = 3, delay_seconds: float = 2.0):
    last_trace = None
    for attempt in range(attempts):
        last_trace = mlflow.get_trace(trace_id=trace_id)
        if last_trace is not None:
            return last_trace
        if attempt < attempts - 1:
            time.sleep(delay_seconds)
    raise RuntimeError(f"Trace {trace_id} is not fully exported yet.")


def assess_trace(
    trace_id: str,
    *,
    judge_names: list[str] | None = None,
):
    configure_mlflow()
    trace = _get_trace_with_retry(trace_id)
    target_judges = judge_names or list(JUDGE_FACTORIES.keys())
    assessments = []
    for judge_name in target_judges:
        judge = JUDGE_FACTORIES[judge_name](JUDGE_MODEL_URI)
        assessments.append(judge(trace=trace))

    for assessment in assessments:
        mlflow.log_assessment(trace_id=trace_id, assessment=assessment)
    return assessments


def assess_experiment(
    experiment_name: str,
    *,
    max_traces: int | None = None,
) -> dict[str, list]:
    unscored = find_unscored_traces(experiment_name, max_traces=max_traces)
    results: dict[str, list] = {}
    for trace_id, judge_names in unscored:
        try:
            results[trace_id] = assess_trace(trace_id, judge_names=judge_names)
        except RuntimeError as exc:
            if "not fully exported yet" in str(exc).lower():
                continue
            raise
    return results


def _format_user_facing_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if any(marker in lowered for marker in ["rate limit", "resource_exhausted", "quota exceeded"]):
        return (
            "Gemini judge invocation failed because the configured "
            f"Google API key has no usable quota for {JUDGE_MODEL_URI}. "
            "Update quota or switch to a different judge model."
        )
    return message


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLflow judges on a trace or experiment.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--trace-id", help="Trace ID to assess.")
    target.add_argument(
        "--experiment-name",
        help="Experiment name to scan for unscored traces.",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of traces to scan when using --experiment-name.",
    )
    args = parser.parse_args()

    try:
        if args.trace_id:
            results = {args.trace_id: assess_trace(args.trace_id)}
        else:
            results = assess_experiment(args.experiment_name, max_traces=args.max_traces)
    except Exception as exc:
        print(_format_user_facing_error(exc), file=sys.stderr)
        raise SystemExit(1) from None

    if not results:
        print("No unscored traces found.")
        return

    for trace_id, assessments in results.items():
        print(f"Trace: {trace_id}")
        for assessment in assessments:
            print(f"  {assessment.name}: {assessment.value}")
            if assessment.rationale:
                print(f"  Rationale: {assessment.rationale}")
        print()
