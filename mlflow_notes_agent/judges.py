"""MLflow LLM judges for post-run trace evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from typing import TYPE_CHECKING

import mlflow
from mlflow.genai.judges import make_judge

from mlflow_notes_agent.tracing import configure_mlflow

if TYPE_CHECKING:
    from mlflow.entities import Trace

DEFAULT_JUDGE_MODEL_URI = "gemini:/gemini-2.5-pro"
DEFAULT_TRACE_WORKERS = 2
DEFAULT_JUDGE_WORKERS = 2


def _judge_model_uri(model_uri: str | None = None) -> str:
    return model_uri or DEFAULT_JUDGE_MODEL_URI


def _message_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _truncate_text(text: str, *, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _summarize_messages(messages: list[dict], *, limit: int = 8) -> list[dict]:
    summary = []
    for message in messages[-limit:]:
        if not isinstance(message, dict):
            continue
        role = message.get("type") or message.get("role") or "message"
        text = _truncate_text(_message_to_text(message.get("content", "")), max_chars=700)
        tool_calls = message.get("tool_calls", [])
        summary.append(
            {
                "role": role,
                "text": text,
                "tool_calls": [
                    {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args"),
                        "id": tool_call.get("id"),
                    }
                    for tool_call in tool_calls
                    if isinstance(tool_call, dict)
                ],
            }
        )
    return summary


def _summarize_trace(trace: Trace) -> tuple[dict, dict]:
    root_span = trace.data.spans[0]
    messages = root_span.inputs.get("messages", []) if isinstance(root_span.inputs, dict) else []

    conversation = _summarize_messages(messages)

    tool_spans = []
    for span in trace.data.spans:
        span_type = getattr(span, "span_type", None)
        if str(span_type).upper().endswith("TOOL"):
            tool_spans.append(
                {
                    "name": span.name,
                    "inputs": span.inputs,
                    "outputs": span.outputs,
                    "status": str(getattr(getattr(span, "status", None), "status_code", "")),
                }
            )

    output_messages = []
    if isinstance(root_span.outputs, dict):
        output_messages = root_span.outputs.get("messages", [])

    final_output = {
        "messages": _summarize_messages(output_messages, limit=4),
        "raw_keys": sorted(root_span.outputs.keys()) if isinstance(root_span.outputs, dict) else [],
    }

    judge_inputs = {
        "trace_id": trace.info.trace_id,
        "conversation": conversation,
    }
    judge_outputs = {
        "tool_calls": tool_spans,
        "final_output": final_output,
    }
    return judge_inputs, judge_outputs


@cache
def get_tool_accuracy_judge(model_uri: str | None = None):
    return make_judge(
        name="tool_accuracy_ok",
        description="Judge whether the agent used tools appropriately and accurately.",
        instructions="""
        Analyze {{ inputs }} and {{ outputs }}.

        Consider:
        - Does the conversation show a clear user request?
        - Did the listed tool calls match the user's request?
        - Were tool arguments reasonable and specific?
        - Did the final output reflect the tool outputs faithfully?
        - Did the agent avoid unnecessary or misleading tool calls?

        Return only a boolean:
        - true if the tool usage was overall accurate and helpful
        - false if the tool usage was incorrect, misleading, or incomplete
        """,
        model=_judge_model_uri(model_uri),
        feedback_value_type=bool,
    )


@cache
def get_user_frustration_judge(model_uri: str | None = None):
    return make_judge(
        name="low_user_frustration",
        description="Judge whether the interaction shows low user frustration.",
        instructions="""
        Analyze {{ inputs }} and {{ outputs }} and assess whether
        the interaction shows low user frustration.

        Consider:
        - Did the user need to repeat or restate requests?
        - Did the assistant ignore instructions or fail to complete the task?
        - Did the final outcome appear to resolve the user's need?
        - Does the conversation show confusion, dissatisfaction, or friction?

        Return only a boolean:
        - true if the interaction shows low or no user frustration
        - false if the interaction suggests the user was frustrated,
          dissatisfied, or their request was not resolved well
        """,
        model=_judge_model_uri(model_uri),
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


def _run_single_judge(
    judge_name: str,
    *,
    judge_inputs: dict,
    judge_outputs: dict,
    model_uri: str | None = None,
):
    judge = JUDGE_FACTORIES[judge_name](_judge_model_uri(model_uri))
    return judge(inputs=judge_inputs, outputs=judge_outputs)


def assess_trace(
    trace_id: str,
    *,
    judge_names: list[str] | None = None,
    model_uri: str | None = None,
    judge_workers: int = DEFAULT_JUDGE_WORKERS,
):
    configure_mlflow()
    trace = _get_trace_with_retry(trace_id)
    judge_inputs, judge_outputs = _summarize_trace(trace)
    target_judges = judge_names or list(JUDGE_FACTORIES.keys())

    if len(target_judges) == 1 or judge_workers <= 1:
        assessments = [
            _run_single_judge(
                judge_name,
                judge_inputs=judge_inputs,
                judge_outputs=judge_outputs,
                model_uri=model_uri,
            )
            for judge_name in target_judges
        ]
    else:
        assessments_by_name = {}
        with ThreadPoolExecutor(max_workers=min(judge_workers, len(target_judges))) as executor:
            future_to_name = {
                executor.submit(
                    _run_single_judge,
                    judge_name,
                    judge_inputs=judge_inputs,
                    judge_outputs=judge_outputs,
                    model_uri=model_uri,
                ): judge_name
                for judge_name in target_judges
            }
            for future in as_completed(future_to_name):
                assessments_by_name[future_to_name[future]] = future.result()
        assessments = [assessments_by_name[judge_name] for judge_name in target_judges]

    for assessment in assessments:
        mlflow.log_assessment(trace_id=trace_id, assessment=assessment)
    return assessments


def _assess_trace_for_experiment(
    trace_id: str,
    judge_names: list[str],
    *,
    model_uri: str | None = None,
    judge_workers: int = DEFAULT_JUDGE_WORKERS,
):
    try:
        return trace_id, assess_trace(
            trace_id,
            judge_names=judge_names,
            model_uri=model_uri,
            judge_workers=judge_workers,
        )
    except RuntimeError as exc:
        if "not fully exported yet" in str(exc).lower():
            return trace_id, None
        raise


def assess_experiment(
    experiment_name: str,
    *,
    max_traces: int | None = None,
    model_uri: str | None = None,
    workers: int = DEFAULT_TRACE_WORKERS,
    judge_workers: int = DEFAULT_JUDGE_WORKERS,
) -> dict[str, list]:
    unscored = find_unscored_traces(experiment_name, max_traces=max_traces)
    results: dict[str, list] = {}

    if len(unscored) <= 1 or workers <= 1:
        for trace_id, judge_names in unscored:
            _, assessments = _assess_trace_for_experiment(
                trace_id,
                judge_names,
                model_uri=model_uri,
                judge_workers=judge_workers,
            )
            if assessments is not None:
                results[trace_id] = assessments
        return results

    with ThreadPoolExecutor(max_workers=min(workers, len(unscored))) as executor:
        future_to_trace_id = {
            executor.submit(
                _assess_trace_for_experiment,
                trace_id,
                judge_names,
                model_uri=model_uri,
                judge_workers=judge_workers,
            ): trace_id
            for trace_id, judge_names in unscored
        }
        for future in as_completed(future_to_trace_id):
            trace_id, assessments = future.result()
            if assessments is not None:
                results[trace_id] = assessments
    return results


def _format_user_facing_error(exc: Exception, *, model_uri: str | None = None) -> str:
    message = str(exc)
    lowered = message.lower()
    if any(marker in lowered for marker in ["rate limit", "resource_exhausted", "quota exceeded"]):
        return (
            "Gemini judge invocation failed because the configured "
            f"Google API key has no usable quota for {_judge_model_uri(model_uri)}. "
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
    parser.add_argument(
        "--judge-model-uri",
        default=DEFAULT_JUDGE_MODEL_URI,
        help="Model URI for the judges.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_TRACE_WORKERS,
        help="Parallel trace workers when using --experiment-name.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=DEFAULT_JUDGE_WORKERS,
        help="Parallel judge workers per trace.",
    )
    args = parser.parse_args()

    try:
        if args.trace_id:
            results = {
                args.trace_id: assess_trace(
                    args.trace_id,
                    model_uri=args.judge_model_uri,
                    judge_workers=args.judge_workers,
                )
            }
        else:
            results = assess_experiment(
                args.experiment_name,
                max_traces=args.max_traces,
                model_uri=args.judge_model_uri,
                workers=args.workers,
                judge_workers=args.judge_workers,
            )
    except Exception as exc:
        print(
            _format_user_facing_error(exc, model_uri=args.judge_model_uri),
            file=sys.stderr,
        )
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
