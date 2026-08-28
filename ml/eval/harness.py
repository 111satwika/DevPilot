"""Runs a model (via a pluggable predictor function) against the held-out
test set and aggregates ml/eval/metrics.py's per-example scores into the
report the fine-tune's own plan asks for: exact tool match, schema-valid
rate, mean argument F1, mode-violation rate, task completion rate, and
p50/p95 latency.

A predictor is any callable `(example: dict) -> list[dict]` returning the
tool calls a model produced for that example (empty list for a refusal/
text response) -- see ml/eval/predictors.py for real implementations.
Timing happens here, around the predictor call, so every predictor
(Ollama HTTP, HF+adapter generate()) is timed identically regardless of
how it actually talks to a model.
"""

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

from ml.eval.metrics import ExampleResult, score_example

Predictor = Callable[[dict], list[dict]]


@dataclass
class EvalReport:
    model_name: str
    num_examples: int
    exact_tool_match_rate: float
    schema_valid_rate: float
    mean_argument_f1: float | None  # None if no example had a comparable tool call
    mode_violation_rate: float
    task_completion_rate: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    per_example: list[ExampleResult]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_example"] = [asdict(r) for r in self.per_example]
        return d

    def summary_line(self) -> str:
        f1 = f"{self.mean_argument_f1:.3f}" if self.mean_argument_f1 is not None else "n/a"
        return (
            f"{self.model_name}: exact_match={self.exact_tool_match_rate:.1%} "
            f"schema_valid={self.schema_valid_rate:.1%} arg_f1={f1} "
            f"mode_violation={self.mode_violation_rate:.1%} "
            f"task_completion={self.task_completion_rate:.1%} "
            f"p50={self.p50_latency_seconds:.2f}s p95={self.p95_latency_seconds:.2f}s "
            f"(n={self.num_examples})"
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct * (len(ordered) - 1)))
    return ordered[index]


def evaluate(
    model_name: str, predictor: Predictor, examples: list[dict], verbose: bool = True
) -> EvalReport:
    results: list[ExampleResult] = []

    for i, example in enumerate(examples, 1):
        start = time.perf_counter()
        predicted_tool_calls = predictor(example)
        elapsed = time.perf_counter() - start

        result = score_example(example, predicted_tool_calls, elapsed)
        results.append(result)

        if verbose:
            status = "OK" if result.task_completed else "MISS"
            print(
                f"  [{i}/{len(examples)}] {status} mode={example['mode']} "
                f"expected={example['completion']['type']} "
                f"({elapsed:.2f}s)"
            )

    latencies = [r.latency_seconds for r in results]
    f1_scores = [r.argument_f1 for r in results if r.argument_f1 is not None]
    # exact_tool_match and mode_violation are each only MEANINGFUL over one
    # subset of the dataset (a refusal-expected example can never be an
    # "exact match" -- there's no tool to match; a tool-call-expected
    # example can never register as a "mode violation" -- see metrics.py).
    # Averaging either over the whole mixed dataset would dilute the rate
    # in a way that's confusing to report, not just imprecise -- caught by
    # a test that expected a perfect predictor to score 100% and got 67%
    # instead, because 1/3 of that test's examples were refusal-expected.
    tool_call_results = [r for r in results if r.expected_type == "tool_call"]
    refusal_results = [r for r in results if r.expected_type == "refusal"]

    return EvalReport(
        model_name=model_name,
        num_examples=len(results),
        exact_tool_match_rate=_rate(tool_call_results, lambda r: r.exact_match),
        schema_valid_rate=_rate(results, lambda r: r.schema_valid),
        mean_argument_f1=statistics.mean(f1_scores) if f1_scores else None,
        mode_violation_rate=_rate(refusal_results, lambda r: r.mode_violation),
        task_completion_rate=_rate(results, lambda r: r.task_completed),
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
        per_example=results,
    )


def _rate(results: list[ExampleResult], predicate: Callable[[ExampleResult], bool]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if predicate(r)) / len(results)
