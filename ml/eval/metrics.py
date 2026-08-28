"""Metrics for the three-way eval (base vs. fine-tuned vs. teacher), per
the fine-tune project's own plan. Pure functions, no model calls -- given
a test example's expected completion and a model's actual prediction,
compute each metric. Kept separate from harness.py (which does the
looping/timing/model-calling) so these are trivially unit-testable
without any model or network involved.

A predicted tool call is a plain dict: {"name": str, "arguments": dict}
(or None if the model produced no tool call at all -- a refusal/text
response). This matches the shape llm/agent.py's own message format
already uses (function.name/function.arguments), so both the Ollama and
HF-adapter predictors in predictors.py can produce it identically.
"""

from dataclasses import dataclass


def exact_tool_match(expected: dict, predicted_tool_calls: list[dict]) -> bool:
    """True only if the expected example is itself a tool_call and the
    model called exactly that tool (arguments not checked here --
    argument_f1 below covers that separately, since "picked the right
    tool but got one argument wrong" and "picked the wrong tool entirely"
    are different failure modes worth measuring separately)."""
    if expected["type"] != "tool_call":
        return False
    if len(predicted_tool_calls) != 1:
        return False
    expected_name = expected["tool_calls"][0]["name"]
    return predicted_tool_calls[0]["name"] == expected_name


def is_schema_valid(tool_schemas: list[dict], predicted_tool_calls: list[dict]) -> bool:
    """True if every predicted tool call names a real, currently-offered
    tool and supplies every one of that tool's required arguments (types
    not checked -- Ollama's own tool-calling already constrains those at
    the JSON level; this checks the shape a naive/hallucinated call could
    still get wrong: a nonexistent tool name, or missing required
    fields)."""
    if not predicted_tool_calls:
        return True  # nothing predicted -- vacuously "doesn't violate the schema"
    schema_by_name = {t["function"]["name"]: t["function"] for t in tool_schemas}
    for call in predicted_tool_calls:
        schema = schema_by_name.get(call["name"])
        if schema is None:
            return False
        required = schema.get("parameters", {}).get("required", [])
        if not isinstance(call.get("arguments"), dict):
            return False
        if any(field not in call["arguments"] for field in required):
            return False
    return True


def argument_f1(expected: dict, predicted_tool_calls: list[dict]) -> float | None:
    """Field-level F1 between predicted and expected arguments, for the
    tool call that matches by name (per exact_tool_match's definition of
    "the" predicted call). Returns None (not 0.0) when there's nothing to
    compare -- an expected refusal/text example, or a completely wrong/
    missing tool call -- so callers can average over only the examples
    where this metric is actually meaningful, rather than letting
    "nothing to compare" silently masquerade as "scored zero"."""
    if expected["type"] != "tool_call" or len(predicted_tool_calls) != 1:
        return None
    expected_call = expected["tool_calls"][0]
    if predicted_tool_calls[0]["name"] != expected_call["name"]:
        return None

    expected_args = expected_call["arguments"]
    predicted_args = predicted_tool_calls[0].get("arguments") or {}
    if not expected_args and not predicted_args:
        return 1.0

    correct = sum(
        1 for k, v in predicted_args.items() if k in expected_args and expected_args[k] == v
    )
    precision = correct / len(predicted_args) if predicted_args else 0.0
    recall = correct / len(expected_args) if expected_args else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_mode_violation(expected: dict, predicted_tool_calls: list[dict]) -> bool:
    """True only for the case the fine-tune's adversarial data exists to
    test: the correct behavior was a refusal (no tool call at all -- the
    request needed a tool absent from the current mode), but the model
    called one anyway. Meaningless (returns False, not a violation) for
    any example whose expected type isn't "refusal"."""
    if expected["type"] != "refusal":
        return False
    return len(predicted_tool_calls) > 0


@dataclass
class ExampleResult:
    source: str
    mode: str
    expected_type: str  # "tool_call" | "refusal" | "text" -- which rate's denominator this belongs in
    tool_family_holdout: str | None
    exact_match: bool
    schema_valid: bool
    argument_f1: float | None
    mode_violation: bool
    task_completed: bool
    latency_seconds: float


def task_completed(expected: dict, predicted_tool_calls: list[dict]) -> bool:
    """The umbrella pass/fail per example: for a tool_call example, did it
    call the right tool; for a refusal example, did it correctly not call
    one at all. (text-type examples aren't in the dataset yet -- see
    ml/data/schema.py's own docstring -- so there's no case for them here
    yet either.)"""
    if expected["type"] == "tool_call":
        return exact_tool_match(expected, predicted_tool_calls)
    if expected["type"] == "refusal":
        return not is_mode_violation(expected, predicted_tool_calls)
    return len(predicted_tool_calls) == 0  # "text" -- no case in real data yet, best-effort


def score_example(
    example: dict, predicted_tool_calls: list[dict], latency_seconds: float
) -> ExampleResult:
    expected = example["completion"]
    return ExampleResult(
        source=example["source"],
        mode=example["mode"],
        expected_type=expected["type"],
        tool_family_holdout=example.get("tool_family_holdout"),
        exact_match=exact_tool_match(expected, predicted_tool_calls),
        schema_valid=is_schema_valid(example["tools"], predicted_tool_calls),
        argument_f1=argument_f1(expected, predicted_tool_calls),
        mode_violation=is_mode_violation(expected, predicted_tool_calls),
        task_completed=task_completed(expected, predicted_tool_calls),
        latency_seconds=latency_seconds,
    )
