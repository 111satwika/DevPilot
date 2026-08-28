"""Tests for ml/eval/harness.py's aggregation/timing logic -- driven with
a fake, instant predictor so these are fast and don't need a real model.
"""

import time

from ml.eval.harness import evaluate


def _example(source="real_trace", mode="agent", expected_tool="read_file", tool_family=None):
    return {
        "mode": mode,
        "system_prompt": "sys",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": expected_tool,
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        "messages": [],
        "user_request": "do something",
        "completion": {"type": "tool_call", "tool_calls": [{"name": expected_tool, "arguments": {}}], "text": ""},
        "source": source,
        "tool_family_holdout": tool_family or expected_tool,
    }


def _refusal_example():
    return {
        "mode": "plan",
        "system_prompt": "sys",
        "tools": [],
        "messages": [],
        "user_request": "write a file",
        "completion": {"type": "refusal", "tool_calls": [], "text": "can't do that"},
        "source": "adversarial_mode_violation",
        "tool_family_holdout": "write_file",
    }


def test_perfect_predictor_scores_100_percent():
    examples = [_example(), _example(expected_tool="list_directory"), _refusal_example()]

    def perfect_predictor(example):
        expected = example["completion"]
        if expected["type"] == "refusal":
            return []
        return expected["tool_calls"]

    report = evaluate("perfect", perfect_predictor, examples, verbose=False)

    assert report.num_examples == 3
    assert report.exact_tool_match_rate == 1.0
    assert report.task_completion_rate == 1.0
    assert report.mode_violation_rate == 0.0
    assert report.mean_argument_f1 == 1.0


def test_always_wrong_predictor_scores_badly():
    examples = [_example(), _refusal_example()]

    def bad_predictor(example):
        return [{"name": "wrong_tool", "arguments": {}}]

    report = evaluate("bad", bad_predictor, examples, verbose=False)

    assert report.exact_tool_match_rate == 0.0
    assert report.task_completion_rate == 0.0
    assert report.mode_violation_rate == 1.0  # the refusal example got a tool call it shouldn't have


def test_latency_is_measured_around_the_predictor_call():
    def slow_predictor(example):
        time.sleep(0.05)
        return []

    report = evaluate("slow", slow_predictor, [_example()], verbose=False)

    assert report.p50_latency_seconds >= 0.05
    assert report.p95_latency_seconds >= 0.05


def test_empty_example_list_does_not_crash():
    report = evaluate("empty", lambda ex: [], [], verbose=False)
    assert report.num_examples == 0
    assert report.exact_tool_match_rate == 0.0
    assert report.mean_argument_f1 is None


def test_summary_line_handles_none_f1_gracefully():
    report = evaluate("empty", lambda ex: [], [], verbose=False)
    assert "n/a" in report.summary_line()


def test_report_round_trips_through_to_dict():
    report = evaluate("x", lambda ex: [], [_refusal_example()], verbose=False)
    d = report.to_dict()
    assert d["model_name"] == "x"
    assert isinstance(d["per_example"], list)
    assert d["per_example"][0]["mode"] == "plan"
