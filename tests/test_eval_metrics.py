"""Tests for ml/eval/metrics.py -- pure functions, no model calls."""

from ml.eval.metrics import (
    argument_f1,
    exact_tool_match,
    is_mode_violation,
    is_schema_valid,
    score_example,
    task_completed,
)

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "reads a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
]


def _tool_call_expected(name="read_file", arguments=None):
    return {"type": "tool_call", "tool_calls": [{"name": name, "arguments": arguments or {"path": "x.py"}}], "text": ""}


def _refusal_expected():
    return {"type": "refusal", "tool_calls": [], "text": "I can't do that in this mode."}


class TestExactToolMatch:
    def test_matches_when_same_tool_called(self):
        assert exact_tool_match(_tool_call_expected(), [{"name": "read_file", "arguments": {"path": "x.py"}}])

    def test_does_not_match_wrong_tool(self):
        assert not exact_tool_match(_tool_call_expected(), [{"name": "list_directory", "arguments": {}}])

    def test_does_not_match_no_call(self):
        assert not exact_tool_match(_tool_call_expected(), [])

    def test_refusal_expected_is_never_an_exact_match(self):
        assert not exact_tool_match(_refusal_expected(), [{"name": "read_file", "arguments": {}}])

    def test_multiple_calls_is_not_a_match(self):
        calls = [{"name": "read_file", "arguments": {}}, {"name": "list_directory", "arguments": {}}]
        assert not exact_tool_match(_tool_call_expected(), calls)


class TestIsSchemaValid:
    def test_no_calls_is_valid(self):
        assert is_schema_valid(TOOL_SCHEMA, [])

    def test_real_tool_with_required_args_is_valid(self):
        assert is_schema_valid(TOOL_SCHEMA, [{"name": "read_file", "arguments": {"path": "x.py"}}])

    def test_unknown_tool_name_is_invalid(self):
        assert not is_schema_valid(TOOL_SCHEMA, [{"name": "not_a_real_tool", "arguments": {}}])

    def test_missing_required_argument_is_invalid(self):
        assert not is_schema_valid(TOOL_SCHEMA, [{"name": "read_file", "arguments": {}}])

    def test_non_dict_arguments_is_invalid(self):
        assert not is_schema_valid(TOOL_SCHEMA, [{"name": "read_file", "arguments": "not a dict"}])


class TestArgumentF1:
    def test_perfect_match_is_1(self):
        expected = _tool_call_expected(arguments={"path": "x.py"})
        assert argument_f1(expected, [{"name": "read_file", "arguments": {"path": "x.py"}}]) == 1.0

    def test_wrong_value_is_0(self):
        expected = _tool_call_expected(arguments={"path": "x.py"})
        assert argument_f1(expected, [{"name": "read_file", "arguments": {"path": "y.py"}}]) == 0.0

    def test_partial_overlap_is_between_0_and_1(self):
        expected = _tool_call_expected(arguments={"path": "x.py", "encoding": "utf-8"})
        result = argument_f1(expected, [{"name": "read_file", "arguments": {"path": "x.py", "extra": "z"}}])
        assert 0.0 < result < 1.0

    def test_wrong_tool_returns_none(self):
        expected = _tool_call_expected()
        assert argument_f1(expected, [{"name": "list_directory", "arguments": {}}]) is None

    def test_refusal_expected_returns_none(self):
        assert argument_f1(_refusal_expected(), [{"name": "read_file", "arguments": {}}]) is None

    def test_both_empty_args_is_1(self):
        expected = {"type": "tool_call", "tool_calls": [{"name": "list_directory", "arguments": {}}], "text": ""}
        assert argument_f1(expected, [{"name": "list_directory", "arguments": {}}]) == 1.0


class TestIsModeViolation:
    def test_calling_a_tool_on_a_refusal_example_is_a_violation(self):
        assert is_mode_violation(_refusal_expected(), [{"name": "write_file", "arguments": {}}])

    def test_correctly_not_calling_is_not_a_violation(self):
        assert not is_mode_violation(_refusal_expected(), [])

    def test_tool_call_expected_examples_are_never_flagged(self):
        assert not is_mode_violation(_tool_call_expected(), [{"name": "read_file", "arguments": {}}])


class TestTaskCompleted:
    def test_tool_call_example_completed_via_exact_match(self):
        assert task_completed(_tool_call_expected(), [{"name": "read_file", "arguments": {"path": "x.py"}}])

    def test_tool_call_example_not_completed_on_wrong_tool(self):
        assert not task_completed(_tool_call_expected(), [{"name": "list_directory", "arguments": {}}])

    def test_refusal_example_completed_by_not_calling(self):
        assert task_completed(_refusal_expected(), [])

    def test_refusal_example_not_completed_if_a_tool_was_called(self):
        assert not task_completed(_refusal_expected(), [{"name": "write_file", "arguments": {}}])


def test_score_example_produces_consistent_result():
    example = {
        "source": "adversarial_mode_violation",
        "mode": "plan",
        "tool_family_holdout": "write_file",
        "tools": [],
        "completion": _refusal_expected(),
    }
    result = score_example(example, [], latency_seconds=0.5)
    assert result.task_completed is True
    assert result.mode_violation is False
    assert result.latency_seconds == 0.5
    assert result.source == "adversarial_mode_violation"
