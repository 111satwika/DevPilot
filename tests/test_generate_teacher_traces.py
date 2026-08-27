"""Tests for ml/data/generate_teacher_traces.py's request-building and
response-parsing logic -- no real OpenAI/Anthropic call is made (no API
key exists in this environment); a fake TeacherClient stands in, same
"fake the network boundary, not the logic" pattern this project already
uses for mcp_servers/github and mcp_servers/browser.
"""

import pytest

from ml.data.generate_teacher_traces import (
    TeacherClient,
    _build_teacher,
    _parse_teacher_response,
    generate_teacher_examples,
)


class _FakeTeacher(TeacherClient):
    def __init__(self, response_by_call_index):
        self._responses = response_by_call_index
        self.prompts = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses[len(self.prompts) - 1]


def test_parse_teacher_response_extracts_valid_pairs():
    raw = '[{"request": "read config.py", "arguments": {"path": "config.py"}}]'
    pairs = _parse_teacher_response(raw, "read_file")
    assert pairs == [("read config.py", {"path": "config.py"})]


def test_parse_teacher_response_tolerates_markdown_fence():
    raw = 'Sure!\n```json\n[{"request": "x", "arguments": {"a": 1}}]\n```\n'
    pairs = _parse_teacher_response(raw, "some_tool")
    assert pairs == [("x", {"a": 1})]


def test_parse_teacher_response_skips_malformed_entries():
    raw = '[{"request": "ok", "arguments": {}}, {"request": 123, "arguments": {}}, "not a dict"]'
    pairs = _parse_teacher_response(raw, "some_tool")
    assert pairs == [("ok", {})]


def test_parse_teacher_response_returns_empty_on_garbage():
    assert _parse_teacher_response("not json at all", "some_tool") == []


@pytest.mark.asyncio
async def test_generate_teacher_examples_builds_training_examples(monkeypatch):
    async def fake_discover_tools():
        return [
            {"function": {"name": "read_file", "description": "reads a file", "parameters": {"type": "object"}}}
        ]

    import ml.data.generate_teacher_traces as mod
    monkeypatch.setattr(mod, "_discover_tools", fake_discover_tools)

    fake = _FakeTeacher([
        '[{"request": "show me config.py", "arguments": {"path": "config.py"}}, '
        '{"request": "read the readme", "arguments": {"path": "README.md"}}]'
    ])

    examples = await generate_teacher_examples(fake, per_tool=2, mode="agent")

    assert len(examples) == 2
    assert examples[0].source == "teacher_generated"
    assert examples[0].completion.type == "tool_call"
    assert examples[0].completion.tool_calls[0].name == "read_file"
    assert examples[0].user_request == "show me config.py"
    assert examples[1].completion.tool_calls[0].arguments == {"path": "README.md"}
    # The prompt sent to the teacher must actually describe the real tool.
    assert "read_file" in fake.prompts[0]
    assert "reads a file" in fake.prompts[0]


def test_build_teacher_fails_closed_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        _build_teacher("openai")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        _build_teacher("anthropic")


def test_build_teacher_rejects_unknown_provider():
    with pytest.raises(SystemExit, match="Unknown teacher provider"):
        _build_teacher("some-other-provider")
