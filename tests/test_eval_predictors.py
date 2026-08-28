"""Tests for ml/eval/predictors.py's make_ollama_predictor -- mocked at
the HTTP boundary (no real Ollama call), same "fake the network, not the
logic" pattern this project already uses for mcp_servers/github and
mcp_servers/browser. make_hf_adapter_predictor needs a real GPU and is
marked `# pragma: no cover`; nothing here tests that one.

parse_tool_call_blocks IS fully testable without a GPU though -- it's
pure text parsing -- and the tests below use the EXACT raw generated
text a real Kaggle adapter produced (Entry 58), not fabricated examples,
to guard against the regression that motivated the fix: correct JSON
missing its <tool_call> wrapper tags was being silently scored as zero
tool calls.
"""

from ml.eval.predictors import make_ollama_predictor, parse_tool_call_blocks


class _FakeResponse:
    def __init__(self, message):
        self._message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": self._message}


class _FakeClient:
    def __init__(self, response, captured):
        self._response = response
        self._captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        self._captured["url"] = url
        self._captured["json"] = json
        return self._response


def _example():
    return {
        "mode": "agent",
        "system_prompt": "You are DevPilot.",
        "tools": [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}}],
        "messages": [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "earlier reply"}],
        "user_request": "what's in requirements.txt?",
        "completion": {"type": "tool_call", "tool_calls": [{"name": "read_file", "arguments": {"path": "requirements.txt"}}], "text": ""},
        "source": "real_trace",
        "tool_family_holdout": "read_file",
    }


def test_parses_a_real_tool_call_response(monkeypatch):
    captured = {}
    fake_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "requirements.txt"}}}],
    }

    import ml.eval.predictors as mod
    monkeypatch.setattr(mod.httpx2, "Client", lambda **kwargs: _FakeClient(_FakeResponse(fake_message), captured))

    predict = make_ollama_predictor("qwen2.5:7b-instruct")
    result = predict(_example())

    assert result == [{"name": "read_file", "arguments": {"path": "requirements.txt"}}]


def test_returns_empty_list_when_no_tool_calls(monkeypatch):
    import ml.eval.predictors as mod
    monkeypatch.setattr(
        mod.httpx2, "Client",
        lambda **kwargs: _FakeClient(_FakeResponse({"role": "assistant", "content": "I can't do that."}), {}),
    )

    predict = make_ollama_predictor("qwen2.5:7b-instruct")
    assert predict(_example()) == []


def test_sends_the_real_example_context_to_ollama(monkeypatch):
    captured = {}
    import ml.eval.predictors as mod
    monkeypatch.setattr(
        mod.httpx2, "Client",
        lambda **kwargs: _FakeClient(_FakeResponse({"role": "assistant", "content": "ok"}), captured),
    )

    predict = make_ollama_predictor("qwen2.5:7b-instruct", host="127.0.0.1", port=11434)
    predict(_example())

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    body = captured["json"]
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "You are DevPilot."}
    # Prior conversation turns must be forwarded, not dropped.
    assert {"role": "user", "content": "earlier"} in body["messages"]
    assert body["messages"][-1] == {"role": "user", "content": "what's in requirements.txt?"}
    assert body["tools"] == _example()["tools"]


class TestParseToolCallBlocks:
    def test_parses_the_wrapped_hermes_style_format(self):
        text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "config.py"}}\n</tool_call>'
        assert parse_tool_call_blocks(text) == [{"name": "read_file", "arguments": {"path": "config.py"}}]

    def test_returns_empty_list_for_plain_refusal_text(self):
        text = "I'm in Plan mode right now, which means I can't access any tools yet."
        assert parse_tool_call_blocks(text) == []

    def test_falls_back_to_bare_json_with_no_wrapper_tags(self):
        """Real raw output from a Kaggle-trained adapter (Entry 58): the
        exact correct tool + arguments, but with no <tool_call> wrapper
        at all -- previously silently scored as zero tool calls."""
        text = '{"name": "git_log", "arguments": {"limit": 10}}'
        assert parse_tool_call_blocks(text) == [{"name": "git_log", "arguments": {"limit": 10}}]

    def test_bare_json_fallback_takes_only_the_first_call_of_a_repeated_run(self):
        """Real raw output from the same adapter: instead of stopping
        after one call, it repeated many (including a hallucinated
        'list_files' tool not in DevPilot's real tool set). Only the
        first, real call should be extracted -- repeating several calls
        is a distinct generation-quality problem this parser should not
        paper over as if they were all valid."""
        text = (
            '{"name": "list_directory", "arguments": {"path": "./backend"}}\n'
            '{"name": "list_files", "arguments": {"owner": "your_username", "repo": "your_repo", '
            '"path": "./backend", "ref": "main"}}'
        )
        assert parse_tool_call_blocks(text) == [{"name": "list_directory", "arguments": {"path": "./backend"}}]

    def test_handles_nested_argument_objects_correctly(self):
        """A naive non-greedy regex (\\{.*?\\}) would truncate at the
        FIRST closing brace, cutting off a nested arguments value -- this
        checks the brace-counting extractor gets the real, full object."""
        text = '<tool_call>{"name": "execute_read_query", "arguments": {"filters": {"status": "done"}}}</tool_call>'
        assert parse_tool_call_blocks(text) == [
            {"name": "execute_read_query", "arguments": {"filters": {"status": "done"}}}
        ]

    def test_wrapped_format_takes_priority_over_bare_json_elsewhere_in_the_text(self):
        text = 'some preamble {"name": "wrong", "arguments": {}} more text <tool_call>{"name": "read_file", "arguments": {"path": "x"}}</tool_call>'
        assert parse_tool_call_blocks(text) == [{"name": "read_file", "arguments": {"path": "x"}}]

    def test_ignores_invalid_json_and_returns_empty_list(self):
        assert parse_tool_call_blocks("<tool_call>{not valid json}</tool_call>") == []
        assert parse_tool_call_blocks("no braces here at all") == []
