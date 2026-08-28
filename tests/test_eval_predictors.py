"""Tests for ml/eval/predictors.py's make_ollama_predictor -- mocked at
the HTTP boundary (no real Ollama call), same "fake the network, not the
logic" pattern this project already uses for mcp_servers/github and
mcp_servers/browser. make_hf_adapter_predictor needs a real GPU and is
marked `# pragma: no cover`; nothing here tests that one.
"""

from ml.eval.predictors import make_ollama_predictor


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
