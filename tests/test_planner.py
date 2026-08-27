"""Tests for Entry 46's planning engine (llm/agent.py's "planner" mode):
generate a plan, hold it for a whole-plan human decision, then either
stop (declined) or fall through into the existing tool-calling loop
(approved). No real Ollama is reachable in this environment, so
_ollama_chat is mocked -- these tests are about the planner's own control
flow (decision plumbing, JSON parsing, fallback parsing), not about model
quality.
"""

import asyncio
import json

import pytest

from llm import agent
from mcp_servers import audit


def _fake_chat_returning(content: str, tool_calls=None):
    """Builds a stand-in for agent._ollama_chat with a fixed response."""
    def _fake(messages, tools):
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return _fake


class TestParsePlanSteps:
    def test_parses_fenced_json_array(self):
        content = 'Sure, here:\n```json\n["Read package.json", "Run tests"]\n```\n'
        assert agent._parse_plan_steps(content) == ["Read package.json", "Run tests"]

    def test_parses_bare_json_array_without_fence(self):
        content = '["Step one", "Step two", "Step three"]'
        assert agent._parse_plan_steps(content) == ["Step one", "Step two", "Step three"]

    def test_caps_at_max_plan_steps(self):
        steps = [f"Step {i}" for i in range(agent.MAX_PLAN_STEPS + 5)]
        content = f"```json\n{json.dumps(steps)}\n```"
        assert len(agent._parse_plan_steps(content)) == agent.MAX_PLAN_STEPS

    def test_falls_back_to_numbered_lines_when_not_valid_json(self):
        content = "1. Read the config file\n2. Run the build\n3. Report results"
        result = agent._parse_plan_steps(content)
        assert result == ["Read the config file", "Run the build", "Report results"]

    def test_falls_back_to_bulleted_lines(self):
        content = "- First do this\n- Then do that"
        assert agent._parse_plan_steps(content) == ["First do this", "Then do that"]

    def test_never_returns_empty(self):
        assert agent._parse_plan_steps("") == ["Investigate the request and report back."]


@pytest.mark.asyncio
async def test_generate_plan_calls_ollama_with_no_tools(monkeypatch):
    """The generation call must never itself be able to invoke a tool --
    tools=[] is passed to _ollama_chat regardless of how many real tools
    exist."""
    captured = {}

    def fake_chat(messages, tools):
        captured["tools"] = tools
        return {"role": "assistant", "content": '```json\n["Do the thing"]\n```'}

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)
    steps = await agent._generate_plan("do something", [
        {"function": {"name": "read_file", "description": "reads a file"}}
    ])
    assert steps == ["Do the thing"]
    assert captured["tools"] == []


@pytest.mark.asyncio
async def test_planner_mode_pauses_for_plan_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_AUDIT_DIR", tmp_path / "audit_log")  # don't pollute the real project's audit log
    monkeypatch.setattr(
        agent, "_ollama_chat",
        _fake_chat_returning('```json\n["Read requirements.txt", "Summarize dependencies"]\n```'),
    )

    async def fake_discover_tools():
        return []

    monkeypatch.setattr(agent, "_discover_tools", fake_discover_tools)

    session = agent.AgentSession(id="s1")

    async def approver():
        while session.pending_plan is None:
            await asyncio.sleep(0.01)
        assert session.status == "awaiting_plan_approval"
        assert session.plan == ["Read requirements.txt", "Summarize dependencies"]
        session.pending_plan.decision.set_result(False)  # decline

    task = asyncio.create_task(approver())
    result = await agent.ask("what does this project depend on?", session=session, mode="planner")
    await task

    assert "not approved" in result.answer
    assert result.tool_calls == []
    assert session.status == "running"  # reset after the decision resolves


@pytest.mark.asyncio
async def test_planner_mode_executes_after_approval(monkeypatch, tmp_path):
    """Once approved, execution should fall through into the normal
    tool-calling loop -- proven here by having the second (post-approval)
    model turn return a plain final answer with no further tool calls."""
    monkeypatch.setattr(audit, "_AUDIT_DIR", tmp_path / "audit_log")  # don't pollute the real project's audit log
    calls = {"n": 0}

    def fake_chat(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant", "content": '```json\n["Just answer directly"]\n```'}
        # Second call is the actual execution turn -- respond with a
        # final answer, no tool_calls, proving the loop reached here.
        return {"role": "assistant", "content": "All done.", "tool_calls": None}

    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)

    async def fake_discover_tools():
        return []

    monkeypatch.setattr(agent, "_discover_tools", fake_discover_tools)

    session = agent.AgentSession(id="s2")

    async def approver():
        while session.pending_plan is None:
            await asyncio.sleep(0.01)
        session.pending_plan.decision.set_result(True)  # approve

    task = asyncio.create_task(approver())
    result = await agent.ask("say hi", session=session, mode="planner")
    await task

    assert result.answer == "All done."
    assert calls["n"] == 2  # one generation call, one execution call


@pytest.mark.asyncio
async def test_planner_mode_without_session_fails_closed():
    result = await agent.ask("do something", session=None, mode="planner")
    assert "requires an active session" in result.answer
