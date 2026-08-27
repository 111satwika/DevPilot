"""Tests for ml/data/mine_real_traces.py against fabricated
conversation_history files shaped exactly like backend/history.py's real
save_turn() output -- there's no real usage data yet (DevPilot has never
actually been run against a live Ollama call that got persisted in this
environment), so this is the only way to prove the miner's extraction
logic is correct before it ever sees real data.
"""

import json

import pytest

from ml.data.mine_real_traces import mine_real_traces


def _write_conversation(directory, filename, messages, turns):
    (directory / filename).write_text(
        json.dumps(
            {
                "id": filename.replace(".json", ""),
                "title": "test conversation",
                "created_at": "2026-08-27T00:00:00+00:00",
                "updated_at": "2026-08-27T00:00:00+00:00",
                "turns": turns,
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_mines_one_example_from_a_real_shaped_conversation(tmp_path):
    messages = [
        {"role": "system", "content": "You are DevPilot..."},
        {"role": "user", "content": "what does this project depend on?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "requirements.txt"}}, "id": "1"}
            ],
        },
        {"role": "tool", "content": "mcp\nhttpx2\nfastapi\n", "tool_call_id": "1"},
        {"role": "assistant", "content": "This project depends on mcp, httpx2, and fastapi.", "tool_calls": None},
    ]
    turns = [
        {
            "question": "what does this project depend on?",
            "answer": "This project depends on mcp, httpx2, and fastapi.",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "requirements.txt"}, "result": "..."}],
            "mode": "agent",
        }
    ]
    _write_conversation(tmp_path, "conv1.json", messages, turns)

    examples, skipped = await mine_real_traces(history_dir=tmp_path)

    assert skipped == 0
    assert len(examples) == 1
    ex = examples[0]
    assert ex.mode == "agent"
    assert ex.source == "real_trace"
    assert ex.user_request == "what does this project depend on?"
    assert ex.completion.type == "tool_call"
    assert ex.completion.tool_calls[0].name == "read_file"
    assert ex.completion.tool_calls[0].arguments == {"path": "requirements.txt"}
    # The prompt (prior messages) must stop BEFORE the tool-calling
    # assistant turn -- otherwise the model would be trained on seeing
    # its own answer already in context.
    assert all(m.get("tool_calls") is None or m["role"] != "assistant" for m in ex.messages)
    # Full tool schema for "agent" mode should be non-empty (real tools
    # discovered from the real MCP servers).
    assert len(ex.tools) > 10


@pytest.mark.asyncio
async def test_skips_turns_with_no_recorded_mode(tmp_path):
    """A conversation saved before mode-per-turn tracking existed has no
    "mode" key -- must be skipped and counted, never guessed at."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {}}, "id": "1"}]},
    ]
    turns = [{"question": "q", "answer": "a", "tool_calls": []}]  # no "mode" key
    _write_conversation(tmp_path, "old_conv.json", messages, turns)

    examples, skipped = await mine_real_traces(history_dir=tmp_path)

    assert examples == []
    assert skipped == 1


@pytest.mark.asyncio
async def test_returns_empty_when_no_history_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    examples, skipped = await mine_real_traces(history_dir=missing)
    assert examples == []
    assert skipped == 0


@pytest.mark.asyncio
async def test_final_text_only_turn_produces_no_example(tmp_path):
    """A turn with no tool call at all isn't mined here -- that's the
    adversarial/synthetic generator's job (generate_adversarial.py), not
    something inferred from a real conversation without knowing why no
    tool was needed."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!", "tool_calls": None},
    ]
    turns = [{"question": "hi", "answer": "Hello!", "tool_calls": [], "mode": "ask"}]
    _write_conversation(tmp_path, "chit_chat.json", messages, turns)

    examples, skipped = await mine_real_traces(history_dir=tmp_path)
    assert examples == []
    assert skipped == 0
