"""Tests for ml/train/format_example.py against the REAL
Qwen2.5-Coder-1.5B-Instruct tokenizer (downloaded from Hugging Face --
no GPU needed to test tokenization/chat-template logic, only to train).
Skipped automatically if no network access to Hugging Face is available
(e.g. a CI runner with no outbound internet), rather than failing.
"""

import json

import pytest

from ml.data.schema import Completion, ToolCall, TrainingExample
from ml.train.format_example import IGNORE_LABEL, format_example

transformers = pytest.importorskip("transformers")

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as exc:  # noqa: BLE001 -- no network / HF unreachable in this environment
        pytest.skip(f"Could not load {BASE_MODEL} tokenizer (no network?): {exc}")


def _tool_schema(name="read_file"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }


def test_tool_call_completion_masks_exactly_the_prompt(tokenizer):
    example = TrainingExample(
        mode="agent",
        system_prompt="You are DevPilot.",
        tools=[_tool_schema()],
        messages=[],
        user_request="what is in requirements.txt?",
        completion=Completion(type="tool_call", tool_calls=[ToolCall(name="read_file", arguments={"path": "requirements.txt"})]),
        source="real_trace",
    ).to_dict()

    result = format_example(example, tokenizer)

    assert result is not None
    assert len(result.input_ids) == len(result.labels) == len(result.attention_mask)

    masked_count = sum(1 for l in result.labels if l == IGNORE_LABEL)
    unmasked_count = sum(1 for l in result.labels if l != IGNORE_LABEL)
    assert masked_count > 0  # the prompt really is masked
    assert unmasked_count > 0  # the completion really is trained on

    # The unmasked tail, decoded, must actually contain the real tool call.
    completion_ids = [t for t in result.labels if t != IGNORE_LABEL]
    decoded_completion = tokenizer.decode(completion_ids)
    assert "read_file" in decoded_completion
    assert "requirements.txt" in decoded_completion
    parsed = json.loads(decoded_completion.split("<tool_call>")[1].split("</tool_call>")[0].strip())
    assert parsed == {"name": "read_file", "arguments": {"path": "requirements.txt"}}


def test_refusal_completion_masks_exactly_the_prompt(tokenizer):
    example = TrainingExample(
        mode="plan",
        system_prompt="You are DevPilot.",
        tools=[],
        messages=[],
        user_request="commit these changes",
        completion=Completion(type="refusal", text="I can't do that in Plan mode."),
        source="adversarial_mode_violation",
    ).to_dict()

    result = format_example(example, tokenizer)

    assert result is not None
    completion_ids = [t for t in result.labels if t != IGNORE_LABEL]
    decoded_completion = tokenizer.decode(completion_ids)
    assert "Plan mode" in decoded_completion
    # A refusal must never contain a tool_call block -- if it does, the
    # completion/prompt split logic (or the example itself) is wrong.
    assert "<tool_call>" not in decoded_completion


def test_prior_conversation_messages_are_part_of_the_masked_prompt(tokenizer):
    example = TrainingExample(
        mode="agent",
        system_prompt="You are DevPilot.",
        tools=[_tool_schema()],
        messages=[
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ],
        user_request="what is in requirements.txt?",
        completion=Completion(type="tool_call", tool_calls=[ToolCall(name="read_file", arguments={"path": "requirements.txt"})]),
        source="real_trace",
    ).to_dict()

    result = format_example(example, tokenizer)
    prompt_only_ids = [i for i, l in zip(result.input_ids, result.labels) if l == IGNORE_LABEL]
    decoded_prompt = tokenizer.decode(prompt_only_ids)
    assert "earlier question" in decoded_prompt
    assert "earlier answer" in decoded_prompt


def test_oversized_example_is_dropped_not_truncated(tokenizer):
    huge_content = "x " * 5000
    example = TrainingExample(
        mode="agent", system_prompt="sys", tools=[], messages=[],
        user_request=huge_content,
        completion=Completion(type="text", text="ok"),
        source="real_trace",
    ).to_dict()

    assert format_example(example, tokenizer, max_seq_len=64) is None
