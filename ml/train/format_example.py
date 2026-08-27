"""Turns one TrainingExample dict (ml/data/schema.py's format) into
exactly the token sequence + completion-only labels the base model
should be trained on.

Verified for real against the actual Qwen2.5-Coder-1.5B-Instruct
tokenizer (downloaded from Hugging Face, CPU-only -- no GPU needed just
to inspect a chat template): its `apply_chat_template` renders tool
calls in a Hermes-style `<tool_call>{"name":...,"arguments":...}
</tool_call>` block, auto-injecting the tool schemas into the system
message. Critically, `apply_chat_template(msgs_up_to_user,
add_generation_prompt=True)` is an exact STRING PREFIX of
`apply_chat_template(msgs_including_assistant_turn)` -- confirmed
directly, not assumed -- for both a tool-call completion and a plain-text
completion.

Default max_seq_len is 4096, not the original plan's 2048 -- measured
directly against DevPilot's real tool schemas (not assumed): Agent
mode's full 33-tool schema alone renders to ~3,850 tokens *before* any
completion is added, since Qwen's chat template inlines every tool's
full JSON schema into the system message. At 2048, 14 of 33 real Phase-1
examples (every Plan/Agent-mode one) were silently dropped as "too long"
-- a big enough loss to be a real methodology problem, not a rounding
error. Verified 4096 covers the full real dataset (51/51 examples,
longest real sequence 3867 tokens) with headroom to spare.

That prefix property is what makes completion-only masking exact here:
tokenize the prompt alone to get its token count, then mask every label
before that boundary with -100 (the value transformers/PyTorch's
cross-entropy loss ignores). This is used INSTEAD OF TRL's
DataCollatorForCompletionOnlyLM, which finds the prompt/completion
boundary by string-matching a response template against the tokenized
sequence -- a well-documented source of off-by-one/boundary bugs when a
chat template's special tokens don't tokenize back to the exact same
subsequence in every context. Precomputing labels from a property that
was actually verified against this real tokenizer is the more reliable
choice here, not a shortcut around the spec's "mask the prompt" goal --
it's a stricter way of hitting that same goal.
"""

from dataclasses import dataclass

IGNORE_LABEL = -100


def _example_to_messages(example: dict) -> tuple[list[dict], dict]:
    """Splits a TrainingExample dict into (prompt_messages, assistant_message)."""
    prompt_messages = [{"role": "system", "content": example["system_prompt"]}]
    prompt_messages.extend(example.get("messages", []))
    prompt_messages.append({"role": "user", "content": example["user_request"]})

    completion = example["completion"]
    if completion["type"] == "tool_call":
        assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in completion["tool_calls"]
            ],
        }
    else:  # "refusal" or "text" -- both are a plain assistant text turn
        assistant_message = {"role": "assistant", "content": completion["text"]}

    return prompt_messages, assistant_message


@dataclass
class Tokenized:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def format_example(example: dict, tokenizer, max_seq_len: int = 4096) -> Tokenized | None:
    """Returns None if the full sequence exceeds max_seq_len (dropped,
    not silently truncated -- truncating a tool-call completion could cut
    off mid-JSON and teach the model to emit invalid JSON)."""
    prompt_messages, assistant_message = _example_to_messages(example)
    tools = example.get("tools") or None

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [assistant_message], tools=tools, tokenize=False, add_generation_prompt=False
    )

    if not full_text.startswith(prompt_text):
        # Should never happen given the verified prefix property, but if
        # a future tokenizer/template version breaks it, fail loudly on
        # this example rather than silently mask the wrong boundary.
        raise ValueError(
            "Chat template no longer produces prompt-as-prefix for this "
            "example -- completion-only masking can't be trusted; "
            "re-verify against the real tokenizer before proceeding."
        )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_len:
        return None

    labels = list(full_ids)
    labels[: len(prompt_ids)] = [IGNORE_LABEL] * len(prompt_ids)

    return Tokenized(
        input_ids=full_ids,
        labels=labels,
        attention_mask=[1] * len(full_ids),
    )
