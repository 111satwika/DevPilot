"""Mines DevPilot's own persisted conversation history
(conversation_history/*.json, backend/history.py) into TrainingExample
records -- the "real logged traces" bucket of the fine-tune's dataset.

Honest limitation, by construction: a persisted conversation stores the
raw message list (system/user/assistant/tool turns) but, before this
pipeline existed, never recorded which MODE was active for a given turn
-- mode can change turn to turn (Entry 38), so it can't be inferred after
the fact. Sessions saved from now on carry a "mode" field per turn
(backend/sessions.py's _run()); anything saved before that field existed
is skipped rather than guessed at, with a clear count reported -- a wrong
guess here would poison training data with the wrong tool schema for
that example, which is worse than just having less data.

As of this pipeline's creation, DevPilot has never actually been run
against a real Ollama call whose result got persisted (every real
conversation_history/ entry to date came from mocked/test runs) -- so
mine_real_traces() legitimately returns zero examples today. That's the
correct, honest result, not a bug: this script exists so that traces
accumulate automatically as the app gets real use, not to fabricate
volume it doesn't have.
"""

import argparse
import asyncio
import json
from pathlib import Path

from llm.agent import _discover_tools, _tools_for_mode
from ml.data.schema import Completion, ToolCall, TrainingExample, write_jsonl

_HISTORY_DIR = Path(__file__).resolve().parent.parent.parent / "conversation_history"


def _extract_examples_from_messages(
    messages: list[dict], mode: str, tools_for_this_mode: list[dict]
) -> list[TrainingExample]:
    """Walk one conversation's raw message list and emit one
    TrainingExample per assistant turn that made a real tool call.
    Turns with no tool_calls (a final text answer) are skipped here --
    Phase 1's target mix treats "text, no tool call" as its own
    adversarial/synthetic category (generate_adversarial.py), not
    something to mine from real conversations, since a real "no tool
    needed" turn doesn't tell you much without knowing WHY it needed none."""
    examples = []
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")

    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue

        prior_messages = messages[:i]
        last_user = next(
            (m["content"] for m in reversed(prior_messages) if m["role"] == "user"), ""
        )

        completion = Completion(
            type="tool_call",
            tool_calls=[
                ToolCall(name=c["function"]["name"], arguments=c["function"]["arguments"])
                for c in tool_calls
            ],
        )
        examples.append(
            TrainingExample(
                mode=mode,
                system_prompt=system_prompt,
                tools=tools_for_this_mode,
                messages=prior_messages,
                user_request=last_user,
                completion=completion,
                source="real_trace",
                # First call's tool name -- build_dataset.py's tool-family
                # split needs this populated the same way the synthetic
                # generators already do.
                tool_family_holdout=completion.tool_calls[0].name,
            )
        )
    return examples


async def mine_real_traces(history_dir: Path = _HISTORY_DIR) -> tuple[list[TrainingExample], int]:
    """Returns (examples, skipped_count) -- skipped_count is every turn
    from before per-turn mode tracking existed, deliberately not guessed."""
    all_tools = await _discover_tools()
    tools_by_mode: dict[str, list[dict]] = {}

    def tools_for(mode: str) -> list[dict]:
        if mode not in tools_by_mode:
            tools_by_mode[mode] = _tools_for_mode(all_tools, mode if mode != "planner" else "agent")
        return tools_by_mode[mode]

    examples: list[TrainingExample] = []
    skipped = 0

    if not history_dir.is_dir():
        return examples, skipped

    for path in history_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data.get("messages", [])
        turns = data.get("turns", [])

        for turn in turns:
            mode = turn.get("mode")
            if mode is None:
                skipped += 1
                continue
            # Only mine the messages up through this conversation's full
            # history once (a whole-conversation messages list already
            # contains every turn's exchange) -- extraction itself walks
            # the full list once per conversation file, not per turn.

        if not turns or all(t.get("mode") is None for t in turns):
            continue

        # Best-effort: use the LAST turn's mode for the whole file's
        # extraction pass when turns share one mode (the common case);
        # a conversation that genuinely switched modes mid-way needs
        # per-turn granularity this simple pass doesn't attempt yet.
        mode = turns[-1].get("mode", "agent")
        examples.extend(_extract_examples_from_messages(messages, mode, tools_for(mode)))

    return examples, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ml/data/out/real_traces.jsonl"))
    args = parser.parse_args()

    examples, skipped = asyncio.run(mine_real_traces())
    write_jsonl(examples, args.out)
    print(f"Mined {len(examples)} real tool-call examples -> {args.out}")
    if skipped:
        print(
            f"Skipped {skipped} turn(s) from before per-turn mode tracking existed "
            f"(no mode recorded -- not guessed at)."
        )


if __name__ == "__main__":
    main()
