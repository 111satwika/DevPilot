"""Shared training-example schema for the DevPilot tool-calling fine-tune.

One example = one real (or synthesized) turn of DevPilot's own agent loop:
the exact system prompt + tool schemas + user request the model actually
saw, and the completion it should have produced. This shape is chosen
deliberately to match what SFTTrainer needs (a prompt, and a completion to
mask-and-train-on) without any translation step between "what DevPilot
logs" and "what the trainer consumes."

completion.type is one of:
  "tool_call" -- the correct action is to call one or more real tools
  "refusal"   -- the correct action is to NOT call a tool at all (a
                 mode-violation request, e.g. asking Plan mode to write a
                 file) -- this is the case a plain "did it pick the right
                 tool" metric would never test, and the whole reason the
                 adversarial-negatives bucket exists (see
                 generate_adversarial.py).
  "text"      -- a plain final answer, no tool call (e.g. the model
                 already has enough information, or -- per DevPilot's own
                 SYSTEM_PROMPT, Entry 29 -- it should explore via
                 list_directory/read_file rather than ask for
                 clarification on a vague request).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Completion:
    type: str  # "tool_call" | "refusal" | "text"
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "tool_calls": [asdict(c) for c in self.tool_calls],
            "text": self.text,
        }


@dataclass
class TrainingExample:
    mode: str  # "ask" | "plan" | "agent" | "planner"
    system_prompt: str
    tools: list[dict]  # Ollama function-schema shape, exactly what the model saw
    messages: list[dict]  # prior conversation turns (may be empty for a single-turn example)
    user_request: str
    completion: Completion
    source: str  # "real_trace" | "adversarial_mode_violation" | "adversarial_explore" | "teacher_generated"
    tool_family_holdout: str | None = None  # filled in by the dataset splitter, not the generators

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "messages": self.messages,
            "user_request": self.user_request,
            "completion": self.completion.to_dict(),
            "source": self.source,
            "tool_family_holdout": self.tool_family_holdout,
        }


def write_jsonl(examples: list[TrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
