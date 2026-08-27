"""Generates the bulk "single tool call" and "multi-step chain" buckets
of the fine-tune's dataset by asking a real teacher model (GPT-4o or
Claude) to produce realistic (user request -> correct tool call) pairs
against DevPilot's own real tool schemas -- this is the knowledge
distillation step, worth naming as exactly that on a resume rather than
just "fine-tuning."

Deliberately does NOT generate mode-violation-refusal or explore-first
examples -- generate_adversarial.py already produces those, grounded
directly in DevPilot's real mode/tool config rather than a model's
guess at what a refusal should look like, which is both cheaper and more
reliable for that specific bucket.

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY to actually run -- this
script cannot be executed without one (no such key exists in the
environment this was written in). The HTTP request-building and
response-parsing logic is unit-tested against a fake HTTP client
(tests/test_generate_teacher_traces.py), same pattern this project
already uses for mcp_servers/github and mcp_servers/browser -- proving
the plumbing is correct without needing a live teacher call to do it.

Uses raw REST calls via httpx2 (already a project dependency) rather
than adding the full openai/anthropic SDKs, matching this project's
established minimal-dependency preference (see mcp_servers/github/
server.py's own httpx2-over-PyGithub choice).
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import httpx2

from llm.agent import SYSTEM_PROMPT, _discover_tools, _tools_for_mode
from ml.data.schema import Completion, ToolCall, TrainingExample, write_jsonl

REQUEST_TIMEOUT_SECONDS = 60

_GENERATION_INSTRUCTIONS = (
    "You are generating training data for a tool-calling coding assistant. "
    "Given ONE real tool's name, description, and JSON argument schema below, "
    "invent {n} realistic, varied user requests a developer might type that "
    "should each result in exactly this tool being called with sensible, "
    "realistic arguments. Vary the phrasing and the argument values across "
    "examples -- don't repeat the same wording.\n\n"
    "Tool: {tool_name}\n"
    "Description: {tool_description}\n"
    "Argument schema: {tool_schema}\n\n"
    "Respond with ONLY a JSON array, each element shaped exactly like:\n"
    '{{"request": "...", "arguments": {{...matching the schema...}}}}\n'
    "Nothing else, no prose, no markdown fence."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class TeacherClient:
    """Minimal interface -- generate(prompt) -> raw text response."""

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAITeacher(TeacherClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str) -> str:
        async with httpx2.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.9,
                },
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicTeacher(TeacherClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-5"):
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str) -> str:
        async with httpx2.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


def _parse_teacher_response(raw_text: str, tool_name: str) -> list[tuple[str, dict]]:
    """Extracts [(request, arguments), ...] from the teacher's raw reply.
    Tolerates a markdown fence around the JSON even though the prompt asks
    for none, since models don't always follow that instruction exactly."""
    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    pairs = []
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("request"), str) and isinstance(item.get("arguments"), dict):
            pairs.append((item["request"], item["arguments"]))
    return pairs


async def generate_teacher_examples(
    teacher: TeacherClient, per_tool: int = 20, mode: str = "agent"
) -> list[TrainingExample]:
    all_tools = await _discover_tools()
    tools_for_mode = _tools_for_mode(all_tools, mode)
    examples = []

    for tool in tools_for_mode:
        fn = tool["function"]
        prompt = _GENERATION_INSTRUCTIONS.format(
            n=per_tool,
            tool_name=fn["name"],
            tool_description=fn["description"],
            tool_schema=json.dumps(fn["parameters"]),
        )
        raw = await teacher.generate(prompt)
        for request, arguments in _parse_teacher_response(raw, fn["name"]):
            examples.append(
                TrainingExample(
                    mode=mode,
                    system_prompt=SYSTEM_PROMPT,
                    tools=tools_for_mode,
                    messages=[],
                    user_request=request,
                    completion=Completion(
                        type="tool_call",
                        tool_calls=[ToolCall(name=fn["name"], arguments=arguments)],
                    ),
                    source="teacher_generated",
                    tool_family_holdout=fn["name"],
                )
            )
    return examples


def _build_teacher(provider: str) -> TeacherClient:
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set -- can't call the teacher model.")
        return OpenAITeacher(key)
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set -- can't call the teacher model.")
        return AnthropicTeacher(key)
    raise SystemExit(f"Unknown teacher provider: {provider!r} (expected 'openai' or 'anthropic')")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--per-tool", type=int, default=20, help="requests to generate per tool")
    parser.add_argument("--mode", default="agent")
    parser.add_argument("--out", type=Path, default=Path("ml/data/out/teacher_traces.jsonl"))
    args = parser.parse_args()

    teacher = _build_teacher(args.provider)
    examples = asyncio.run(generate_teacher_examples(teacher, per_tool=args.per_tool, mode=args.mode))
    write_jsonl(examples, args.out)
    print(f"Generated {len(examples)} teacher-distilled examples -> {args.out}")


if __name__ == "__main__":
    main()
