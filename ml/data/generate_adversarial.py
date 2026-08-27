"""Generates the two adversarial buckets from the fine-tune's data plan --
built entirely from DevPilot's OWN real mode/tool configuration
(llm/agent.py's PLAN_MODE_ALLOWED_TOOLS, GATED_TOOLS, MODE_PROMPTS), not a
teacher API, so this needs no API key and can run right now.

1. Mode-violation refusals: a request that would need a tool NOT offered
   in the current mode. The correct completion is a refusal explaining
   why, never a tool call -- this is the case a plain "did it pick the
   right tool" accuracy metric never tests, and it's the bucket that
   connects the fine-tune directly back to DevPilot's own security model
   (Entry 38's structural mode enforcement, Entries 41/42/45's gated-tool
   approval). Teaching the model to refuse here is defense in depth on
   top of the API-layer enforcement that already makes the tool
   unreachable -- belt AND suspenders, not a replacement for either.

2. Explore-first examples: DevPilot's real SYSTEM_PROMPT (Entry 29) does
   NOT want the model asking for clarification on a vague request ("fix
   my code", "optimize this") -- it wants list_directory called first.
   The generic version of this fine-tune plan calls this bucket
   "ambiguous -> clarify"; that would directly contradict Entry 29's own
   fix if copied verbatim, so this generator produces the *opposite*
   correction on purpose: vague request -> tool_call(list_directory),
   not a clarifying question.

Both buckets are built by combining a small set of hand-written request
templates with realistic slot values (filenames, branch names, commit
messages, package names, ...) -- enough combinatorial variety to reach
real volume without needing an LLM to phrase each one.
"""

import argparse
import asyncio
import itertools
import random
from pathlib import Path

from llm.agent import (
    GATED_TOOLS,
    PLAN_MODE_ALLOWED_TOOLS,
    SYSTEM_PROMPT,
    _discover_tools,
    _tools_for_mode,
)
from ml.data.schema import Completion, ToolCall, TrainingExample, write_jsonl

RANDOM_SEED = 42  # reproducible dataset generation

# Requests that map to a specific mutating/gated tool -- {slot} placeholders
# filled combinatorially from SLOT_VALUES below. Grounded in each tool's
# REAL signature (mcp_servers/*/server.py), not guessed.
MUTATING_REQUEST_TEMPLATES: dict[str, list[str]] = {
    "write_file": [
        "create a file called {filename} with a short description of this project",
        "write {filename} with the text 'hello world'",
        "save this summary to {filename}",
    ],
    "execute_command": [
        "install {package} as a dependency",
        "run npm install {package}",
        "run pip install {package}",
    ],
    "git_commit": [
        "commit the staged changes with message '{commit_message}'",
        "commit this with the message: {commit_message}",
    ],
    "git_push": [
        "push these commits to {remote}",
        "push my changes to the {remote} remote",
    ],
    "git_delete_branch": [
        "delete the branch {branch_name}",
        "remove the {branch_name} branch, it's not needed anymore",
    ],
    "build_image": [
        "build a docker image tagged {tag} from the Dockerfile",
        "build the docker image and call it {tag}",
    ],
    "run_container": [
        "start a container named {container_name} from the {image} image",
        "run the {image} image as a container called {container_name}",
    ],
    "stop_container": [
        "stop the {container_name} container",
        "shut down container {container_name}",
    ],
}

# Requests that map to a real, ungated but Plan-mode-excluded action
# (git_create_branch is ungated per Entry 16, but deliberately excluded
# from Plan mode per Entry 38 -- it's still a real mutation).
PLAN_EXCLUDED_ONLY_TEMPLATES: dict[str, list[str]] = {
    "git_create_branch": [
        "create a new branch called {branch_name}",
        "make a branch named {branch_name} and switch to it",
    ],
}

# A handful of requests mapping to plain READ-ONLY tools -- used only for
# Ask mode (which excludes everything, mutating or not) so its refusal
# examples aren't exclusively about mutations.
READ_ONLY_REQUEST_TEMPLATES: dict[str, list[str]] = {
    "read_file": ["what's in {filename}?", "show me the contents of {filename}"],
    "git_log": ["what are the last few commits?", "show me the recent git history"],
    "list_tables": ["what tables are in the database?"],
}

SLOT_VALUES = {
    "filename": ["notes.txt", "summary.md", "config.json", "output.log", "README_new.md"],
    "package": ["lodash", "requests", "left-pad", "axios", "numpy"],
    "commit_message": ["fix bug", "update dependencies", "wip", "add feature", "cleanup"],
    "remote": ["origin", "upstream"],
    "branch_name": ["feature/login", "bugfix/typo", "release-1.2", "experiment"],
    "tag": ["myapp:latest", "backend:v2", "test-image"],
    "container_name": ["web-1", "test-container", "app-dev"],
    "image": ["nginx", "python:3.12", "myapp:latest"],
}

VAGUE_REQUESTS = [
    "analyze this code",
    "optimize the project",
    "clean this up",
    "make this better",
    "review my code",
    "what's wrong with this project?",
    "improve performance",
    "help me understand this codebase",
]

_MODE_REFUSAL_TEXT = {
    "ask": (
        "I'm in Ask mode right now, which has no tool access at all -- I can't "
        "check, read, or look anything up. Switch to Plan or Agent mode if you "
        "want me to actually investigate this."
    ),
    "plan": (
        "I'm in Plan mode, which is read-only by design -- I can explore and "
        "propose an approach, but I can't write, modify, execute, or otherwise "
        "change anything. Switch to Agent mode if you want this carried out."
    ),
}


def _fill(template: str, rng: random.Random) -> str:
    slots = {name: rng.choice(values) for name, values in SLOT_VALUES.items()}
    return template.format(**slots)


def _mode_violation_examples(
    mode: str, templates: dict[str, list[str]], tools: list[dict], system_prompt: str, rng: random.Random
) -> list[TrainingExample]:
    examples = []
    for tool_name, phrasings in templates.items():
        for phrasing in phrasings:
            request = _fill(phrasing, rng)
            examples.append(
                TrainingExample(
                    mode=mode,
                    system_prompt=system_prompt,
                    tools=tools,
                    messages=[],
                    user_request=request,
                    completion=Completion(type="refusal", text=_MODE_REFUSAL_TEXT[mode]),
                    source="adversarial_mode_violation",
                    tool_family_holdout=tool_name,
                )
            )
    return examples


def _explore_first_examples(mode: str, tools: list[dict], system_prompt: str) -> list[TrainingExample]:
    examples = []
    for request in VAGUE_REQUESTS:
        examples.append(
            TrainingExample(
                mode=mode,
                system_prompt=system_prompt,
                tools=tools,
                messages=[],
                user_request=request,
                completion=Completion(
                    type="tool_call",
                    tool_calls=[ToolCall(name="list_directory", arguments={"path": "."})],
                ),
                source="adversarial_explore",
                tool_family_holdout="list_directory",
            )
        )
    return examples


async def generate_adversarial_examples(system_prompt: str = SYSTEM_PROMPT) -> list[TrainingExample]:
    # Catches drift: if a new gated tool is ever added to llm/agent.py
    # without a matching template here, fail loudly rather than silently
    # under-covering it.
    missing = GATED_TOOLS - set(MUTATING_REQUEST_TEMPLATES)
    assert not missing, f"GATED_TOOLS has no request template(s) for: {missing}"

    rng = random.Random(RANDOM_SEED)
    all_tools = await _discover_tools()

    ask_tools = _tools_for_mode(all_tools, "ask")  # always []
    plan_tools = _tools_for_mode(all_tools, "plan")
    agent_tools = _tools_for_mode(all_tools, "agent")

    examples: list[TrainingExample] = []

    # Ask mode: everything is a violation -- both mutating and read-only requests.
    examples += _mode_violation_examples("ask", MUTATING_REQUEST_TEMPLATES, ask_tools, system_prompt, rng)
    examples += _mode_violation_examples("ask", READ_ONLY_REQUEST_TEMPLATES, ask_tools, system_prompt, rng)

    # Plan mode: only genuinely excluded tools -- GATED_TOOLS (all mutating
    # templates) plus the Plan-specific exclusions (git_create_branch).
    # Sanity-check against the real PLAN_MODE_ALLOWED_TOOLS set rather
    # than assuming the template dicts stayed in sync with it.
    plan_excluded_templates = {
        name: tmpl for name, tmpl in MUTATING_REQUEST_TEMPLATES.items()
        if name not in PLAN_MODE_ALLOWED_TOOLS
    }
    plan_excluded_templates.update(
        {name: tmpl for name, tmpl in PLAN_EXCLUDED_ONLY_TEMPLATES.items() if name not in PLAN_MODE_ALLOWED_TOOLS}
    )
    examples += _mode_violation_examples("plan", plan_excluded_templates, plan_tools, system_prompt, rng)

    # Explore-first: vague requests in Agent mode should trigger
    # list_directory, never a clarifying question (Entry 29).
    examples += _explore_first_examples("agent", agent_tools, system_prompt)

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ml/data/out/adversarial.jsonl"))
    args = parser.parse_args()

    examples = asyncio.run(generate_adversarial_examples())
    write_jsonl(examples, args.out)

    by_source = {}
    for ex in examples:
        by_source[ex.source] = by_source.get(ex.source, 0) + 1
    print(f"Generated {len(examples)} adversarial examples -> {args.out}")
    for source, count in by_source.items():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
