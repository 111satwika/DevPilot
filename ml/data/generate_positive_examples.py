"""Generates realistic, correctly-labeled tool_call examples across MANY
different real tools -- the fix for a real problem Entry 51 found: the
dataset's only tool_call examples came from generate_adversarial.py's
explore-first bucket, and every single one of those targets
list_directory. Combined with ~27 refusal examples, that's a roughly
4.5:1 class imbalance, and the working hypothesis for the fine-tuned
model's exact_tool_match regression (0%, both examples wrong) is that
this imbalance taught it "refuse when in doubt" at the cost of actual
tool-calling proficiency.

Same no-API-key, hand-written-template + slot-filling approach already
proven in generate_adversarial.py, deliberately NOT the teacher-
distillation script -- this fixes the class-imbalance hypothesis
directly and immediately, without depending on an API key nobody has
supplied. (generate_teacher_traces.py still exists for whoever wants to
add real natural-language variety and volume on top of this later.)

Argument shapes are taken directly from each tool's real signature
(mcp_servers/*/server.py), not guessed -- a wrong argument key here would
train the model to call real tools with fake parameters.

A real bug caught before this ever ran once, not after: the first draft
picked slot values for the request TEXT and for the tool-call ARGUMENTS
via two independent rng.choice() calls -- meaning a request could say
"what's in config.py?" while the labeled correct tool call pointed at a
completely different file, since nothing forced the two draws to agree.
Fixed by picking one set of slot values ONCE per example and building
both the text and the arguments from that same dict, so they can't
diverge by construction.
"""

import argparse
import asyncio
import random
from pathlib import Path

from llm.agent import SYSTEM_PROMPT, _discover_tools, _tools_for_mode
from ml.data.schema import Completion, ToolCall, TrainingExample, write_jsonl

RANDOM_SEED = 43  # different from generate_adversarial.py's 42, deliberately -- independent generator

SLOTS = {
    "filename": ["config.py", "README.md", "package.json", "app.py", "utils.py", "index.ts"],
    "dirname": ["src", "tests", "mcp_servers", "backend"],
    "ext": [".py", ".ts", ".json", ".md"],
    "n_commits": [5, 10, 3],
    "table": ["stages", "users", "orders"],
    "container": ["web-1", "app-dev", "test-container"],
    "pr_number": [1, 42, 7],
    "sha": ["abc1234", "deadbeef", "0f1e2d3"],
    "web_query": ["FastAPI dependency injection", "Python asyncio best practices", "SQLite read-only connection"],
    "url": ["https://docs.python.org/3/library/asyncio.html", "https://fastapi.tiangolo.com/tutorial/"],
}

OWNER_REPOS = [
    ("modelcontextprotocol", "python-sdk"),
    ("anthropics", "anthropic-sdk-python"),
    ("openai", "openai-python"),
]


def _draw_slots(rng: random.Random) -> dict:
    """One consistent set of slot values for one example -- text and
    arguments are both built from this SAME dict (see module docstring),
    never from independent rng.choice() calls."""
    slots = {name: rng.choice(values) for name, values in SLOTS.items()}
    owner, repo = rng.choice(OWNER_REPOS)
    slots["owner"] = owner
    slots["repo"] = repo
    return slots


# Each entry: tool name -> list of (request_template, arguments_builder).
# arguments_builder takes the SAME slots dict _fill() renders the request
# text from, so a filename/table/owner/etc. mentioned in the request is
# guaranteed to be the exact one named in the labeled tool call.
POSITIVE_TEMPLATES: dict[str, list[tuple[str, "callable"]]] = {
    "read_file": [
        ("what's in {filename}?", lambda s: {"path": s["filename"]}),
        ("show me the contents of {filename}", lambda s: {"path": s["filename"]}),
        ("open {filename} and tell me what it does", lambda s: {"path": s["filename"]}),
    ],
    "list_directory": [
        ("what files are in {dirname}?", lambda s: {"path": s["dirname"]}),
        ("list the contents of the {dirname} folder", lambda s: {"path": s["dirname"]}),
    ],
    "get_file_info": [
        ("how big is {filename}?", lambda s: {"path": s["filename"]}),
        ("when was {filename} last modified?", lambda s: {"path": s["filename"]}),
    ],
    "search_files": [
        ("find files with {ext} in the name", lambda s: {"query": s["ext"], "path": "."}),
        ("search this project for files matching {ext}", lambda s: {"query": s["ext"], "path": "."}),
    ],
    "git_status": [
        ("what's changed in the working tree?", lambda s: {}),
        ("show me git status", lambda s: {}),
        ("are there any uncommitted changes?", lambda s: {}),
    ],
    "git_log": [
        ("show me the last {n_commits} commits", lambda s: {"limit": s["n_commits"]}),
        ("what's the recent commit history?", lambda s: {"limit": 10}),
    ],
    "git_diff": [
        ("show me the diff for {filename}", lambda s: {"path": s["filename"]}),
        ("what's changed in the working tree overall?", lambda s: {}),
    ],
    "git_list_branches": [
        ("what branches exist?", lambda s: {}),
        ("list all local and remote branches", lambda s: {}),
    ],
    "list_tables": [
        ("what tables are in the database?", lambda s: {}),
        ("show me the database schema overview", lambda s: {}),
    ],
    "describe_table": [
        ("what columns does the {table} table have?", lambda s: {"table": s["table"]}),
        ("describe the {table} table", lambda s: {"table": s["table"]}),
    ],
    "execute_read_query": [
        ("how many rows are in the {table} table?", lambda s: {"query": f"SELECT COUNT(*) FROM {s['table']}"}),
    ],
    "list_containers": [
        ("what docker containers are running?", lambda s: {"all": True}),
        ("list all containers, including stopped ones", lambda s: {"all": True}),
    ],
    "inspect_container": [
        ("show me details about the {container} container", lambda s: {"container": s["container"]}),
    ],
    "get_container_logs": [
        ("show me logs for {container}", lambda s: {"container": s["container"], "tail": 100}),
    ],
    "get_repository": [
        ("tell me about the {owner}/{repo} repository on GitHub", lambda s: {"owner": s["owner"], "repo": s["repo"]}),
    ],
    "list_pull_requests": [
        ("what pull requests are open on {owner}/{repo}?", lambda s: {"owner": s["owner"], "repo": s["repo"], "state": "open"}),
    ],
    "get_pull_request": [
        ("show me PR #{pr_number} on {owner}/{repo}", lambda s: {"owner": s["owner"], "repo": s["repo"], "number": s["pr_number"]}),
    ],
    "list_commits": [
        ("show me the recent commits on {owner}/{repo}", lambda s: {"owner": s["owner"], "repo": s["repo"]}),
    ],
    "get_commit": [
        ("show me commit {sha} on {owner}/{repo}", lambda s: {"owner": s["owner"], "repo": s["repo"], "sha": s["sha"]}),
    ],
    "search_web": [
        ("search the web for {web_query}", lambda s: {"query": s["web_query"]}),
        ("look up documentation on {web_query}", lambda s: {"query": s["web_query"]}),
    ],
    "fetch_page": [
        ("fetch the content of {url}", lambda s: {"url": s["url"]}),
    ],
}


async def generate_positive_examples(
    mode: str = "agent", system_prompt: str = SYSTEM_PROMPT
) -> list[TrainingExample]:
    rng = random.Random(RANDOM_SEED)
    all_tools = await _discover_tools()
    tools_for_mode = _tools_for_mode(all_tools, mode)
    offered_names = {t["function"]["name"] for t in tools_for_mode}

    missing = set(POSITIVE_TEMPLATES) - offered_names
    assert not missing, f"POSITIVE_TEMPLATES has entries for tools not offered in mode={mode!r}: {missing}"

    examples = []
    for tool_name, phrasings in POSITIVE_TEMPLATES.items():
        for phrasing, args_builder in phrasings:
            slots = _draw_slots(rng)
            request = phrasing.format(**slots)
            arguments = args_builder(slots)
            examples.append(
                TrainingExample(
                    mode=mode,
                    system_prompt=system_prompt,
                    tools=tools_for_mode,
                    messages=[],
                    user_request=request,
                    completion=Completion(
                        type="tool_call",
                        tool_calls=[ToolCall(name=tool_name, arguments=arguments)],
                    ),
                    source="positive_template",
                    tool_family_holdout=tool_name,
                )
            )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ml/data/out/positive.jsonl"))
    parser.add_argument("--mode", default="agent")
    args = parser.parse_args()

    examples = asyncio.run(generate_positive_examples(mode=args.mode))
    write_jsonl(examples, args.out)

    by_tool: dict[str, int] = {}
    for ex in examples:
        by_tool[ex.tool_family_holdout] = by_tool.get(ex.tool_family_holdout, 0) + 1
    print(f"Generated {len(examples)} positive tool_call examples across {len(by_tool)} tools -> {args.out}")
    for tool, count in sorted(by_tool.items()):
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
