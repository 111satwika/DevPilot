"""Stage 9: a lightweight planner -- the first client that orchestrates
multiple MCP servers in one script.

The plan is fixed and hand-written, not LLM-generated -- no LLM integration
exists in this project yet. This proves the *execution* half of planning:
represent a plan as data, show it before running it, execute steps across
multiple real servers, pass results forward, keep going past an individual
step failure. Full rationale logged in DevPilot_AI_Implementation_Log.html
Entry 13.

Stage 10: wired in SessionMemory (this run only) and ProjectMemory
(persists across separate runs via a JSON file) -- see memory.py and
Entry 14.

Entry 41 (gap-fix, 2026-08-23): the original plan ran `git status`/`git
log` through Terminal MCP's execute_command, since Git MCP didn't exist
yet at Stage 9. Terminal no longer allows `git` at all (it was a real
approval-bypass around Git MCP's own gates -- see mcp_servers/terminal/
server.py), so this plan now calls Git MCP's git_status/git_log directly
instead -- still a 2-server plan (filesystem + git), same teaching point,
now via the tool that's actually meant to answer it.

Run from the project root: python -m mcp_client.planner
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from mcp import Client

from mcp_client.memory import ProjectMemory, SessionMemory
from mcp_servers.filesystem.server import mcp as filesystem_mcp
from mcp_servers.git.server import mcp as git_mcp

SERVERS = {"filesystem": filesystem_mcp, "git": git_mcp}


@dataclass
class Step:
    description: str
    server: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


PLAN = [
    Step("Check what's changed", "git", "git_status", {}),
    Step("Check recent history", "git", "git_log", {"limit": 5}),
    Step(
        "Read current dependencies",
        "filesystem",
        "read_file",
        {"path": "requirements.txt"},
    ),
]


def _print_plan(steps: list[Step]) -> None:
    print("Plan:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step.description}  [{step.server}.{step.tool}]")


async def _run_step(step: Step) -> dict:
    server = SERVERS[step.server]
    async with Client(server) as client:
        result = await client.call_tool(step.tool, step.args)

    if result.is_error:
        return {"ok": False, "error": result.content[0].text}

    if result.structured_content is not None:
        value = result.structured_content
    else:
        raw = result.content[0].text
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            value = raw

    return {"ok": True, "value": value}


async def run_plan(steps: list[Step], session: SessionMemory | None = None) -> list[dict]:
    """Execute steps in order. A failed step is logged, not fatal --
    later steps in this plan don't depend on earlier ones succeeding."""
    _print_plan(steps)
    results = []
    for i, step in enumerate(steps, 1):
        print(f"\nRunning step {i}: {step.description}...")
        outcome = await _run_step(step)
        print("  done." if outcome["ok"] else f"  FAILED: {outcome['error']}")
        if session is not None:
            session.record(step.description, outcome)
        results.append({"step": step, "outcome": outcome})
    return results


def _build_report(results: list[dict]) -> str:
    """Plain string formatting over the collected results -- NOT an LLM
    summary. No LLM is wired into this project yet; calling this a
    'summary' would overclaim what the code does."""
    lines = ["# Project Status Report", ""]
    for entry in results:
        step, outcome = entry["step"], entry["outcome"]
        lines.append(f"## {step.description}")
        if not outcome["ok"]:
            lines.append(f"FAILED: {outcome['error']}")
        else:
            value = outcome["value"]
            if isinstance(value, dict) and "stdout" in value:
                text = value["stdout"].strip() or value["stderr"].strip() or "(no output)"
            elif isinstance(value, dict) and "result" in value:
                text = str(value["result"])
            else:
                text = str(value).strip()
            lines.append(text or "(no output)")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    # ProjectMemory: JSON-file-backed, persists across separate process runs.
    project = ProjectMemory()
    print("Project memory (persists across runs):")
    for key in ("project", "run_command", "test_command", "stages_completed"):
        print(f"  {key}: {project.get(key)}")

    # SessionMemory: in-process only, starts empty every single run,
    # regardless of what ProjectMemory remembers.
    session = SessionMemory()

    results = await run_plan(PLAN, session=session)
    print("\n" + "=" * 60)
    print(_build_report(results))

    print("\n" + "=" * 60)
    print(session.summary())

    stages_completed = project.get("stages_completed", 0) + 1
    project.set("stages_completed", stages_completed)
    print(f"\nProjectMemory updated: stages_completed = {stages_completed} (persisted to disk)")

    # Failure-handling test: same plan, but step 3 targets a file that
    # doesn't exist -- confirms the orchestrator logs the failure and
    # still completes the plan instead of crashing.
    print("\n" + "=" * 60)
    print("Failure-handling test (step 3 targets a nonexistent file):\n")
    broken_plan = [
        PLAN[0],
        PLAN[1],
        Step(
            "Read current dependencies",
            "filesystem",
            "read_file",
            {"path": "does_not_exist.txt"},
        ),
    ]
    broken_session = SessionMemory()
    broken_results = await run_plan(broken_plan, session=broken_session)
    print("\n" + "=" * 60)
    print(_build_report(broken_results))
    print("\n" + "=" * 60)
    print(broken_session.summary())


if __name__ == "__main__":
    asyncio.run(main())
