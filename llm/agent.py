"""DevPilot's first real tool-selecting agent.

Ollama decides which MCP tools to call, in what order, from a free-text
request -- no hardcoded plan (contrast with Stage 9's planner.py). This is
a small, hand-rolled equivalent of Anthropic's Tool Runner, since Ollama
has no such helper: discover tools, offer them to the model, execute
whatever it calls, feed results back, repeat until it stops calling tools.

Scope: only read-only tools from Filesystem + Database MCP (no approval-
gated tools routed through this loop yet). Full rationale logged in
DevPilot_AI_Implementation_Log.html Entry 18.

Entry 19: added a system prompt after testing showed the model could
select and chain tools correctly but still drop part of a multi-part
question during final-answer synthesis. Prompting-level mitigation only,
tried once, not a structural guarantee -- see Entry 19 for why.

Entry 21: ask() now returns AgentResult (answer + tool_calls) instead of a
bare string, so the frontend can render a tool-execution trace -- the
trace already existed as console prints, this just surfaces it to callers.

Entry 23: all 6 MCP servers wired in, not just Filesystem + Database.
Every server's own ctx.elicit()-gated tools were excluded explicitly via
EXCLUDED_TOOLS, same mechanism Entry 18 used for write_file -- adding a
server never required solving human-in-the-loop-for-an-LLM-driven-call,
it just required knowing which of its tools are already gated.

Entry 24: real human-in-the-loop approval. The 7 gated tools (renamed
EXCLUDED_TOOLS -> GATED_TOOLS) are back on the model's tool list -- it can
now genuinely propose a write_file/build_image/git_commit/etc. call. A
gated call is routed through a real stdio ClientSession opened just for
that call, with an elicitation_callback that pauses on an AgentSession's
asyncio.Future until a human resolves it (via the FastAPI backend's
/session/{id}/approve route) -- the same accept/decline shape proven live
in Stage 8's stage8_client.py, not a client-side shortcut that skips the
server's own ctx.elicit() gate. ask() accepts an optional AgentSession;
without one, gated calls fail closed with an explanatory message instead
of either silently running or silently vanishing.

Entry 29: SYSTEM_PROMPT extended to tell the model to explore the project
(list_directory, then read_file) before answering a vague request that
doesn't name a specific file, instead of immediately asking the user for
one -- a real live run of "analyze and optimize code" with no specifics
had produced zero tool calls and a generic "please provide the code"
non-answer. Prompting-level mitigation, tried once, same discipline as
Entry 19.

Entry 30: _ollama_chat's blocking subprocess.run() call, made directly
from inside async def ask() with no await/executor, was freezing the
entire single-threaded event loop for the whole duration of every model
turn -- during a slow question, the server couldn't even answer /health,
which the VS Code extension polls and reacts to by trying to spawn a
competing backend on the same port. Fixed with asyncio.to_thread() at the
call site so the blocking call runs on a worker thread instead of the
event loop itself.

Entry 31: every prior call to ask() built a brand-new messages list from
scratch -- zero memory of any earlier question, even within what looked
like one ongoing conversation in the panel. AgentSession now carries
messages across calls; passing a session whose messages list is already
populated continues that real conversation instead of starting over with
no context of what was just discussed.

Entry 33: AgentSession also carries turns (display-friendly per-exchange
records) so backend/history.py can persist a whole conversation to disk,
scoped per project -- a conversation now survives "New conversation" and
a backend restart, not just this process's lifetime.

Entry 34: confirmed via Ollama's /api/ps and nvidia-smi that this model
runs CPU-only, no GPU path exists in this environment -- a real,
permanent speed ceiling, not a bug. Raised REQUEST_TIMEOUT_SECONDS
300->600 and added options.num_ctx=8192 to give the 26-tool schema plus
growing conversation history real headroom instead of Ollama's small
4096 default.

Entry 35: confirmed live that the model can describe a tool call as JSON
text in its answer instead of actually invoking Ollama's real
tool-calling mechanism -- a claimed write_file that never ran, with a
fabricated "confirmation" of a file that was never written. When a turn
has no real tool_calls, _extract_hallucinated_tool_calls() now looks for
a fenced ```json block naming a real, currently-offered tool and, if
found, routes it through the exact same execution path a genuine tool
call would use (still approval-gated for gated tools) instead of
returning the false claim as a final answer.

Entry 38: ask() takes a mode ("ask" | "plan" | "agent"), matching
Copilot Chat's mode dropdown. _tools_for_mode() filters what's even
offered to the model this turn -- Ask gets nothing, Plan gets an
explicit read-only allow-list, Agent is unchanged. This is the real
enforcement (Ollama can't call a tool that isn't in the request), not
just a prompting request the model could ignore -- and it also closes
Entry 35's hallucination-rescue path for excluded tools, since that path
only trusts names already in the offered tool list. No auto-accept-edits
option exists or is planned -- every gated mutation stays
manual-approval-only in every mode, unchanged since Entry 24.

Entry 46 (2026-08-24): a real planning engine, closing the gap the design
doc itself named as a Future Enhancement ("Add lightweight planner") and
finishing Concept #19's separation ("plan generation vs. plan execution
are separable" -- Stage 9's planner.py proved execution only, against a
hardcoded plan). New "planner" mode: _generate_plan() asks the model for
a short, ordered, human-readable plan (a JSON array of step
descriptions, no tool calls made yet) given the user's request and the
full tool list as context. The plan is shown to the user and held at a
new AgentSession status ("awaiting_plan_approval") until a human
approves or declines the *whole plan* -- a second, coarser-grained
approval layer sitting in front of the existing per-tool-call one
(GATED_TOOLS calls inside an approved plan still separately pause for
their own approval when actually executed, unchanged). On approval, the
plan is folded into the turn's message as guidance and execution falls
through into the exact same iterative tool-calling loop every other mode
already uses -- deliberately not a separate step-executor, so real
intermediate results (a file's real content, a command's real exit code)
still drive what the model actually does next, rather than locking
execution to arguments guessed before anything was inspected. Full
rationale and design tradeoffs in DevPilot_AI_Learning_Log.md.

Entry 41 (gap-fix, 2026-08-23): two additions.

1. `execute_command` moved into GATED_TOOLS. Terminal MCP's own allow-list
   used to let `git` run with zero approval, completely bypassing Git
   MCP's dedicated commit/push/branch-delete approval gates -- git was
   removed from Terminal's allow-list entirely (see
   mcp_servers/terminal/server.py), and mutating npm/pip subcommands
   (install, uninstall, ...) now require approval the same way. Routing
   ALL execute_command calls (not just mutating ones) through the gated
   stdio path also closes an unrelated secret-leak: the in-memory Client
   path used for every ungated tool runs inside the live backend
   process and inherits its full environment, so a `python -c
   "import os;print(os.environ)"` call could have dumped GITHUB_TOKEN or
   any other secret straight into the model's context. stdio_client's
   fixed OS-level env allowlist (verified in Entry 22) closes that for
   free.

2. Every tool call -- gated or not -- is now recorded to
   mcp_servers/audit.py's append-only per-project log (arguments and
   result preview secret-redacted). Previously the only record of what
   the AI actually did was console output, not persisted or queryable
   after the fact -- a real gap against the design doc's own audit-log
   requirement.
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field

import httpx2
import mcp.types as types
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_servers import audit
from mcp_servers.browser.server import mcp as browser_mcp
from mcp_servers.database.server import mcp as database_mcp
from mcp_servers.docker.server import mcp as docker_mcp
from mcp_servers.filesystem.server import mcp as filesystem_mcp
from mcp_servers.git.server import mcp as git_mcp
from mcp_servers.github.server import mcp as github_mcp
from mcp_servers.terminal.server import mcp as terminal_mcp
from mcp_servers.workspace import forwarded_env

# Entry 44: this used to resolve a Docker container's IP through a WSL
# `docker inspect` bridge (OLLAMA_CONTAINER, _resolve_ollama_host below,
# both removed) -- a deployment topology this environment doesn't
# actually have. Confirmed directly, repeatedly: Ollama runs as a native
# Windows install, reachable at 127.0.0.1 with no WSL/Docker bridge at
# all -- the same address ml/eval/predictors.py's make_ollama_predictor
# has been using successfully via direct HTTP this entire project. This
# was a real bug in the primary application path (untouched by this
# project's own test suite, which only ever monkeypatches _ollama_chat
# wholesale) -- found via portfolio review, not by anyone running the
# app and hitting it.
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:7b-instruct"
# Entry 34: confirmed via Ollama's own /api/ps ("size_vram": 0) and a
# direct nvidia-smi check (not found in WSL or the container) that this
# model runs on CPU-only inference -- no GPU path exists in this
# environment at all. That's a real, permanent speed ceiling, not a bug;
# 300s (set for the smaller Entry 18/19 tool schema, before Entries 23/24
# grew it to 26 tools) was cutting off requests that would have finished.
REQUEST_TIMEOUT_SECONDS = 600
# Entry 34 raised this 4096 -> 8192 to reduce truncation risk against the
# 26-tool schema. Entry 37 reverted it: real evidence showed timeouts
# recurring *more* often afterward -- Ollama's own /api/ps reported the
# loaded model ~400MB bigger at num_ctx=8192 than at 4096, on a machine
# where WSL free memory was repeatedly measured near zero. Truncation was
# a theoretical risk that was never actually observed; the extra memory
# pressure was a real, measured cost on the actual bottleneck. Reverted
# to Ollama's own default rather than guessing at a middle value.
OLLAMA_CONTEXT_LENGTH = 4096
# Entry 29: was 5. A real run on an actual multi-folder project (Entry
# 29's fix made the model genuinely explore via list_directory instead of
# giving up immediately) used all 5 turns just browsing directories and
# never reached reading a file or answering. Raised to give real
# browse-then-read-then-answer sequences room to finish, at the cost of
# more time on questions that do end up needing every turn.
MAX_ITERATIONS = 12
# Entry 46: caps a generated plan's length -- a runaway/degenerate plan
# (the model listing 30 trivial steps) would just mean 30x the manual
# approval-reading effort for no real benefit; same "bound the loop"
# instinct as MAX_ITERATIONS, applied to plan generation instead of tool
# calling.
MAX_PLAN_STEPS = 8

SYSTEM_PROMPT = (
    "You are DevPilot AI's tool-using assistant. Use the available tools to "
    "gather facts before answering -- never guess, and never answer from "
    "memory when a tool could confirm the real answer. "
    "If the user's question has multiple parts, before giving your final "
    "answer, explicitly re-read the original question and check that your "
    "answer addresses every part of it -- not just the part you most "
    "recently gathered information about. Do not silently drop a part of "
    "the question just because you already answered a different part. "
    "If the user's request is broad or doesn't name a specific file (for "
    "example 'analyze this code' or 'optimize the project'), do not ask "
    "them to paste the code or specify a file first -- explore the open "
    "project yourself: call list_directory to see what's there, then "
    "read_file on whichever files actually look relevant, before giving "
    "your answer. Only ask the user for specifics if exploring the "
    "project genuinely doesn't turn up anything relevant to their request. "
    "When you want to use a tool, you MUST call it through the real "
    "tool-calling mechanism -- never write out a tool call as JSON text in "
    "your answer instead of actually invoking it. Never claim you created, "
    "wrote, modified, or deleted a file unless you actually called the "
    "corresponding tool and got a real result back confirming it worked."
)

SERVERS = {
    "filesystem": filesystem_mcp,
    "database": database_mcp,
    "terminal": terminal_mcp,
    "docker": docker_mcp,
    "git": git_mcp,
    "github": github_mcp,
    "browser": browser_mcp,
}

# Every server splits its own tools into ungated and ctx.elicit()-gated
# ones. These 7 are the gated ones, confirmed against each server's real
# @mcp.tool() names, not assumed. A gated call is routed through a real
# stdio session with an elicitation callback (_call_gated_tool), not the
# in-memory Client used for everything else.
# Entry 41: execute_command joined this set -- not because every call
# needs approval (most don't), but because routing it through the same
# real-stdio mechanism as the others closes a secret-leak vector inherent
# to the in-memory path, and lets it request approval itself for mutating
# npm/pip subcommands. See mcp_servers/terminal/server.py for the actual
# approval logic.
GATED_TOOLS = {
    "write_file", "build_image", "run_container", "stop_container",
    "git_commit", "git_delete_branch", "git_push", "execute_command",
}

GATED_TOOL_MODULE = {
    "write_file": "mcp_servers.filesystem.server",
    "build_image": "mcp_servers.docker.server",
    "run_container": "mcp_servers.docker.server",
    "stop_container": "mcp_servers.docker.server",
    "git_commit": "mcp_servers.git.server",
    "git_delete_branch": "mcp_servers.git.server",
    "git_push": "mcp_servers.git.server",
    "execute_command": "mcp_servers.terminal.server",
}

# Entry 38: Ask/Plan/Agent modes, matching Copilot Chat's mode dropdown.
# Ask -- zero tools, pure conversation. Plan -- an explicit allow-list of
# read-only tools (not a denylist -- a new tool added later is excluded
# by default until explicitly added here). Agent -- today's full
# behavior, unchanged. Plan deliberately excludes execute_command (can
# run allow-listed but still side-effecting commands like `pip install`)
# and git_create_branch (a real mutation, ungated since Entry 16) on top
# of every already-gated tool -- nothing in Plan mode can change
# anything, by construction.
PLAN_MODE_ALLOWED_TOOLS = {
    "read_file", "list_directory", "get_file_info", "search_files",
    "list_tables", "describe_table", "execute_read_query",
    "list_containers", "inspect_container", "get_container_logs",
    "git_status", "git_log", "git_diff", "git_list_branches",
    # Entry 43: GitHub MCP grew from 1 to 8 tools (design doc §8's full
    # read-only set) -- all seven new ones are read-only, same as
    # get_repository, so they belong in Plan mode too.
    "get_repository", "list_files", "search_code", "get_file",
    "list_commits", "get_commit", "list_pull_requests", "get_pull_request",
    "search_web", "fetch_page",
}

MODE_PROMPTS = {
    "ask": (
        "You are in Ask mode: answer directly from this conversation. "
        "You have no tools available -- do not claim to check, read, or "
        "look anything up."
    ),
    "plan": (
        "You are in Plan mode: explore the project using only the "
        "read-only tools available to you and produce a clear, concrete "
        "plan of what should be done and why. Do not attempt to write, "
        "modify, delete, or execute anything -- that's not possible in "
        "this mode. If the user wants the plan carried out, tell them to "
        "switch to Agent mode."
    ),
    "agent": "",
    # Entry 46: this is the instruction used for the EXECUTION half of
    # Planner mode (after the human has already approved a shown plan) --
    # generation itself uses a separate, dedicated prompt (see
    # PLAN_GENERATION_PROMPT below), not this one.
    "planner": "",
}

# Entry 46: used only to GENERATE a plan -- a one-shot, tool-free text
# response (tools=[] is passed to Ollama for this call), never used for
# actually executing anything. Asks for a JSON array so parsing is
# mechanical, same fenced-block convention Entry 35's hallucination
# rescue already established for this codebase.
PLAN_GENERATION_PROMPT = (
    "You are DevPilot AI's planning assistant. Given the user's request "
    "and the list of tools available below, do NOT call any tools or take "
    "any action yet. Instead, produce a short, ordered plan (at most "
    f"{MAX_PLAN_STEPS} steps) describing, in plain language, what you "
    "will do to accomplish the request -- e.g. \"Read package.json to "
    "find the test command\", \"Run the test suite\", \"Inspect the "
    "failure output\". Each step should be a single, concrete action, not "
    "a vague goal. Respond with ONLY a JSON array of short strings inside "
    "a fenced code block, nothing else, for example:\n"
    "```json\n"
    '["Read package.json to find the test command", "Run the test suite", '
    '"Explain any failures found"]\n'
    "```"
)

_tool_server: dict[str, str] = {}


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class PendingApproval:
    """A gated tool call waiting on a human decision."""
    tool: str
    arguments: dict
    message: str
    decision: "asyncio.Future[bool]"


@dataclass
class PendingPlanApproval:
    """Entry 46: a *whole generated plan* waiting on a human decision --
    coarser-grained than PendingApproval above (which gates one tool
    call). Approving a plan doesn't skip individual GATED_TOOLS calls
    encountered while carrying it out; both layers apply independently."""
    steps: list[str]
    decision: "asyncio.Future[bool]"


@dataclass
class AgentSession:
    """Tracks one conversation. Status: running -> awaiting_approval (0+
    times) -> done | error, then back to running if continued with
    another message. messages holds the real conversation history
    (Entry 31) -- ask() appends to this same list object across calls, so
    a later /ask carrying this session's id genuinely continues the
    conversation instead of starting over with no memory of it.

    Entry 33: turns holds one display-friendly {question, answer,
    tool_calls} record per completed exchange -- backend/sessions.py
    appends to it and persists it via backend/history.py so a
    conversation survives a backend restart, not just this process's
    lifetime.

    Entry 46: plan/pending_plan support "planner" mode's extra phase --
    status can also become "awaiting_plan_approval" before any tool is
    ever called. Both fields are scoped to the CURRENT turn only (unlike
    messages/turns, which persist across the whole conversation) -- a new
    planner-mode question generates and holds a fresh plan, it doesn't
    reuse a previous turn's."""
    id: str
    status: str = "running"
    pending: PendingApproval | None = None
    plan: list[str] | None = None
    pending_plan: PendingPlanApproval | None = None
    result: AgentResult | None = None
    error: str | None = None
    messages: list[dict] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)


async def check_ollama_reachable(timeout: float = 3.0) -> str:
    """Best-effort reachability check for GET /status -- hits Ollama's own
    lightweight GET /api/tags (no chat call), capped at `timeout` so a
    slow/unreachable Ollama can't make /status itself hang. Never raises
    -- always returns a short human-readable status string. Deliberately
    NOT part of GET /health, which the VS Code extension polls every
    500ms expecting a near-instant reply (Entry 27)."""
    try:
        async with httpx2.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags")
        response.raise_for_status()
        return f"ok ({OLLAMA_HOST})"
    except Exception as exc:  # noqa: BLE001 -- this is a status probe, any failure means "unreachable"
        return f"unreachable: {exc}"


def _ollama_chat(messages: list[dict], tools: list[dict]) -> dict:
    """Direct HTTP call to the local Ollama install -- see Entry 44 above
    for why this replaced a WSL-bridged subprocess call. Synchronous
    (blocking) on purpose; the one caller runs it via asyncio.to_thread."""
    try:
        with httpx2.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_CONTEXT_LENGTH},
                },
            )
        response.raise_for_status()
    except httpx2.HTTPError as exc:
        raise ConnectionError(f"Could not reach Ollama at {OLLAMA_HOST}:{OLLAMA_PORT}: {exc}") from exc

    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    return data["message"]


async def _discover_tools() -> list[dict]:
    """Convert every MCP tool from every server into Ollama's function-schema
    shape. Gated tools are included -- Entry 24 needs the model able to
    propose them, unlike Entry 23/18 which hid them entirely."""
    ollama_tools = []
    for server_name, server in SERVERS.items():
        async with Client(server) as client:
            result = await client.list_tools()
            for tool in result.tools:
                _tool_server[tool.name] = server_name
                ollama_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
    return ollama_tools


def _tools_for_mode(all_tools: list[dict], mode: str) -> list[dict]:
    """Entry 38: the real enforcement for Ask/Plan modes -- Ollama can
    only call a tool that's actually in the request's tools array, so
    this is a structural restriction, not just a prompting request the
    model could ignore."""
    if mode == "ask":
        return []
    if mode == "plan":
        return [t for t in all_tools if t["function"]["name"] in PLAN_MODE_ALLOWED_TOOLS]
    return all_tools  # "agent" (and any unrecognized mode) -- today's full behavior


async def _call_gated_tool(name: str, arguments: dict, session: AgentSession) -> str:
    """Open a real stdio session just for this call, with an elicitation
    callback that pauses on session.pending until a human resolves the
    approval Future -- exactly the accept/decline shape stage8_client.py
    proved live, not a client-side shortcut around ctx.elicit().

    Entry 41: every call here is audited (mcp_servers/audit.py) regardless
    of outcome -- approved, declined, or errored -- via the finally block,
    and whether approval was actually *requested* is tracked separately
    from whether it was granted, since some GATED_TOOLS calls (a
    non-mutating execute_command) never call ctx.elicit() at all."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", GATED_TOOL_MODULE[name]],
        env=forwarded_env(),
    )

    approval_requested = False
    approval_granted: bool | None = None

    async def elicitation_callback(context, params):
        nonlocal approval_requested, approval_granted
        approval_requested = True
        decision: "asyncio.Future[bool]" = asyncio.get_running_loop().create_future()
        session.pending = PendingApproval(
            tool=name, arguments=arguments, message=params.message, decision=decision
        )
        session.status = "awaiting_approval"
        approved = await decision
        approval_granted = approved
        session.pending = None
        session.status = "running"
        if approved:
            return types.ElicitResult(action="accept", content={})
        return types.ElicitResult(action="decline")

    result_text = ""
    error_text: str | None = None
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(
                read, write, elicitation_callback=elicitation_callback
            ) as mcp_session:
                await mcp_session.initialize()
                result = await mcp_session.call_tool(name, arguments)

        if result.is_error:
            error_text = result.content[0].text
            result_text = error_text
        elif result.structured_content is not None:
            result_text = json.dumps(result.structured_content)
        else:
            result_text = result.content[0].text
        return result_text
    except Exception as exc:
        error_text = str(exc)
        raise
    finally:
        audit.record(
            session_id=session.id,
            tool=name,
            arguments=arguments,
            gated=True,
            approved=approval_granted if approval_requested else None,
            result_preview=result_text,
            error=error_text,
        )


async def _call_mcp_tool(name: str, arguments: dict, session: AgentSession | None) -> str:
    if name in GATED_TOOLS:
        if session is None:
            return (
                f"Error: '{name}' requires human approval, which isn't "
                f"available in this context."
            )
        return await _call_gated_tool(name, arguments, session)

    server_name = _tool_server.get(name)
    if server_name is None:
        return f"Error: unknown tool '{name}'"

    server = SERVERS[server_name]
    result_text = ""
    error_text: str | None = None
    try:
        async with Client(server) as client:
            result = await client.call_tool(name, arguments)

        if result.is_error:
            error_text = result.content[0].text
            result_text = error_text
        elif result.structured_content is not None:
            result_text = json.dumps(result.structured_content)
        else:
            result_text = result.content[0].text
        return result_text
    except Exception as exc:
        error_text = str(exc)
        raise
    finally:
        audit.record(
            session_id=session.id if session is not None else "no-session",
            tool=name,
            arguments=arguments,
            gated=False,
            approved=None,
            result_preview=result_text,
            error=error_text,
        )


_JSON_TOOL_CALL_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_hallucinated_tool_calls(content: str, known_tool_names: set[str]) -> list[dict]:
    """Entry 35: find fenced ```json blocks in the model's own answer text
    shaped like a tool call ({"name": ..., "arguments": {...}}) naming a
    real, currently-available tool. This is how the model was observed
    describing a write_file call instead of actually invoking it --
    claiming success on a file that was never written. Only tool names
    the caller actually offered this turn are eligible, so this can't be
    used to conjure a call to something that was never on the table."""
    calls = []
    for match in _JSON_TOOL_CALL_BLOCK_RE.finditer(content):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if name in known_tool_names and isinstance(arguments, dict):
            calls.append({"name": name, "arguments": arguments})
    return calls


_JSON_ARRAY_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def _parse_plan_steps(content: str) -> list[str]:
    """Extract a JSON array of step strings from the model's plan-
    generation response. Tries the fenced-block convention first (what
    PLAN_GENERATION_PROMPT actually asks for); falls back to scanning the
    raw text for a bare JSON array; falls back again to treating each
    non-empty line as one step (stripping a leading number/bullet) so a
    model that ignores the JSON instruction still produces *something*
    usable instead of a hard failure. Always capped at MAX_PLAN_STEPS."""
    match = _JSON_ARRAY_BLOCK_RE.search(content)
    candidate = match.group(1) if match else content

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            steps = [s.strip() for s in parsed if s.strip()]
            if steps:
                return steps[:MAX_PLAN_STEPS]
    except json.JSONDecodeError:
        pass

    fallback_steps = []
    for line in content.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.):])\s*", "", line).strip()
        if cleaned:
            fallback_steps.append(cleaned)
    return fallback_steps[:MAX_PLAN_STEPS] or ["Investigate the request and report back."]


async def _generate_plan(user_message: str, all_tools: list[dict]) -> list[str]:
    """One-shot, tool-free call: ask the model to describe its intended
    approach before doing anything. tools=[] is passed to _ollama_chat
    deliberately -- this call must never itself invoke a real tool, only
    describe a plan to invoke tools later, once approved."""
    tool_summary = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}" for t in all_tools
    )
    messages = [
        {"role": "system", "content": PLAN_GENERATION_PROMPT},
        {
            "role": "user",
            "content": f"Available tools:\n{tool_summary}\n\nUser request: {user_message}",
        },
    ]
    message = await asyncio.to_thread(_ollama_chat, messages, [])
    return _parse_plan_steps(message.get("content", ""))


async def ask(
    user_message: str, session: AgentSession | None = None, mode: str = "agent"
) -> AgentResult:
    """Ask the agent something. It decides which tools (if any) to call.

    Pass a session to allow gated tool calls to pause for real human
    approval (session.status flips to "awaiting_approval" and this
    function's await simply doesn't resume until the Future backend/
    sessions.py resolves is done). Without a session, gated calls fail
    closed rather than running or vanishing silently.

    Entry 31: if the session already has messages (a prior turn of this
    same conversation), this continues that same conversation instead of
    starting fresh -- messages is the SAME list object as
    session.messages, so every append below (including the final answer)
    persists onto the session automatically for the next continuation.

    Entry 38: mode ("ask" | "plan" | "agent") controls which tools are
    even offered this turn (see _tools_for_mode) -- applied fresh every
    call, so mode can differ turn to turn within the same conversation,
    same granularity Copilot's own mode dropdown uses. A mode reminder is
    injected into this turn's own message content (not the stored
    session.turns shown in the UI, which stays the user's raw text) so
    the model is told about the current mode even mid-conversation,
    where the original system prompt was fixed on turn 1.

    Entry 46: mode "planner" runs an extra phase before any of the above --
    generate a plan (_generate_plan), hold it for a whole-plan human
    decision (session.status = "awaiting_plan_approval"), then either
    stop (declined) or fold the approved plan into this turn's message
    and fall through into the exact same execution path "agent" mode
    uses, with full tool access."""
    all_tools = await _discover_tools()

    if mode == "planner":
        if session is None:
            return AgentResult(
                answer="Planner mode requires an active session (nowhere to hold the plan for approval).",
                tool_calls=[],
            )
        plan_steps = await _generate_plan(user_message, all_tools)
        decision: "asyncio.Future[bool]" = asyncio.get_running_loop().create_future()
        session.plan = plan_steps
        session.pending_plan = PendingPlanApproval(steps=plan_steps, decision=decision)
        session.status = "awaiting_plan_approval"
        approved = await decision
        session.pending_plan = None
        session.status = "running"
        audit.record(
            session_id=session.id,
            tool="__plan__",
            arguments={"steps": plan_steps},
            gated=True,
            approved=approved,
            result_preview="approved -- proceeding to execute" if approved else "declined",
        )
        if not approved:
            return AgentResult(
                answer="The plan was not approved -- no tools were called and nothing was changed.",
                tool_calls=[],
            )

        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(plan_steps, 1))
        user_message = (
            f"You already proposed this plan and the user approved it:\n{plan_text}\n\n"
            f"Carry it out now, step by step, using the available tools -- call a tool "
            f"for each step in order, and briefly say what you're doing at each step. "
            f"If a step turns out to need a different action once you see real results "
            f"from an earlier step, adapt it and explain why. Original request: {user_message}"
        )
        mode = "agent"  # execution gets full tool access, same as Agent mode

    tools = _tools_for_mode(all_tools, mode)
    print(
        f"Mode={mode!r}: offering {len(tools)}/{len(all_tools)} tools: "
        + ", ".join(t["function"]["name"] for t in tools)
    )

    mode_instruction = MODE_PROMPTS.get(mode, "")
    turn_content = f"{mode_instruction}\n\n{user_message}" if mode_instruction else user_message

    if session is not None and session.messages:
        messages = session.messages
        messages.append({"role": "user", "content": turn_content})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": turn_content},
        ]
        if session is not None:
            session.messages = messages
    trace: list[dict] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Turn {iteration}: asking the model (can take ~1 min on this hardware) ---")
        # _ollama_chat is a blocking call (httpx2.Client, synchronous) --
        # run it off the event loop (Entry 30) so the server stays
        # responsive to every other request (health checks, other
        # sessions' polling) for the full duration of a slow model turn,
        # instead of freezing the whole single-threaded asyncio loop.
        message = await asyncio.to_thread(_ollama_chat, messages, tools)

        tool_calls = message.get("tool_calls")
        rescued = False
        if not tool_calls:
            # Entry 35: the model can describe a tool call as JSON text in
            # its answer instead of actually invoking Ollama's real
            # tool-calling mechanism -- confirmed live with a claimed
            # write_file that never ran, plus a fabricated "confirmation"
            # of content that was never written. Rescue any such block
            # naming a real, currently-available tool into the exact same
            # execution path below (still approval-gated for gated tools)
            # instead of returning the false claim as a final answer.
            known_tool_names = {t["function"]["name"] for t in tools}
            hallucinated = _extract_hallucinated_tool_calls(message.get("content", ""), known_tool_names)
            if not hallucinated:
                messages.append(message)  # persisted for a future continuation
                return AgentResult(answer=message.get("content", ""), tool_calls=trace)
            rescued = True
            tool_calls = [
                {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in hallucinated
            ]

        messages.append(message)
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"  -> {'(rescued from text) ' if rescued else ''}model called {name}({args})")
            result_text = await _call_mcp_tool(name, args, session)
            print(f"     result: {result_text[:200]}")
            trace.append({"name": name, "arguments": args, "result": result_text})
            messages.append(
                {"role": "tool", "content": result_text, "tool_call_id": call.get("id", "")}
            )

    return AgentResult(
        answer="(gave up after max iterations without a final answer)", tool_calls=trace
    )
