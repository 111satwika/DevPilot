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
"""

import asyncio
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field

import mcp.types as types
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_servers.browser.server import mcp as browser_mcp
from mcp_servers.database.server import mcp as database_mcp
from mcp_servers.docker.server import mcp as docker_mcp
from mcp_servers.filesystem.server import mcp as filesystem_mcp
from mcp_servers.git.server import mcp as git_mcp
from mcp_servers.github.server import mcp as github_mcp
from mcp_servers.terminal.server import mcp as terminal_mcp
from mcp_servers.workspace import forwarded_env

OLLAMA_CONTAINER = "githubcodebaseintelligenceplatform-ollama-1"
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
GATED_TOOLS = {
    "write_file", "build_image", "run_container", "stop_container",
    "git_commit", "git_delete_branch", "git_push",
}

GATED_TOOL_MODULE = {
    "write_file": "mcp_servers.filesystem.server",
    "build_image": "mcp_servers.docker.server",
    "run_container": "mcp_servers.docker.server",
    "stop_container": "mcp_servers.docker.server",
    "git_commit": "mcp_servers.git.server",
    "git_delete_branch": "mcp_servers.git.server",
    "git_push": "mcp_servers.git.server",
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
    "get_repository", "search_web", "fetch_page",
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
}

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
    lifetime."""
    id: str
    status: str = "running"
    pending: PendingApproval | None = None
    result: AgentResult | None = None
    error: str | None = None
    messages: list[dict] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)


def _resolve_ollama_host() -> str:
    """Same pattern as llm/client.py -- resolve fresh, never hardcode."""
    result = subprocess.run(
        [
            "wsl", "-d", "Ubuntu", "-e", "docker", "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            OLLAMA_CONTAINER,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    ip = result.stdout.strip()
    if not ip:
        raise ConnectionError(f"Could not resolve an IP for container '{OLLAMA_CONTAINER}'.")
    return ip


def _ollama_chat(messages: list[dict], tools: list[dict]) -> dict:
    host = _resolve_ollama_host()
    url = f"http://{host}:{OLLAMA_PORT}/api/chat"
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"num_ctx": OLLAMA_CONTEXT_LENGTH},
        }
    )

    result = subprocess.run(
        [
            "wsl", "-d", "Ubuntu", "-e", "curl", "-s", "-X", "POST", url,
            "-H", "Content-Type: application/json", "-d", "@-",
        ],
        input=payload,
        capture_output=True,
        text=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if result.returncode != 0:
        raise ConnectionError(f"curl failed: {result.stderr.strip()}")

    data = json.loads(result.stdout)
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
    proved live, not a client-side shortcut around ctx.elicit()."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", GATED_TOOL_MODULE[name]],
        env=forwarded_env(),
    )

    async def elicitation_callback(context, params):
        decision: "asyncio.Future[bool]" = asyncio.get_running_loop().create_future()
        session.pending = PendingApproval(
            tool=name, arguments=arguments, message=params.message, decision=decision
        )
        session.status = "awaiting_approval"
        approved = await decision
        session.pending = None
        session.status = "running"
        if approved:
            return types.ElicitResult(action="accept", content={})
        return types.ElicitResult(action="decline")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, elicitation_callback=elicitation_callback
        ) as mcp_session:
            await mcp_session.initialize()
            result = await mcp_session.call_tool(name, arguments)

    if result.is_error:
        return result.content[0].text
    if result.structured_content is not None:
        return json.dumps(result.structured_content)
    return result.content[0].text


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
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)

    if result.is_error:
        return result.content[0].text
    if result.structured_content is not None:
        return json.dumps(result.structured_content)
    return result.content[0].text


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
    where the original system prompt was fixed on turn 1."""
    all_tools = await _discover_tools()
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
        # _ollama_chat is a blocking call (subprocess.run) -- run it off
        # the event loop (Entry 30) so the server stays responsive to
        # every other request (health checks, other sessions' polling)
        # for the full duration of a slow model turn, instead of freezing
        # the whole single-threaded asyncio loop.
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
