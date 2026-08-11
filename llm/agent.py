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
"""

import json
import subprocess
from dataclasses import dataclass, field

from mcp import Client

from mcp_servers.database.server import mcp as database_mcp
from mcp_servers.filesystem.server import mcp as filesystem_mcp

OLLAMA_CONTAINER = "githubcodebaseintelligenceplatform-ollama-1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:7b-instruct"
REQUEST_TIMEOUT_SECONDS = 180
MAX_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are DevPilot AI's tool-using assistant. Use the available tools to "
    "gather facts before answering -- never guess, and never answer from "
    "memory when a tool could confirm the real answer. "
    "If the user's question has multiple parts, before giving your final "
    "answer, explicitly re-read the original question and check that your "
    "answer addresses every part of it -- not just the part you most "
    "recently gathered information about. Do not silently drop a part of "
    "the question just because you already answered a different part."
)

SERVERS = {"filesystem": filesystem_mcp, "database": database_mcp}

# write_file is approval-gated (Stage 8) -- this loop has no elicitation
# callback wired in, so exclude it explicitly rather than relying on that
# to fail safe. Scope is read-only tools only, on purpose, not by accident.
EXCLUDED_TOOLS = {"write_file"}

_tool_server: dict[str, str] = {}


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)


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
        {"model": OLLAMA_MODEL, "messages": messages, "tools": tools, "stream": False}
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
    """Convert every MCP tool from every server into Ollama's function-schema shape."""
    ollama_tools = []
    for server_name, server in SERVERS.items():
        async with Client(server) as client:
            result = await client.list_tools()
            for tool in result.tools:
                if tool.name in EXCLUDED_TOOLS:
                    continue
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


async def _call_mcp_tool(name: str, arguments: dict) -> str:
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


async def ask(user_message: str) -> AgentResult:
    """Ask the agent something. It decides which tools (if any) to call."""
    tools = await _discover_tools()
    print(
        f"Discovered {len(tools)} tools across {len(SERVERS)} servers: "
        + ", ".join(t["function"]["name"] for t in tools)
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    trace: list[dict] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Turn {iteration}: asking the model (can take ~1 min on this hardware) ---")
        message = _ollama_chat(messages, tools)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return AgentResult(answer=message.get("content", ""), tool_calls=trace)

        messages.append(message)
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            print(f"  -> model called {name}({args})")
            result_text = await _call_mcp_tool(name, args)
            print(f"     result: {result_text[:200]}")
            trace.append({"name": name, "arguments": args, "result": result_text})
            messages.append(
                {"role": "tool", "content": result_text, "tool_call_id": call.get("id", "")}
            )

    return AgentResult(
        answer="(gave up after max iterations without a final answer)", tool_calls=trace
    )
