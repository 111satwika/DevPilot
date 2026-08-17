"""Stage 8: exercising human-in-the-loop approval (MCP elicitation).

Unlike every prior stage's client, this one connects over REAL stdio
transport (StdioServerParameters + stdio_client + ClientSession), not the
in-memory Client(mcp) shortcut used since Stage 1. That shortcut has no
back-channel for server-initiated requests (confirmed via NoBackChannelError
during implementation) -- elicitation needs the server to send a request TO
the client mid-call, which only a real transport supports. This is the MCP
Transport lesson, arrived at out of necessity rather than by plan.

Run from the project root: python -m mcp_client.stage8_client
"""

import asyncio
import sys

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError

from mcp_servers.workspace import forwarded_env

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_servers.filesystem.server"],
    env=forwarded_env(),
)


def _print_result(label: str, result) -> None:
    print(f"\n{label}:")
    if result.is_error:
        print("  is_error: True")
        print(" ", result.content[0].text)
    elif result.structured_content is not None:
        print(" ", result.structured_content)
    else:
        print(" ", result.content[0].text)


async def approve_callback(context, params):
    print(f"  [client asked to approve]: {params.message}")
    return types.ElicitResult(action="accept", content={})


async def decline_callback(context, params):
    print(f"  [client asked to approve]: {params.message}")
    return types.ElicitResult(action="decline")


async def _call_write_file(elicitation_callback, path: str, content: str):
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(
            read, write, elicitation_callback=elicitation_callback
        ) as session:
            await session.initialize()
            return await session.call_tool(
                "write_file", {"path": path, "content": content}
            )


async def main() -> None:
    # 1. Approve path -- file should actually get written.
    result = await _call_write_file(
        approve_callback, "stage8_approved.txt", "written after approval"
    )
    _print_result("write_file, approved", result)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            confirm = await session.call_tool(
                "read_file", {"path": "stage8_approved.txt"}
            )
    _print_result("read_file('stage8_approved.txt') -- confirms it was really written", confirm)

    # 2. Decline path -- file must NOT be written.
    result = await _call_write_file(
        decline_callback, "stage8_declined.txt", "should never appear"
    )
    _print_result("write_file, declined", result)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            confirm = await session.call_tool(
                "read_file", {"path": "stage8_declined.txt"}
            )
    _print_result(
        "read_file('stage8_declined.txt') -- should fail, file must not exist", confirm
    )

    # 3. No callback registered at all -- must fail safe, not silently write.
    # Unlike every prior error path (which comes back as a normal CallToolResult
    # with is_error=True), this fails at the protocol level: MCPError raised
    # directly out of call_tool(), wrapped in an ExceptionGroup by anyio's
    # structured concurrency -- needs except* (Python 3.11+), not except.
    print("\nwrite_file, NO elicitation_callback registered:")
    try:
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:  # no elicitation_callback
                await session.initialize()
                await session.call_tool(
                    "write_file", {"path": "stage8_no_callback.txt", "content": "x"}
                )
        print("  UNEXPECTED: call succeeded with no callback registered")
    except* MCPError as eg:
        for exc in eg.exceptions:
            print(f"  Failed safe at the protocol level: {exc}")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            confirm = await session.call_tool(
                "read_file", {"path": "stage8_no_callback.txt"}
            )
    _print_result(
        "read_file('stage8_no_callback.txt') -- should fail, file must not exist",
        confirm,
    )

    # 4. Sandbox still runs first -- traversal must be rejected before any approval prompt.
    result = await _call_write_file(
        approve_callback, "../outside_stage8.txt", "should never be asked about"
    )
    _print_result(
        "write_file('../outside_stage8.txt') -- sandbox should reject before elicit()",
        result,
    )


if __name__ == "__main__":
    asyncio.run(main())
