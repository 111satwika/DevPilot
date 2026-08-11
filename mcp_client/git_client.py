"""Test client for Git MCP.

This project folder is not a git repository (established in Stage 9), and
that's not being changed just to make this test look cleaner -- every
read-only/write call below is expected to return a clean
"fatal: not a git repository" result, which is itself the correct,
honest test of the error path, not a workaround. The approval flow
(decline/approve) is tested independently of whether the underlying git
command can actually succeed here.

Run from the project root: python -m mcp_client.git_client
"""

import asyncio
import sys

import mcp.types as types
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_servers.git.server"],
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


async def _call(elicitation_callback, tool: str, args: dict):
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(
            read, write, elicitation_callback=elicitation_callback
        ) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


async def main() -> None:
    from mcp_servers.git.server import mcp as git_mcp

    async with Client(git_mcp) as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools.tools)} tools:")
        for t in tools.tools:
            print(f"  - {t.name}")

        for tool_name, args in [
            ("git_status", {}),
            ("git_log", {"limit": 5}),
            ("git_diff", {}),
            ("git_list_branches", {}),
        ]:
            result = await client.call_tool(tool_name, args)
            _print_result(f"{tool_name}({args})", result)

    # Decline path -- commit.
    result = await _call(decline_callback, "git_commit", {"message": "test commit"})
    _print_result("git_commit, declined", result)

    # Approve path -- commit. Expected to fail cleanly with "not a git
    # repository" (this folder has no .git) -- that's the honest, correct
    # outcome, and it still proves approval -> execution wiring works.
    result = await _call(approve_callback, "git_commit", {"message": "test commit"})
    _print_result("git_commit, approved (expect clean git error, not a crash)", result)

    # Decline path -- push.
    result = await _call(decline_callback, "git_push", {"remote": "origin"})
    _print_result("git_push, declined", result)

    # Decline path -- delete branch.
    result = await _call(
        decline_callback, "git_delete_branch", {"name": "some-branch", "force": False}
    )
    _print_result("git_delete_branch, declined", result)


if __name__ == "__main__":
    asyncio.run(main())
