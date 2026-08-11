"""Stage 3: exercising the Terminal MCP server.

Run from the project root: python -m mcp_client.stage3_client
"""

import asyncio

from mcp import Client

from mcp_servers.terminal.server import mcp


def _print_result(label: str, result) -> None:
    print(f"\n{label}:")
    if result.is_error:
        print("  is_error: True")
        print(" ", result.content[0].text)
    elif result.structured_content is not None:
        print(" ", result.structured_content)
    else:
        print(" ", result.content[0].text)


async def main() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools.tools)} tools:")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
            print(f"    schema: {tool.input_schema}")

        result = await client.call_tool(
            "execute_command", {"command": "git", "args": ["--version"]}
        )
        _print_result("git --version (happy path)", result)

        result = await client.call_tool(
            "execute_command",
            {"command": "python", "args": ["-c", "import sys; sys.exit(1)"]},
        )
        _print_result(
            "python exiting 1 on purpose (should NOT be is_error)", result
        )

        result = await client.call_tool(
            "execute_command", {"command": "rm", "args": ["-rf", "/"]}
        )
        _print_result("rm -rf / (should be rejected, not executed)", result)

        result = await client.call_tool(
            "execute_command",
            {
                "command": "python",
                "args": ["-c", "import time; time.sleep(5)"],
                "timeout_seconds": 1,
            },
        )
        _print_result("sleep 5s with 1s timeout (should time out)", result)


if __name__ == "__main__":
    asyncio.run(main())
