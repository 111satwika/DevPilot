"""Stage 4: exercising the GitHub MCP server.

Run from the project root: python -m mcp_client.stage4_client
"""

import asyncio

from mcp import Client

from mcp_servers.github.server import mcp


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
            "get_repository",
            {"owner": "modelcontextprotocol", "repo": "python-sdk"},
        )
        _print_result(
            "get_repository(modelcontextprotocol/python-sdk) — happy path", result
        )

        result = await client.call_tool(
            "get_repository",
            {"owner": "this-owner-does-not-exist-xyz123", "repo": "nope"},
        )
        _print_result(
            "get_repository (nonexistent repo, expect 404 -> ValueError)", result
        )


if __name__ == "__main__":
    asyncio.run(main())
