"""Stage 2: exercising tool discovery/selection across multiple tools.

Run from the project root: python -m mcp_client.stage2_client
"""

import asyncio

from mcp import Client

from mcp_servers.filesystem.server import mcp


def _print_result(label: str, result) -> None:
    """Prefer structured_content (the real return value) over content[0].text,
    since list/dict-returning tools split across multiple content blocks."""
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

        result = await client.call_tool("list_directory", {"path": "."})
        _print_result("list_directory('.')", result)

        result = await client.call_tool("get_file_info", {"path": "requirements.txt"})
        _print_result("get_file_info('requirements.txt')", result)

        result = await client.call_tool("search_files", {"query": "server"})
        _print_result("search_files(query='server')", result)

        result = await client.call_tool("search_files", {"query": "x", "path": ".."})
        _print_result("search_files traversal attempt (path='..')", result)


if __name__ == "__main__":
    asyncio.run(main())
