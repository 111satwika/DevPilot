"""Lesson 1: the smallest possible MCP client.

Connects to the Filesystem MCP server, discovers its tools, then invokes
read_file. Run this from the project root: python -m mcp_client.lesson1_client
"""

import asyncio

from mcp import Client

from mcp_servers.filesystem.server import mcp


async def main() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Discovered tools:")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")

        result = await client.call_tool(
            "read_file", {"path": "requirements.txt"}
        )
        print("\nresult of calling read_file('requirements.txt'):")
        print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
