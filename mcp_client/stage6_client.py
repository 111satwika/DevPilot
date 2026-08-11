"""Stage 6: exercising the Database MCP server.

Run from the project root: python -m mcp_client.stage6_client
"""

import asyncio

from mcp import Client

from mcp_servers.database.server import mcp


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

        result = await client.call_tool("list_tables", {})
        _print_result("list_tables()", result)

        result = await client.call_tool("describe_table", {"table": "stages"})
        _print_result("describe_table('stages')", result)

        result = await client.call_tool(
            "execute_read_query",
            {
                "query": "SELECT stage_number, name, status FROM stages "
                "WHERE status = 'done' ORDER BY stage_number"
            },
        )
        _print_result("execute_read_query(... WHERE status = 'done')", result)

        result = await client.call_tool(
            "execute_read_query",
            {
                "query": "INSERT INTO stages (stage_number, name, status) "
                "VALUES (99, 'hacked', 'done')"
            },
        )
        _print_result(
            "execute_read_query(INSERT ...) — should be rejected by prefix check",
            result,
        )


if __name__ == "__main__":
    asyncio.run(main())
