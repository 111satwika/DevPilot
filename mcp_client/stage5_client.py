"""Stage 5: exercising the Browser / Documentation MCP server.

Run from the project root: python -m mcp_client.stage5_client
"""

import asyncio

from mcp import Client

from mcp_servers.browser.server import mcp


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
            "search_web", {"query": "python requests timeout", "max_results": 3}
        )
        _print_result("search_web('python requests timeout')", result)

        first_url = None
        if result.structured_content and result.structured_content.get("result"):
            first_url = result.structured_content["result"][0]["url"]

        if first_url:
            page_result = await client.call_tool("fetch_page", {"url": first_url})
            _print_result(f"fetch_page({first_url!r})", page_result)
        else:
            print("\n(no search result URL available to test fetch_page happy path)")

        result = await client.call_tool("fetch_page", {"url": "http://localhost/"})
        _print_result("fetch_page('http://localhost/') — should be rejected", result)

        result = await client.call_tool(
            "fetch_page", {"url": "http://169.254.169.254/"}
        )
        _print_result(
            "fetch_page('http://169.254.169.254/') — cloud metadata IP, should be rejected",
            result,
        )


if __name__ == "__main__":
    asyncio.run(main())
