"""Stage 7: exercising the Docker MCP server, including a real
build -> run -> logs -> stop lifecycle, now that Docker is genuinely
available (bridged through WSL).

Uses real stdio transport for the approval-gated tools (same reason as
Stage 8: elicitation needs a back-channel the in-memory Client lacks).

Run from the project root: python -m mcp_client.stage7_client
"""

import asyncio
import sys

import mcp.types as types
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_servers.docker.server"],
)

TEST_TAG = "devpilot-stage7-test:latest"
TEST_CONTAINER = "devpilot-stage7-test-container"


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
    from mcp_servers.docker.server import mcp as docker_mcp

    # 1. A non-shell-injectable argument -- shell=False means these
    # characters are just a literal (nonexistent) container name, not
    # shell syntax.
    async with Client(docker_mcp) as client:
        result = await client.call_tool(
            "inspect_container", {"container": "x; rm -rf /"}
        )
        _print_result(
            "inspect_container with a shell-metacharacter-laced arg "
            "(should just be treated as a literal, non-existent container name)",
            result,
        )

    # 2. Sandbox rejection for build_image -- path outside the workspace.
    result = await _call(
        approve_callback,
        "build_image",
        {"dockerfile_dir": "../outside_docker_test", "tag": "should-not-build"},
    )
    _print_result("build_image('../outside_docker_test') -- sandbox should reject first", result)

    # 3. Decline path -- confirm no docker build is attempted.
    result = await _call(
        decline_callback,
        "build_image",
        {"dockerfile_dir": "mcp_servers/docker/_test_build", "tag": TEST_TAG},
    )
    _print_result("build_image, declined", result)

    # 4. Full real lifecycle: build -> run -> logs -> stop.
    result = await _call(
        approve_callback,
        "build_image",
        {"dockerfile_dir": "mcp_servers/docker/_test_build", "tag": TEST_TAG},
    )
    _print_result("build_image, approved (real docker build)", result)

    result = await _call(
        approve_callback,
        "run_container",
        {"image": TEST_TAG, "name": TEST_CONTAINER},
    )
    _print_result("run_container, approved (real docker run)", result)

    async with Client(docker_mcp) as client:
        result = await client.call_tool(
            "get_container_logs", {"container": TEST_CONTAINER, "tail": 10}
        )
    _print_result("get_container_logs(devpilot-stage7-test-container)", result)

    result = await _call(
        approve_callback, "stop_container", {"container": TEST_CONTAINER}
    )
    _print_result("stop_container, approved (real docker stop)", result)

    async with Client(docker_mcp) as client:
        result = await client.call_tool("list_containers", {"all": True})
    _print_result(
        "list_containers() -- devpilot-stage7-test-container should show Exited", result
    )


if __name__ == "__main__":
    asyncio.run(main())
