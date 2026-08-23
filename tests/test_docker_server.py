"""Tests for DevPilot Docker MCP's Entry 44 gap-fix: container/image/tag
identifiers reached argv as bare positional arguments with no
validation, risking a leading '-' being read as a docker flag instead of
the name it's supposed to be (same class of issue as Git MCP's Entry 42
fix, no ext::-style remote-execution equivalent found for Docker).

These don't require a real Docker/WSL install -- the rejection happens
before _run_docker() is ever called, which is exactly what's being tested.
"""

import sys

import pytest
from mcp import Client, ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from mcp_servers.docker import server as docker_server


async def _call_ungated(tool: str, args: dict):
    async with Client(docker_server.mcp) as client:
        return await client.call_tool(tool, args)


async def _call_gated_approving_everything(tool: str, args: dict):
    async def approve_everything(context, params):
        return types.ElicitResult(action="accept", content={})

    server_params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.docker.server"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, elicitation_callback=approve_everything) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


@pytest.mark.asyncio
async def test_inspect_container_rejects_flag_like_name():
    result = await _call_ungated("inspect_container", {"container": "--format={{json .}}"})
    assert result.is_error
    assert "must not start with" in result.content[0].text


@pytest.mark.asyncio
async def test_get_container_logs_rejects_flag_like_name():
    result = await _call_ungated("get_container_logs", {"container": "-a"})
    assert result.is_error


@pytest.mark.asyncio
async def test_build_image_rejects_flag_like_tag_before_approval():
    """Rejected even though the approver would accept anything -- proving
    the guard runs independent of, and before, the human decision."""
    result = await _call_gated_approving_everything(
        "build_image", {"dockerfile_dir": ".", "tag": "--pull"}
    )
    assert result.is_error
    assert "must not start with" in result.content[0].text


@pytest.mark.asyncio
async def test_run_container_rejects_flag_like_image_before_approval():
    result = await _call_gated_approving_everything(
        "run_container", {"image": "--privileged", "name": "test"}
    )
    assert result.is_error
    assert "must not start with" in result.content[0].text


@pytest.mark.asyncio
async def test_stop_container_rejects_flag_like_name_before_approval():
    result = await _call_gated_approving_everything("stop_container", {"container": "-f"})
    assert result.is_error
    assert "must not start with" in result.content[0].text


@pytest.mark.asyncio
async def test_normal_container_name_is_not_rejected_by_the_guard():
    """The guard itself must not block ordinary names -- the actual
    docker call will fail with a connectivity/not-found error in this
    environment (no real Docker here), which is a *different*, expected
    failure, not the validation error this test is checking for."""
    result = await _call_ungated("inspect_container", {"container": "my-real-container"})
    if result.is_error:
        assert "must not start with" not in result.content[0].text
