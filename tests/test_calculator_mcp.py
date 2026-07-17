import asyncio
import sys
from pathlib import Path

from lorcy_code.mcp.manager import MCPManager
from lorcy_code.mcp.models import MCPServerConfig, MCPServerState, MCPServerStatus


def test_calculator_server_over_real_stdio(tmp_path):
    async def run():
        server_path = (
            Path(__file__).parents[1]
            / "examples"
            / "mcp_servers"
            / "calculator_server.py"
        )
        config = MCPServerConfig.from_raw(
            "calculator",
            {
                "command": sys.executable,
                "args": [str(server_path)],
                "timeoutSeconds": 20,
            },
            source="user",
        )
        manager = MCPManager(tmp_path)
        manager.states[config.name] = MCPServerState(
            config, MCPServerStatus.CONNECTING
        )
        try:
            assert await manager.connect(config.name)
            tools = {tool.name: tool for tool in manager.get_tools()}
            assert set(tools) == {
                "mcp__calculator__add",
                "mcp__calculator__subtract",
                "mcp__calculator__multiply",
                "mcp__calculator__divide",
                "mcp__calculator__power",
                "mcp__calculator__square_root",
            }
            add_result = await tools["mcp__calculator__add"].ainvoke(
                {"a": 2, "b": 3}
            )
            sqrt_result = await tools["mcp__calculator__square_root"].ainvoke(
                {"value": 81}
            )
            assert add_result[0]["text"] == "5.0"
            assert sqrt_result[0]["text"] == "9.0"
        finally:
            await manager.close()

    asyncio.run(run())
