import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lorcy_code.config import storage
from lorcy_code.mcp import config as mcp_config
from lorcy_code.mcp.config import MCPConfigStore
from lorcy_code.mcp.manager import MCPManager
from lorcy_code.mcp.models import (
    MCPConfigError,
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MissingEnvironmentVariable,
)
from lorcy_code.mcp.security import (
    is_workspace_server_trusted,
    safe_subprocess_environment,
    server_fingerprint,
    trust_workspace_server,
)


def _stdio_raw(command="python"):
    return {
        "enabled": True,
        "transport": "stdio",
        "command": command,
        "args": ["server.py"],
        "cwd": "${workspace}",
        "env": {"TOKEN": "${env:TEST_MCP_TOKEN}"},
        "timeoutSeconds": 10,
        "toolFilter": {"include": ["*"], "exclude": ["delete_*"]},
    }


def test_config_merges_workspace_over_user(tmp_path, monkeypatch):
    user_path = tmp_path / "home" / "mcp.json"
    workspace = tmp_path / "project"
    workspace.mkdir()
    project_path = workspace / ".lorcy" / "mcp.json"
    project_path.parent.mkdir()
    user_path.parent.mkdir()
    user_path.write_text(
        json.dumps({"version": 1, "servers": {"shared": _stdio_raw("user-python")}}),
        encoding="utf-8",
    )
    project_path.write_text(
        json.dumps({"version": 1, "servers": {"shared": _stdio_raw("project-python")}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_config, "MCP_CONFIG", user_path)

    store = MCPConfigStore(workspace)
    config = store.load()["shared"]

    assert config.source == "workspace"
    assert config.command == "project-python"
    assert not store.errors


def test_invalid_config_is_not_overwritten_by_save(tmp_path, monkeypatch):
    user_path = tmp_path / "home" / "mcp.json"
    user_path.parent.mkdir()
    user_path.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(mcp_config, "MCP_CONFIG", user_path)
    store = MCPConfigStore(tmp_path)
    config = MCPServerConfig.from_raw("demo", _stdio_raw(), source="user")
    with pytest.raises(MCPConfigError):
        store.save_server(config)
    assert user_path.read_text(encoding="utf-8") == "{invalid"


def test_config_validation_and_environment_resolution(tmp_path):
    with pytest.raises(MCPConfigError):
        MCPServerConfig.from_raw("bad name", _stdio_raw(), source="user")

    config = MCPServerConfig.from_raw("demo", _stdio_raw(), source="user")
    with pytest.raises(MissingEnvironmentVariable):
        config.resolve(str(tmp_path), {})
    resolved = config.resolve(str(tmp_path), {"TEST_MCP_TOKEN": "secret"})
    assert resolved.cwd == str(tmp_path)
    assert resolved.env["TOKEN"] == "secret"

    cloud = MCPServerConfig.from_raw(
        "cloud",
        {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer plaintext-secret"},
        },
        source="user",
    )
    assert cloud.transport == "streamable_http"


def test_common_mcpservers_format_is_loaded_without_conversion(tmp_path, monkeypatch):
    user_path = tmp_path / "home" / "mcp.json"
    user_path.parent.mkdir()
    user_path.write_text(
        json.dumps(
            {
                "note": "preserve-me",
                "mcpServers": {
                    "weather": {
                        "command": "uvx",
                        "args": [
                            "--from",
                            "git+https://github.com/adhikasp/mcp-weather.git",
                            "mcp-weather",
                        ],
                        "env": {"ACCUWEATHER_API_KEY": "your_api_key_here"},
                    },
                    "bing-search": {
                        "command": "npx",
                        "args": ["-y", "bing-cn-mcp"],
                        "alwaysAllow": ["search"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_config, "MCP_CONFIG", user_path)

    store = MCPConfigStore(tmp_path)
    configs = store.load()

    assert set(configs) == {"weather", "bing-search"}
    assert configs["weather"].transport == "stdio"
    assert configs["bing-search"].command == "npx"
    assert not store.errors
    assert any("ACCUWEATHER_API_KEY" in warning for warning in store.warnings)
    configs["bing-search"].enabled = False
    store.save_server(configs["bing-search"])
    saved = json.loads(user_path.read_text(encoding="utf-8"))
    assert saved["note"] == "preserve-me"
    assert saved["mcpServers"]["bing-search"]["alwaysAllow"] == ["search"]
    assert saved["mcpServers"]["bing-search"]["enabled"] is False


def test_new_config_is_saved_in_portable_mcpservers_format(tmp_path, monkeypatch):
    user_path = tmp_path / "home" / "mcp.json"
    monkeypatch.setattr(mcp_config, "MCP_CONFIG", user_path)
    store = MCPConfigStore(tmp_path)
    config = MCPServerConfig.from_raw(
        "bing-search",
        {"command": "npx", "args": ["-y", "bing-cn-mcp"]},
        source="user",
    )
    store.save_server(config)
    saved = json.loads(user_path.read_text(encoding="utf-8"))
    assert saved == {
        "mcpServers": {
            "bing-search": {"command": "npx", "args": ["-y", "bing-cn-mcp"]}
        }
    }


def test_security_environment_and_fingerprint(tmp_path):
    environment = safe_subprocess_environment(
        {"PATH": "bin", "GITHUB_TOKEN": "hidden"}, {"EXPLICIT": "yes"}
    )
    assert environment["PATH"] == "bin"
    assert environment["EXPLICIT"] == "yes"
    assert "GITHUB_TOKEN" not in environment

    config = MCPServerConfig.from_raw("demo", _stdio_raw(), source="workspace")
    first = server_fingerprint(config, tmp_path)
    config.args.append("--changed")
    assert server_fingerprint(config, tmp_path) != first


def test_workspace_trust_is_invalidated_by_executable_change(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SETTING_JSON", tmp_path / "settings.json")
    config = MCPServerConfig.from_raw("demo", _stdio_raw(), source="workspace")
    assert not is_workspace_server_trusted(config, tmp_path)
    trust_workspace_server(config, tmp_path)
    assert is_workspace_server_trusted(config, tmp_path)
    config.args.append("--changed")
    assert not is_workspace_server_trusted(config, tmp_path)


def test_stdio_server_discovery_and_invocation(tmp_path):
    async def run():
        fixture = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
        config = MCPServerConfig.from_raw(
            "math",
            {
                "enabled": True,
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(fixture)],
                "timeoutSeconds": 20,
            },
            source="user",
        )
        manager = MCPManager(tmp_path)
        manager.states["math"] = MCPServerState(config, MCPServerStatus.CONNECTING)
        try:
            assert await manager.connect("math")
            tools = manager.get_tools()
            assert [tool.name for tool in tools] == ["mcp__math__add"]
            result = await tools[0].ainvoke({"a": 2, "b": 5})
            assert result[0]["text"] == "7"
        finally:
            await manager.close()

    asyncio.run(run())


def test_streamable_http_server_discovery_and_invocation(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    fixture = Path(__file__).parent / "fixtures" / "mcp_http_server.py"
    env = {**os.environ, "MCP_TEST_PORT": str(port)}
    process = subprocess.Popen(
        [sys.executable, str(fixture)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if process.poll() is not None:
                pytest.fail("HTTP MCP test server exited before startup")
            time.sleep(0.05)
        else:
            pytest.fail("HTTP MCP test server did not start")

        async def run():
            config = MCPServerConfig.from_raw(
                "http_math",
                {
                    "enabled": True,
                    "transport": "streamable_http",
                    "url": f"http://127.0.0.1:{port}/mcp",
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
                tool = manager.get_tools()[0]
                assert tool.name == "mcp__http_math__multiply"
                result = await tool.ainvoke({"a": 3, "b": 4})
                assert result[0]["text"] == "12"
            finally:
                await manager.close()

        asyncio.run(run())
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_concurrent_start_and_close_owns_transport_contexts(tmp_path, monkeypatch):
    async def run():
        fixture = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
        configs = {
            name: MCPServerConfig.from_raw(
                name,
                {
                    "enabled": True,
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(fixture)],
                    "timeoutSeconds": 20,
                },
                source="user",
            )
            for name in ("math_one", "math_two")
        }
        manager = MCPManager(tmp_path)
        monkeypatch.setattr(manager.store, "load", lambda: configs)
        await manager.start()
        try:
            assert all(
                state.status is MCPServerStatus.READY
                for state in manager.states.values()
            )
            assert len(manager.get_tools()) == 2
        finally:
            await manager.close()

    asyncio.run(run())
