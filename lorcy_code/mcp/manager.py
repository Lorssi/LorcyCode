from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import tempfile
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from lorcy_code.tools.registry import ALL_TOOLS

from .config import MCPConfigStore
from .models import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MissingEnvironmentVariable,
    SENSITIVE_KEY_RE,
)
from .security import (
    is_workspace_server_trusted,
    redact,
    safe_subprocess_environment,
    trust_workspace_server,
)


TrustCallback = Callable[[MCPServerConfig], Awaitable[bool]]


@dataclass(slots=True)
class _Connection:
    session: ClientSession
    raw_tools: dict[str, BaseTool]
    owner_task: asyncio.Task
    close_event: asyncio.Event
    errlog: object | None = None
    stderr_position: int = 0
    secrets: tuple[str, ...] = ()


def _normalize_tool_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return normalized.strip("_") or "tool"


class MCPManager:
    """Own MCP sessions and expose stable, namespaced LangChain tools."""

    def __init__(
        self,
        workspace: Path,
        *,
        trust_callback: TrustCallback | None = None,
    ):
        self.workspace = workspace.resolve()
        self.store = MCPConfigStore(self.workspace)
        self.trust_callback = trust_callback
        self.states: dict[str, MCPServerState] = {}
        self._connections: dict[str, _Connection] = {}

    async def start(self) -> None:
        configs = self.store.load()
        self.states = {
            name: MCPServerState(
                config=config,
                status=MCPServerStatus.CONNECTING if config.enabled else MCPServerStatus.DISABLED,
            )
            for name, config in configs.items()
        }
        trusted: list[MCPServerConfig] = []
        for config in configs.values():
            if not config.enabled:
                continue
            if not await self._ensure_trusted(config):
                state = self.states[config.name]
                state.status = MCPServerStatus.FAILED
                state.last_error = "项目级 stdio MCP 尚未信任"
                state.add_diagnostic(state.last_error)
                continue
            trusted.append(config)
        await asyncio.gather(*(self.connect(config.name) for config in trusted))

    async def _ensure_trusted(self, config: MCPServerConfig) -> bool:
        if is_workspace_server_trusted(config, self.workspace):
            return True
        if self.trust_callback is None or not await self.trust_callback(config):
            return False
        trust_workspace_server(config, self.workspace)
        return True

    async def reload(self) -> None:
        """Apply config changes, closing removed/disabled servers."""
        configs = self.store.load()
        previous_configs = {name: state.config for name, state in self.states.items()}
        for name in list(self._connections):
            if (
                name not in configs
                or not configs[name].enabled
                or previous_configs.get(name) != configs[name]
            ):
                await self.disconnect(name)
        for name in list(self.states):
            if name not in configs:
                self.states.pop(name, None)
        for name, config in configs.items():
            previous = self.states.get(name)
            self.states[name] = MCPServerState(
                config=config,
                status=(
                    previous.status
                    if previous and config.enabled and name in self._connections
                    else MCPServerStatus.DISABLED if not config.enabled
                    else MCPServerStatus.CONNECTING
                ),
                tools=previous.tools if previous and name in self._connections else [],
                connected_ms=previous.connected_ms if previous else None,
                last_error=previous.last_error if previous else None,
                diagnostics=previous.diagnostics if previous else [],
            )
            if config.enabled and name not in self._connections:
                if await self._ensure_trusted(config):
                    await self.connect(name)
                else:
                    self.states[name].status = MCPServerStatus.FAILED
                    self.states[name].last_error = "项目级 stdio MCP 尚未信任"

    async def test_config(self, config: MCPServerConfig) -> tuple[bool, str | None, int]:
        """Test an unsaved config without changing this manager's catalog."""
        if not await self._ensure_trusted(config):
            return False, "项目级 stdio MCP 尚未信任", 0
        tester = MCPManager(self.workspace, trust_callback=self.trust_callback)
        tester.states[config.name] = MCPServerState(config, MCPServerStatus.CONNECTING)
        try:
            ok = await tester.connect(config.name)
            state = tester.states[config.name]
            return ok, state.last_error, len(state.tools)
        finally:
            await tester.close()

    async def connect(self, name: str) -> bool:
        state = self.states.get(name)
        if state is None:
            config = self.store.load().get(name)
            if config is None:
                return False
            state = self.states[name] = MCPServerState(config, MCPServerStatus.CONNECTING)
        if not state.config.enabled:
            state.status = MCPServerStatus.DISABLED
            return False

        await self.disconnect(name, keep_state=True)
        state.status = MCPServerStatus.CONNECTING
        state.last_error = None
        state.tools = []
        started = time.perf_counter()
        secrets: tuple[str, ...] = ()
        try:
            config = state.config.resolve(str(self.workspace), dict(os.environ))
            secret_values = []
            for raw_mapping, resolved_mapping in (
                (state.config.env, config.env),
                (state.config.headers, config.headers),
            ):
                for key, raw_value in raw_mapping.items():
                    if (
                        ("${env:" in raw_value or SENSITIVE_KEY_RE.search(key))
                        and resolved_mapping.get(key)
                    ):
                        secret_values.append(resolved_mapping[key])
            secrets = tuple(secret_values)
            ready = asyncio.get_running_loop().create_future()
            close_event = asyncio.Event()
            owner_task = asyncio.create_task(
                self._connection_owner(config, state, ready, close_event, secrets),
                name=f"mcp-{name}",
            )
            try:
                connection = await asyncio.wait_for(
                    asyncio.shield(ready), timeout=config.timeout_seconds
                )
            except BaseException:
                close_event.set()
                owner_task.cancel()
                await asyncio.gather(owner_task, return_exceptions=True)
                raise
            connection.owner_task = owner_task
            filtered = connection.raw_tools
            proxies = self._make_proxy_tools(name, filtered)
            self._connections[name] = connection
            state.tools = proxies
            state.status = MCPServerStatus.READY
            state.connected_ms = (time.perf_counter() - started) * 1000
            state.add_diagnostic(f"已连接，发现 {len(proxies)} 个工具")
            return True
        except MissingEnvironmentVariable as exc:
            state.status = MCPServerStatus.AUTH_REQUIRED
            state.last_error = str(exc)
        except Exception as exc:  # one server must not block the host
            state.status = MCPServerStatus.FAILED
            state.last_error = self._sanitize(str(exc), secrets)
        state.add_diagnostic(state.last_error or "连接失败")
        return False

    async def _connection_owner(
        self,
        config: MCPServerConfig,
        state: MCPServerState,
        ready: asyncio.Future,
        close_event: asyncio.Event,
        secrets: tuple[str, ...],
    ) -> None:
        """Enter and exit transport contexts in the same long-lived task."""
        stack = AsyncExitStack()
        errlog = None
        connection: _Connection | None = None
        try:
            if config.transport == "stdio":
                errlog = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
                params = StdioServerParameters(
                    command=config.command or "",
                    args=config.args,
                    cwd=config.cwd,
                    env=safe_subprocess_environment(dict(os.environ), config.env),
                )
                read, write = await stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
            else:
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=config.headers,
                        timeout=httpx.Timeout(
                            connect=config.timeout_seconds,
                            read=max(config.timeout_seconds, 300),
                            write=config.timeout_seconds,
                            pool=config.timeout_seconds,
                        ),
                    )
                )
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(config.url or "", http_client=http_client)
                )
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=config.timeout_seconds)
            loaded = await asyncio.wait_for(
                load_mcp_tools(
                    session, server_name=config.name, handle_tool_errors=True
                ),
                timeout=config.timeout_seconds,
            )
            filtered = {
                tool.name: tool
                for tool in loaded
                if self._tool_allowed(tool.name, config)
            }
            connection = _Connection(
                session=session,
                raw_tools=filtered,
                owner_task=asyncio.current_task(),
                close_event=close_event,
                errlog=errlog,
                secrets=secrets,
            )
            if not ready.done():
                ready.set_result(connection)
            await close_event.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                state.add_diagnostic(self._sanitize(str(exc), secrets))
        finally:
            try:
                await stack.aclose()
            except Exception as exc:
                state.add_diagnostic(self._sanitize(str(exc), secrets))
            if errlog is not None:
                position = connection.stderr_position if connection else 0
                self._read_stderr(state, errlog, position, secrets)
                errlog.close()

    @staticmethod
    def _tool_allowed(name: str, config: MCPServerConfig) -> bool:
        included = any(fnmatch.fnmatchcase(name, pattern) for pattern in config.tool_filter.include)
        excluded = any(fnmatch.fnmatchcase(name, pattern) for pattern in config.tool_filter.exclude)
        return included and not excluded

    def _make_proxy_tools(
        self, server_name: str, raw_tools: dict[str, BaseTool]
    ) -> list[BaseTool]:
        result: list[BaseTool] = []
        seen = {tool.name for tool in ALL_TOOLS}
        state = self.states[server_name]
        for raw_name, raw_tool in raw_tools.items():
            public_name = f"mcp__{_normalize_tool_name(server_name)}__{_normalize_tool_name(raw_name)}"
            if public_name in seen:
                state.add_diagnostic(f"工具名称冲突，已忽略: {public_name}")
                continue
            seen.add(public_name)

            async def invoke(_raw_name: str = raw_name, **kwargs):
                return await self.invoke(server_name, _raw_name, kwargs)

            metadata = dict(raw_tool.metadata or {})
            metadata.update(mcp_server=server_name, mcp_tool=raw_name)
            result.append(
                StructuredTool(
                    name=public_name,
                    description=raw_tool.description or f"MCP tool {server_name}/{raw_name}",
                    args_schema=raw_tool.args_schema,
                    coroutine=invoke,
                    metadata=metadata,
                )
            )
        return result

    async def invoke(self, server_name: str, raw_name: str, arguments: dict):
        connection = self._connections.get(server_name)
        if connection is None:
            raise ConnectionError(f"MCP 服务 {server_name} 未连接")
        tool = connection.raw_tools.get(raw_name)
        if tool is None:
            raise KeyError(f"MCP 工具 {server_name}/{raw_name} 不存在")
        try:
            return await tool.ainvoke(arguments)
        except Exception:
            # Retry once with a newly initialized session on transport failures.
            if not await self.connect(server_name):
                state = self.states[server_name]
                raise ConnectionError(state.last_error or f"MCP 服务 {server_name} 重连失败")
            tool = self._connections[server_name].raw_tools.get(raw_name)
            if tool is None:
                raise KeyError(f"重连后 MCP 工具 {server_name}/{raw_name} 不存在")
            return await tool.ainvoke(arguments)

    async def refresh(self, name: str | None = None) -> bool:
        if name is not None:
            return await self.connect(name)
        enabled = [key for key, state in self.states.items() if state.config.enabled]
        results = await asyncio.gather(*(self.connect(key) for key in enabled))
        return all(results)

    async def disconnect(self, name: str, *, keep_state: bool = False) -> None:
        connection = self._connections.pop(name, None)
        state = self.states.get(name)
        if connection is not None:
            if state is not None and connection.errlog is not None:
                connection.stderr_position = self._read_stderr(
                    state, connection.errlog, connection.stderr_position,
                    connection.secrets,
                )
            connection.close_event.set()
            await asyncio.gather(connection.owner_task, return_exceptions=True)
        if state is not None and not keep_state:
            state.tools = []
            state.status = (
                MCPServerStatus.DISABLED
                if not state.config.enabled
                else MCPServerStatus.FAILED
            )

    @staticmethod
    def _read_stderr(
        state: MCPServerState,
        errlog,
        position: int,
        secrets: tuple[str, ...] = (),
    ) -> int:
        try:
            errlog.flush()
            errlog.seek(position)
            for line in errlog.read().splitlines():
                state.add_diagnostic(MCPManager._sanitize(line, secrets))
            return errlog.tell()
        except Exception:
            return position

    @staticmethod
    def _sanitize(value: str, secrets: tuple[str, ...]) -> str:
        value = redact(value)
        for secret in secrets:
            if secret:
                value = value.replace(secret, "***")
        return value

    def get_tools(self) -> list[BaseTool]:
        return [tool for state in self.states.values() for tool in state.tools]

    def get_state(self, name: str) -> MCPServerState | None:
        state = self.states.get(name)
        connection = self._connections.get(name)
        if state is not None and connection is not None and connection.errlog is not None:
            connection.stderr_position = self._read_stderr(
                state, connection.errlog, connection.stderr_position,
                connection.secrets,
            )
        return state

    async def close(self) -> None:
        for name in list(self._connections):
            await self.disconnect(name)
