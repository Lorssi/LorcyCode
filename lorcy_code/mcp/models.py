from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal


SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
ENV_REF_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
SENSITIVE_KEY_RE = re.compile(
    r"authorization|(^|[_-])(token|secret|password|api[_-]?key)($|[_-])",
    re.I,
)


class MCPConfigError(ValueError):
    """Raised when an MCP configuration is invalid."""


class MissingEnvironmentVariable(MCPConfigError):
    def __init__(self, names: set[str]):
        self.names = names
        super().__init__("缺少环境变量: " + ", ".join(sorted(names)))


class MCPServerStatus(str, Enum):
    DISABLED = "disabled"
    CONNECTING = "connecting"
    READY = "ready"
    AUTH_REQUIRED = "auth_required"
    FAILED = "failed"


@dataclass(slots=True)
class MCPToolFilter:
    include: list[str] = field(default_factory=lambda: ["*"])
    exclude: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any) -> "MCPToolFilter":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise MCPConfigError("toolFilter 必须是对象")
        include = raw.get("include", ["*"])
        exclude = raw.get("exclude", [])
        if not isinstance(include, list) or not all(isinstance(v, str) for v in include):
            raise MCPConfigError("toolFilter.include 必须是字符串数组")
        if not isinstance(exclude, list) or not all(isinstance(v, str) for v in exclude):
            raise MCPConfigError("toolFilter.exclude 必须是字符串数组")
        return cls(include=include, exclude=exclude)

    def to_raw(self) -> dict[str, list[str]]:
        return {"include": self.include, "exclude": self.exclude}


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    enabled: bool
    transport: Literal["stdio", "streamable_http"]
    source: Literal["user", "workspace"] = "user"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    tool_filter: MCPToolFilter = field(default_factory=MCPToolFilter)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls, name: str, raw: Any, *, source: Literal["user", "workspace"]
    ) -> "MCPServerConfig":
        if not SERVER_NAME_RE.fullmatch(name):
            raise MCPConfigError(f"无效 MCP 服务名: {name!r}")
        if not isinstance(raw, dict):
            raise MCPConfigError(f"MCP 服务 {name} 的配置必须是对象")
        transport = raw.get("transport", raw.get("type"))
        if transport is None:
            transport = "stdio" if raw.get("command") else "streamable_http" if raw.get("url") else None
        transport = {
            "http": "streamable_http",
            "streamable-http": "streamable_http",
        }.get(transport, transport)
        if transport not in {"stdio", "streamable_http"}:
            raise MCPConfigError(f"{name}.transport 必须是 stdio 或 streamable_http")
        disabled = raw.get("disabled", False)
        if not isinstance(disabled, bool):
            raise MCPConfigError(f"{name}.disabled 必须是布尔值")
        enabled = raw.get("enabled", not disabled)
        if not isinstance(enabled, bool):
            raise MCPConfigError(f"{name}.enabled 必须是布尔值")
        args = raw.get("args", [])
        env = raw.get("env", {})
        headers = raw.get("headers", {})
        timeout = raw.get("timeoutSeconds", 60)
        if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
            raise MCPConfigError(f"{name}.args 必须是字符串数组")
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise MCPConfigError(f"{name}.env 必须是字符串映射")
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise MCPConfigError(f"{name}.headers 必须是字符串映射")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise MCPConfigError(f"{name}.timeoutSeconds 必须大于 0")

        command = raw.get("command")
        url = raw.get("url")
        cwd = raw.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise MCPConfigError(f"{name}.cwd 必须是字符串")
        if transport == "stdio" and (not isinstance(command, str) or not command.strip()):
            raise MCPConfigError(f"{name}.command 不能为空")
        if transport == "streamable_http" and (
            not isinstance(url, str) or not url.startswith(("http://", "https://"))
        ):
            raise MCPConfigError(f"{name}.url 必须是 HTTP(S) URL")

        known_fields = {
            "enabled", "disabled", "transport", "type", "command", "args", "cwd",
            "env", "url", "headers", "timeoutSeconds", "toolFilter",
        }
        return cls(
            name=name,
            enabled=enabled,
            transport=transport,
            source=source,
            command=command,
            args=list(args),
            cwd=cwd,
            env=dict(env),
            url=url,
            headers=dict(headers),
            timeout_seconds=float(timeout),
            tool_filter=MCPToolFilter.from_raw(raw.get("toolFilter")),
            extra={key: value for key, value in raw.items() if key not in known_fields},
        )

    def to_raw(self, *, portable: bool = False) -> dict[str, Any]:
        if portable:
            raw: dict[str, Any] = dict(self.extra)
            if not self.enabled:
                raw["enabled"] = False
            if self.transport == "stdio":
                raw.update(command=self.command, args=self.args)
                if self.cwd:
                    raw["cwd"] = self.cwd
                if self.env:
                    raw["env"] = self.env
            else:
                raw.update(type="http", url=self.url)
                if self.headers:
                    raw["headers"] = self.headers
            if self.timeout_seconds != 60:
                raw["timeoutSeconds"] = self.timeout_seconds
            if self.tool_filter.include != ["*"] or self.tool_filter.exclude:
                raw["toolFilter"] = self.tool_filter.to_raw()
            return raw

        raw: dict[str, Any] = {
            **self.extra,
            "enabled": self.enabled,
            "transport": self.transport,
            "timeoutSeconds": self.timeout_seconds,
            "toolFilter": self.tool_filter.to_raw(),
        }
        if self.transport == "stdio":
            raw.update(command=self.command, args=self.args, env=self.env)
            if self.cwd:
                raw["cwd"] = self.cwd
        else:
            raw.update(url=self.url, headers=self.headers)
        return raw

    def resolve(self, workspace: str, environ: dict[str, str]) -> "MCPServerConfig":
        missing: set[str] = set()

        def expand(value: str | None) -> str | None:
            if value is None:
                return None
            value = value.replace("${workspace}", workspace)

            def repl(match: re.Match[str]) -> str:
                name = match.group(1)
                if name not in environ:
                    missing.add(name)
                    return ""
                return environ[name]

            return ENV_REF_RE.sub(repl, value)

        resolved = replace(
            self,
            command=expand(self.command),
            args=[expand(v) or "" for v in self.args],
            cwd=expand(self.cwd),
            env={k: expand(v) or "" for k, v in self.env.items()},
            url=expand(self.url),
            headers={k: expand(v) or "" for k, v in self.headers.items()},
        )
        if missing:
            raise MissingEnvironmentVariable(missing)
        return resolved


@dataclass(slots=True)
class MCPServerState:
    config: MCPServerConfig
    status: MCPServerStatus
    tools: list[Any] = field(default_factory=list)
    connected_ms: float | None = None
    last_error: str | None = None
    diagnostics: list[str] = field(default_factory=list)

    def add_diagnostic(self, message: str) -> None:
        self.diagnostics.append(message)
        if len(self.diagnostics) > 100:
            del self.diagnostics[:-100]
