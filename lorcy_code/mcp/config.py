from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from lorcy_code.config.paths import CONFIG_DIR, workspace_config_dir
from lorcy_code.shared.json import atomic_write_json

from .models import ENV_REF_RE, SENSITIVE_KEY_RE, MCPConfigError, MCPServerConfig


MCP_CONFIG = CONFIG_DIR / "mcp.json"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MCPConfigError(f"JSON 中存在重复字段: {key}")
        result[key] = value
    return result


class MCPConfigStore:
    """Read and update user/workspace MCP configuration files."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.user_path = MCP_CONFIG
        self.workspace_path = workspace_config_dir(self.workspace) / "mcp.json"
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _path(self, source: Literal["user", "workspace"]) -> Path:
        return self.user_path if source == "user" else self.workspace_path

    def _load_document(
        self,
        source: Literal["user", "workspace"],
        *,
        strict: bool = False,
    ) -> dict:
        path = self._path(source)
        if not path.exists():
            return {"version": 1, "servers": {}, "_root_key": "mcpServers"}
        try:
            data = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
            )
            if not isinstance(data, dict):
                raise MCPConfigError("MCP 配置必须是对象")
            if "servers" in data and "mcpServers" in data:
                raise MCPConfigError("servers 与 mcpServers 不能同时存在")
            if "mcpServers" in data:
                root_key = "mcpServers"
            elif "servers" in data:
                root_key = "servers"
                if data.get("version", 1) != 1:
                    raise MCPConfigError("仅支持 MCP 配置 version=1")
            elif not data:
                root_key = "mcpServers"
            else:
                raise MCPConfigError("缺少 mcpServers 或 servers")
            servers = data.get(root_key, {})
            if not isinstance(servers, dict):
                raise MCPConfigError(f"{root_key} 必须是对象")
            extra = {
                key: value
                for key, value in data.items()
                if key not in {root_key, "version"}
            }
            return {
                "version": 1,
                "servers": dict(servers),
                "_root_key": root_key,
                "_extra": extra,
            }
        except (OSError, json.JSONDecodeError, MCPConfigError) as exc:
            if strict:
                raise MCPConfigError(f"无法更新无效配置 {path}: {exc}") from exc
            self.errors.append(f"{path}: {exc}")
            return {"version": 1, "servers": {}, "_root_key": "mcpServers"}

    def load(self) -> dict[str, MCPServerConfig]:
        self.errors.clear()
        self.warnings.clear()
        merged: dict[str, MCPServerConfig] = {}
        for source in ("user", "workspace"):
            document = self._load_document(source)
            for name, raw in document["servers"].items():
                try:
                    merged[name] = MCPServerConfig.from_raw(name, raw, source=source)
                    if isinstance(raw, dict):
                        for key, value in {
                            **raw.get("env", {}), **raw.get("headers", {})
                        }.items():
                            if (
                                isinstance(value, str)
                                and SENSITIVE_KEY_RE.search(str(key))
                                and not ENV_REF_RE.search(value)
                            ):
                                self.warnings.append(
                                    f"{self._path(source)} [{name}]: {key} 使用明文；建议改为 ${{env:NAME}}"
                                )
                except MCPConfigError as exc:
                    self.errors.append(f"{self._path(source)} [{name}]: {exc}")
        return merged

    def save_server(self, config: MCPServerConfig) -> None:
        document = self._load_document(config.source, strict=True)
        portable = document["_root_key"] == "mcpServers"
        document["servers"][config.name] = config.to_raw(portable=portable)
        self._save_document(config.source, document)

    def remove_server(self, name: str, source: Literal["user", "workspace"]) -> bool:
        document = self._load_document(source, strict=True)
        if name not in document["servers"]:
            return False
        del document["servers"][name]
        self._save_document(source, document)
        return True

    def _save_document(self, source: Literal["user", "workspace"], document: dict) -> None:
        if document["_root_key"] == "mcpServers":
            data = {**document.get("_extra", {}), "mcpServers": document["servers"]}
        else:
            data = {
                **document.get("_extra", {}),
                "version": 1,
                "servers": document["servers"],
            }
        atomic_write_json(self._path(source), data, indent=4, ensure_dir=True)

    def set_enabled(self, name: str, enabled: bool) -> MCPServerConfig:
        config = self.load().get(name)
        if config is None:
            raise KeyError(name)
        config.enabled = enabled
        self.save_server(config)
        return config
