from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lorcy_code.config.storage import _load_setting, _update_setting

from .models import MCPServerConfig


SAFE_ENV_KEYS = {
    "PATH", "PATHEXT", "SYSTEMROOT", "COMSPEC", "WINDIR", "TEMP", "TMP",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "HOME", "LANG",
    "LC_ALL", "TMPDIR", "VIRTUAL_ENV", "UV_CACHE_DIR", "UV_PYTHON",
}


def safe_subprocess_environment(
    environ: dict[str, str], explicit: dict[str, str]
) -> dict[str, str]:
    result = {key: value for key, value in environ.items() if key.upper() in SAFE_ENV_KEYS}
    result.update(explicit)
    return result


def server_fingerprint(config: MCPServerConfig, workspace: Path) -> str:
    data = {
        "workspace": str(workspace.resolve()),
        "name": config.name,
        "command": config.command,
        "args": config.args,
        "cwd": config.cwd,
        "env": config.env,
    }
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_workspace_server_trusted(config: MCPServerConfig, workspace: Path) -> bool:
    if config.source != "workspace" or config.transport != "stdio":
        return True
    trusted = _load_setting().get("trusted_mcp_servers", {})
    key = f"{workspace.resolve()}::{config.name}"
    return trusted.get(key) == server_fingerprint(config, workspace)


def trust_workspace_server(config: MCPServerConfig, workspace: Path) -> None:
    settings = _load_setting()
    trusted = dict(settings.get("trusted_mcp_servers", {}))
    key = f"{workspace.resolve()}::{config.name}"
    trusted[key] = server_fingerprint(config, workspace)
    _update_setting(trusted_mcp_servers=trusted)


def redact(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in ("bearer ", "token", "api_key", "secret", "password")):
        if ":" in value:
            return value.split(":", 1)[0] + ": ***"
        return "***"
    return value
