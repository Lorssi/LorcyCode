from typer.testing import CliRunner

import lorcy_code.cli.app as cli_app
from lorcy_code.config import storage
from lorcy_code.cli.app import app
from lorcy_code.cli.input import SLASH_COMMANDS, SlashCommandCompleter
from lorcy_code.cli.repl import ChatREPL
from types import SimpleNamespace


def test_cli_help_keeps_public_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "config" in result.stdout
    assert "gui" in result.stdout


def test_slash_command_registry():
    assert {"/help", "/model", "/skill", "/git", "/quit"} <= set(SLASH_COMMANDS)
    assert SlashCommandCompleter()


def test_bottom_toolbar_mcp_count_uses_enabled_servers():
    repl = ChatREPL()
    repl.mcp_manager = SimpleNamespace(
        states={
            "one": SimpleNamespace(config=SimpleNamespace(enabled=True)),
            "two": SimpleNamespace(config=SimpleNamespace(enabled=True)),
            "off": SimpleNamespace(config=SimpleNamespace(enabled=False)),
        }
    )
    assert repl._mcp_status_text() == "MCP: 2"


def test_yolo_option_is_forwarded(monkeypatch):
    received = []

    async def fake_run_agent(*, yolo=False):
        received.append(yolo)

    monkeypatch.setattr(cli_app, "_run_agent", fake_run_agent)
    result = CliRunner().invoke(app, ["--yolo"])
    assert result.exit_code == 0
    assert received == [True]


def test_config_show_masks_api_key(tmp_path, monkeypatch):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        '{"default":{"model":"demo","api_key":"abcdefghijkl"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(storage, "MODEL_JSON", model_path)
    monkeypatch.setattr(storage, "_model_json_cache", None)

    result = CliRunner().invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "demo" in result.stdout
    assert "abcdef...ijkl" in result.stdout
    assert "abcdefghijkl" not in result.stdout
