"""Slash-command routing kept separate from the REPL lifecycle."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from .display import render_warning


class CommandHost(Protocol):
    """Methods required by the command router."""


async def dispatch_command(host: CommandHost, raw_command: str) -> None:
    parts = raw_command.strip().split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1] if len(parts) > 1 else ""
    handlers: dict[str, Callable[[str], Awaitable[None]]] = {
        "/new": host._cmd_new,
        "/history": host._cmd_history,
        "/model": host._cmd_model,
        "/compress": host._cmd_compress,
        "/messages": host._cmd_messages,
        "/skill": host._cmd_skill,
        "/mode": host._cmd_mode,
        "/git": host._cmd_git,
        "/workdir": host._cmd_workdir,
        "/tools": host._cmd_tools,
        "/help": host._cmd_help,
        "/quit": host._cmd_quit,
    }
    handler = handlers.get(command)
    if handler is None:
        render_warning(f"未知命令: {command}，输入 /help 查看帮助")
        return
    await handler(argument)
