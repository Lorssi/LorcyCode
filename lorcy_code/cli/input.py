"""Prompt-toolkit helpers used by the interactive REPL."""

import datetime
import re
from pathlib import Path
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

SLASH_COMMANDS = {
    "/new": "新会话", "/history": "历史会话", "/model": "模型管理（新建/编辑/切换）",
    "/messages": "管理历史消息（编辑/分叉/删除）", "/compress": "压缩会话",
    "/skill": "技能启用与管理", "/mode": "模式切换（Common/Yolo）", "/git": "Git 状态",
    "/workdir": "切换工作目录", "/tools": "显示内置工具", "/help": "显示帮助", "/quit": "退出",
}
_TAG_SPLIT = re.compile(r"(\[/?[^\]]+\])")
_TAG_OPEN = re.compile(r"^\[([^\]]+)\]$")
_TAG_CLOSE = re.compile(r"^\[/([^\]]*)\]$")
_TAG_MAP = {"bold": "b", "italic": "i", "red": "fg:red", "green": "fg:green", "yellow": "fg:yellow", "blue": "fg:blue", "dim": "fg:#888888"}

def rich_to_html(value: str) -> str:
    opened, result = [], []
    for part in _TAG_SPLIT.split(value):
        close_match = _TAG_CLOSE.match(part)
        open_match = _TAG_OPEN.match(part) if not close_match else None
        if close_match:
            while opened:
                result.append(f"</{opened.pop()}>")
        elif open_match:
            for tag in open_match.group(1).split():
                mapped = _TAG_MAP.get(tag)
                if mapped:
                    if mapped.startswith("fg:"):
                        result.append(f'<style fg="{mapped[3:]}">'); opened.append("style")
                    else:
                        result.append(f"<{mapped}>"); opened.append(mapped)
        else:
            result.append(part)
    return "".join(result)

class LimitedFileHistory(FileHistory):
    MAX_ENTRIES = 50
    def store_string(self, string):
        Path(self.filename).parent.mkdir(exist_ok=True)
        super().store_string(string)
        strings = list(self.load_history_strings())
        if len(strings) > self.MAX_ENTRIES:
            keep = strings[:self.MAX_ENTRIES]; self._loaded_strings = keep; self._rewrite(keep)
    def _rewrite(self, keep):
        Path(self.filename).parent.mkdir(exist_ok=True)
        with open(self.filename, "wb") as history_file:
            for value in reversed(keep):
                history_file.write(f"\n# {datetime.datetime.now()}\n".encode())
                for line in value.split("\n"):
                    history_file.write(f"+{line}\n".encode())

class SlashCommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        partial = document.text_before_cursor.lower()
        if partial.startswith("/"):
            for command, description in SLASH_COMMANDS.items():
                if command.startswith(partial):
                    yield Completion(command, start_position=-len(partial), display=command, display_meta=description)
