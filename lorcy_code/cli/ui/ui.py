from pathlib import Path
from rich.console import Console

from .display import (
    console,
)

class ChatREPL:
    def __init__(self):
        self.workplace_path: Path | None = None
        pass

    async def initialize(self):
        from lorcy_code.cli.config.config import ensure_config_dir
        ensure_config_dir() # 确保全局配置目录存在

        self.workplace_path = Path.cwd() # 获取当前目录路径

        # 确保当前目录下的.chat配置目录存在
        chat_dir = self.workplace_path / ".chat"
        chat_dir.mkdir(exist_ok=True)
        (chat_dir / "sessions").mkdir(exist_ok=True)
        (chat_dir / "skills").mkdir(exist_ok=True)

        # 构建 agent（可能较慢，放线程）
        console.print(
            "[dim cyan]"
            "██╗         ██████╗   ██████╗    ███████╗   ██╗   ██╗   ███████╗   ██████╗   █████╗    ████████╗\n"
            "██║        ██╔═══██╗  ██╔══██╗   ██╔═════╝  ╚██╗ ██╔╝  ██╔═════╝  ██╔═══██╗  ██╔══██╗  ██╔═════╝\n"
            "██║        ██║   ██║  ██████╔╝   ██║         ╚████╔╝   ██║        ██║   ██║  ██║  ██╗  ████████╗\n"
            "██║        ██║   ██║  ██╔══██╗   ██║          ╚██╔╝    ██║        ██║   ██║  ██║  ██╔╝ ██╔═════╝\n"
            "████████╗  ╚██████╔╝  ██║  ██║   ████████╗     ██║     ████████╗  ╚██████╔╝  █████╔═╝  ████████╗\n"
            "╚═══════╝   ╚═════╝   ╚═╝  ╚═╝    ╚══════╝     ╚═╝      ╚══════╝   ╚═════╝   ╚════╝    ╚══════╝ \n"
            "[dim cyan]"
        )