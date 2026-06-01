from pathlib import Path
from rich.console import Console

from .display import (
    console,
    render_welcome,
)

from lorcy_code.cli.config.config import (
    first_run_configure,
)
from lorcy_code.core.environment.build_env import (
    ensure_home_config_dir,
    ensure_chat_config_dir,
    load_model_config,
)

class ChatREPL:
    def __init__(self):
        self.workplace_path: Path | None = None # 当前工作目录路径
        self.model_config: dict = {}  # 模型参数

    async def initialize(self):
        # 确保配置目录存在
        ensure_home_config_dir()
        ensure_chat_config_dir()

        self.model_config = load_model_config()
        if not self.model_config:
            config = await first_run_configure()
            if config is None:
                return False
            self.model_config = config
            

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

        return True

    async def run(self):
        render_welcome()