from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live

console = Console()

def render_welcome() -> None:
    """渲染欢迎信息"""
    console.print()
    console.print(
        Panel(
            "[bold]LorcyCode[/bold] — Terminal-based AI Coding Agent\n"
            "Enter 发送 | Ctrl+Enter 换行 | /help 查看命令\n"
            "Ctrl+C 中断生成 | Tab 切换模式 | /quit 退出",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()