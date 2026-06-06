import asyncio
import contextvars
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich._spinners import SPINNERS

_subagent_count = 0
_subagent_count_lock = threading.Lock()
_subagent_parallel = False

_current_agent_tag: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_agent_tag", default=None
)
_agent_progress: dict[str, dict] = {}
_agent_progress_lock = threading.Lock()
_progress_live: Live | None = None
_progress_task: asyncio.Task | None = None

_DOTS = SPINNERS["dots"]["frames"]
_DOTS_MS = SPINNERS["dots"]["interval"]

console = Console()

def _suppress_in_subagent(fn):
    """Decorator: suppress output when subagents are active (parallel or count > 0)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _subagent_parallel or _subagent_count > 0:
            return
        return fn(*args, **kwargs)

    return wrapper

# ─── 消息渲染 ──────────────────────────────────────────

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

def render_error(message: str) -> None:
    """渲染错误信息"""
    console.print(Text("Error: ", style="red bold"), Text(message, style="red bold"))

def render_ai_start():
    """AI 回复开始"""
    global _subagent_parallel
    if _subagent_count == 0:
        _finalize_progress()
        with _agent_progress_lock:
            _agent_progress.clear()
    _subagent_parallel = False
    if _subagent_count > 0:
        return
    console.print()

def _finalize_progress():
    """停止进度显示并清理资源"""
    global _progress_live, _progress_task

    if _progress_task is not None and not _progress_task.done():
        _progress_task.cancel()
        _progress_task = None

    if _progress_live is not None:
        _update_progress()
        _progress_live.stop()
        _progress_live = None

    with _agent_progress_lock:
        _agent_progress.clear()

def _update_progress():
    if not _progress_live:
        return
    with _agent_progress_lock:
        if not _agent_progress:
            _progress_live.update("")
            return
        frame = _DOTS[int(time.time() * 1000 / _DOTS_MS) % len(_DOTS)]
        lines = []
        for tag, info in _agent_progress.items():
            calls = info.get("calls", 0)
            calls_str = f" ({calls} calls)" if calls else ""
            if info.get("failed"):
                lines.append(f"  [red]✗ {tag}[/red]{calls_str}")
            elif info.get("done"):
                lines.append(f"  [green]✓ {tag}[/green]{calls_str}")
            else:
                lines.append(f"  [cyan]{frame}[/cyan] {tag}{calls_str}")
    _progress_live.update("\n".join(lines))

@_suppress_in_subagent
def render_ai_end() -> None:
    """AI 回复结束"""
    console.print()

@_suppress_in_subagent
def render_ai_chunk(content: str) -> None:
    """渲染 AI 回复片段（流式）"""
    console.print(content, end="", style="white")

# ─── 消息列表渲染（加载历史） ─────────────────────────────

# ─── 上下文用量 ──────────────────────────────────────────