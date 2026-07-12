"""
Rich 输出渲染 — Markdown、流式输出、状态栏、消息样式
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich._spinners import SPINNERS
from rich.table import Table
from rich.theme import Theme
from rich.box import ROUNDED

import asyncio
import contextvars
import threading
import time

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

THEME = Theme(
    {
        "brand": "bold #5eead4",
        "accent": "#60a5fa",
        "muted": "#7c8aa5",
        "surface": "#111827",
        "success": "bold #34d399",
        "warning": "bold #fbbf24",
        "danger": "bold #fb7185",
        "tool": "#a78bfa",
    }
)

console = Console(theme=THEME, highlight=False)


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


def render_human(message: str) -> None:
    """渲染用户消息"""
    console.print(
        Panel(
            Markdown(message),
            border_style="accent",
            title="[bold #60a5fa] YOU [/bold #60a5fa]",
            title_align="right",
            padding=(0, 1),
            box=ROUNDED,
        )
    )


@_suppress_in_subagent
def render_ai_chunk(content: str) -> None:
    """渲染 AI 回复片段（流式）"""
    console.print(content, end="", style="white")


def begin_model_output() -> None:
    """Prepare the terminal for visible reasoning or answer output.

    A completed single sub-agent leaves a Live result spinner running while the
    parent model starts composing its response.  Stop that Live display before
    writing *any* model text; otherwise Rich redraws the spinner through the
    streamed reasoning chunks.
    """
    global _subagent_parallel
    if _subagent_count == 0:
        _finalize_progress()
    _subagent_parallel = False


def render_ai_start():
    """AI 回复开始"""
    begin_model_output()
    if _subagent_count > 0:
        return
    console.print()


@_suppress_in_subagent
def render_ai_end() -> None:
    """AI 回复结束"""
    console.print()


@_suppress_in_subagent
def render_reasoning(reasoning: str) -> None:
    """渲染推理/思考内容（灰色斜体，折叠）"""
    console.print(
        Panel(
            Text(reasoning, style="dim italic"),
            border_style="muted",
            title="[muted] THINKING [/muted]",
            title_align="left",
            padding=(0, 1),
            box=ROUNDED,
        )
    )


def _start_progress():
    global _progress_live
    if _progress_live is None:
        _live_console = Console(file=console.file)
        _progress_live = Live("", transient=False, console=_live_console, refresh_per_second=12)
        _progress_live.start()


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


async def _progress_updater():
    try:
        while True:
            await asyncio.sleep(_DOTS_MS / 1000)
            if _progress_live is None:
                break
            _update_progress()
    except asyncio.CancelledError:
        pass


async def _result_spinner_updater():
    try:
        while True:
            await asyncio.sleep(_DOTS_MS / 1000)
            if _progress_live is None:
                break
            frame = _DOTS[int(time.time() * 1000 / _DOTS_MS) % len(_DOTS)]
            _progress_live.update(f"  [cyan]{frame}[/cyan] 正在整理结果...")
    except asyncio.CancelledError:
        pass


def _start_result_spinner():
    """单 agent 完成后，显示整理结果的加载圈"""
    global _progress_live, _progress_task
    if _progress_live is None:
        _live_console = Console(file=console.file)
        _progress_live = Live("", transient=False, console=_live_console, refresh_per_second=12)
        _progress_live.start()
    if _progress_task is None or _progress_task.done():
        _progress_task = asyncio.ensure_future(_result_spinner_updater())


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


def force_reset_display() -> None:
    """异常退出时强制重置所有显示状态"""
    global _subagent_count, _subagent_parallel
    _subagent_count = 0
    _subagent_parallel = False
    console.quiet = False
    _finalize_progress()


def finalize_turn_display() -> None:
    """Clean up display resources left behind at the end of a turn."""
    if _subagent_count == 0:
        _finalize_progress()


def render_tool_call(name: str, summary: str) -> None:
    tag = _current_agent_tag.get()
    if tag:
        with _agent_progress_lock:
            if tag in _agent_progress:
                _agent_progress[tag]["calls"] += 1
        return
    if _subagent_parallel:
        return
    if len(summary) > 120:
        summary = summary[:117] + "..."
    if _subagent_count == 1:
        console.print(Text(f"  [{name}] {summary}", style="dim cyan"))
        return
    line = Text("\n  ◆ ", style="tool")
    line.append(name, style="bold #a78bfa")
    line.append(f"  {summary}", style="muted")
    console.print(line)


@_suppress_in_subagent
def render_tool(name: str, content: str) -> None:
    """渲染工具调用结果"""
    # 截断过长内容
    lines = content.split("\n")
    if len(lines) > 50:
        content = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more lines)"
    console.print(
        Panel(
            Text(content, style="white"),
            border_style="tool",
            title=f"[bold #a78bfa] ◆ {name} [/bold #a78bfa]",
            title_align="left",
            padding=(0, 1),
            box=ROUNDED,
        )
    )


@_suppress_in_subagent
def render_error(message: str) -> None:
    """渲染错误信息"""
    console.print(Text("  ✕  ", style="danger"), Text(message, style="danger"))


@_suppress_in_subagent
def render_info(message: str) -> None:
    """渲染信息"""
    console.print(Text("  ●  ", style="accent"), Text(message, style="white"))


@_suppress_in_subagent
def render_success(message: str) -> None:
    """渲染成功信息"""
    console.print(Text("  ✓  ", style="success"), Text(message, style="success"))


@_suppress_in_subagent
def render_warning(message: str) -> None:
    """渲染警告信息"""
    console.print(Text("  !  ", style="warning"), Text(message, style="warning"))


def render_separator() -> None:
    """渲染分隔线"""
    console.print(Rule(style="muted"))


def render_welcome(
    *,
    model: str | None = None,
    workdir: str | None = None,
    yolo: bool = False,
) -> None:
    """渲染 Claude Code 风格的横向欢迎卡片。"""
    from lorcy_code import __version__

    glyphs = {
        "L": ("█    ", "█    ", "█    ", "█    ", "█████"),
        "O": (" ███ ", "█   █", "█   █", "█   █", " ███ "),
        "R": ("████ ", "█   █", "████ ", "█  █ ", "█   █"),
        "C": (" ████", "█    ", "█    ", "█    ", " ████"),
        "D": ("████ ", "█   █", "█   █", "█   █", "████ "),
        "E": ("█████", "█    ", "████ ", "█    ", "█████"),
        "Y": ("█   █", " █ █ ", "  █  ", "  █  ", "  █  "),
    }
    wordmark = "LORCY CODE"
    colors = (
        "#5eead4", "#4adecf", "#38d0dd", "#22c3ee", "#38bdf8",
        "#4eaffb", "#60a5fa", "#718ff7", "#818cf8",
    )
    # 文本内部必须左对齐；逐行居中会根据行尾空白重新计算宽度，造成块字左右抖动。
    mark = Text(justify="left", no_wrap=True)
    for row in range(5):
        color_index = 0
        for index, letter in enumerate(wordmark):
            if letter == " ":
                mark.append("   ")
            else:
                # 所有字形严格占五列，样式只作用于块字符，不使用背景色。
                mark.append(glyphs[letter][row], style=f"bold {colors[color_index]}")
                color_index += 1
            if index < len(wordmark) - 1 and wordmark[index + 1] != " ":
                mark.append(" ")
        if row < 4:
            mark.append("\n")

    info = Table.grid(padding=(0, 1))
    info.add_column(style="muted", no_wrap=True)
    info.add_column(style="white", overflow="ellipsis")

    heading = Text("Lorcy Code", style="bold white")
    heading.append(f"  v{__version__}", style="muted")
    info.add_row(heading, "")
    info.add_row("模型", model or "未配置")
    mode = Text("Yolo", style="danger") if yolo else Text("Common", style="success")
    info.add_row("模式", mode)
    if workdir:
        info.add_row("目录", Text(workdir, overflow="ellipsis", no_wrap=True))
    info.add_row("提示", Text("/help 查看命令  ·  Ctrl+C 中断", style="accent"))

    divider = Text("│\n│\n│\n│\n│", style="#334155", justify="center")

    body = Table.grid(expand=True, padding=(0, 2))
    body.add_column(width=58, justify="center", vertical="middle", no_wrap=True)
    body.add_column(width=1, justify="center", vertical="middle", no_wrap=True)
    body.add_column(ratio=1, vertical="middle")
    body.add_row(mark, divider, info)

    console.print()
    console.print(
        Panel(
            body,
            border_style="muted",
            box=ROUNDED,
            padding=(1, 2),
            title="[brand] Welcome back [/brand]",
            subtitle="[muted]terminal-native coding agent[/muted]",
        )
    )
    console.print()


# ─── 消息列表渲染（加载历史） ─────────────────────────────


def render_conversation(messages: list) -> None:
    """渲染完整对话历史"""
    top_flag = True
    for i, message in enumerate(messages):
        if message.additional_kwargs.get("hide", ""):
            continue
        msg_type = message.type
        content = message.content
        from lorcy_code.shared.text import get_text_content
        content = get_text_content(content)

        if msg_type == "human":
            if top_flag:
                top_flag = False
            else:
                render_separator()
            render_human(content or "")

        elif msg_type == "ai":
            reasoning = message.additional_kwargs.get("reasoning")
            if reasoning:
                render_reasoning(reasoning)
            if content:
                render_ai_start()
                console.print(Markdown(content))
                render_ai_end()

        elif msg_type == "tool":
            if content:
                render_tool(message.name or "tool", content)

    console.print()


# ─── 上下文用量 ──────────────────────────────────────────


def _format_tokens(n: int) -> str:
    """格式化 token 数：123456 → 123.5K"""
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


def get_context_usage_text(messages: list, max_context: int) -> str:
    """
    从消息列表计算上下文占用，返回带样式的文本。

    取最后一次 AIMessage 的 input_tokens 作为上下文快照
    （因为每次请求的 input_tokens 包含了完整上下文）。
    """
    input_tokens = 0
    for message in reversed(messages):
        from langchain_core.messages import AIMessage

        if isinstance(message, AIMessage):
            usage = message.usage_metadata
            if usage and usage.get("input_tokens"):
                input_tokens = usage["input_tokens"]
                break

    if input_tokens == 0:
        return ""

    pct = input_tokens / max_context
    used_str = _format_tokens(input_tokens)
    max_str = _format_tokens(max_context)
    pct_str = f"{pct * 100:.0f}%"

    if pct < 0.7:
        style = "yellow"
    elif pct < 0.9:
        style = "bold yellow"
    else:
        style = "bold red"

    return f"[{style}]{used_str}/{max_str} {pct_str}[/{style}]"
