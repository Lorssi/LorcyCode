import shutil
import re
import asyncio
import openai
import lorcy_code.cli.ui.display as _display
from pathlib import Path
from rich.console import Console
from rich.text import Text

from prompt_toolkit.styles import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.layout.dimension import Dimension

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    RemoveMessage,
    HumanMessage,
    BaseMessage,
)
from langgraph.types import Command

from .display import (
    console,
    render_welcome,
    render_error,
    render_ai_start,
    render_ai_end,
    render_ai_chunk,
)

from lorcy_code.cli.config.config import (
    first_run_configure,
)
from lorcy_code.core.environment.build_env import (
    ensure_home_config_dir,
    ensure_chat_config_dir,
    load_model_config,
)
from lorcy_code.cli.ui.display import (
    console,
    render_error,
    render_welcome,
)
from lorcy_code.core.utils.agent_setup import SkillAgentContext, ModelSwitchError
from lorcy_code.core.utils.text_utils import get_text_content
from lorcy_code.core.agent.main_agent import build_agent
from lorcy_code.core.utils.session_manager import SessionManager
from lorcy_code.core.utils.model_retry import ModelRetryException, fallback_manager

SLASH_COMMANDS = {
    "/new": "新会话",
    "/history": "历史会话",
    "/model": "模型管理（新建/编辑/切换）",
    "/messages": "管理历史消息（编辑/分叉/删除）",
    "/compress": "压缩会话",
    "/skill": "技能管理",
    "/search": "配置 Tavily 搜索 API Key",
    "/workdir": "切换工作目录",
    "/tools": "显示内置工具",
    "/help": "显示帮助",
    "/quit": "退出",
}

# ─── 辅助函数 ──────────────────────────────────────────

# 简易的 BBCode 风格标记语言解析 （论坛或聊天软件）
_RE_TAG_SPLIT = re.compile(r"(\[/?[^\]]+\])")
_RE_TAG_OPEN = re.compile(r"^\[([^\]]+)\]$")
_RE_TAG_CLOSE = re.compile(r"^\[/([^\]]*)\]$")

_RICH_TAG_MAP = {
    "bold": "b",
    "italic": "i",
    "red": "fg:red",
    "green": "fg:green",
    "yellow": "fg:yellow",
    "blue": "fg:blue",
    "dim": "fg:#888888",
}


# 将BBCode 风格标记语言 渲染成 html 样式
def _rich_to_html(text: str) -> str:
    parts = _RE_TAG_SPLIT.split(text)
    opened: list[str] = []
    result: list[str] = []

    for part in parts:
        close_m = _RE_TAG_CLOSE.match(part)
        open_m = _RE_TAG_OPEN.match(part) if not close_m else None
        if close_m:
            while opened:
                tag = opened.pop()
                result.append(f"</{tag}>")
        elif open_m:
            tags = open_m.group(1).split()
            for t in tags:
                mapped = _RICH_TAG_MAP.get(t)
                if mapped:
                    if mapped.startswith("fg:"):
                        result.append(f'<style fg="{mapped[3:]}">')
                        opened.append("style")
                    else:
                        result.append(f"<{mapped}>")
                        opened.append(mapped)
        else:
            result.append(part)

    return "".join(result)

class _LimitedFileHistory(FileHistory):
    MAX_ENTRIES = 50

    def store_string(self, string):
        Path(self.filename).parent.mkdir(exist_ok=True)
        super().store_string(string)
        strings = list(self.load_history_strings())
        if len(strings) > self.MAX_ENTRIES:
            keep = strings[:self.MAX_ENTRIES]
            self._loaded_strings = keep
            self._rewrite(keep)

    def _rewrite(self, keep):
        import datetime as _dt
        Path(self.filename).parent.mkdir(exist_ok=True)
        with open(self.filename, "wb") as f:
            for s in reversed(keep):
                f.write(f"\n# {_dt.datetime.now()}\n".encode())
                for line in s.split("\n"):
                    f.write(f"+{line}\n".encode())

class SlashCommandCompleter(Completer):
    """斜杠命令自动补全器 - 输入 / 时触发下拉列表"""

    def get_completions(self, document, complete_event):
        # 获取光标前的完整文本
        text = document.text_before_cursor

        # 当输入 / 时触发补全
        if text.startswith("/"):
            # 把输入的文本中的字母转化成小写来处理（大小写不敏感）
            partial = text.lower()
            # 遍历预先定义的斜杠命令字典
            for cmd, desc in SLASH_COMMANDS.items():
                # 如果转化成小写的输入框中文本 被字典里 命令名 的 前缀匹配 到
                if cmd.startswith(partial):
                    # 生成命令
                    yield Completion(
                        cmd,  # 返回完整的命令
                        start_position=-len(partial),  # 返回前清空输入框已有输入
                        display=cmd,  # 下拉框显示的命令名
                        display_meta=desc,  # 下拉框显示的命令名的描述
                    )

class ChatREPL:
    def __init__(self):
        self.workplace_path: Path | None = None # 当前工作目录路径
        self.model_config: dict = {}  # 模型参数
        self._prompt_session = None # 初始化 prompt-toolkit 会话（用于命令自动补全）
        self._context_text: str = "" # 上下文用量缓存
        self._processing = False
        self._stop_requested = False  # 暂停agent的flag

        self._edit_buffer: str | None = None # 编辑缓冲区（用于 /edit 命令）
        self._interrupt_buffer: str | None = None # 中断恢复缓冲区（中断时将内容填回输入框，不进入编辑模式）

        self.agent = None  # agent实例
        self.checkpointer = None  # 检查点实例

    async def initialize(self):
        # 确保配置目录存在
        ensure_home_config_dir()
        ensure_chat_config_dir()

        self.workplace_path = Path.cwd()  # 获取当前目录路径

        self.model_config = load_model_config()
        if not self.model_config:
            config = await first_run_configure()
            if config is None:
                return False
            self.model_config = config
            

        # 构建 agent（可能较慢，放线程）
        console.print()
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

        # 创建 checkpointer
        from lorcy_code.core.utils.agent_setup import create_checkpointer
        self.session_mgr:SessionManager = SessionManager(self.workplace_path)  # 初始化历史会话管理器
        db_path = self.session_mgr.sessions_dir / "checkpointer.db"
        self.checkpointer = await create_checkpointer(db_path)

        # 构建 agent（可能较慢，放线程）
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
        )

        return True

    async def run(self):
        render_welcome()

        while True:
            try:
                user_input = await self._get_input()
                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # 斜杠命令
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # 正常对话
                await self._process_input(user_input)

            except KeyboardInterrupt:
                if self._processing:
                    self._stop_requested = True
                else:
                    console.print(Text("\n再见！", style="dim"))
                    break
            except EOFError:
                break
            except Exception as e:
                render_error(f"Unexpected error: {e}")

    async def _get_input(self) -> str | None:
        # 初始化 prompt session（带命令自动补全 + 底部状态栏）
        if self._prompt_session is None:
            completer = SlashCommandCompleter()

            # 自定义按键：Enter 提交，Ctrl+Enter 换行
            kb = KeyBindings()

            @kb.add("enter")
            def _submit(event):
                event.current_buffer.validate_and_handle() # 验证并提交缓冲区内容

            @kb.add("c-j")  # Ctrl+Enter → 换行
            def _newline(event):
                event.current_buffer.insert_text("\n") # 向缓冲区插入换行

            _last_width = 0
            _last_width_time = 0.0

            def _bottom_toolbar():
                nonlocal _last_width, _last_width_time
                import time as _time
                now = _time.monotonic()
                if now - _last_width_time > 1.0:
                    _last_width = shutil.get_terminal_size().columns
                    _last_width_time = now
                width = _last_width or shutil.get_terminal_size().columns
                sep = "\u2500" * width
                parts = []
                model = self.model_config.get("model", "未设置")
                parts.append(model)
                if self._context_text:
                    styled = _rich_to_html(self._context_text)
                    parts.append(styled)
                parts.append(
                    "普通模式"
                )
                # if self.git and self.git_manager and self.git_manager.is_repo():
                #     parts.append(f"Git ({self._git_cp_count} cp)")
                wp = str(self.workplace_path) if self.workplace_path else ""
                if wp:
                    parts.append(f"cwd: {wp}")
                status = "  │  ".join(parts)
                ratelimit_line = ""

                return HTML(f"<ansiblue>{sep}</ansiblue>\n{status}{ratelimit_line}")
            
            self._prompt_session:PromptSession = PromptSession(
                history=_LimitedFileHistory(str(Path.home() / ".lorcy" / "history")),
                multiline=True,
                key_bindings=kb,
                completer=completer,
                complete_while_typing=True,
                reserve_space_for_menu=0,
                bottom_toolbar=_bottom_toolbar,
                refresh_interval=0.1,
                style=Style.from_dict(
                    {
                        "completion-menu.completion": "bg:#008888 #ffffff",
                        "completion-menu.completion.current": "bg:#00aaaa #000000",
                        "completion-menu.meta.completion": "bg:#008888 #ffffff",
                        "completion-menu.meta.completion.current": "bg:#00aaaa #000000",
                        "bottom-toolbar": "noreverse bg:#1a1a2e #aaaaaa",
                    }
                ),
            )

            # 动态缓存区高度
            def _dynamic_buffer_height():
                buff = self._prompt_session.default_buffer
                if buff.complete_state is not None:
                    n = len(buff.complete_state.completions)
                    needed = min(n + 2, 10)
                    return Dimension(min=needed, max=needed)
                line_count = buff.text.count("\n") + 1
                return Dimension(min=line_count, max=line_count)
            
            # 寻找缓存区窗口
            def _find_buffer_window(container):
                from prompt_toolkit.layout.containers import Window
                from prompt_toolkit.layout.controls import BufferControl

                if isinstance(container, Window):
                    if isinstance(getattr(container, "content", None), BufferControl):
                        return container
                for attr in ("content", "children", "alternative_content"):
                    child = getattr(container, attr, None)
                    if child is None:
                        continue
                    children = child if isinstance(child, list) else [child]
                    for c in children:
                        result = _find_buffer_window(c)
                        if result:
                            return result
                return None
            
            buffer_window = _find_buffer_window(
                self._prompt_session.app.layout.container
            )
            if buffer_window:
                buffer_window.height = _dynamic_buffer_height

        try:
            # 如果有编辑缓冲区，预填充到输入框
            if self._edit_buffer is not None:
                default_text = self._edit_buffer
                self._edit_buffer = None  # 清除缓冲区
            # 如果有中断恢复缓冲区，也预填充到输入框
            elif self._interrupt_buffer is not None:
                default_text = self._interrupt_buffer
                self._interrupt_buffer = None  # 清除缓冲区
            # 如果都没有，则不填充
            else:
                default_text = ""

            width = shutil.get_terminal_size().columns # 获取终端大小的列数，确保分隔线始终覆盖整个宽度
            sep = "\u2500" * width # 即为 width个 ─ , 效果：───────────────（这个是输入框的顶栏，由于prompt_toolkit不支持顶栏，所以需要自己构造）
            prompt_text = f"{sep}\n > " # 构造顶栏和 > 提示符

            # 使用 prompt-toolkit 获取输入（支持命令自动补全）
            result = await self._prompt_session.prompt_async( # 显示顶栏和 > 提示符，并等待用户输入，返回值也是用户的输入
                HTML(f"<ansiblue>{prompt_text}</ansiblue>"),
                default=default_text, # 返回的默认值，代替用户输入或可能为空
            )
            return result
        except (EOFError, KeyboardInterrupt):
            return None

    async def _handle_command(self, cmd: str) -> None:
        pass

    async def _process_input(self, user_input: str) -> None:
        """处理用户输入并调用 agent"""
        self._processing = True
        self._stop_requested = False

        accumulated_content = ""
        ai_started = False

        try: 
            input_data = {"messages": user_input}

            # 保存原始输入，用于模型切换后重试时重置 input_data
            _original_input_data = input_data


            skill_agent_context = SkillAgentContext(
                working_directory=self.workplace_path,
                model_config=self.model_config,
                thread_id=self.session_mgr.thread_id,
            )

            while True:
                interrupt_chunk = None

                try:
                    async for m, i in self.agent.astream(
                        input_data,
                        self.session_mgr.config,
                        stream_mode=["messages", "updates"],
                        context=skill_agent_context,
                    ):
                        if self._stop_requested:
                            raise asyncio.CancelledError()

                        if m == "messages":
                            content = get_text_content(i[0].content)
                            additional_kwargs = i[0].additional_kwargs

                            if additional_kwargs.get("hide", ""):
                                continue

                            if isinstance(i[0], AIMessageChunk):
                                reasoning = additional_kwargs.get("reasoning")
                                if reasoning:
                                    if (
                                        not _display._subagent_parallel
                                        and _display._subagent_count == 0
                                    ):
                                        console.print(reasoning, end="", style="dim")
                                if not ai_started:
                                    if not content:
                                        continue
                                    ai_started = True
                                    render_ai_start()
                                render_ai_chunk(content or "")
                                accumulated_content += content or ""

                            elif isinstance(i[0], ToolMessage):
                                ai_started = False

                        elif m == "updates" and "__interrupt__" in i:
                            interrupt_chunk = i

                except asyncio.CancelledError:
                    await self._handle_cancel(user_input)
                    _display.force_reset_display()
                    console.print(Text("\n[已中断]", style="dim"), "\n")
                    break
                except ModelSwitchError:
                    # 需要切换到备用模型
                    fallback = fallback_manager.get_fallback_model()
                    if fallback:
                        console.print(f"[yellow]正在切换到备用模型: {fallback.get('model', 'unknown')}[/yellow]")
                        self.model_config = fallback
                        fallback_manager.advance_fallback()
                        # 持久化到 model.json，确保模型列表显示一致
                        import copy
                        from lorcy_code.core.environment.build_env import load_model_json, save_model_json
                        _data = copy.deepcopy(load_model_json())
                        _old_default = _data.get("default", {})
                        _old_model = _old_default.get("model", "")
                        if _old_model and _old_model not in _data.get("fallback", {}):
                            _data.setdefault("fallback", {})[_old_model] = _old_default
                        _data["default"] = fallback
                        save_model_json(_data)
                        try:
                            await self._rebuild_agent()
                            # 重建 context 以使用新模型配置
                            skill_agent_context = SkillAgentContext(
                                working_directory=self.workplace_path,
                                model_config=self.model_config,
                                thread_id=self.session_mgr.thread_id,
                            )
                            # 如果当前 input_data 是已消费的 Command(resume=...)，
                            # 重置为原始输入，避免复用已消费的 Command
                            if isinstance(input_data, Command):
                                input_data = _original_input_data
                            console.print("[green]已切换到备用模型，自动重试中...[/green]")
                            continue  # 用备用模型重试当前请求
                        except Exception as e:
                            render_error(f"切换模型失败: {e}")
                    else:
                        render_error("没有更多备用模型可用")
                        await self._handle_agent_error(ModelSwitchError("所有模型均失败"))
                    break
                except openai.APIError as e:
                    render_error(f"Agent 执行错误: {e}")
                    await self._handle_agent_error(e)
                    break
                except Exception as e:
                    render_error(f"Agent 执行错误: {e}")
                    await self._handle_agent_error(e)
                    break

                if self._stop_requested:
                    break

                if interrupt_chunk is None:
                    break

                # HITL 审批
                decisions = await self._collect_decisions_async(interrupt_chunk)
                input_data = Command(resume={"decisions": decisions})

            if ai_started:
                render_ai_end()

            # 后处理（上下文更新 + Git 提交）放到后台，不阻塞输入框
            # asyncio.create_task(self._post_process())

        finally:
            self._processing = False

    async def _cleanup_last_turn(self, append_msg: str | None = None) -> list[BaseMessage] | None:
        """查找最后一组消息：若无 AIMessage 则删除整组并返回该组，否则追加错误消息返回 None

        用于统一 _handle_agent_error 和 _handle_cancel 的共同逻辑：
        找到最后一组消息（以最后一个 HumanMessage 开头），
        判断当前组是否有 AIMessage，分别处理。
        """
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages: list[BaseMessage] = state.values.get("messages", [])

            last_human_idx = -1
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    last_human_idx = i

            if last_human_idx >= 0:
                current_group = messages[last_human_idx:]
                has_ai = any(isinstance(m, AIMessage) for m in current_group)

                if not has_ai:
                    await self._delete_messages([m.id for m in current_group])
                    return current_group

            if append_msg:
                error_msg = AIMessage(
                    append_msg,
                    additional_kwargs={"error": True, "composed": True},
                )
                await self.agent.aupdate_state(
                    self.session_mgr.config,
                    {"messages": [error_msg]},
                    as_node="model",
                )
        except Exception:
            pass
        return None

    async def _handle_agent_error(self, error: Exception) -> None:
        """Agent 出错时：当前组无 AIMessage 则删除整组，否则保存错误消息"""
        deleted = await self._cleanup_last_turn(f"Agent 执行错误: {error}")
        # 如果没有删除整组（已有 AIMessage），错误消息已在 _cleanup_last_turn 中追加