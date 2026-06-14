import shutil
import re
import asyncio
import openai
import os
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
    render_warning,
    render_success,
    render_info,
    get_context_usage_text,
)
from lorcy_code.cli.ui.prompts import select, text, confirm
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
from lorcy_code.core.utils.agent_setup import (
    SkillAgentContext, 
    ModelSwitchError, 
    get_context_window_size,
)
from lorcy_code.core.utils.text_utils import get_text_content
from lorcy_code.core.agent.main_agent import build_agent, update_summarization_model
from lorcy_code.core.utils.session_manager import SessionManager
from lorcy_code.core.utils.model_retry import ModelRetryException, fallback_manager
from lorcy_code.core.tools.tool_result_pipeline import (
    reset_budget_state,
)
from lorcy_code.core.utils.enhanced_chat_openai import EnhancedChatOpenAI

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

    # ─── 清理 ────────────────────────────────────────

    async def close_checkpointer(self) -> None:
        """安全关闭 checkpointer 连接"""
        if self.checkpointer is not None:
            try:
                await self.checkpointer.conn.close()
            except Exception:
                pass
            finally:
                self.checkpointer = None

    async def _rebuild_agent(self, *, rebuild_session: bool = False) -> None:
        """重建 agent（可选重建 session/checkpointer）"""
        from lorcy_code.core.utils.agent_setup import create_checkpointer
        if rebuild_session:
            await self.close_checkpointer() # 关闭当前会话数据库连接
            self.session_mgr:SessionManager = SessionManager(self.workplace_path) # 创建会话管理器
            db_path = self.session_mgr.sessions_dir/ "checkpointer.db" # 创建新的会话数据库（一般是进入新工作目录才这样）
            self.checkpointer = await create_checkpointer(db_path) # 创建数据库连接
        self.agent = await asyncio.to_thread(  # 异步新线程构建agent
            build_agent,
            self.model_config,
            self.checkpointer,
        )

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
        # console.print()
        # console.print(
        #     "[dim cyan]"
        #     "██╗         ██████╗   ██████╗    ███████╗   ██╗   ██╗   ███████╗   ██████╗   █████╗    ████████╗\n"
        #     "██║        ██╔═══██╗  ██╔══██╗   ██╔═════╝  ╚██╗ ██╔╝  ██╔═════╝  ██╔═══██╗  ██╔══██╗  ██╔═════╝\n"
        #     "██║        ██║   ██║  ██████╔╝   ██║         ╚████╔╝   ██║        ██║   ██║  ██║  ██╗  ████████╗\n"
        #     "██║        ██║   ██║  ██╔══██╗   ██║          ╚██╔╝    ██║        ██║   ██║  ██║  ██╔╝ ██╔═════╝\n"
        #     "████████╗  ╚██████╔╝  ██║  ██║   ████████╗     ██║     ████████╗  ╚██████╔╝  █████╔═╝  ████████╗\n"
        #     "╚═══════╝   ╚═════╝   ╚═╝  ╚═╝    ╚══════╝     ╚═╝      ╚══════╝   ╚═════╝   ╚════╝    ╚══════╝ \n"
        #     "[dim cyan]"
        # )

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
        """处理斜杠命令"""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/new": self._cmd_new,
            "/history": self._cmd_history,
            "/model": self._cmd_model,
            "/compress": self._cmd_compress,
            "/messages": None,
            "/skill": None,
            "/search": None,
            "/workdir": self._cmd_workdir,
            "/tools": self._cmd_tools,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
        }

        handler = handlers.get(command)
        if handler:
            await handler(arg)
        else:
            render_warning(f"未知命令: {command}，输入 /help 查看帮助")

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
            asyncio.create_task(self._post_process())

        finally:
            self._processing = False

    async def _post_process(self) -> None:
        """流式输出后的后台处理：更新上下文用量、Git 提交"""
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            model_name = self.model_config.get("model", "")
            max_ctx = get_context_window_size(model_name)
            self._context_text = get_context_usage_text(messages, max_ctx)
        except Exception:
            pass

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

    # ------ slash command handlers ------
    async def _cmd_new(self, _arg: str) -> None:
        reset_budget_state()
        self.session_mgr.new_session()
        # render_success("新会话已开始")
        render_welcome()
        # self._render_status_bar()

    async def _cmd_tools(self, _arg: str) -> None:
        from lorcy_code.core.tools.tools import ALL_TOOLS
        from rich.table import Table
        table = Table(title="内置工具")
        table.add_column("工具", style="cyan")
        table.add_column("说明")

        for t in ALL_TOOLS:
            name = t.name
            desc = t.description.split("\n")[0] if t.description else ""
            table.add_row(name, desc)

        console.print(table)

    async def _cmd_quit(self, _arg: str) -> None:
        render_warning("Bye!")
        raise EOFError()
    
    async def _cmd_help(self, _arg: str) -> None:
        from rich.table import Table

        table = Table(title="命令列表")
        table.add_column("命令", style="cyan")
        table.add_column("说明")
        for cmd, desc in SLASH_COMMANDS.items():
            table.add_row(cmd, desc)
        console.print(table)

    # 模型配置
    async def _cmd_model(self, arg: str) -> None:
        from lorcy_code.cli.config.config import (
            configure_new_model,
            edit_current_model,
            switch_model,
        )
        from lorcy_code.cli.ui.prompts import select
        if arg == "new":
            config = await configure_new_model()
        elif arg == "edit":
            config = await edit_current_model()
        elif arg == "switch":
            config = await switch_model()
        else:
            action = await select(
                "模型管理:",
                [
                    "新建模型 (/model new)",
                    "编辑当前模型 (/model edit)",
                    "切换模型 (/model switch)",
                ],
            )
            if action is None:
                return
            if "新建" in action:
                config = await configure_new_model()
            elif "编辑" in action:
                config = await edit_current_model()
            elif "切换" in action:
                config = await switch_model()
            else:
                return

        if config:
            self.model_config = config
            from lorcy_code.core.agent.main_agent import update_summarization_model
            # 同步更新摘要模型
            update_summarization_model(config)

    async def _cmd_workdir(self, _arg: str) -> None:
        from lorcy_code.core.environment.build_env import load_workplace, save_workplace
        from lorcy_code.cli.ui.prompts import select_or_custom
        saved = load_workplace()
        choices = [str(saved)] if saved else []

        result = await select_or_custom(
            "选择工作目录:",
            choices,
            custom_label="自定义路径...",
            custom_prompt="请输入工作目录路径: ",
        )
        if not result:
            return

        new_path = Path(result)
        if not new_path.exists():
            render_error("路径不存在")
            return

        self.workplace_path = new_path
        self._skill_loader = None  # 工作目录变了，失效缓存
        os.chdir(self.workplace_path)
        save_workplace(self.workplace_path)

        # 重建子目录
        ensure_chat_config_dir(self.workplace_path)

        # 关闭旧 checkpointer 连接，重建会话和 agent
        await self._rebuild_agent(rebuild_session=True)

        # await self._init_git()
        render_success(f"工作目录: {self.workplace_path}")

    async def _cmd_history(self, _arg: str) -> None:
        # ------------------- 1.选择会话--------------------------------------
        if not self.session_mgr or not self.checkpointer or not self.agent:
            return
        sessions = await self.session_mgr.list_sessions(self.checkpointer) # 通过检查点从数据库（sqlite）中获取所有会话（实际为会话线程id）
        if not sessions:
            render_warning("没有历史会话")
            return

        sessions = sessions[-50:] # 取倒数50个会话并倒序排序（从新到旧）

        display_names = await self.session_mgr.get_display_names(sessions, self.agent) # 渲染所有会话的名称，返回一个 {tid: display_name} 的字典
        label_to_tid: dict[str, str] = {} # 初始化 <标签：会话线程id> 键值字典
        labels: list[str] = [] # 初始化标签列表（展示给用户的会话名）
        for tid in sessions:  # 遍历会话线程id
            name = display_names.get(tid, tid) # 通过 会话线程id 获取渲染的 会话名 ，如果没有则直接用 线程id 代替 空的 会话名
            label = name if name == tid else f"{name}  ({tid})" # 拼接 会话名 和 线程id 成 新的会话名，确保会话名 绝对的 唯一性
            label_to_tid[label] = tid # 构建 <新的会话名：会话线程id>字典
            labels.append(label) # 构建 会话名 列表（展示给用户）
        labels.reverse() # 将 会话名 列表倒序（从新到旧）
        labels.append("返回") # 在 会话名 列表最后 加上 返回 选项

        action = await select("选择历史会话:", labels) # 获取用户 选择的 会话名
        if action is None or action == "返回": # 如果是返回直接退出
            return

        selected_tid = label_to_tid[action] # 根据 用户选择 的会话名 在 之前构建好的<会话名：会话线程id>字典中 获取 会话名 对应的 会话线程id

        # ------------------- 2.操作选择的会话--------------------------------------
        match await select("操作:", ["加载此会话", "重命名此会话", "删除此会话", "返回"]): # 可以对会话进行 这4个操作
            case "加载此会话":
                self.session_mgr.set_thread(selected_tid) # 设置会话管理器 的 线程id 属性为 选中的 会话 对应的 线程id
                await self._load_conversation() # 加载会话历史消息 （通过 线程id 从 agent 的 state 中取）
            case "重命名此会话":
                try:
                    cur = self.session_mgr._load_names().get(selected_tid, "") # 尝试获取 已经可能被 更改过的 会话名  | names.json 通过 _save_names 保存。其在两个地方被调用：1. rename_session— 用户重命名会话时写入。  2. delete_session— 删除会话时从 names 里移除对应条目再写回
                except Exception: # 获取失败（说明当前会话尚未被改过名）
                    cur = ""
                new_name = await text("输入新名称（留空恢复默认）:", default=cur)
                if new_name is not None:
                    self.session_mgr.rename_session(selected_tid, new_name) # 将 新会话名 和 对应的 线程id 持久化到 name.json中
                    render_success("会话已重命名")
            case "删除此会话":
                ok = await confirm(f"确定删除会话 {selected_tid}？", default=False)
                if ok:
                    await self.session_mgr.delete_session(selected_tid, self.checkpointer)
                    render_success("会话已删除")
                    if selected_tid == self.session_mgr.thread_id:
                        await self._cmd_new("")
            case _: # 返回 或 Ctrl C 都回到上一步（重新加载历史会话）
                await self._cmd_history(_arg)

    async def _cmd_compress(self, _arg: str) -> None:
        import json
        if not self.model_config:
            render_warning("请先配置模型")
            return

        if not await confirm("确定压缩当前会话？", default=True):
            return

        render_info("压缩中...")
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages: list[BaseMessage] = state.values["messages"]

            # 分离历史消息和最近消息
            recent_messages = []
            recent_message_ids = []
            recent_count = 0
            for msg in reversed(messages):
                recent_messages.append(msg)
                recent_message_ids.append(msg.id)
                if isinstance(msg, HumanMessage):
                    recent_count += 1
                    if recent_count == 2:
                        break

            pre_messages = []
            for msg in messages:
                if msg.id not in recent_message_ids:
                    msg.additional_kwargs["composed"] = True
                    # 压缩时去掉 base64 图片/视频，避免 payload 过大导致 API 返回空 choices
                    if isinstance(msg.content, list):
                        clean_blocks = [
                            b for b in msg.content
                            if not isinstance(b, dict)
                            or b.get("type") not in ("image_url", "video_url")
                        ]
                        if clean_blocks != msg.content:
                            msg = msg.model_copy(update={"content": clean_blocks})
                    pre_messages.append(msg)

            model = EnhancedChatOpenAI(**self.model_config)

            human_msg = HumanMessage(
                content='以你的角度用第二人称压缩会话，严格按以下JSON格式输出，不要使用markdown代码块：\n{{"summary": "压缩内容"}}',
                additional_kwargs={"hide": True, "composed": True},
            )

            try:
                raw_resp = await asyncio.to_thread(
                    model.invoke, pre_messages + [human_msg]
                )

                content = raw_resp.content.strip()
                # 去除 markdown 代码块包裹
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                    content = re.sub(r"\n?```\s*$", "", content)
                # 提取包含 "summary" 的 JSON 对象（模型可能在 JSON 前输出思考内容）
                json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', content)
                if json_match:
                    content = json_match.group()
                else:
                    # 可能 summary 值中包含嵌套对象，用逐字符括号匹配兜底
                    # NOTE: 不处理字符串内的 `}`，但模型 summary 含 `}` 的概率极低，暂不改
                    depth = 0
                    start = -1
                    for i, ch in enumerate(content):
                        if ch == '{':
                            if depth == 0:
                                start = i
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0 and start >= 0:
                                candidate = content[start:i+1]
                                if '"summary"' in candidate:
                                    content = candidate
                                    break
                data = json.loads(content)
                ai_content = data.get("summary", "")
                if isinstance(ai_content, dict):
                    ai_content = json.dumps(ai_content, ensure_ascii=False)
                if not ai_content:
                    ai_content = "会话压缩失败: LLM 返回结果缺少 summary 字段"
            except Exception as e:
                ai_content = f"会话压缩失败: {e}"
                human_msg.additional_kwargs["composed"] = True

            if ai_content.startswith("会话压缩失败"):
                ai_message = AIMessage(
                    ai_content,
                    additional_kwargs={"error": True, "composed": True},
                    usage_metadata={
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )
            else:
                ai_message = AIMessage(
                    f"历史对话已压缩: {ai_content}",
                    additional_kwargs={"hide": True},
                )

            await self.agent.aupdate_state(
                self.session_mgr.config,
                {"messages": pre_messages + [human_msg, ai_message] + recent_messages},
                as_node="model",
            )
            await self._load_conversation()
            render_success("会话压缩完成")
        except Exception as e:
            render_error(f"压缩失败: {e}")

    async def _load_conversation(self) -> None:
        """加载当前会话的对话历史并渲染"""
        from lorcy_code.cli.ui.display import render_conversation
        if not self.agent:
            return
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            render_conversation(messages)
        except Exception as e:
            render_error(f"加载对话失败: {e}")