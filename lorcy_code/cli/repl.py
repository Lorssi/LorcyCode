import shutil
import asyncio
import openai
import os
import re
import lorcy_code.cli.display as _display
from pathlib import Path
from rich.console import Console
from rich.text import Text
from rich.box import SIMPLE_HEAVY

from prompt_toolkit.styles import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from .input import LimitedFileHistory, SLASH_COMMANDS, SlashCommandCompleter, rich_to_html
from .commands import dispatch_command

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
from lorcy_code.cli.prompts import select, text, confirm, checkbox, select_or_custom
from lorcy_code.config.models import (
    first_run_configure,
)
from lorcy_code.config.storage import (
    ensure_home_config_dir,
    ensure_chat_config_dir,
    load_model_config,
)
from lorcy_code.cli.display import (
    console,
    render_error,
    render_welcome,
)
from lorcy_code.agents.context import (
    SkillAgentContext,
    get_context_window_size,
)
from lorcy_code.agents.errors import ModelSwitchError
from lorcy_code.shared.text import get_text_content
from lorcy_code.agents.builder import build_agent, update_summarization_model
from lorcy_code.sessions.manager import SessionManager
from lorcy_code.agents.retry import ModelRetryException, fallback_manager
from lorcy_code.tools.result_pipeline import (
    reset_budget_state,
)
from lorcy_code.agents.model import EnhancedChatOpenAI
from lorcy_code.skills.loader import SkillLoader
from lorcy_code.integrations.git import GitManager, check_git_availability

class ChatREPL:
    def __init__(self, *, yolo: bool = False):
        self.workplace_path: Path | None = None # 当前工作目录路径
        self.model_config: dict = {}  # 模型参数
        self._prompt_session = None # 初始化 prompt-toolkit 会话（用于命令自动补全）
        self._context_text: str = "" # 上下文用量缓存
        self._processing = False
        self._stop_requested = False  # 暂停agent的flag
        self.yolo = yolo  # Yolo模式

        self.git_manager: GitManager | None = None  # git管理器
        self.git = False  # git是否激活
        self._git_cp_count = 0  # git提交数

        self._edit_buffer: str | None = None # 编辑缓冲区（用于 /edit 命令）
        self._interrupt_buffer: str | None = None # 中断恢复缓冲区（中断时将内容填回输入框，不进入编辑模式）

        self.agent = None  # agent实例
        self.checkpointer = None  # 检查点实例

        self._skill_loader: SkillLoader | None = None # SkillLoader 复用，避免每条消息重建

        self.session_mgr: SessionManager | None = None  # 会话管理器
        self.mcp_manager = None  # MCP 连接与动态工具目录

    def _create_skill_loader(self) -> SkillLoader:
        from lorcy_code.config.storage import load_skill_selection

        selection = load_skill_selection(self.workplace_path)
        return SkillLoader(
            [
                self.workplace_path / ".lorcy/skills",
                Path.home() / ".lorcy/skills",
            ],
            selection_mode=selection.get("mode", "all"),
            enabled_skills=selection.get("skills", []),
        )

    def _ensure_skill_loader(self) -> SkillLoader:
        if self._skill_loader is None:
            self._skill_loader = self._create_skill_loader()
        return self._skill_loader

    def _skill_status_text(self) -> str:
        loader = self._skill_loader
        if loader is None:
            return "skills: all"
        selection = loader.get_skill_selection()
        if selection.get("mode") == "all":
            return "skills: all"
        return f"skills: {len(loader.get_enabled_skill_names())}"

    def _mcp_status_text(self) -> str:
        states = self.mcp_manager.states.values() if self.mcp_manager else []
        enabled = sum(1 for state in states if state.config.enabled)
        return f"MCP: {enabled}"

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

    async def close(self) -> None:
        """关闭会话相关的外部资源。"""
        if self.mcp_manager is not None:
            await self.mcp_manager.close()
            self.mcp_manager = None
        await self.close_checkpointer()

    def _effective_tools(self) -> list:
        from lorcy_code.tools.registry import ALL_TOOLS
        mcp_tools = self.mcp_manager.get_tools() if self.mcp_manager else []
        return [*ALL_TOOLS, *mcp_tools]

    async def _confirm_mcp_trust(self, config) -> bool:
        command = " ".join([config.command or "", *config.args]).strip()
        return await confirm(
            f"项目请求运行 MCP 服务 {config.name}: {command}\n是否信任此配置？",
            default=False,
        )

    async def _rebuild_agent(self, *, rebuild_session: bool = False) -> None:
        """重建 agent（可选重建 session/checkpointer）"""
        from lorcy_code.agents.context import create_checkpointer
        if rebuild_session:
            await self.close_checkpointer() # 关闭当前会话数据库连接
            self.session_mgr:SessionManager = SessionManager(self.workplace_path) # 创建会话管理器
            db_path = self.session_mgr.sessions_dir / "checkpointer.db" # 创建新的会话数据库（一般是进入新工作目录才这样）
            self.checkpointer = await create_checkpointer(db_path) # 创建数据库连接
        self.agent = await asyncio.to_thread(  # 异步新线程构建agent
            build_agent,
            self.model_config,
            self.checkpointer,
            self.yolo,
            self._effective_tools(),
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

        self._skill_loader = self._create_skill_loader()

        # 创建 checkpointer
        from lorcy_code.agents.context import create_checkpointer
        self.session_mgr:SessionManager = SessionManager(self.workplace_path)  # 初始化历史会话管理器
        db_path = self.session_mgr.sessions_dir / "checkpointer.db"
        self.checkpointer = await create_checkpointer(db_path)

        # MCP 服务独立初始化；单个服务失败不会阻止主 Agent 启动。
        from lorcy_code.mcp import MCPManager
        self.mcp_manager = MCPManager(
            self.workplace_path,
            trust_callback=self._confirm_mcp_trust,
        )
        await self.mcp_manager.start()
        for error in self.mcp_manager.store.errors:
            render_warning(f"MCP 配置错误: {error}")
        for warning in self.mcp_manager.store.warnings:
            render_warning(f"MCP 安全提示: {warning}")

        # 构建 agent（可能较慢，放线程）
        self.agent = await asyncio.to_thread(
            build_agent,
            self.model_config,
            self.checkpointer,
            self.yolo,
            self._effective_tools(),
        )

        # 初始化 Git（subprocess.run 会阻塞事件循环）
        await self._init_git()

        return True
    
    async def _init_git(self) -> None:
        """初始化 Git"""
        from lorcy_code.integrations.git import GitManager, check_git_availability
        is_available, status, version = await asyncio.to_thread(check_git_availability) # 检查是否安装了Git且为可用状态
        if is_available: # 如果可用
            self.git_manager = GitManager(str(self.workplace_path)) # 初始Git管理器
            if not self.git_manager.is_repo(): # 如果没有Git仓库
                await asyncio.to_thread(self.git_manager.init) # 就初始化Git
            else: # 如果已经有Git仓库了
                await asyncio.to_thread(self.git_manager._ensure_init_checkpoint) # 就确保 仓库至少有一条提交供回溯
            self.git = True # 设置Git为可用状态，此后回溯消息会自动回溯工作目录
            self._git_cp_count = self.git_manager.count_checkpoints() # 记录提交数

    async def run(self):
        render_welcome(
            model=self.model_config.get("model"),
            workdir=str(self.workplace_path) if self.workplace_path else None,
            yolo=self.yolo,
        )

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

            @kb.add("tab")
            def _tab_toggle_mode(event):
                pass

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
                    styled = rich_to_html(self._context_text)
                    parts.append(styled)
                parts.append(
                    "普通模式" if not self.yolo else "<ansired>YOLO 模式</ansired>"
                )
                if self.git and self.git_manager and self.git_manager.is_repo():
                    parts.append(f"Git ({self._git_cp_count} cp)")
                parts.append(self._skill_status_text())
                parts.append(self._mcp_status_text())
                wp = str(self.workplace_path) if self.workplace_path else ""
                if wp:
                    parts.append(f"cwd: {wp}")
                status = "  │  ".join(parts)
                ratelimit_line = ""

                return HTML(f'<style fg="#334155">{sep}</style>\n  {status}{ratelimit_line}')
            
            self._prompt_session:PromptSession = PromptSession(
                history=LimitedFileHistory(str(Path.home() / ".lorcy" / "history")),
                multiline=True,
                key_bindings=kb,
                completer=completer,
                complete_while_typing=True,
                reserve_space_for_menu=0,
                bottom_toolbar=_bottom_toolbar,
                refresh_interval=0.1,
                style=Style.from_dict(
                    {
                        "completion-menu.completion": "bg:#111827 #cbd5e1",
                        "completion-menu.completion.current": "bg:#0f766e #ffffff bold",
                        "completion-menu.meta.completion": "bg:#111827 #7c8aa5",
                        "completion-menu.meta.completion.current": "bg:#0f766e #ccfbf1",
                        "bottom-toolbar": "noreverse bg:#0b1220 #7c8aa5",
                    }
                ),
            )
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
            prompt_text = f'<style fg="#334155">{sep}</style>\n<style fg="#5eead4"><b>  ❯ </b></style>'

            # 使用 prompt-toolkit 获取输入（支持命令自动补全）
            result = await self._prompt_session.prompt_async( # 显示顶栏和 > 提示符，并等待用户输入，返回值也是用户的输入
                HTML(prompt_text),
                default=default_text, # 返回的默认值，代替用户输入或可能为空
            )
            return result
        except (EOFError, KeyboardInterrupt):
            return None

    async def _handle_command(self, cmd: str) -> None:
        """处理斜杠命令"""
        await dispatch_command(self, cmd)

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

            skill_loader = self._ensure_skill_loader()

            skill_agent_context = SkillAgentContext(
                skill_loader=skill_loader,
                working_directory=self.workplace_path,
                model_config=self.model_config,
                thread_id=self.session_mgr.thread_id,
                extra={"mcp_tools": self.mcp_manager.get_tools() if self.mcp_manager else []},
                yolo=self.yolo,
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
                                        _display.begin_model_output()
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
                        from lorcy_code.config.storage import load_model_json, save_model_json
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
                                skill_loader=skill_loader,
                                working_directory=self.workplace_path,
                                model_config=self.model_config,
                                thread_id=self.session_mgr.thread_id,
                                extra={"mcp_tools": self.mcp_manager.get_tools() if self.mcp_manager else []},
                                yolo=self.yolo,
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
            _display.finalize_turn_display()
            self._processing = False

    async def _post_process(self) -> None:
        """流式输出后的后台处理：更新上下文用量、Git 提交"""
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            model_name = self.model_config.get("model", "")
            max_ctx = get_context_window_size(model_name)
            self._context_text = get_context_usage_text(messages, max_ctx)

            def find_and_slice_from_end(lst, x):
                """从后往前查找第一个 type==x 的元素，返回从该元素到末尾的切片"""
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i].type == x:
                        return lst[i:]
                return []

            if self.git and self.git_manager:
                new_msgs = find_and_slice_from_end(messages, "human")
                ids = [m.id for m in new_msgs]
                result = await asyncio.to_thread(
                    self.git_manager.add_commit, "&".join(ids)
                )
                if isinstance(result, int) and not isinstance(result, bool):
                    self._git_cp_count = result
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

    async def _handle_cancel(self, user_input: str) -> None:
        """取消时：当前组无 AIMessage 则删除整组并回填输入框，否则追加停止消息"""
        deleted = await self._cleanup_last_turn("该消息意外停止")
        if deleted is not None:
            self._interrupt_buffer = user_input.strip()

    # --------------------------------------------------------------------------------
    # 中断处理
    # --------------------------------------------------------------------------------

    async def _collect_decisions_async(self, interrupt_chunk) -> list[dict]:
        """收集 HITL 决策"""
        console.print()  # 确保 AI 输出和 HITL 之间有换行
        decisions = []
        for interrupt in interrupt_chunk["__interrupt__"]:
            action_requests = interrupt.value["action_requests"]

            for action_request in action_requests:
                name = action_request["name"]
                args = action_request["args"]

                content = ""
                match name:
                    case "bash":
                        content = args.get("command", "")
                    case "write_file":
                        content = f"写入文件: {args.get('file_path')}\n内容: {args.get('content', '')[:200]}"
                    case "edit":
                        file_path = args.get("file_path", "")
                        old_str = args.get("old_string", "")
                        new_str = args.get("new_string", "")
                        render_warning(f"[HITL] edit  修改文件: {file_path}")
                        import difflib
                        from rich.table import Table

                        # 查找 old_str 在文件中的起始行号
                        start_line = 1
                        try:
                            content = await asyncio.to_thread(
                                Path(file_path).read_text, encoding="utf-8"
                            )
                            for i, line in enumerate(content.splitlines(), 1):
                                if old_str.splitlines()[0] in line:
                                    start_line = i
                                    break
                        except Exception:
                            pass
                        old_lines = old_str.splitlines()
                        new_lines = new_str.splitlines()
                        table = Table(
                            show_header=False,
                            show_edge=False,
                            padding=(0, 1),
                            border_style="dim",
                        )
                        table.add_column("old", ratio=1)
                        table.add_column("new", ratio=1)
                        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
                        old_num = start_line
                        new_num = start_line
                        for tag, i1, i2, j1, j2 in sm.get_opcodes():
                            if tag == "equal":
                                for k in range(i2 - i1):
                                    table.add_row(
                                        Text(
                                            f"  {old_num:>3}  {old_lines[i1 + k]}",
                                            style="dim",
                                        ),
                                        Text(
                                            f"  {new_num:>3}  {new_lines[j1 + k]}",
                                            style="dim",
                                        ),
                                    )
                                    old_num += 1
                                    new_num += 1
                            elif tag == "replace":
                                max_len = max(i2 - i1, j2 - j1)
                                for k in range(max_len):
                                    old_text = (
                                        Text(
                                            f"{old_num:>3} - {old_lines[i1 + k]}",
                                            style="red",
                                        )
                                        if k < i2 - i1
                                        else None
                                    )
                                    new_text = (
                                        Text(
                                            f"{new_num:>3} + {new_lines[j1 + k]}",
                                            style="green",
                                        )
                                        if k < j2 - j1
                                        else None
                                    )
                                    table.add_row(old_text, new_text)
                                    if k < i2 - i1:
                                        old_num += 1
                                    if k < j2 - j1:
                                        new_num += 1
                            elif tag == "delete":
                                for k in range(i2 - i1):
                                    table.add_row(
                                        Text(
                                            f"{old_num:>3} - {old_lines[i1 + k]}",
                                            style="red",
                                        )
                                    )
                                    old_num += 1
                            elif tag == "insert":
                                for k in range(j2 - j1):
                                    table.add_row(
                                        None,
                                        Text(
                                            f"{new_num:>3} + {new_lines[j1 + k]}",
                                            style="green",
                                        ),
                                    )
                                    new_num += 1
                        console.print(table)
                        content = None  # 已直接渲染，跳过通用渲染

                if self.yolo:
                    select_action = True
                else:
                    if content is not None:
                        render_warning(f"[HITL] {name}")
                        console.print(Text(f"  {content[:500]}", style="dim"))
                    result = await select(
                        "操作:",
                        ["approve (批准)", "reject (拒绝)"],
                    )
                    select_action = result != "reject (拒绝)" if result else False

                extra = {}
                if not select_action:
                    extra["message"] = "用户已拒绝"
                decision = {"type": "approve" if select_action else "reject"}
                decision.update(extra)
                decisions.append(decision)

        return decisions

    # --------------------------------------------------------------------------------
    # 命令处理函数（斜杠命令的具体实现） - 每个函数对应一个斜杠命令，参数为命令后面的字符串
    # --------------------------------------------------------------------------------
    async def _cmd_new(self, _arg: str) -> None:
        reset_budget_state()
        self.session_mgr.new_session()
        # render_success("新会话已开始")
        render_welcome(
            model=self.model_config.get("model"),
            workdir=str(self.workplace_path) if self.workplace_path else None,
            yolo=self.yolo,
        )
        # self._render_status_bar()

    async def _cmd_tools(self, _arg: str) -> None:
        from rich.table import Table
        table = Table(
            title="◆ 可用工具",
            box=SIMPLE_HEAVY,
            border_style="muted",
            header_style="bold #60a5fa",
            row_styles=["", "dim"],
        )
        table.add_column("工具", style="tool", no_wrap=True)
        table.add_column("来源", style="cyan", no_wrap=True)
        table.add_column("说明", style="white")

        for t in self._effective_tools():
            name = t.name
            desc = t.description.split("\n")[0] if t.description else ""
            metadata = getattr(t, "metadata", None) or {}
            source = metadata.get("mcp_server", "内置")
            table.add_row(name, source, desc)

        console.print(table)

    async def _cmd_mcp(self, arg: str) -> None:
        """管理 MCP 服务。"""
        from rich.table import Table
        from lorcy_code.mcp.models import MCPConfigError, MCPServerConfig, MCPToolFilter

        usage = (
            "/mcp [list|add|remove|enable|disable|status|test|tools|refresh|logs] [name]"
        )
        parts = arg.strip().split(maxsplit=1)
        action = parts[0].lower() if parts else ""
        name = parts[1].strip() if len(parts) > 1 else ""

        if not action:
            choice = await select(
                "MCP 管理:",
                ["查看服务", "添加服务", "启用服务", "停用服务", "刷新服务", "查看日志"],
            )
            if not choice:
                return
            action = {
                "查看服务": "list", "添加服务": "add", "启用服务": "enable",
                "停用服务": "disable", "刷新服务": "refresh", "查看日志": "logs",
            }[choice]

        states = self.mcp_manager.states if self.mcp_manager else {}

        async def choose_server(message: str, predicate=lambda _state: True) -> str:
            choices = [key for key, state in states.items() if predicate(state)]
            if not choices:
                return ""
            return await select(message, choices) or ""

        if action in {"list", "status"}:
            if name and name not in states:
                render_error(f"MCP 服务不存在: {name}")
                return
            table = Table(title="◆ MCP 服务", box=SIMPLE_HEAVY, border_style="muted")
            for column in ("服务", "来源", "传输", "状态", "工具", "连接耗时", "最近错误"):
                table.add_column(column)
            selected = {name: states[name]} if name else states
            for server_name, state in selected.items():
                self.mcp_manager.get_state(server_name)
                table.add_row(
                    server_name,
                    state.config.source,
                    state.config.transport,
                    state.status.value,
                    str(len(state.tools)),
                    f"{state.connected_ms:.0f}ms" if state.connected_ms is not None else "-",
                    state.last_error or "-",
                )
            console.print(table)
            for error in self.mcp_manager.store.errors:
                render_warning(f"配置错误: {error}")
            for warning in self.mcp_manager.store.warnings:
                render_warning(f"安全提示: {warning}")
            return

        if action == "add":
            if not name:
                name = (await text("服务名称: ")).strip()
            if name in states and not await confirm(
                f"MCP 服务 {name} 已存在，是否覆盖？", default=False
            ):
                return
            source_label = await select("配置作用域:", ["用户级", "项目级"])
            transport_label = await select("传输类型:", ["stdio", "streamable_http"])
            if not source_label or not transport_label:
                return
            try:
                kwargs = {
                    "name": name,
                    "enabled": True,
                    "transport": transport_label,
                    "source": "user" if source_label == "用户级" else "workspace",
                    "timeout_seconds": float((await text("超时秒数: ", "60")).strip()),
                    "tool_filter": MCPToolFilter(),
                }
                if transport_label == "stdio":
                    kwargs["command"] = (await text("启动命令: ")).strip()
                    raw_args = await text("命令参数: ")
                    kwargs["args"] = self._split_mcp_args(raw_args)
                    kwargs["cwd"] = (await text("工作目录: ", "${workspace}")).strip() or None
                    kwargs["env"] = self._parse_mcp_pairs(await text("环境变量（逗号分隔 KEY=VALUE）: "))
                else:
                    kwargs["url"] = (await text("MCP URL: ")).strip()
                    kwargs["headers"] = self._parse_mcp_pairs(
                        await text("Headers（逗号分隔 KEY=VALUE）: ")
                    )
                config = MCPServerConfig.from_raw(
                    name,
                    MCPServerConfig(**kwargs).to_raw(),
                    source=kwargs["source"],
                )
            except (ValueError, MCPConfigError) as exc:
                render_error(f"配置无效: {exc}")
                return
            ok, error, count = await self.mcp_manager.test_config(config)
            if not ok:
                render_error(f"连接测试失败: {error or '未知错误'}")
                return
            try:
                self.mcp_manager.store.save_server(config)
            except MCPConfigError as exc:
                render_error(str(exc))
                return
            await self.mcp_manager.reload()
            await self._rebuild_agent()
            render_success(f"已添加 MCP 服务 {name}，发现 {count} 个工具")
            return

        if action == "refresh" and not name:
            await self.mcp_manager.reload()
            ok = await self.mcp_manager.refresh()
            await self._rebuild_agent()
            if ok:
                render_success("已刷新全部 MCP 服务")
            else:
                render_warning("刷新完成，部分 MCP 服务连接失败；可用服务不受影响")
            return

        if action in {"enable", "disable", "remove", "test", "refresh", "tools", "logs"}:
            if not name:
                name = await choose_server("选择 MCP 服务:")
            if not name:
                render_warning("没有可用的 MCP 服务")
                return
            if name not in states:
                render_error(f"MCP 服务不存在: {name}")
                return

        if action in {"enable", "disable"}:
            enabled = action == "enable"
            try:
                self.mcp_manager.store.set_enabled(name, enabled)
            except (MCPConfigError, KeyError) as exc:
                render_error(f"无法更新 MCP 配置: {exc}")
                return
            await self.mcp_manager.reload()
            await self._rebuild_agent()
            render_success(f"已{'启用' if enabled else '停用'} MCP 服务 {name}")
            return

        if action == "remove":
            state = states[name]
            if not await confirm(f"确定删除 MCP 服务 {name}？", default=False):
                return
            await self.mcp_manager.disconnect(name)
            try:
                self.mcp_manager.store.remove_server(name, state.config.source)
            except MCPConfigError as exc:
                await self.mcp_manager.reload()
                render_error(str(exc))
                return
            await self.mcp_manager.reload()
            await self._rebuild_agent()
            render_success(f"已删除 MCP 服务 {name}")
            return

        if action in {"test", "refresh"}:
            ok = await self.mcp_manager.refresh(name)
            await self._rebuild_agent()
            state = self.mcp_manager.states[name]
            if ok:
                render_success(f"{name} 连接正常，发现 {len(state.tools)} 个工具")
            else:
                render_error(f"{name} 连接失败: {state.last_error or '未知错误'}")
            return

        if action == "tools":
            state = states[name]
            if not state.tools:
                render_warning(f"{name} 当前没有可用工具")
                return
            table = Table(title=f"◆ {name} 工具", box=SIMPLE_HEAVY)
            table.add_column("工具", style="tool")
            table.add_column("说明")
            for tool in state.tools:
                table.add_row(tool.name, (tool.description or "").split("\n")[0])
            console.print(table)
            return

        if action == "logs":
            state = self.mcp_manager.get_state(name)
            if not state.diagnostics:
                render_info(f"{name} 暂无诊断日志")
            else:
                console.print("\n".join(f"  {line}" for line in state.diagnostics[-100:]))
            return

        render_warning(f"未知 MCP 子命令。用法: {usage}")

    @staticmethod
    def _parse_mcp_pairs(value: str) -> dict[str, str]:
        result: dict[str, str] = {}
        if not value.strip():
            return result
        for item in value.split(","):
            if "=" not in item:
                raise ValueError(f"应为 KEY=VALUE: {item.strip()}")
            key, item_value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("变量名不能为空")
            result[key] = item_value.strip()
        return result

    @staticmethod
    def _split_mcp_args(value: str) -> list[str]:
        import shlex

        if os.name != "nt":
            return shlex.split(value)
        parts = shlex.split(value, posix=False)
        return [
            part[1:-1]
            if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}
            else part
            for part in parts
        ]

    async def _cmd_quit(self, _arg: str) -> None:
        render_warning("Bye!")
        raise EOFError()
    
    async def _cmd_help(self, _arg: str) -> None:
        from rich.table import Table
        table = Table(
            title="✦ 命令列表",
            box=SIMPLE_HEAVY,
            border_style="muted",
            header_style="bold #60a5fa",
            row_styles=["", "dim"],
        )
        table.add_column("命令", style="brand", no_wrap=True)
        table.add_column("说明", style="white")
        for cmd, desc in SLASH_COMMANDS.items():
            table.add_row(cmd, desc)
        console.print(table)

    # 模型配置
    async def _cmd_model(self, arg: str) -> None:
        from lorcy_code.config.models import (
            configure_new_model,
            edit_current_model,
            switch_model,
        )
        from lorcy_code.cli.prompts import select
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
            from lorcy_code.agents.builder import update_summarization_model
            # 同步更新摘要模型
            update_summarization_model(config)

    async def _cmd_workdir(self, _arg: str) -> None:
        from lorcy_code.config.storage import load_workplace, save_workplace
        from lorcy_code.cli.prompts import select_or_custom
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

        if self.mcp_manager is not None:
            await self.mcp_manager.close()
        self.workplace_path = new_path
        os.chdir(self.workplace_path)
        save_workplace(self.workplace_path)

        # 重建子目录
        ensure_chat_config_dir(self.workplace_path)
        self._skill_loader = self._create_skill_loader()

        from lorcy_code.mcp import MCPManager
        self.mcp_manager = MCPManager(
            self.workplace_path,
            trust_callback=self._confirm_mcp_trust,
        )
        await self.mcp_manager.start()

        # 关闭旧 checkpointer 连接，重建会话和 agent
        await self._rebuild_agent(rebuild_session=True)

        await self._init_git()
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
                    else:
                        await self._cmd_history(_arg) # 删除后回到历史会话列表
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
        from lorcy_code.cli.display import render_conversation
        if not self.agent:
            return
        try:
            state = await self.agent.aget_state(self.session_mgr.config)
            messages = state.values.get("messages", [])
            render_conversation(messages)
        except Exception as e:
            render_error(f"加载对话失败: {e}")

    async def _cmd_skill(self, _arg: str) -> None:
        if not self.session_mgr:
            render_error("请先初始化工作目录")
            return
        from lorcy_code.skills.manager import (
            manage_skills,
            list_workspace_skills,
            render_workspace_skill_table,
            choose_workspace_skills,
            save_workspace_skill_selection,
            format_skill_selection_status,
        )

        arg = _arg.strip()
        loader = self._ensure_skill_loader()

        async def _apply_selection(mode: str, names: list[str]) -> tuple[dict, list[str], int]:
            selection, invalid = save_workspace_skill_selection(
                self.session_mgr,
                mode,
                names,
            )
            loader.set_skill_selection(selection["mode"], selection["skills"])
            installed_count = len(list_workspace_skills(self.session_mgr))
            return selection, invalid, installed_count

        if not arg:
            action = await select(
                "Skill 设置:",
                [
                    "选择当前工作区启用的 skills",
                    "查看 skills 状态",
                    "恢复全部 skills",
                    "清空已启用 skills",
                    "技能管理",
                    "返回",
                ],
            )
            if action is None or action == "返回":
                return
            if action == "选择当前工作区启用的 skills":
                result = await choose_workspace_skills(self.session_mgr)
                if result is None:
                    return
                selection, invalid = result
                loader.set_skill_selection(selection["mode"], selection["skills"])
                installed_count = len(list_workspace_skills(self.session_mgr))
                render_success(
                    format_skill_selection_status(selection, installed_count)
                )
                if invalid:
                    render_warning(f"已忽略不存在的 skill: {', '.join(invalid)}")
                return
            if action == "查看 skills 状态":
                render_workspace_skill_table(self.session_mgr)
                return
            if action == "恢复全部 skills":
                selection, invalid, installed_count = await _apply_selection("all", [])
                render_success(format_skill_selection_status(selection, installed_count))
                if invalid:
                    render_warning(f"已忽略不存在的 skill: {', '.join(invalid)}")
                return
            if action == "清空已启用 skills":
                selection, invalid, installed_count = await _apply_selection("selected", [])
                render_success(format_skill_selection_status(selection, installed_count))
                if invalid:
                    render_warning(f"已忽略不存在的 skill: {', '.join(invalid)}")
                return
            if action == "技能管理":
                await manage_skills(self.session_mgr)
                loader.scan_all_skills(force=True)
                return

        parts = arg.split()
        subcmd = parts[0].lower()
        names = parts[1:]

        if subcmd == "list":
            render_workspace_skill_table(self.session_mgr)
            return
        if subcmd == "use":
            selection, invalid, installed_count = await _apply_selection("selected", names)
            render_success(format_skill_selection_status(selection, installed_count))
            if invalid:
                render_warning(f"已忽略不存在的 skill: {', '.join(invalid)}")
            return
        if subcmd == "all":
            selection, invalid, installed_count = await _apply_selection("all", [])
            render_success(format_skill_selection_status(selection, installed_count))
            if invalid:
                render_warning(f"已忽略不存在的 skill: {', '.join(invalid)}")
            return
        if subcmd == "clear":
            selection, invalid, installed_count = await _apply_selection("selected", [])
            render_success(format_skill_selection_status(selection, installed_count))
            if invalid:
                render_warning(f"已忽略不存在的 skill: {', '.join(invalid)}")
            return
        if subcmd == "manage":
            await manage_skills(self.session_mgr)
            loader.scan_all_skills(force=True)
            return

        render_warning("未知 /skill 子命令，可用: list, use, all, clear, manage")

    async def _cmd_git(self, _arg: str) -> None:
        if not self.git_manager:
            from lorcy_code.integrations.git import check_git_availability
            is_available, status, version = await asyncio.to_thread(
                check_git_availability
            )
            if is_available:
                render_success(f"Git {version}")
                await self._init_git()
            else:
                render_error(f"Git 不可用: {status}")
                return

        if self.git_manager.is_repo():
            count = self.git_manager.count_checkpoints()
            self._git_cp_count = count
            render_success(f"Git 仓库已初始化 ({count} 个检查点)")
        else:
            render_warning("Git 仓库未初始化")

    # ─── 消息管理命令 ──────────────────────────────────

    async def _cmd_messages(self, _arg: str) -> None:
        """管理历史消息：编辑、分叉、删除"""
        from lorcy_code.config.storage import load_workplace, save_workplace
        from lorcy_code.shared.messages import (
            _group_messages_by_turn,
            _get_group_display,
            _collect_ids_from_group,
        )

        if not self.agent or not self.session_mgr:
            render_error("Agent 未初始化")
            return

        state = await self.agent.aget_state(self.session_mgr.config)
        messages: list[BaseMessage] = state.values.get("messages", [])

        groups = _group_messages_by_turn(messages)
        if not groups:
            render_warning("没有可管理的消息")
            return

        while True:
            # 第一步：选择操作类型
            action = await select("选择操作:", ["编辑消息", "分叉消息", "删除消息","返回"])
            if not action or action == "返回":
                return

            # 构建选项列表（带返回选项）
            options = []
            for idx, group in enumerate(groups):
                display = _get_group_display(group)
                options.append(f"[{idx + 1}] {display}")

            if action == "删除消息":
                # 多选
                chosen_list = await checkbox(
                    "选择要删除的消息组（空格选择，回车确认）:", options
                )
                if not chosen_list:
                    continue  # 返回操作选择

                ok = await confirm(
                    f"确定删除 {len(chosen_list)} 个消息组？", default=False
                )
                if not ok:
                    continue

                delete_ids = []
                for chosen in chosen_list:
                    try:
                        sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
                        if 0 <= sel_idx < len(groups):
                            delete_ids.extend([m.id for m in groups[sel_idx]])
                    except (ValueError, IndexError):
                        continue

                if not delete_ids:
                    render_error("没有有效的选择")
                    continue

                await self._delete_messages(delete_ids)
                render_success(f"已删除 {len(chosen_list)} 个消息组")
                return

            # 编辑 / 分叉：单选一条消息组
            if action == "编辑消息":
                hint = "选择要编辑的消息组（编辑后将删除此消息组之后的所有内容）:"
            else:
                hint = "选择 Fork 点（此消息组将保留在分支中）:"

            select_options = options + ["返回"]
            chosen = await select(hint, select_options)
            if not chosen:
                return
            if chosen == "返回":
                continue

            # 解析选择
            try:
                sel_idx = int(chosen.split("]")[0].replace("[", "")) - 1
                if sel_idx < 0 or sel_idx >= len(groups):
                    render_error("无效的选择")
                    continue
            except (ValueError, IndexError):
                render_error("无效的选择")
                continue

            if action == "编辑消息":
                target_group = groups[sel_idx]
                edit_msg = None
                for msg in target_group:
                    if msg.type == "human":
                        edit_msg = msg
                        break

                if not edit_msg:
                    render_warning("该组没有 HumanMessage")
                    continue

                ok = await confirm(
                    "确定编辑此消息组？编辑后将删除此消息组之后的所有内容。",
                    default=False,
                )
                if not ok:
                    continue

                no_need_ids, all_ids = _collect_ids_from_group(
                    sel_idx, groups
                )

                if self.git and self.git_manager:
                    try:
                        await asyncio.to_thread(
                            self.git_manager.rollback, no_need_ids, all_ids
                        )
                    except Exception as e:
                        render_warning(f"Git 回滚失败: {e}")

                await self._delete_messages(no_need_ids)

                self._edit_buffer = get_text_content(edit_msg.content)
                render_success("消息已加载到输入框，修改后发送即可重新生成")
                return

            elif action == "分叉消息":
                ok = await confirm(
                    f"确定从第 {sel_idx + 1} 条消息组创建分支？", default=True
                )
                if not ok:
                    continue

                no_need_ids, all_ids = _collect_ids_from_group(
                    sel_idx, groups
                )

                saved = load_workplace()
                if saved:
                    choices = [str(saved), "自定义路径..."]
                else:
                    choices = ["自定义路径..."]

                new_path_str = await select_or_custom("选择新工作目录:", choices)
                if not new_path_str:
                    continue

                new_path = Path(new_path_str)
                if not new_path.exists():
                    render_error("路径不存在")
                    continue

                old_path = self.workplace_path

                self.workplace_path = new_path
                os.chdir(self.workplace_path)
                save_workplace(self.workplace_path)

                from lorcy_code.config.storage import ensure_chat_config_dir
                ensure_chat_config_dir(self.workplace_path)

                if old_path != new_path:
                    render_info("复制工作目录文件...")
                    try:
                        await asyncio.to_thread(self._copy_dir, old_path, new_path)
                        # 复制 .git 目录以保留检查点数据
                        old_git = old_path / ".git"
                        new_git = new_path / ".git"
                        if old_git.exists() and old_git.is_dir():
                            await asyncio.to_thread(
                                shutil.copytree, old_git, new_git, dirs_exist_ok=True
                            )
                        sessions_path = self.workplace_path / ".lorcy" / "sessions"
                        if sessions_path.exists():
                            await asyncio.to_thread(shutil.rmtree, sessions_path)
                            sessions_path.mkdir(exist_ok=True)
                    except Exception:
                        import traceback

                        tb = traceback.format_exc()
                        render_error(f"复制文件失败:\n{tb}")
                        self.workplace_path = old_path
                        os.chdir(self.workplace_path)
                        return

                await self._rebuild_agent(rebuild_session=True)

                need_messages = []
                for i, group in enumerate(groups):
                    need_messages.extend(group)
                    if i == sel_idx:
                        break

                await self.agent.aupdate_state(
                    self.session_mgr.config,
                    {"messages": need_messages},
                )

                # 先初始化 git
                await self._init_git()

                # 回滚工作目录
                if self.git and self.git_manager:
                    try:
                        await asyncio.to_thread(
                            self.git_manager.rollback, no_need_ids, all_ids
                        )
                    except Exception as e:
                        render_warning(f"Git 回滚失败: {e}")

                render_success(f"分支已创建！工作目录: {self.workplace_path}")
                await self._load_conversation()
                return
            
    async def _delete_messages(self, message_ids: list[str]) -> None:
        """删除指定消息"""
        if not self.agent or not self.session_mgr:
            return

        # 使用 RemoveMessage 删除
        remove_messages = [RemoveMessage(id=mid) for mid in message_ids]
        await self.agent.aupdate_state(
            self.session_mgr.config,
            {"messages": remove_messages},
        )

    # ---------------------------------------------------------------------------------
    # Yolo 模式切换命令 - 允许用户在 Common 模式（需要审批）和 Yolo 模式（自动批准）之间切换
    # ---------------------------------------------------------------------------------

    async def _cmd_mode(self, _arg: str) -> None:
        action = await select(
            "选择模式:",
            ["Common (手动批准风险操作)", "Yolo (自动批准所有操作)"],
        )
        if action is None:
            return
        self.yolo = "Yolo" in action
        from lorcy_code.agents.middleware import update_hitl_config

        update_hitl_config(self.yolo, self._effective_tools())
        mode_str = "Yolo" if self.yolo else "Common"
        render_success(f"已切换到 {mode_str} 模式")
