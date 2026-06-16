import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Callable

from lorcy_code.cli.ui.display import console
from lorcy_code.core.utils.enhanced_chat_openai import EnhancedChatOpenAI
from lorcy_code.core.tools.tool_result_pipeline import (
    clean_tool_output,
    truncate_large_result,
    enforce_per_turn_budget,
)

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    HumanInTheLoopMiddleware,
)
from langchain.agents.middleware.context_editing import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

class ModelSwitchError(Exception):
    """标记需要切换模型的异常"""
    pass

_IPC_SOCK: socket.socket | None = None
_IPC_ADDR = ("127.0.0.1", 19876)

RETRY_DELAYS = [3, 10, 30, 60]


def _ipc_send(event: dict) -> None:
    global _IPC_SOCK
    try:
        if _IPC_SOCK is None:
            _IPC_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = json.dumps(event, ensure_ascii=False).encode("utf-8")
        _IPC_SOCK.sendto(data, _IPC_ADDR)
    except Exception:
        pass

@wrap_model_call
async def emit_thinking_events(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    _ipc_send({"type": "thinking_start", "ts": time.time()})
    try:
        result = await handler(request)
        _ipc_send({"type": "thinking_end", "ts": time.time()})
        return result
    except Exception:
        _ipc_send({"type": "thinking_end", "ts": time.time()})
        raise

@wrap_tool_call
async def emit_tool_events(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    tool_name = request.tool_call.get("name", "")
    args = request.tool_call.get("args", {})
    summary = ""
    for key in ("command", "file_path", "pattern", "query", "url", "question",
                "task", "filePath", "skill_name", "path", "prompt", "image_path"):
        if key in args:
            summary = str(args[key])[:80]
            break
    if not summary and "todos" in args:
        todos = args["todos"]
        if isinstance(todos, list) and todos:
            first = todos[0]
            if isinstance(first, dict):
                summary = first.get("content", str(first))[:80]
            else:
                summary = str(first)[:80]

    start_evt: dict = {"type": "tool_start", "tool": tool_name, "summary": summary, "ts": time.time()}
    if tool_name == "agent":
        sa_type = args.get("subagent_type", "general-purpose")
        sa_desc = args.get("description", "")[:30]
        start_evt["subagent_type"] = sa_type
        start_evt["subagent_tag"] = f"{sa_type}: {sa_desc}"
    try:
        from lorcy_code.cli.ui.display import _current_agent_tag
        tag = _current_agent_tag.get(None)
    except Exception:
        tag = None
    if tag:
        start_evt["subagent"] = tag

    _ipc_send(start_evt)
    try:
        result = await handler(request)
        ok = not (isinstance(result, ToolMessage) and getattr(result, "status", None) == "error")
        end_evt: dict = {"type": "tool_end", "tool": tool_name, "success": ok, "ts": time.time()}
        if tool_name == "agent":
            end_evt["subagent_type"] = args.get("subagent_type", "general-purpose")
            end_evt["subagent_tag"] = start_evt.get("subagent_tag", "")
        if tag:
            end_evt["subagent"] = tag
        _ipc_send(end_evt)
        return result
    except Exception:
        end_evt = {"type": "tool_end", "tool": tool_name, "success": False, "ts": time.time()}
        if tool_name == "agent":
            end_evt["subagent_type"] = args.get("subagent_type", "general-purpose")
            end_evt["subagent_tag"] = start_evt.get("subagent_tag", "")
        _ipc_send(end_evt)
        raise

@wrap_model_call
async def load_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """动态加载模型"""
    model_config = request.runtime.context.model_config
    kwargs = dict(model_config)

    return await handler(request.override(model=EnhancedChatOpenAI(**kwargs)))

@wrap_tool_call
async def handle_tool_errors(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            f"Tool error: Please check your input and try again ({e})",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    
@wrap_model_call
async def fix_messages(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """过滤隐藏消息"""
    messages = request.messages
    real_messages = [m for m in messages if not m.additional_kwargs.get("composed", "")]
    if len(real_messages) == len(messages):
        return await handler(request)
    return await handler(request.override(messages=real_messages))

@dynamic_prompt
async def load_skills(request: ModelRequest) -> str:
    """构建 system prompt — Level 1: 注入所有 Skills 元数据"""
    skill_loader = request.runtime.context.skill_loader
    os_name = sys.platform

    base_prompt = f"""You are a coding assistant. OS: {os_name}. CWD: {request.runtime.context.working_directory}.

Tools:
- bash: execute shell commands and scripts. Stop immediately if the user refuses.
- read_file: view file content; write_file: create or save files; edit: modify existing files. Always read before write, prefer edit over write_file.
- glob: find files by name pattern; grep: search file contents with regex; list_dir: browse directory structure.
- todo_write: create and manage a task list for complex multi-step work.
- load_skill: when a request matches a skill's description, load it first to get detailed instructions.

"""

    # 动态注入可用子 agent 列表
    yolo = request.runtime.context.yolo
    agents_section = "\n\nSub-agents:\n- Explore: codebase exploration and search\n- Plan: design implementation plans"
    if yolo:
        agents_section += "\n- general-purpose: full-capability tasks including reading, writing, and executing code"
    base_prompt += agents_section

    return await asyncio.to_thread(skill_loader.build_system_prompt, base_prompt)
    
@wrap_model_call
async def model_retry_with_backoff(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """指数级退避重试中间件 — 每次调用独立计数"""
    from lorcy_code.core.utils.model_retry import fallback_manager
    max_retries = 4
    retry_count = 0

    while True:
        try:
            return await handler(request)
        except Exception as e:
            retry_count += 1

            if retry_count >= max_retries:
                fallback = fallback_manager.get_or_load_fallback_model()
                if fallback:
                    console.print(f"[yellow]主模型重试{retry_count}次失败，切换到备用模型...[/yellow]")
                    raise ModelSwitchError("切换到备用模型")
                console.print(f"[red]请求失败，无备用模型可用，放弃请求\n  {e}[/red]")
                raise

            delay_idx = min(retry_count - 1, len(RETRY_DELAYS) - 1)
            delay = RETRY_DELAYS[delay_idx]

            console.print(f"[yellow]请求失败 ({retry_count}/{max_retries}), {delay}秒后重试...\n  {e}[/yellow]")

            await asyncio.sleep(delay)

@wrap_model_call
async def detect_parallel_agents(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    result = await handler(request)
    if not result.result:
        return result
    ai_msg = result.result[0]
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        from lorcy_code.cli.ui import display as _d
        agent_count = sum(1 for tc in ai_msg.tool_calls if tc.get("name") == "agent")
        if agent_count >= 2:
            _d._subagent_parallel = True
    return result

# ---------------------------------------------------------------------------
# Human-in-the-loop 运行时更新配置相关（无需重建 agent）
# ---------------------------------------------------------------------------

class AsyncHITL(HumanInTheLoopMiddleware):
    """异步 HITL 中间件 — 审批在 chat loop 中处理"""

    async def awrap_model_call(self, request, handler):
        return await handler(request)
    
_hitl_middleware: AsyncHITL | None = None

def _build_interrupt_on(yolo: bool) -> dict:
    return (
        {}
        if yolo
        else {
            "bash": {"allowed_decisions": ["approve", "reject"]},
            "edit": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject"]},
        }
    )

def update_hitl_config(yolo: bool) -> None:
    """运行时更新 HITL interrupt_on 配置，无需重建 agent"""
    if _hitl_middleware is not None:
        _hitl_middleware.interrupt_on = _build_interrupt_on(yolo)
    from lorcy_code.core.tools.tools import update_agent_tool_desc
    update_agent_tool_desc(yolo)

@wrap_tool_call
async def restrict_agent_type(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Command]
) -> Command | ToolMessage:
    if request.tool_call.get("name") == "agent":
        args = request.tool_call.get("args", {})
        if args.get("subagent_type") == "general-purpose":
            if _hitl_middleware is not None and _hitl_middleware.interrupt_on:
                args["subagent_type"] = "Explore"
    return await handler(request)

# ---------------------------------------------------------------------------
# Tool 结果截断和预算控制中间件
# ---------------------------------------------------------------------------

@wrap_model_call
async def tool_result_budget(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """工具结果截断和 token 预算控制"""
    workplace = request.runtime.context.working_directory
    messages = list(request.messages)
    changed = False
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            if msg.additional_kwargs.get("_budget_ok"):
                continue
            cleaned = clean_tool_output(msg.content)
            truncated = truncate_large_result(
                cleaned,
                msg.name or "",
                msg.tool_call_id,
                workplace=workplace,
            )
            new_kwargs = {**msg.additional_kwargs, "_budget_ok": True}
            messages[i] = msg.model_copy(update={"content": truncated, "additional_kwargs": new_kwargs})
            changed = True
    if changed:
        messages = enforce_per_turn_budget(messages, budget=200_000, workplace=workplace)
        return await handler(request.override(messages=messages))
    return await handler(request)
