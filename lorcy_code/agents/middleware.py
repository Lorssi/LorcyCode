import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Callable

from lorcy_code.cli.display import console
from lorcy_code.agents.model import EnhancedChatOpenAI
from lorcy_code.agents.errors import ModelSwitchError
from lorcy_code.tools.result_pipeline import (
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

RETRY_DELAYS = [3, 10, 30, 60]

# ---------------------------------------------------------------------------
# 动态加载模型
# ---------------------------------------------------------------------------

@wrap_model_call
async def load_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """动态加载模型"""
    model_config = request.runtime.context.model_config
    kwargs = dict(model_config)

    return await handler(request.override(model=EnhancedChatOpenAI(**kwargs)))

# ---------------------------------------------------------------------------
# 工具调用Error处理
# ---------------------------------------------------------------------------

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
    
# ---------------------------------------------------------------------------
# 消息过滤与修正中间件
# ---------------------------------------------------------------------------
    
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

# ---------------------------------------------------------------------------
# 系统提示构建中间件
# ---------------------------------------------------------------------------

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
    
# ---------------------------------------------------------------------------
# 重试与Back Up机制
# ---------------------------------------------------------------------------

@wrap_model_call
async def model_retry_with_backoff(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """指数级退避重试中间件 — 每次调用独立计数"""
    from lorcy_code.agents.retry import fallback_manager
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

# ---------------------------------------------------------------------------
# 多Agent输出控制
# ---------------------------------------------------------------------------

@wrap_model_call
async def detect_parallel_agents(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    result = await handler(request)
    if not result.result:
        return result
    ai_msg = result.result[0]
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        from lorcy_code.cli import display as _d
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

_active_tools: list = []


def _build_interrupt_on(yolo: bool, tools: list | None = None) -> dict:
    global _active_tools
    if tools is not None:
        _active_tools = tools
    if yolo:
        return {}
    result = {
        "bash": {"allowed_decisions": ["approve", "reject"]},
        "edit": {"allowed_decisions": ["approve", "reject"]},
        "write_file": {"allowed_decisions": ["approve", "reject"]},
    }
    for tool in _active_tools:
        name = getattr(tool, "name", "")
        if not name.startswith("mcp__"):
            continue
        metadata = getattr(tool, "metadata", None) or {}
        read_only = metadata.get("readOnlyHint", metadata.get("read_only_hint", False))
        if read_only is not True:
            result[name] = {"allowed_decisions": ["approve", "reject"]}
    return result

def update_hitl_config(yolo: bool, tools: list | None = None) -> None:
    """运行时更新 HITL interrupt_on 配置，无需重建 agent"""
    global _active_tools
    if tools is not None:
        _active_tools = tools
    if _hitl_middleware is not None:
        _hitl_middleware.interrupt_on = _build_interrupt_on(yolo, _active_tools)
    from lorcy_code.tools.registry import update_agent_tool_desc
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
