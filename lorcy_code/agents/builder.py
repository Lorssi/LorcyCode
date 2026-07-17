from lorcy_code.agents.model import EnhancedChatOpenAI
from lorcy_code.agents.retry import RETRY_DELAYS, fallback_manager
from lorcy_code.agents.context import SkillAgentContext

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.agents.middleware.context_editing import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

from lorcy_code.agents.middleware import (
    load_skills,
    load_model,
    fix_messages,
    handle_tool_errors,
    model_retry_with_backoff,
    detect_parallel_agents,
    _build_interrupt_on,
    restrict_agent_type,
    tool_result_budget,
    _hitl_middleware,
    AsyncHITL,
)
from lorcy_code.tools.registry import ALL_TOOLS

def _dummy_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="placeholder", api_key="sk-placeholder", max_retries=0)

def build_agent(
    model_config: dict | None = None,
    checkpointer: AsyncSqliteSaver | None = None,
    yolo: bool = False,
    tools: list | None = None,
) -> object:
    """构建 agent 实例"""
    global _summarization_model
    cfg = model_config
    model = _dummy_model()

    effective_tools = tools if tools is not None else ALL_TOOLS
    _hitl_middleware = AsyncHITL(
        interrupt_on=_build_interrupt_on(yolo, effective_tools)
    )
    _summarization_model = EnhancedChatOpenAI(**cfg)

    # 加载 fallback 模型配置
    from lorcy_code.agents.context import get_context_window_size

    current_model = cfg.get("model", "")
    fallback_manager.load_fallback_models(current_model=current_model)

    # 摘要触发阈值 = 上下文窗口的 90%
    model_name = cfg.get("model", "")
    ctx_window = get_context_window_size(model_name)
    summary_trigger = int(ctx_window * 0.9)

    agent = create_agent(
        model=model,
        tools=effective_tools,
        middleware=[
            restrict_agent_type,
            handle_tool_errors,
            detect_parallel_agents,
            tool_result_budget,
            load_skills,
            load_model,
            model_retry_with_backoff,
            fix_messages,
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=100_000,
                        keep=3,
                        exclude_tools=["read_file"],
                        placeholder="[Old tool result content cleared]",
                    )
                ]
            ),
            SummarizationMiddleware(
                model=_summarization_model,
                trigger=("tokens", summary_trigger),
                keep=("messages", 20),
            ),
            _hitl_middleware,
        ],
        context_schema=SkillAgentContext,
        checkpointer=checkpointer,
    )
    return agent

# ---------------------------------------------------------------------------
# Summarization Model 运行时更新配置相关（无需重建 agent）
# ---------------------------------------------------------------------------

_summarization_model: EnhancedChatOpenAI | None = None

def update_summarization_model(model_config: dict) -> None:
    """运行时更新 SummarizationMiddleware 的模型"""
    if _summarization_model is not None:
        new_model = EnhancedChatOpenAI(**model_config)
        for key in new_model.model_fields_set:
            try:
                if key in new_model.__dict__:
                    setattr(_summarization_model, key, new_model.__dict__[key])
            except (AttributeError, TypeError):
                pass
