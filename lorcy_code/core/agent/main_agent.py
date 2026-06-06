
from lorcy_code.core.utils.enhanced_chat_openai import EnhancedChatOpenAI
from lorcy_code.core.utils.model_retry import RETRY_DELAYS, fallback_manager
from lorcy_code.core.utils.agent_setup import SkillAgentContext

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.agents.middleware.context_editing import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    HumanInTheLoopMiddleware,
)

from lorcy_code.core.agent.middleware import (
    load_model,
)

_summarization_model: EnhancedChatOpenAI | None = None


def _dummy_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="placeholder", api_key="sk-placeholder", max_retries=0)


def build_agent(
    model_config: dict | None = None,
    checkpointer: AsyncSqliteSaver | None = None,
) -> object:
    """构建 agent 实例"""
    global _summarization_model
    cfg = model_config
    model = _dummy_model()

    _summarization_model = EnhancedChatOpenAI(**cfg)

    # 加载 fallback 模型配置
    from lorcy_code.core.utils.agent_setup import get_context_window_size

    current_model = cfg.get("model", "")
    fallback_manager.load_fallback_models(current_model=current_model)

    # 摘要触发阈值 = 上下文窗口的 90%
    model_name = cfg.get("model", "")
    ctx_window = get_context_window_size(model_name)
    summary_trigger = int(ctx_window * 0.9)

    agent = create_agent(
        model,
        middleware=[
            load_model,
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
        ],
        context_schema=SkillAgentContext,
        checkpointer=checkpointer,
    )
    return agent