import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Callable

from lorcy_code.core.utils.enhanced_chat_openai import EnhancedChatOpenAI

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

@wrap_model_call
async def load_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """动态加载模型"""
    model_config = request.runtime.context.model_config
    kwargs = dict(model_config)

    return await handler(request.override(model=EnhancedChatOpenAI(**kwargs)))