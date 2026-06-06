import aiosqlite

from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CONTEXT_WINDOW_SIZES: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000,
    "deepseek-chat": 65536,
    "deepseek-v3.2": 128000,
    "deepseek-r1-0528": 65536,
    "deepseek-v4-pro": 1048576,
    "deepseek-v4-flash": 1048576,
    "glm-5.1": 200000,
    "glm-5": 200000,
    "glm-4.7": 200000,
    "minimax-m2": 204800,
    "minimax-m2.5": 200000,
    "kimi-k2": 262144,
    "mimo-v2-flash": 262144,
    "qwen3.5-plus": 1048576,
    "qwen3.6-plus": 1048576,
    "qwen": 256000,
    "longcat-2.0-preview": 1048576,
    "longcat-flash-chat": 262144,
    "longcat-flash-thinking": 262144,
    "longcat-flash-lite": 262144,
}

_DEFAULT_CONTEXT_WINDOW = 204800

class ModelSwitchError(Exception):
    """标记需要切换模型的异常"""
    pass

@dataclass
class SkillAgentContext:
    """
    Agent 运行时上下文

    通过 ToolRuntime[SkillAgentContext] 在 tool 中访问
    """

    model_config: dict
    working_directory: Path
    thread_id: str = ""
    extra: dict = field(default_factory=dict)

def get_context_window_size(model_name: str) -> int:
    """根据模型名获取上下文窗口大小，无匹配时返回默认值"""
    if not model_name:
        return _DEFAULT_CONTEXT_WINDOW
    # 精确匹配
    if model_name in CONTEXT_WINDOW_SIZES:
        return CONTEXT_WINDOW_SIZES[model_name]
    # 前缀匹配（去掉 org/ 前缀后匹配）
    short = model_name.split("/")[-1].lower()
    if short in CONTEXT_WINDOW_SIZES:
        return CONTEXT_WINDOW_SIZES[short]
    for key, size in CONTEXT_WINDOW_SIZES.items():
        if key in model_name.lower():
            return size
    return _DEFAULT_CONTEXT_WINDOW

async def create_checkpointer(db_path: Path) -> AsyncSqliteSaver:
    """创建异步 SQLite checkpointer"""
    conn = await aiosqlite.connect(str(db_path))
    return AsyncSqliteSaver(conn)