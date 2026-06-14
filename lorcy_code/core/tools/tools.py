import asyncio
import json
import os
import platform
import re
import time
from pathlib import Path

from langchain.tools import tool, ToolRuntime

from lorcy_code.core.utils.agent_setup import (
    SkillAgentContext, 
)

@tool
async def bash(
    command: str,
    runtime: ToolRuntime[SkillAgentContext],
    timeout: int = 300,
    workdir: str | None = None,
) -> str:
    """
    Execute a shell command with automatic platform detection and CWD tracking.

    On Windows: uses Git Bash if available, falls back to PowerShell.
    On Linux/Mac: uses the system shell (bash/zsh).

    The working directory is tracked across commands within the same session.
    Use 'workdir' to override the working directory for a specific command
    without affecting the session's tracked CWD.

    Output is automatically truncated if it exceeds 2000 lines or 51200 bytes.
    Certain exit codes are interpreted semantically (e.g., grep exit 1 = no matches).

    Args:
        command: The shell command to execute
        timeout: Timeout in seconds (default 300, max 600)
        workdir: Working directory override (default: project root)
    """
    pass


ALL_TOOLS = [
    bash,
]