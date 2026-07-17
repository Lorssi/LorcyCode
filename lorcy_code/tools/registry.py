import asyncio
import json
import os
import platform
import re
import time
import aiofiles
from pathlib import Path
from pydantic import BaseModel, BeforeValidator, Field
from langchain.tools import tool, ToolRuntime

from lorcy_code.agents.context import (
    SkillAgentContext, 
)
from lorcy_code.cli.display import render_tool_call, console, Text
from lorcy_code.tools.shell.session import ShellSession
from lorcy_code.tools.shell.provider import BashProvider, PowerShellProvider
from lorcy_code.tools.shell.semantics import interpret_command_result
from lorcy_code.tools.config import (
    _GREP_MAX_FILE_SIZE, 
    _GREP_EXCLUDED_DIRS, 
    _GREP_BINARY_EXT, 
    _STATUS_MARKERS,
    _TODO_STORAGE_DIR,
)


def resolve_path(file_path: str, working_directory: Path) -> Path:
    """
    解析文件路径，处理相对路径和 ~ 展开

    Args:
        file_path: 文件路径（绝对或相对，支持 ~ 表示用户主目录）
        working_directory: 工作目录

    Returns:
        解析后的绝对路径
    """
    path = Path(file_path).expanduser()  # 处理 ~ 展开
    if not path.is_absolute():
        path = working_directory / path
    return path

# ---------------------------------------------------------------------------
# bash — 执行 shell 命令，自动跟踪工作目录
# ---------------------------------------------------------------------------

_shell_sessions: dict[str, ShellSession] = {}

def _create_shell_session(workdir: str) -> ShellSession | None:
    providers = [BashProvider()]
    if platform.system() == "Windows":
        providers.append(PowerShellProvider())
    for provider in providers:
        if provider.is_available:
            session = ShellSession(provider)
            session.cwd = workdir
            return session
    return None

def _get_shell_session(workdir: str) -> ShellSession | None:
    key = workdir
    session = _shell_sessions.get(key)
    if session is not None:
        return session
    session = _create_shell_session(workdir)
    if session is not None:
        _shell_sessions[key] = session
        return session
    return None

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
    cwd = str(runtime.context.working_directory)
    render_tool_call("bash", command)

    timeout = min(timeout, 600)
    timeout_ms = timeout * 1000

    session = _get_shell_session(cwd)
    if session is None:
        return "bash:\n[FAILED] No shell available on this system"

    exec_workdir = workdir if workdir else None
    result, truncated = await asyncio.to_thread(
        session.execute, command, timeout=timeout_ms, workdir=exec_workdir
    )

    interpretation = interpret_command_result(command, result.exit_code)

    parts = []
    if result.exit_code == 0:
        parts.append(f"[OK] ({session.provider_name})")
    elif interpretation.message and not interpretation.is_error:
        parts.append(f"[OK] ({session.provider_name}) {interpretation.message}")
    else:
        parts.append(
            f"[FAILED] Exit code: {result.exit_code} ({session.provider_name})"
        )
    parts.append("")

    output = truncated.content if truncated.truncated else result.stdout
    if output and output.strip():
        parts.append(output.rstrip())

    if result.stderr and result.stderr.strip():
        if output and output.strip():
            parts.append("")
        parts.append("--- stderr ---")
        parts.append(result.stderr.rstrip())

    if result.timed_out:
        parts.append("")
        parts.append(f"Command timed out after {timeout}s")

    if not output or not output.strip():
        if not result.stderr or not result.stderr.strip():
            parts.append("(no output)")

    return "bash:\n" + "\n".join(parts)

# ---------------------------------------------------------------------------
# read_file & write_file — 读写文件内容
# ---------------------------------------------------------------------------

@tool
async def read_file(file_path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Read the contents of a file.

    Use this to:
    - Read skill documentation files
    - View script output files
    - Inspect any text file

    Args:
        file_path: Path to the file (absolute or relative to working directory)
    """
    path = resolve_path(file_path, runtime.context.working_directory)
    render_tool_call("read_file", file_path)

    if not path.exists():
        return f"read:\n[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"read:\n[FAILED] Not a file: {file_path}"

    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        lines = content.split("\n")

        numbered_lines = []
        for i, line in enumerate(lines[:2000], 1):
            numbered_lines.append(f"{i:4d}| {line}")

        if len(lines) > 2000:
            numbered_lines.append(f"... ({len(lines) - 2000} more lines)")

        result = "\n".join(numbered_lines)
        if len(lines) > 2000:
            return f"read:\n[OK] ({len(lines)} lines, showing first 2000)\n\n{result}"
        return f"read:\n[OK]\n\n{result}"

    except UnicodeDecodeError:
        return f"read:\n[FAILED] Cannot read file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"read:\n[FAILED] Failed to read file: {str(e)}"
    
@tool
async def write_file(
    file_path: str, content: str, runtime: ToolRuntime[SkillAgentContext]
) -> str:
    """
    Write content to a file.

    Use this to:
    - Save generated content
    - Create new files
    - Modify existing files

    Args:
        file_path: Path to the file (absolute or relative to working directory)
        content: Content to write to the file
    """
    path = resolve_path(file_path, runtime.context.working_directory)
    render_tool_call("write_file", file_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(content)
        return f"write:\n[OK] File written: {path}"

    except Exception as e:
        return f"write:\n[FAILED] Failed to write file: {str(e)}"
    
# ---------------------------------------------------------------------------
# glob & grep — 文件搜索工具
# ---------------------------------------------------------------------------
    
@tool
async def glob(pattern: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Find files matching a glob pattern.

    Use this to:
    - Find files by name pattern (e.g., "**/*.py" for all Python files)
    - List files in a directory with wildcards
    - Discover project structure

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.md")
    """
    cwd = runtime.context.working_directory
    render_tool_call("glob", pattern)

    try:
        loop = asyncio.get_running_loop()
        matches = await loop.run_in_executor(None, lambda: sorted(cwd.glob(pattern)))

        if not matches:
            return f"glob:\n[FAILED] No files matching pattern: {pattern}"

        max_results = 100
        result_lines = []

        for path in matches[:max_results]:
            try:
                rel_path = path.relative_to(cwd)
                result_lines.append(str(rel_path))
            except ValueError:
                result_lines.append(str(path))

        result = "\n".join(result_lines)

        if len(matches) > max_results:
            result += f"\n... and {len(matches) - max_results} more files"

        return f"glob:\n[OK] ({len(matches)} matches)\n\n{result}"

    except Exception as e:
        return f"glob:\n[FAILED] {str(e)}"
    
@tool
async def grep(pattern: str, path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Search for a pattern in files.

    Use this to:
    - Find code containing specific text or regex
    - Search for function/class definitions
    - Locate usages of variables or imports

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (use "." for current directory)
    """
    cwd = runtime.context.working_directory
    render_tool_call("grep", pattern)
    search_path = resolve_path(path, cwd)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"grep:\n[FAILED] Invalid regex pattern: {e}"

    max_results = 50

    def _sync_grep() -> tuple[list[str], int]:
        results = []
        files_searched = 0

        def _search_file(file_path: Path):
            nonlocal files_searched
            try:
                size = file_path.stat().st_size
                if size > _GREP_MAX_FILE_SIZE:
                    return
            except OSError:
                return

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                lines = content.split("\n")
                files_searched += 1

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        try:
                            rel_path = file_path.relative_to(cwd)
                        except ValueError:
                            rel_path = file_path
                        results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")

                        if len(results) >= max_results:
                            return
            except (PermissionError, IsADirectoryError):
                pass

        try:
            if search_path.is_file():
                _search_file(search_path)
            else:
                for p in search_path.rglob("*"):
                    if len(results) >= max_results:
                        break
                    if not p.is_file():
                        continue
                    parts = p.parts
                    if any(
                        part.startswith(".") or part in _GREP_EXCLUDED_DIRS
                        for part in parts
                    ):
                        continue
                    if p.suffix.lower() in _GREP_BINARY_EXT:
                        continue
                    _search_file(p)
        except Exception:
            pass

        return results, files_searched

    loop = asyncio.get_running_loop()
    results, files_searched = await loop.run_in_executor(None, _sync_grep)

    if not results:
        return f"grep:\n[FAILED] No matches found for pattern: {pattern} (searched {files_searched} files)"

    output = "\n".join(results)
    if len(results) >= max_results:
        output += f"\n... (truncated, showing first {max_results} matches)"

    return f"grep:\n[OK] ({len(results)} matches in {files_searched} files)\n\n{output}"

# ---------------------------------------------------------------------------
# edit — 编辑文件内容
# ---------------------------------------------------------------------------

@tool
async def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime[SkillAgentContext],
) -> str:
    """
    Edit a file by replacing text.

    Use this to:
    - Modify existing code
    - Fix bugs by replacing incorrect code
    - Update configuration values

    The old_string must match exactly (including whitespace/indentation).
    For safety, the old_string must be unique in the file.

    Args:
        file_path: Path to the file to edit
        old_string: The exact text to find and replace
        new_string: The text to replace it with
    """
    path = resolve_path(file_path, runtime.context.working_directory)
    render_tool_call("edit", file_path)

    if not path.exists():
        return f"edit:\n[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"edit:\n[FAILED] Not a file: {file_path}"

    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()

        count = content.count(old_string)

        if count == 0:
            return "edit:\n[FAILED] String not found in file. Make sure the text matches exactly including whitespace."

        if count > 1:
            return f"edit:\n[FAILED] String appears {count} times in file. Please provide more context to make it unique."

        new_content = content.replace(old_string, new_string, 1)

        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(new_content)

        old_lines = len(old_string.split("\n"))
        new_lines = len(new_string.split("\n"))

        return f"edit:\n[OK] Edited {path.name}: replaced {old_lines} lines with {new_lines} lines"

    except UnicodeDecodeError:
        return f"edit:\n[FAILED] Cannot edit file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"edit:\n[FAILED] {str(e)}"
    
# ---------------------------------------------------------------------------
# list_dir — 列出目录内容
# ---------------------------------------------------------------------------
    
@tool
async def list_dir(path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    List contents of a directory.

    Use this to:
    - Explore directory structure
    - See what files exist in a folder
    - Check if files/folders exist

    Args:
        path: Directory path (use "." for current directory)
    """
    dir_path = resolve_path(path, runtime.context.working_directory)
    render_tool_call("list_dir", path)

    if not dir_path.exists():
        return f"ls:\n[FAILED] Directory not found: {path}"

    if not dir_path.is_dir():
        return f"ls:\n[FAILED] Not a directory: {path}"

    def _sync_list_dir() -> list[tuple[str, bool, int]]:
        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        result = []
        for entry in entries:
            try:
                is_dir = entry.is_dir()
                size = entry.stat().st_size if not is_dir else 0
                result.append((entry.name, is_dir, size))
            except Exception:
                result.append((entry.name, False, 0))
        return result

    loop = asyncio.get_running_loop()
    entries = await loop.run_in_executor(None, _sync_list_dir)

    result_lines = []
    for name, is_dir, size in entries[:100]:
        if is_dir:
            result_lines.append(f"{name}/")
        else:
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size // 1024}KB"
            else:
                size_str = f"{size // (1024 * 1024)}MB"
            result_lines.append(f"   {name} ({size_str})")

    if len(entries) > 100:
        result_lines.append(f"... and {len(entries) - 100} more entries")

    return f"ls:\n[OK] ({len(entries)} entries)\n\n{chr(10).join(result_lines)}"

# ---------------------------------------------------------------------------
# todo_write — 创建和管理结构化任务列表
# ---------------------------------------------------------------------------

class TodoItem(BaseModel):
    """单个任务项"""

    content: str = Field(description="Brief description of the task")
    status: str = Field(
        default="pending",
        description="Current status: pending, in_progress, completed, cancelled",
    )
    priority: str = Field(
        default="medium",
        description="Priority level: high, medium, low",
    )

@tool
async def todo_write(
    todos: list[TodoItem],
    runtime: ToolRuntime[SkillAgentContext],
) -> str:
    """\
Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

When to use:
- Complex multistep tasks (3+ steps)
- User provides multiple tasks
- After receiving new instructions
- After completing a task (mark complete, add follow-ups)

When NOT to use:
- Single, straightforward tasks
- Trivial tasks with no organizational benefit
- Purely conversational or informational requests

Task states: pending, in_progress, completed, cancelled
Priority levels: high, medium, low
Only ONE task should be in_progress at any time. Mark tasks complete immediately after finishing.

Args:
        todos: The updated todo list. Each item has content (str), status (str), and priority (str).
    """
    session_id = runtime.context.thread_id or "default"

    # Convert Pydantic models to dicts
    todo_dicts = [t.model_dump() for t in todos]

    # Persist (delete file if empty, matching opencode's delete-then-insert pattern)
    def _todo_path(session_id: str) -> str:
        return os.path.join(_TODO_STORAGE_DIR, f"ses_{session_id}.json")
    
    async def _save_todos(session_id: str, todos: list[dict]) -> None:
        os.makedirs(_TODO_STORAGE_DIR, exist_ok=True)
        now = int(time.time() * 1000)
        for i, todo in enumerate(todos):
            todo["position"] = i
            todo["time_updated"] = now
            if "time_created" not in todo:
                todo["time_created"] = now
        async with aiofiles.open(_todo_path(session_id), "w", encoding="utf-8") as f:
            await f.write(json.dumps(todos, ensure_ascii=False, indent=2))
    
    path = _todo_path(session_id)
    if os.path.isfile(path) and len(todo_dicts) == 0:
        os.remove(path)
    else:
        await _save_todos(session_id, todo_dicts)

    active = sum(1 for t in todo_dicts if t.get("status") != "completed")

    render_tool_call("todo_write", f"{active} active todos")

    lines = []
    for t in todo_dicts:
        status = t.get("status", "pending")
        content = t.get("content", "")
        priority = t.get("priority", "medium")
        marker = _STATUS_MARKERS.get(status, "[ ]")
        lines.append(f"  {marker} {content} (priority: {priority})")

    output = "\n".join(lines)
    if active > 0:
        output = f"{active} active todo(s):\n{output}"
    else:
        output = "All todos completed." if todo_dicts else "Todo list cleared."

    if todo_dicts:
        console.print(Text(f"\n  {active} active todo(s):", style="bold green"))
        for t in todo_dicts:
            status = t.get("status", "pending")
            content = t.get("content", "")
            priority = t.get("priority", "medium")
            marker = _STATUS_MARKERS.get(status, "[ ]")
            ps = {"high": "red bold", "medium": "yellow", "low": "dim"}.get(
                priority, ""
            )
            line = Text(f"    {marker} {content} ")
            line.append(f"({priority})", style=ps)
            console.print(line)
    else:
        console.print(Text("  Todo list cleared.", style="dim"))

    return output

# ---------------------------------------------------------------------------
# agent — 启动子 Agent 来执行任务
# ---------------------------------------------------------------------------

_AGENT_DESC_NORMAL = """Launch a sub-agent to perform a task autonomously.

Available sub-agent types:
- "Explore": For codebase exploration, searching code, finding files.
- "Plan": For designing implementation plans and architectural analysis.

Args:
    prompt: The task description for the sub-agent.
    subagent_type: Type of sub-agent to launch ("Explore", "Plan", or a custom agent name).
    description: Short description of what this sub-agent invocation does (for display purposes).
    timeout_seconds: Maximum seconds the sub-agent can run before being terminated. Default 300 (5 minutes). Must be greater than 300 (5 minutes) to allow sufficient execution time.
"""

_AGENT_DESC_YOLO = """Launch a sub-agent to perform a task autonomously.

Available sub-agent types:
- "Explore": For codebase exploration, searching code, finding files.
- "Plan": For designing implementation plans and architectural analysis.
- "general-purpose": For full-capability tasks including reading, writing, and executing code.

Args:
    prompt: The task description for the sub-agent.
    subagent_type: Type of sub-agent to launch ("Explore", "Plan", "general-purpose", or a custom agent name).
    description: Short description of what this sub-agent invocation does (for display purposes).
    timeout_seconds: Maximum seconds the sub-agent can run before being terminated. Default 300 (5 minutes). Must be greater than 300 (5 minutes) to allow sufficient execution time.
"""


def update_agent_tool_desc(yolo: bool) -> None:
    agent.__doc__ = _AGENT_DESC_YOLO if yolo else _AGENT_DESC_NORMAL

@tool
async def agent(
    prompt: str,
    subagent_type: str = "Explore",
    description: str = "",
    timeout_seconds: int = 300,
    runtime: ToolRuntime[SkillAgentContext] = None,
) -> str:
    """
    Launch a sub-agent to perform a task autonomously.

    Available sub-agent types:
    - "Explore": For codebase exploration, searching code, finding files.
    - "Plan": For designing implementation plans and architectural analysis.

    Args:
        prompt: The task description for the sub-agent.
        subagent_type: Type of sub-agent to launch ("Explore", "Plan", or a custom agent name).
        description: Short description of what this sub-agent invocation does (for display purposes).
        timeout_seconds: Maximum seconds the sub-agent can run before being terminated. Default 300 (5 minutes). Must be greater than 300 (5 minutes) to allow sufficient execution time.
    """
    import lorcy_code.cli.display as _display
    from lorcy_code.agents.loader import load_agents
    from lorcy_code.agents.subagents import run_subagent

    tag = f"{subagent_type}: {(description or '')[:30]}"

    all_agents = load_agents()
    agent_def = all_agents.get(subagent_type)

    if agent_def is None:
        available = ", ".join(sorted(all_agents.keys()))
        return f"Unknown agent type '{subagent_type}'. Available types: {available}"

    model_config = runtime.context.model_config
    working_directory = runtime.context.working_directory
    skill_loader = runtime.context.skill_loader

    with _display._subagent_count_lock:
        _display._subagent_count += 1
        if _display._subagent_count >= 2:
            _display._subagent_parallel = True
            _display.console.quiet = True
            _display._start_progress()
            if _display._progress_task is None or _display._progress_task.done():
                _display._progress_task = asyncio.ensure_future(
                    _display._progress_updater()
                )

    _display._current_agent_tag.set(tag)
    with _display._agent_progress_lock:
        _display._agent_progress[tag] = {
            "failed": False,
            "calls": 0,
        }

    # 让所有并行 agent 先完成同步初始化，再判断是否打印 [agent] 行
    await asyncio.sleep(0)
    if not _display._subagent_parallel and _display._subagent_count == 1:
        _display.console.print(
            Text(f"\n[agent] {subagent_type}: {description or prompt[:60]}", style="bold cyan")
        )

    try:
        result, is_error = await run_subagent(
            prompt=prompt,
            agent_def=agent_def,
            model_config=model_config,
            working_directory=working_directory,
            skill_loader=skill_loader,
            timeout_seconds=timeout_seconds,
            description=description,
            extra_tools=runtime.context.extra.get("mcp_tools", []),
        )

        with _display._agent_progress_lock:
            if tag in _display._agent_progress:
                if is_error:
                    _display._agent_progress[tag]["failed"] = True
                else:
                    _display._agent_progress[tag]["done"] = True
        _display._update_progress()

    finally:
        _display._current_agent_tag.set(None)
        with _display._subagent_count_lock:
            _display._subagent_count -= 1
            if _display._subagent_count == 0:
                was_parallel = _display._subagent_parallel
                _display._subagent_parallel = False
                _display.console.quiet = False
                if was_parallel:
                    _display._finalize_progress()
                else:
                    _display._start_result_spinner()

    return result

@tool
async def load_skill(skill_name: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Load a skill's detailed instructions.

    This tool reads the SKILL.md file for the specified skill and returns
    its complete instructions. Use this when the user's request matches
    a skill's description from the available skills list.

    The skill's instructions will guide you on how to complete the task,
    which may include running scripts via the bash tool.

    Args:
        skill_name: Name of the skill to load (e.g., 'news-extractor')
    """
    loader = runtime.context.skill_loader
    render_tool_call("load_skill", skill_name)

    # 尝试加载 skill
    skill_content = loader.load_skill(skill_name)

    if not skill_content:
        if loader.has_skill(skill_name):
            enabled = loader.get_enabled_skill_names()
            if enabled:
                return (
                    f"Skill '{skill_name}' is installed but not enabled for this workspace. "
                    f"Enabled skills: {', '.join(enabled)}"
                )
            return (
                f"Skill '{skill_name}' is installed but not enabled for this workspace. "
                "No skills are currently enabled."
            )

        skills = loader.scan_skills()
        if skills:
            available = [s.name for s in skills]
            return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
        return f"Skill '{skill_name}' not found. No skills are currently available."

    # 获取 skill 路径信息
    skill_path = skill_content.metadata.skill_path
    scripts_dir = skill_path / "scripts"

    scripts_info = (
        f"""
- **Scripts Directory**: `{scripts_dir}`

**Important**: When running scripts, use absolute paths like:
```bash
uv run {scripts_dir}/script_name.py [args]
```"""
        if scripts_dir.exists()
        else ""
    )

    path_info = (
        f"""
## Skill Path Info

- **Skill Directory**: `{skill_path}`"""
        + scripts_info
    )

    # 返回 instructions 和路径信息
    return f"""# Skill: {skill_name}

## Instructions

{skill_content.instructions}
{path_info}
"""

ALL_TOOLS = [
    bash,
    read_file,
    write_file,
    glob,
    grep,
    edit,
    list_dir,
    todo_write,
    agent,
    load_skill,
]
