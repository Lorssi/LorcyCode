"""
技能管理 — 扫描/列表/查看详情/安装/删除，全部用下拉列表交互
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
from rich.table import Table
from rich.text import Text

from lorcy_code.cli.display import console
from lorcy_code.cli.prompts import select, confirm, text, checkbox
from lorcy_code.config.storage import (
    load_skill_selection,
    save_skill_selection,
)
from .loader import (
    scan_all_skills,
    validate_skill_package,
    install_skill,
)

if TYPE_CHECKING:
    from lorcy_code.sessions.manager import SessionManager


async def manage_skills(session: SessionManager) -> None:
    """技能管理主菜单"""
    while True:
        action = await select(
            "技能管理:",
            ["查看已安装技能", "安装新技能", "返回"],
        )
        if action is None or action == "返回":
            return

        if action == "查看已安装技能":
            await _list_skills(session)
        elif action == "安装新技能":
            await _install_skill(session)


def get_workspace_skill_selection(session: SessionManager) -> dict:
    return load_skill_selection(session.workplace_path)


def list_workspace_skills(session: SessionManager) -> list[dict]:
    skills = scan_all_skills(session.workplace_path)
    selection = get_workspace_skill_selection(session)
    enabled = set(selection.get("skills", []))
    mode = selection.get("mode", "all")

    for skill in skills:
        skill["enabled"] = mode == "all" or skill["name"] in enabled

    return skills


def save_workspace_skill_selection(
    session: SessionManager,
    mode: str,
    skill_names: list[str],
) -> tuple[dict, list[str]]:
    installed = {skill["name"] for skill in scan_all_skills(session.workplace_path)}
    valid_names = sorted({name for name in skill_names if name in installed})
    invalid_names = sorted({name for name in skill_names if name not in installed})

    normalized = {
        "mode": "all" if mode != "selected" else "selected",
        "skills": [] if mode != "selected" else valid_names,
    }
    save_skill_selection(session.workplace_path, normalized)
    return normalized, invalid_names


def format_skill_selection_status(selection: dict, installed_count: int) -> str:
    mode = selection.get("mode", "all")
    skills = selection.get("skills", [])
    if mode == "all":
        return f"已恢复全部 skill（共 {installed_count} 个）"
    return f"已启用 {len(skills)} 个 skill"


def render_workspace_skill_table(session: SessionManager) -> list[dict]:
    skills = list_workspace_skills(session)
    if not skills:
        console.print("[yellow]没有发现已安装的技能[/yellow]")
        return skills

    enabled_count = sum(bool(skill.get("enabled")) for skill in skills)
    table = Table(
        title="[bold brand]Skills[/bold brand]  [muted]当前工作区[/muted]",
        title_style="",
        caption=(
            f"[success]● {enabled_count} 已启用[/success]  "
            f"[muted]○ {len(skills) - enabled_count} 已禁用 · 共 {len(skills)} 个[/muted]"
        ),
        caption_justify="left",
        caption_style="",
        box=box.ROUNDED,
        border_style="muted",
        header_style="bold #cbd5e1",
        row_styles=("", "on #111827"),
        padding=(0, 1),
        collapse_padding=True,
        expand=True,
    )
    table.add_column("技能", style="bold cyan", no_wrap=True, min_width=14)
    table.add_column("状态", justify="center", no_wrap=True, width=9)
    table.add_column("范围", justify="center", no_wrap=True, width=6)
    table.add_column("描述", style="white", ratio=3, overflow="ellipsis")
    table.add_column("位置", style="muted", ratio=2, overflow="ellipsis")
    for s in skills:
        enabled = bool(s.get("enabled"))
        status = Text("● 已启用" if enabled else "○ 已禁用")
        status.stylize("success" if enabled else "muted")
        scope = Text(f"{'◆' if s['type'] == '项目' else '◇'} {s['type']}")
        scope.stylize("accent" if s["type"] == "项目" else "tool")

        path = Path(s["path"])
        try:
            display_path = f"./{path.relative_to(session.workplace_path).as_posix()}"
        except ValueError:
            try:
                display_path = f"~/{path.relative_to(Path.home()).as_posix()}"
            except ValueError:
                display_path = str(path)

        table.add_row(
            Text(str(s["name"]), style="bold cyan", overflow="ellipsis"),
            status,
            scope,
            Text(str(s["description"] or "暂无描述"), overflow="ellipsis"),
            Text(display_path, style="muted", overflow="ellipsis"),
        )
    console.print(table)
    return skills


async def choose_workspace_skills(session: SessionManager) -> tuple[dict, list[str]] | None:
    skills = list_workspace_skills(session)
    if not skills:
        return None

    current_selection = get_workspace_skill_selection(session)
    current_enabled = set(current_selection.get("skills", []))
    choices = []
    for skill in skills:
        checked = (
            current_selection.get("mode", "all") == "all"
            or skill["name"] in current_enabled
        )
        choices.append(
            {
                "name": f"{skill['name']} ({skill['type']})",
                "value": skill["name"],
                "checked": checked,
            }
        )

    selected_names = await checkbox("选择当前工作区要启用的 skills:", choices)
    normalized, invalid_names = save_workspace_skill_selection(
        session,
        "selected",
        selected_names,
    )
    return normalized, invalid_names


async def _list_skills(session: SessionManager) -> None:
    """列出所有已安装技能，支持下拉选择操作"""
    skills = render_workspace_skill_table(session)
    if not skills:
        return

    # 选择操作
    names = [f"{s['name']} ({s['type']})" for s in skills]
    action = await select(
        "选择技能进行操作:",
        names + ["返回"],
    )
    if action is None or action == "返回":
        return

    # 找到选中的技能
    selected_name = action.split(" (")[0]
    skill = next((s for s in skills if s["name"] == selected_name), None)
    if not skill:
        return

    op = await select(
        f"对技能 '{skill['name']}' 的操作:",
        ["查看详情", "删除技能", "返回"],
    )
    if op == "查看详情":
        await _show_skill_detail(skill)
    elif op == "删除技能":
        await _delete_skill(skill, session)
    elif op == "返回":
        return


async def _show_skill_detail(skill: dict) -> None:
    """查看技能详情"""
    skill_md = Path(skill["path"]) / "SKILL.md"
    if not skill_md.exists():
        console.print("[red]技能文件不存在[/red]")
        return

    content = skill_md.read_text(encoding="utf-8")
    console.print(
        Panel(
            Markdown(content),
            title=f"技能: {skill['name']}",
            border_style="cyan",
            padding=(1, 2),
        )
    )


async def _delete_skill(skill: dict, session: SessionManager) -> None:
    """删除技能"""
    ok = await confirm(
        f"确定删除技能 '{skill['name']}'？此操作不可撤销！", default=False
    )
    if not ok:
        return

    import shutil

    skill_path = Path(skill["path"])
    try:
        shutil.rmtree(skill_path)
        console.print(f"[green]技能 '{skill['name']}' 已删除[/green]")
    except Exception as e:
        console.print(f"[red]删除失败: {e}[/red]")


async def _install_skill(session: SessionManager) -> None:
    """安装技能"""
    file_path = await text("输入技能压缩包路径 (.zip/.tar.gz/.tgz):")
    if not file_path:
        return

    path = Path(file_path)
    if not path.exists():
        console.print("[red]文件不存在[/red]")
        return

    # 验证
    console.print("[yellow]验证技能包...[/yellow]")
    skill_info = validate_skill_package(str(path))
    if not skill_info:
        console.print("[red]无效的技能包，必须包含 SKILL.md[/red]")
        return

    # 选择安装位置
    location = await select(
        "选择安装位置:",
        ["项目级 (当前工作目录)", "全局级 (用户目录)"],
    )
    if location is None:
        return

    if "项目级" in location:
        install_path = session.workplace_path / ".lorcy" / "skills"
    else:
        install_path = Path.home() / ".lorcy" / "skills"

    install_path.mkdir(parents=True, exist_ok=True)

    console.print("[yellow]安装中...[/yellow]")
    if install_skill(str(path), install_path):
        name = skill_info["name"]
        console.print(f"[green]技能 '{name}' 安装成功！[/green]")
    else:
        console.print("[red]安装失败[/red]")
