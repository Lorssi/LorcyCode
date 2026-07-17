import typer
import asyncio
from rich.console import Console
from rich.table import Table

from lorcy_code import __version__

app = typer.Typer(
    name="lorcy_code",
    help="Terminal-based AI coding agent",
    no_args_is_help=False,
)
console = Console()

async def _run_agent(*, yolo: bool = False) -> None:
    from .repl import ChatREPL
    repl = ChatREPL(yolo=yolo)

    try:
        ok = await repl.initialize()
    except Exception:
        await repl.close()
        console.print_exception()
        raise typer.Exit(1)
    
    if not ok:
        console.print("[red]初始化失败[/red]")
        raise typer.Exit(1)
    
    try:
        await repl.run()
    finally:
        await repl.close()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    yolo: bool = typer.Option(
        False, "--yolo", "-y", help="启用 Yolo 模式（自动批准所有操作）"
    ),
    version: bool = typer.Option(False, "--version", "-v", help="显示版本"),
):
    """ChCode — 终端 AI 编程助手"""
    if version:
        console.print("LorcyCode " + __version__)
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    asyncio.run(_run_agent(yolo=yolo))

@app.command()
def config(
    action: str = typer.Argument(
        "show", help="操作：show、new、edit 或 switch"
    ),
):
    """查看或管理模型配置。"""
    from lorcy_code.config.models import (
        configure_new_model,
        edit_current_model,
        switch_model,
    )
    from lorcy_code.config.storage import load_model_json
    from lorcy_code.shared.text import mask_api_key

    normalized = action.lower()
    actions = {
        "new": configure_new_model,
        "edit": edit_current_model,
        "switch": switch_model,
    }
    if normalized in actions:
        asyncio.run(actions[normalized]())
        return
    if normalized != "show":
        raise typer.BadParameter("action 必须是 show、new、edit 或 switch")

    data = load_model_json()
    current = data.get("default")
    if not current:
        console.print("[yellow]尚未配置默认模型，可运行 lorcy_code config new。[/yellow]")
        return
    table = Table(title="当前模型配置", show_header=False)
    for key in ("model", "base_url", "api_key", "temperature", "max_tokens"):
        if key in current:
            value = mask_api_key(str(current[key])) if key == "api_key" else str(current[key])
            table.add_row(key, value)
    console.print(table)

@app.command()
def gui():
    """显示 GUI 功能状态。"""
    console.print("[yellow]GUI 模式尚未实现，请使用终端交互模式。[/yellow]")

if __name__ == "__main__":
    app()
