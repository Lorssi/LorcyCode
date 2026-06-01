import typer
import asyncio
import os
import sys
from rich.console import Console

VERSION = "0.1.0"

app = typer.Typer(
    name="lorcy_code",
    help="Terminal-based AI coding agent",
    no_args_is_help=False,
)
console = Console()

async def _run_agent():
    from .cli.ui.ui import ChatREPL
    repl = ChatREPL()

    try:
        ok = await repl.initialize()
    except Exception as e:
        console.print_exception()
        raise typer.Exit(1)
    
    if not ok:
        console.print("[red]初始化失败[/red]")
        raise typer.Exit(1)
    
    try:
        await repl.run()
    finally:
        pass

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
        console.print("LorcyCode " + VERSION)
        raise typer.Exit()

    if ctx.invoked_subcommand is not None:
        return

    asyncio.run(_run_agent())

@app.command()
def config():
    """模型配置管理"""
    console.print("配置管理功能待实现")
    pass

@app.command()
def gui():
    """模型配置管理"""
    console.print("GUI 模式待实现")
    pass

if __name__ == "__main__":
    app()
