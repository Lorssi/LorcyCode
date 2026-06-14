import json
import os
import traceback
import asyncio
from typing import Any
from pathlib import Path
from rich.panel import Panel
from pathlib import Path
from rich.console import Console
from lorcy_code.cli.ui.prompts import select, confirm, text
from lorcy_code.core.environment.build_env import (
    CONFIG_DIR,
    MODEL_JSON,
    ENV_TO_CONFIG,
    ensure_chat_config_dir,
    ensure_home_config_dir,
    _merge_and_save_config,
    load_model_json,
    save_model_json,
)

from lorcy_code.cli.ui.prompts import (
    model_config_form,
)

console = Console()

def detect_env_api_keys() -> list[dict]:
    """检测环境变量中的 API Key，返回推荐配置列表"""
    results = []
    for var, cfg in ENV_TO_CONFIG.items():
        key = os.getenv(var, "")
        if key:
            results.append({"env_var": var, "api_key": key, **cfg})
    return results

async def first_run_configure() -> dict | None:
    """首次运行配置引导"""
    console.print()
    console.print(
        Panel(
            "[bold]LorcyCode[/bold] — 终端 AI 编程助手\n\n"
            "首次运行需要配置 AI 模型连接。\n"
            "设置环境变量后可自动检测（推荐），或手动填写配置。",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    detected = detect_env_api_keys()

    if detected:
        choices = [f"{d['name']} (检测到 {d['env_var']})" for d in detected]
        choices.append("手动配置...")
        choices.append("退出")

        result = await select("选择配置方式:", choices)
        if result is None or "退出" in result:
            console.print(
                "[dim]设置环境变量后重新运行.[/dim]"
            )
            return None

        if "手动" in result:
            return await configure_new_model()

        idx = choices.index(result)
        chosen = detected[idx]

        model_list = chosen["models"]
        model = await select("选择模型:", model_list)
        if model is None:
            return None

        config: dict[str, Any] = {
            "model": model,
            "base_url": chosen["base_url"],
            "api_key": chosen["api_key"],
            "stream_usage": True,
        }

        if not await _test_connection(config, brief=True):
            return None

        _merge_and_save_config(config)
        console.print(f"[green]配置完成: {model}[/green]")

        return config
    
    else:
        console.print("[yellow]未检测到环境变量中的 API Key[/yellow]")
        choices = ["手动配置...", "退出"]
        result = await select("选择:", choices)
        if result is None or "退出" in result:
            console.print("[dim]提示: 在环境变量中设置 API Key 后重新运行，例如:[/dim]")
            console.print("[dim]  set BIGMODEL_API_KEY=your_key  [/dim]")
            console.print("[dim]或执行 chcode config new 手动配置[/dim]")
            return None

        return await configure_new_model()
    
async def configure_new_model() -> dict | None:
    """新建模型配置（交互式表单）"""
    ensure_home_config_dir()
    result = await select("配置方式:", ["手动配置..."])
    if result is None:
        return None
    
    config = await model_config_form()
    if config is None:
        return None

    if not await _test_connection(config):
        return None

    _merge_and_save_config(config)
    console.print(f"[green]模型配置已保存: {config['model']}[/green]")

    return config

async def edit_current_model() -> dict | None:
    """编辑当前默认模型"""
    data = load_model_json()
    current = data.get("default", {})
    if not current:
        console.print("[yellow]没有当前模型配置，请新建[/yellow]")
        return await configure_new_model()

    config = await model_config_form(existing_config=current)
    if config is None:
        return None

    if not await _test_connection(config):
        return None

    data["default"] = config
    save_model_json(data)
    console.print(f"[green]模型配置已更新: {config['model']}[/green]")
    return config

async def switch_model() -> dict | None:
    """切换模型（从 fallback 列表选择）"""
    data = load_model_json()
    default = data.get("default", {})
    fallback = data.get("fallback", {})

    if not default:
        console.print("[yellow]请先配置默认模型[/yellow]")
        return await configure_new_model()

    if not fallback:
        console.print("[yellow]没有备用模型可切换[/yellow]")
        return None

    # 构建选项列表
    current_name = default.get("model", "")
    choices = []
    for name in fallback:
        tag = " (当前默认)" if name == current_name else ""
        choices.append(f"{name}{tag}")

    result = await select("选择要使用的模型:", choices)
    if result is None:
        return None

    # 提取模型名（去掉 " (当前默认)" 后缀）
    selected_name = result.replace(" (当前默认)", "")

    ok = await confirm(f"确定切换到 {selected_name}？当前默认将移至备用列表")
    if not ok:
        return None

    selected_config = fallback.pop(selected_name)
    if default and current_name not in fallback:
        fallback[current_name] = default

    data["default"] = selected_config
    data["fallback"] = fallback
    save_model_json(data)
    console.print(f"[green]已切换到: {selected_name}[/green]")
    return selected_config

async def _test_connection(
    config: dict, *, quiet: bool = False, brief: bool = False, return_error: bool = False
) -> bool | str:
    """测试模型连接，成功返回 True。

    quiet=True 时不打印任何输出（用于重试循环）。
    brief=True 时只打印简短错误（不输出 traceback）。
    return_error=True 时，失败返回错误信息字符串而非 False。
    """
    if not quiet:
        console.print("[yellow]测试连接中...[/yellow]")
    try:
        from lorcy_code.core.utils.enhanced_chat_openai import EnhancedChatOpenAI

        model = EnhancedChatOpenAI(**config)
        await asyncio.to_thread(model.invoke, "你好")
    except Exception as e:
        err_msg = str(e)
        if "null value" in err_msg and "choices" in err_msg:
            return True
        if not quiet:
            console.print(f"[red]连接测试失败: {err_msg}[/red]")
            if not brief:
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
        if return_error:
            return f"{err_msg}\n{traceback.format_exc()}"
        return False
    return True