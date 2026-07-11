import json

from pathlib import Path

from lorcy_code.shared.json import atomic_write_json
from .paths import CONFIG_DIR, MODEL_JSON, SETTING_JSON, skill_selection_path, workspace_config_dir

_model_json_cache: tuple[float, dict] | None = None

ENV_TO_CONFIG: dict[str, dict[str, str | list[str]]] = {
    "BIGMODEL_API_KEY": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4.7", "glm-5","glm-5-turbo","glm-5.1"],
    },
    "OPENAI_API_KEY": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4", "gpt-5.3"],
    },
    "DEEPSEEK_API_KEY": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat"],
    },
    "DASHSCOPE_API_KEY": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.5-plus", "qwen-turbo"],
    },
    "ANTHROPIC_API_KEY": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4.6"],
    },
}

# ========== 环境初始化 ==========

# 确保.lorcy配置目录存在
def ensure_home_config_dir():
    # 确保当前目录下的.lorcy配置目录存在
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def ensure_chat_config_dir(workplace: Path | None = None):
    """确保工作目录下 .lorcy/sessions 和 .lorcy/skills 子目录存在。"""
    chat_dir = workspace_config_dir(workplace)
    (chat_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (chat_dir / "skills").mkdir(parents=True, exist_ok=True)


# ========== 模型配置管理 ==========
def get_cwd():
    workplace_path = Path.cwd()
    return workplace_path

def load_model_config():
    model_config = get_default_model_config() or {}
    return model_config

def get_default_model_config() -> dict | None:
    """获取当前默认模型配置"""
    data = load_model_json()
    return data.get("default") or None

def load_model_json() -> dict:
    """加载 model.json，带 mtime 缓存"""
    global _model_json_cache
    if not MODEL_JSON.exists():
        return {}
    try:
        mtime = MODEL_JSON.stat().st_mtime
        if _model_json_cache and _model_json_cache[0] == mtime:
            return _model_json_cache[1]
        data = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
        _model_json_cache = (mtime, data)
        return data
    except Exception:
        return {}

def save_model_json(data: dict) -> None:
    global _model_json_cache
    atomic_write_json(MODEL_JSON, data, indent=4, ensure_dir=True)
    _model_json_cache = None

def _merge_and_save_config(
    new_config: dict, fallback_updates: dict | None = None
) -> None:
    """将新配置合并到 model.json：old default → fallback，new_config → default。"""
    data = load_model_json()
    old_default = data.get("default")
    fallback = data.get("fallback", {})

    if old_default:
        old_name = old_default.get("model", "")
        if old_name and old_name not in fallback:
            fallback[old_name] = old_default
    if fallback_updates:
        fallback.update(fallback_updates)

    data["default"] = new_config
    data["fallback"] = fallback
    save_model_json(data)

# ========== 工作目录管理 ==========

def load_workplace() -> Path | None:
    """加载上次的工作目录"""
    data = _load_setting()
    wp = data.get("workplace_path", "")
    return Path(wp) if wp else None
    

def _load_setting() -> dict:
    """读取 SETTING_JSON，失败返回空 dict。"""
    if SETTING_JSON.exists():
        try:
            return json.loads(SETTING_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _update_setting(**kwargs) -> None:
    """更新 SETTING_JSON 中的指定字段。"""
    ensure_home_config_dir()

    data = _load_setting()
    data.update(kwargs)
    atomic_write_json(SETTING_JSON, data, indent=4, ensure_dir=True)

def save_workplace(path: Path) -> None:
    _update_setting(workplace_path=str(path))


def get_skill_selection_path(workplace: Path | None = None) -> Path:
    return skill_selection_path(workplace)


def load_skill_selection(workplace: Path | None = None) -> dict:
    path = get_skill_selection_path(workplace)
    if not path.exists():
        return {"mode": "all", "skills": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "all", "skills": []}

    mode = data.get("mode", "all")
    if mode not in {"all", "selected"}:
        mode = "all"

    skills = data.get("skills", [])
    if not isinstance(skills, list):
        skills = []

    normalized_skills = [
        str(skill).strip() for skill in skills if str(skill).strip()
    ]
    return {"mode": mode, "skills": normalized_skills}


def save_skill_selection(workplace: Path, data: dict) -> None:
    ensure_chat_config_dir(workplace)
    path = get_skill_selection_path(workplace)
    mode = data.get("mode", "all")
    if mode not in {"all", "selected"}:
        mode = "all"
    skills = data.get("skills", [])
    if not isinstance(skills, list):
        skills = []
    normalized = {
        "mode": mode,
        "skills": [str(skill).strip() for skill in skills if str(skill).strip()],
    }
    atomic_write_json(path, normalized, indent=4, ensure_dir=True)
