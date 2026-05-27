from pathlib import Path

CONFIG_DIR = Path.home() / ".chat"

# 确保.chat配置目录存在
def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR