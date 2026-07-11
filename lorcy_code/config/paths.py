from pathlib import Path

# User-level configuration paths.
CONFIG_DIR = Path.home() / ".lorcy"
MODEL_JSON = CONFIG_DIR / "model.json"
SETTING_JSON = CONFIG_DIR / "lorcyagent.json"


def workspace_config_dir(workplace: Path | None = None) -> Path:
    """Return the active workspace's ``.lorcy`` directory."""
    return (workplace or Path.cwd()) / ".lorcy"


def skill_selection_path(workplace: Path | None = None) -> Path:
    return workspace_config_dir(workplace) / "skill_selection.json"


