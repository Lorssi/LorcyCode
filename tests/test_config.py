import json
from pathlib import Path

from lorcy_code.config import storage
from lorcy_code.shared.json import atomic_write_json


def test_skill_selection_defaults_and_normalization(tmp_path: Path):
    assert storage.load_skill_selection(tmp_path) == {"mode": "all", "skills": []}
    storage.save_skill_selection(tmp_path, {"mode": "selected", "skills": [" one ", ""]})
    assert storage.load_skill_selection(tmp_path) == {"mode": "selected", "skills": ["one"]}


def test_default_workspace_path_is_evaluated_at_call_time(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert storage.get_skill_selection_path() == tmp_path / ".lorcy" / "skill_selection.json"


def test_atomic_json_write_creates_nested_parents(tmp_path: Path):
    target = tmp_path / "nested" / "config.json"
    atomic_write_json(target, {"ok": True}, ensure_dir=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
