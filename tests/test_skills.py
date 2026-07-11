import stat
import zipfile
from pathlib import Path

from lorcy_code.skills.loader import install_skill, validate_skill_package


def test_install_valid_skill_and_replace_existing_version(tmp_path: Path):
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(
            "skill/SKILL.md",
            "---\nname: demo\ndescription: valid\n---\nnew body",
        )
    install_dir = tmp_path / "installed"
    old_dir = install_dir / "demo"
    old_dir.mkdir(parents=True)
    (old_dir / "SKILL.md").write_text("old body", encoding="utf-8")

    assert install_skill(str(archive), install_dir)
    assert "new body" in (old_dir / "SKILL.md").read_text(encoding="utf-8")


def test_install_rejects_unsafe_skill_name(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(
            "skill/SKILL.md",
            '---\nname: "../outside"\ndescription: unsafe\n---\nbody',
        )
    assert validate_skill_package(str(archive)) is None
    assert not install_skill(str(archive), tmp_path / "installed")
    assert not (tmp_path / "outside").exists()


def test_validation_rejects_zip_symlink(tmp_path: Path):
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("skill/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("skill/SKILL.md", "---\nname: demo\n---\nbody")
        package.writestr(link, "../../outside")
    assert validate_skill_package(str(archive)) is None
