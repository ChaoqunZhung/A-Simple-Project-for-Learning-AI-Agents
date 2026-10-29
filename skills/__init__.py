"""Project-local research workflow skills."""
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent


def list_skills() -> list[str]:
    return sorted(path.parent.name for path in SKILLS_DIR.glob("*/SKILL.md"))


def read_skill(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("Invalid skill name")
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown project skill: {name}")
    return path.read_text(encoding="utf-8")


__all__ = ["list_skills", "read_skill"]
