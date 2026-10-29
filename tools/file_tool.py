"""Read-only access to approved text/data files under data/."""
import csv
import io
import json
from pathlib import Path

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CHARS = 16000


def resolve_data_file(data_dir: Path, relative_path: str, suffixes: set[str]) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("path must be a non-empty relative path under data/")
    supplied = Path(relative_path)
    if supplied.is_absolute() or supplied.drive:
        raise ValueError("Absolute paths are forbidden")
    root = data_dir.resolve()
    path = (root / supplied).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes data/")
    if path.suffix.lower() not in suffixes or not path.is_file():
        raise ValueError("File does not exist or its format is unsupported")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 2 MiB limit")
    return path


def file_tool(data_dir: Path, action: str = "list", path: str = "") -> dict:
    root = data_dir.resolve()
    if action == "list":
        files = []
        truncated = False
        for item in sorted(root.rglob("*")):
            if item.is_file() and item.resolve().is_relative_to(root):
                if len(files) >= 200:
                    truncated = True
                    break
                files.append(item.relative_to(root).as_posix())
        return {"files": files, "truncated": truncated}
    if action != "read":
        raise ValueError("action must be list or read")
    target = resolve_data_file(root, path, {".txt", ".csv", ".json"})
    text = target.read_text(encoding="utf-8-sig")
    if target.suffix.lower() == ".json":
        content = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    elif target.suffix.lower() == ".csv":
        # Keep CSV as text for readability, while checking that it can be parsed.
        list(csv.reader(io.StringIO(text)))
        content = text
    else:
        content = text
    return {"source": target.relative_to(root).as_posix(),
            "content": content[:MAX_OUTPUT_CHARS], "truncated": len(content) > MAX_OUTPUT_CHARS}
