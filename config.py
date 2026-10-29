"""Environment configuration; paths never depend on the launch directory."""
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    base_url: str = "https://allincode.top/v1"
    api_key: str = field(default="", repr=False)
    model: str = ""
    tavily_api_key: str = field(default="", repr=False)
    max_tool_loops: int = 8
    memory_turns: int = 10
    max_sessions: int = 1000
    llm_timeout_seconds: int = 60
    python_timeout_seconds: int = 30
    mcp_server_command: str = ""
    mcp_server_args: tuple[str, ...] = ()
    mcp_server_url: str = ""
    mcp_timeout_seconds: float = 10.0
    data_dir: Path = field(default_factory=lambda: PROJECT_DIR / "data")

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_DIR / ".env", override=False)
        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", "https://allincode.top/v1").strip(),
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_MODEL", "").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            max_tool_loops=_integer("MAX_TOOL_LOOPS", 8, 1, 16),
            memory_turns=_integer("MEMORY_TURNS", 10, 1, 100),
            max_sessions=_integer("MAX_SESSIONS", 1000, 1, 10000),
            llm_timeout_seconds=_integer("LLM_TIMEOUT_SECONDS", 60, 1, 300),
            python_timeout_seconds=_integer("PYTHON_TIMEOUT_SECONDS", 30, 1, 120),
            mcp_server_command=os.getenv("MCP_SERVER_COMMAND", "").strip(),
            mcp_server_args=tuple(os.getenv("MCP_SERVER_ARGS", "").split()),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "").strip(),
            mcp_timeout_seconds=float(os.getenv("MCP_TIMEOUT_SECONDS", "10")),
        )


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    log_path = Path(os.getenv("LOG_FILE", str(PROJECT_DIR / "logs" / "agent.log"))).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    # HTTP client logs can contain provider URLs; never log headers or prompts.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
