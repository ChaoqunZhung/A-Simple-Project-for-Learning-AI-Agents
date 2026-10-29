"""FastAPI HTTP API for the research agent runtime.

The API keeps the original single-agent contract while also accepting an
``agent_id`` (or the shorter ``agent`` alias) and an optional ``use_mcp`` flag.
This lets a future multi-agent runtime be plugged in without changing API
clients. The adapter functions deliberately keep the first-party
``ResearchAgent`` usable on its own.
"""
from contextlib import asynccontextmanager
import inspect
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import FileResponse

from agent import AgentError, ResearchAgent
from config import Settings, configure_logging
from memory import MemoryCapacityError
from multi_agent import MultiAgentResearch

logger = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _agent_map(app: FastAPI) -> dict[str, Any]:
    registered = getattr(app.state, "agents", None)
    if isinstance(registered, dict) and registered:
        return registered
    primary = getattr(app.state, "agent", None)
    return {"research": primary} if primary is not None else {}


def _tool_names(runtime: Any) -> list[str]:
    definitions = getattr(getattr(runtime, "tools", None), "definitions", None)
    if isinstance(definitions, list):
        names: list[str] = []
        for item in definitions:
            try:
                names.append(str(item["function"]["name"]))
            except (KeyError, TypeError):
                continue
        return names
    names = getattr(runtime, "tool_names", None)
    return list(names) if isinstance(names, (list, tuple, set)) else []


def _agent_description(agent_id: str, runtime: Any) -> dict[str, Any]:
    metadata = getattr(runtime, "metadata", None)
    if callable(metadata):
        try:
            metadata = metadata()
        except Exception:
            metadata = None
    metadata = metadata if isinstance(metadata, dict) else {}
    role = metadata.get("role", getattr(runtime, "role", "科研助手"))
    name = metadata.get("name", getattr(runtime, "name", agent_id))
    tools = metadata.get("tools", _tool_names(runtime))
    if agent_id == "multi" and not metadata:
        name = "multi-agent orchestrator"
        role = "planner / researcher / analyst / synthesizer"
        tools = ["PlannerAgent", "ResearcherAgent", "AnalystAgent", "SynthesizerAgent"]
    return {
        "id": agent_id,
        "name": name,
        "role": role,
        "status": metadata.get("status", "ready"),
        "tools": tools,
    }


def _mcp_status(app: FastAPI) -> dict[str, Any]:
    runtime = getattr(app.state, "mcp", None)
    if runtime is None:
        for agent in _agent_map(app).values():
            tools = getattr(agent, "tools", None)
            runtime = (getattr(agent, "mcp", None)
                       or getattr(agent, "mcp_client", None)
                       or getattr(tools, "mcp", None))
            if runtime is not None:
                break
    servers: Any = None
    tool_count: int | None = None
    if runtime is not None:
        for attr in ("servers", "server_names", "configured_servers"):
            value = getattr(runtime, attr, None)
            if value is not None:
                servers = value() if callable(value) else value
                break
        for attr in ("tool_count", "num_tools"):
            value = getattr(runtime, attr, None)
            if value is not None:
                tool_count = value() if callable(value) else value
                break
        if tool_count is None and isinstance(getattr(runtime, "tools", None), (list, tuple, dict, set)):
            tool_count = len(runtime.tools)
    if servers is None:
        configured = os.getenv("MCP_SERVERS", "")
        servers = [item.strip() for item in configured.split(",") if item.strip()]
    if isinstance(servers, dict):
        servers = list(servers)
    elif not isinstance(servers, list):
        servers = list(servers) if isinstance(servers, (tuple, set)) else []
    result = {
        "enabled": runtime is not None or _truthy(os.getenv("MCP_ENABLED", "")),
        "servers": [str(item) for item in servers],
        "tool_count": tool_count if tool_count is not None else 0,
    }
    status = getattr(runtime, "status", None) if runtime is not None else None
    if callable(status):
        try:
            details = status()
            if isinstance(details, dict):
                result.update({key: details[key] for key in ("configured", "connected", "error")
                               if key in details})
        except Exception:
            pass
    return result


def _invoke_chat(runtime: Any, body: "ChatRequest") -> str:
    chat = getattr(runtime, "chat", None)
    if not callable(chat):
        raise AgentError("所选 Agent 没有可用的 chat 方法")
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(chat).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "agent_id" in parameters:
        kwargs["agent_id"] = body.agent_id
    if "use_mcp" in parameters:
        kwargs["use_mcp"] = body.use_mcp
    return chat(body.message, body.session_id, **kwargs)


def _close_runtimes(app: FastAPI) -> None:
    seen: set[int] = set()
    for runtime in _agent_map(app).values():
        if runtime is None or id(runtime) in seen:
            continue
        seen.add(id(runtime))
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
    mcp = getattr(app.state, "mcp", None)
    close = getattr(mcp, "close", None) if mcp is not None else None
    if callable(close) and id(mcp) not in seen:
        close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    configure_logging()
    primary = ResearchAgent(settings)
    app.state.agent = primary
    app.state.multi_agent = MultiAgentResearch(primary)
    app.state.agents = {"research": primary, "multi": app.state.multi_agent}
    yield
    _close_runtimes(app)


app = FastAPI(title="Research Agent", version="1.1.0", lifespan=lifespan)
UI_FILE = Path(__file__).parent / "static" / "index.html"


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    message: str = Field(min_length=1, max_length=16000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    agent_id: str = Field(default="research", alias="agent", min_length=1, max_length=64)
    mode: str = Field(default="single", pattern="^(single|multi)$",
                      description="single 使用主 Agent；multi 启用 Planner/Researcher/Analyst/Synthesizer")
    use_mcp: bool = Field(default=True, description="是否允许本次请求使用已注册的 MCP 工具")

    @field_validator("message", "session_id", "agent_id")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def status(request: Request) -> dict:
    settings = request.app.state.agent.settings
    agents = _agent_map(request.app)
    return {
        "name": "research-agent", "status": "running",
        "llm_configured": settings.llm_configured,
        "web_search_configured": bool(settings.tavily_api_key),
        "memory": "in-process", "max_tool_loops": settings.max_tool_loops,
        "default_agent": "research" if "research" in agents else next(iter(agents), None),
        "agent_count": len(agents),
        "agents": [_agent_description(agent_id, runtime) for agent_id, runtime in agents.items()],
        "mcp": _mcp_status(request.app),
    }


@app.get("/ui", response_class=FileResponse, include_in_schema=False)
def research_ui() -> FileResponse:
    return FileResponse(UI_FILE, media_type="text/html; charset=utf-8")


@app.get("/agents")
def agents(request: Request) -> dict:
    return {"agents": [_agent_description(agent_id, runtime)
                       for agent_id, runtime in _agent_map(request.app).items()]}


@app.get("/mcp")
def mcp_status(request: Request) -> dict:
    return _mcp_status(request.app)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    selected_id = "multi" if body.mode == "multi" else body.agent_id
    runtime = _agent_map(request.app).get(selected_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"未知 Agent: {selected_id}")
    try:
        return ChatResponse(answer=_invoke_chat(runtime, body))
    except AgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MemoryCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected chat failure (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="内部错误，请检查服务日志。") from exc
