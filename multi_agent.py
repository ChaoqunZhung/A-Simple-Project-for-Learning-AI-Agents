"""Small multi-agent orchestration layer for :mod:`agent`.

The orchestrator deliberately stays independent of LangChain/MCP.  Every
specialist delegates its actual model/tool work to the existing
``ResearchAgent`` so the normal tool validation, memory and OpenAI-compatible
client are reused.  This makes the module useful in the CLI, FastAPI and in a
future MCP server without changing the existing agent loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Iterable
import uuid

from agent import ResearchAgent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentTask:
    """A unit of work assigned to one specialist."""

    name: str
    role: str
    prompt: str
    depends_on: tuple[str, ...] = ()
    parallel: bool = True


@dataclass
class AgentResult:
    """Safe, serializable result from one specialist."""

    name: str
    role: str
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ok": self.ok}


class PlannerAgent:
    """Deterministic router.

    Routing is intentionally local and predictable.  The specialist agents
    still use the LLM and tools, while planning never consumes an extra model
    call or invents a tool result.
    """

    _PAPER_WORDS = ("论文", "文献", "paper", "pdf", "研究", "引用", "方法")
    _WEB_WORDS = ("最新", "新闻", "互联网", "网页", "web", "online", "今天", "最近")
    _DATA_WORDS = ("数据", "统计", "均值", "标准差", "相关", "回归", "r²", "r2", "分析", "csv")

    def plan(self, question: str) -> list[AgentTask]:
        q = question.lower()
        tasks: list[AgentTask] = []
        if any(word in q for word in self._PAPER_WORDS):
            tasks.append(AgentTask(
                name="local_research", role="researcher",
                prompt=("优先使用 paper_search 检索本地论文。回答问题时只依据工具返回的片段，"
                        "保留 source 和 page，并指出证据不足之处。\n用户问题：" + question),
            ))
        if any(word in q for word in self._WEB_WORDS):
            tasks.append(AgentTask(
                name="web_research", role="researcher",
                prompt=("优先使用 web_search 获取最新互联网资料。列出实际返回的 URL，"
                        "区分网页事实和推断，并说明摘要局限。\n用户问题：" + question),
            ))
        if any(word in q for word in self._DATA_WORDS):
            tasks.append(AgentTask(
                name="data_analysis", role="analyst",
                prompt=("优先使用 file_tool 读取 data/ 内数据，再使用 python_calculate 完成统计。"
                        "报告输入文件、公式、样本限制和计算结果。\n用户问题：" + question),
            ))
        if not tasks:
            tasks.append(AgentTask(
                name="general_research", role="researcher",
                prompt=("作为科研研究员拆解问题，必要时选择 paper_search、web_search 或其它工具，"
                        "不得伪造证据。\n用户问题：" + question),
            ))
        # Analysis must see all evidence produced by independent researchers.
        research_names = tuple(task.name for task in tasks if task.role == "researcher")
        if research_names and not any(task.role == "analyst" for task in tasks):
            tasks.append(AgentTask(
                name="evidence_analysis", role="analyst",
                prompt=("检查下面研究结果中的证据质量、冲突和缺口，不要补造事实。\n"
                        "原始问题：" + question),
                depends_on=research_names, parallel=False,
            ))
        return tasks


class SpecialistAgent:
    """Adapter that gives every role an explicit system-like instruction."""

    role = "specialist"

    def __init__(self, research_agent: ResearchAgent):
        self.research_agent = research_agent

    def run(self, prompt: str, session_id: str) -> str:
        instruction = (
            f"你是多 Agent 科研系统中的 {self.role}。只报告真实工具结果和清晰推理，"
            "不伪造工具调用；如果证据不足必须明确说明。\n" + prompt
        )
        return self.research_agent.chat(instruction, session_id=session_id)


class ResearcherAgent(SpecialistAgent):
    role = "researcher（证据检索员）"


class AnalystAgent(SpecialistAgent):
    role = "analyst（数据与证据分析员）"


class SynthesizerAgent(SpecialistAgent):
    role = "synthesizer（最终综合员）"

    def synthesize(self, question: str, results: Iterable[AgentResult], session_id: str) -> str:
        evidence = "\n\n".join(
            f"[{result.name}]\n{result.output if result.ok else '任务失败：' + (result.error or '')}"
            for result in results
        )
        prompt = (
            "综合以下多 Agent 结果回答原问题。最终回答必须分为：事实与证据、计算结果（如有）、"
            "分析与局限；引用本地论文时保留 source/page，引用网页时保留 URL。"
            "不要声称执行过未返回的工具。\n\n原问题：" + question + "\n\n研究结果：\n" + evidence
        )
        return self.run(prompt, session_id)


class MultiAgentResearch:
    """Route and execute research tasks with bounded parallelism and loops."""

    def __init__(
        self,
        research_agent: ResearchAgent | None = None,
        *,
        max_workers: int = 4,
        max_loops: int = 3,
    ):
        if max_workers < 1 or max_workers > 16:
            raise ValueError("max_workers must be between 1 and 16")
        if max_loops < 1 or max_loops > 8:
            raise ValueError("max_loops must be between 1 and 8")
        self.agent = research_agent or ResearchAgent()
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent(self.agent)
        self.analyst = AnalystAgent(self.agent)
        self.synthesizer = SynthesizerAgent(self.agent)
        self.max_workers = max_workers
        self.max_loops = max_loops

    def _execute(self, task: AgentTask, question_session: str, context: str = "") -> AgentResult:
        prompt = task.prompt
        if context:
            prompt += "\n\n依赖任务结果（仅作为待核查证据）：\n" + context
        specialist: SpecialistAgent = self.analyst if task.role == "analyst" else self.researcher
        try:
            output = specialist.run(prompt, session_id=f"{question_session}:{task.name}")
            return AgentResult(task.name, task.role, output, metadata={"parallel": task.parallel})
        except Exception as exc:  # specialists are isolated; synthesis can continue
            logger.exception("Multi-agent task failed: %s", task.name)
            return AgentResult(task.name, task.role, error=f"{type(exc).__name__}: {str(exc)[:500]}")

    def run(self, question: str, session_id: str = "default") -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be blank")
        if len(question) > 16000:
            raise ValueError("question is too long")
        run_id = f"multi-{session_id}-{uuid.uuid4().hex[:8]}"
        tasks = self.planner.plan(question.strip())
        results: dict[str, AgentResult] = {}
        loops = 0

        # Execute dependency-free tasks in parallel.  Dependent tasks are
        # released in later rounds, bounded by max_loops to avoid deadlocks.
        while len(results) < len(tasks) and loops < self.max_loops:
            loops += 1
            ready = [
                task for task in tasks
                if task.name not in results and all(dep in results for dep in task.depends_on)
            ]
            if not ready:
                logger.error("No runnable tasks remain; dependency cycle suspected")
                break
            parallel = [task for task in ready if task.parallel]
            serial = [task for task in ready if not task.parallel]
            if parallel:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(parallel))) as pool:
                    futures = {pool.submit(self._execute, task, run_id): task for task in parallel}
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            results[task.name] = future.result()
                        except Exception as exc:
                            results[task.name] = AgentResult(task.name, task.role, error=str(exc)[:500])
            for task in serial:
                context = "\n".join(
                    f"[{dep}] {results[dep].output if results[dep].ok else results[dep].error}"
                    for dep in task.depends_on
                )
                results[task.name] = self._execute(task, run_id, context)

        ordered = [results[task.name] for task in tasks if task.name in results]
        missing = [task.name for task in tasks if task.name not in results]
        try:
            answer = self.synthesizer.synthesize(question.strip(), ordered, f"{run_id}:synthesis")
        except Exception as exc:
            logger.exception("Synthesis failed")
            answer = "无法完成多 Agent 综合：" + f"{type(exc).__name__}: {str(exc)[:500]}"
        return {
            "answer": answer,
            "plan": [asdict(task) for task in tasks],
            "results": [result.as_dict() for result in ordered],
            "missing_tasks": missing,
            "loops": loops,
            "run_id": run_id,
        }

    def chat(self, message: str, session_id: str = "default") -> str:
        """Drop-in convenience method matching ``ResearchAgent.chat``."""
        return self.run(message, session_id)["answer"]


# A short alias is convenient for callers that prefer ``MultiAgentAgent``.
MultiAgentAgent = MultiAgentResearch

__all__ = [
    "AgentTask", "AgentResult", "PlannerAgent", "ResearcherAgent",
    "AnalystAgent", "SynthesizerAgent", "MultiAgentResearch", "MultiAgentAgent",
]
