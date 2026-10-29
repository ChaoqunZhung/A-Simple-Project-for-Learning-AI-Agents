"""Tool definitions and a single validated dispatch point."""
import json
import logging

from config import Settings
from rag import PaperIndex
from tools.calculator import calculator
from tools.file_tool import file_tool
from tools.python_tool import python_calculate
from tools.web_search import web_search
from tools.mcp_bridge import MCPBridge, config_from_settings

logger = logging.getLogger(__name__)


def _definition(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


TOOL_DEFINITIONS = [
    _definition("calculator", "计算简单数学表达式，支持四则运算、幂、sqrt/log/sin、pi/e。", {
        "expression": {"type": "string"},
    }, ["expression"]),
    _definition("python_calculate", (
        "执行受限 Python 科研计算，支持 pandas DataFrame、均值/标准差/中位数、Pearson、R²。"
        "已提供 np、pd、stats、metrics、linear_model、math、statistics；用 print 输出结果。"
        "可选 data_file 为 data/ 内的 CSV 相对路径，自动载入 df。禁止文件/网络/系统操作，"
        "禁止 read_csv、to_csv、eval、query、apply、getattr、私有属性；每次执行环境独立。"
    ), {"code": {"type": "string"}, "data_file": {"type": "string"}}, ["code"]),
    _definition("file_tool", "列出 data/ 文件，或读取 UTF-8 txt/csv/json。path 相对于 data/。", {
        "action": {"type": "string", "enum": ["list", "read"]},
        "path": {"type": "string"},
    }, ["action"]),
    _definition("paper_search", "自动解析并检索 data/papers/ 中的 PDF，返回带 source、page 的相关片段。", {
        "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
    }, ["query"]),
    _definition("web_search", "使用 Tavily 搜索最新互联网资料，返回标题、URL 和摘要。", {
        "query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
    }, ["query"]),
]


class ToolRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.papers = PaperIndex(settings.data_dir / "papers")
        self.mcp = MCPBridge(config_from_settings(settings))
        self.definitions = [*TOOL_DEFINITIONS, *self.mcp.definitions]
        self._schemas = {item["function"]["name"]: item["function"]["parameters"] for item in self.definitions}
        self._handlers = {
            "calculator": calculator,
            "python_calculate": lambda **args: python_calculate(
                **args, data_dir=settings.data_dir, timeout=settings.python_timeout_seconds),
            "file_tool": lambda **args: file_tool(**args, data_dir=settings.data_dir),
            "paper_search": self.papers.search,
            "web_search": lambda **args: web_search(**args, api_key=settings.tavily_api_key),
        }
        self._handlers.update({
            item["function"]["name"]: lambda _name=item["function"]["name"], **args: self.mcp.call(_name, args)
            for item in self.mcp.definitions
        })

    def close(self) -> None:
        """Stop an optional MCP stdio child process."""
        self.mcp.close()

    def execute(self, name: str, arguments: str) -> dict:
        try:
            if name not in self._handlers:
                raise ValueError("Unknown tool")
            if not isinstance(arguments, str) or len(arguments) > 20000:
                raise ValueError("Invalid or oversized tool arguments")
            args = json.loads(arguments)
            schema = self._schemas[name]
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            if set(args) - set(schema["properties"]) or set(schema["required"]) - set(args):
                raise ValueError("Unexpected or missing tool arguments")
            for key, value in args.items():
                rule = schema["properties"][key]
                if rule["type"] == "string" and not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
                if rule["type"] == "integer":
                    if type(value) is not int or not rule["minimum"] <= value <= rule["maximum"]:
                        raise ValueError(f"{key} is outside its allowed integer range")
                if "enum" in rule and value not in rule["enum"]:
                    raise ValueError(f"Invalid {key}")
            result = self._handlers[name](**args)
            logger.info("Tool completed: %s (error=%s)", name, "error" in result)
            return result
        except Exception as exc:
            logger.warning("Tool failed: %s (%s)", name, type(exc).__name__)
            if isinstance(exc, (ValueError, SyntaxError, ArithmeticError)):
                return {"error": f"{type(exc).__name__}: {str(exc)[:1000]}"}
            return {"error": f"工具执行失败（{type(exc).__name__}），请检查输入和本地依赖。"}
