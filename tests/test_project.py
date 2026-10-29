"""Offline regression tests; no real LLM/Tavily credentials or requests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import OpenAI
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from agent import AgentError, ResearchAgent
from api.server import app
from config import Settings
from memory import ConversationMemory, MemoryCapacityError
from rag import PaperIndex
from tools import ToolRegistry
from tools.calculator import calculator
from tools.file_tool import file_tool
from tools.python_tool import python_calculate, validate_code
from tools.web_search import web_search


def make_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): writer._add_object(font),
    })})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def tool_call(name="calculator", arguments='{"expression":"2+3"}', call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def reply(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        # Snapshot: the live message list is appended to after each request.
        self.requests.append(json.loads(json.dumps(kwargs)))
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.replies))])

    def close(self):
        pass


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.papers = self.data / "papers"
        self.papers.mkdir(parents=True)
        self.settings = Settings(data_dir=self.data, model="test-model")
        (self.data / "sample.csv").write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")


class ToolTests(WorkspaceTest):
    def test_calculator(self):
        self.assertEqual(calculator("sqrt(16) + 2**3")["result"], 12)
        for expression in ("__import__('os')", "2**10000000", "(1).__class__", "1/0"):
            with self.subTest(expression=expression), self.assertRaises((ValueError, ZeroDivisionError)):
                calculator(expression)

    def test_file_reads_and_traversal(self):
        (self.data / "a.json").write_text('{"answer":42}', encoding="utf-8")
        (self.data / "note.txt").write_text("论文笔记", encoding="utf-8")
        self.assertIn("sample.csv", file_tool(self.data)["files"])
        self.assertIn("x,y", file_tool(self.data, "read", "sample.csv")["content"])
        self.assertIn("42", file_tool(self.data, "read", "a.json")["content"])
        self.assertEqual(file_tool(self.data, "read", "note.txt")["content"], "论文笔记")
        (self.root / "outside.txt").write_text("private", encoding="utf-8")
        for path in ("../outside.txt", str(self.root / "outside.txt")):
            with self.assertRaises(ValueError):
                file_tool(self.data, "read", path)

    def test_python_statistics_and_csv(self):
        result = python_calculate("""
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
print(df['x'].mean())
print(df['x'].std())
print(df['x'].median())
print(pearsonr(df['x'], df['y']).statistic)
print(r2_score(df['y'], df['x'] * 2))
""", self.data, "sample.csv", timeout=60)
        self.assertNotIn("error", result, result)
        actual = [float(item) for item in result["stdout"].splitlines()]
        for value, expected in zip(actual, [2, 1, 2, 1, 1], strict=True):
            self.assertAlmostEqual(value, expected)

    def test_python_dataframe_creation(self):
        result = python_calculate(
            "df = pd.DataFrame({'x': [1, 2, 3]})\nprint(df['x'].median())",
            self.data, timeout=60)
        self.assertEqual(result.get("stdout"), "2.0\n", result)

    def test_python_rejects_system_and_indirect_access(self):
        attacks = [
            "import os; os.system('whoami')", "import subprocess", "import shutil", "import socket",
            "open('secret.txt')", "__import__('os')", "pd.read_csv('secret.csv')",
            "df.to_csv('out.csv')", "df.query('x > 1')", "pd.eval('1+1')",
            "np.load('secret.npy')", "np.ctypeslib", "df.__class__",
            "getattr(pd, 'read_csv')('x')", "from pandas import read_pickle",
            "import numpy as _np", "del df['x']", "df.mean = print",
            "np.fromfile('secret')", "pd.DataFrame.from_records([])",
        ]
        for code in attacks:
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_code(code)

    def test_python_timeout(self):
        result = python_calculate("for i in range(1000000000000):\n    pass", self.data, timeout=1)
        self.assertIn("超时", result.get("error", ""))

    def test_python_errors_are_structured(self):
        result = python_calculate("print(1 / 0)", self.data, timeout=60)
        self.assertIn("ZeroDivisionError", result.get("error", ""))

    def test_dispatch_validates_arguments(self):
        registry = ToolRegistry(self.settings)
        for name, arguments in [("missing", "{}"), ("calculator", "{broken"),
                                ("calculator", "[]"), ("calculator", '{"expression":2}'),
                                ("paper_search", '{"query":"x","top_k":true}'),
                                ("python_calculate", '{"code":"print(1)","timeout":500}')]:
            with self.subTest(name=name, arguments=arguments):
                self.assertIn("error", registry.execute(name, arguments))

    def test_web_search_without_key(self):
        self.assertIn("TAVILY_API_KEY", web_search("latest papers", "")["error"])

    def test_web_search_network_error(self):
        with patch("tools.web_search.httpx.Client.post", side_effect=httpx.ConnectError("offline")):
            self.assertIn("error", web_search("research", "test-placeholder"))

    def test_web_search_success(self):
        response = httpx.Response(200, request=httpx.Request("POST", "https://api.tavily.com/search"),
                                  json={"results": [{"title": "Paper", "url": "https://example.org/paper",
                                                     "content": "Evidence", "score": 0.9}]})
        with patch("tools.web_search.httpx.Client.post", return_value=response):
            result = web_search("research", "test-placeholder")
        self.assertEqual(result["results"][0]["content"], "Evidence")


class RagTests(WorkspaceTest):
    def test_discover_update_delete_and_sources(self):
        index = PaperIndex(self.papers)
        self.assertEqual(index.search("photosynthesis")["results"], [])
        pdf = self.papers / "biology.pdf"
        make_pdf(pdf, "Photosynthesis converts sunlight into chemical energy in plants.")
        result = index.search("photosynthesis sunlight", top_k=1)["results"][0]
        self.assertEqual(result["source"], "biology.pdf")
        self.assertEqual(result["page"], 1)
        self.assertIn("sunlight", result["text"])
        make_pdf(pdf, "Quantum entanglement links particle measurement outcomes across distance.")
        result = index.search("quantum entanglement")["results"][0]
        self.assertIn("Quantum", result["text"])
        pdf.unlink()
        self.assertEqual(index.search("quantum")["results"], [])

    def test_bad_and_scanned_pdfs_do_not_break_good_pdf(self):
        (self.papers / "broken.pdf").write_bytes(b"broken")
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        with (self.papers / "scan.pdf").open("wb") as stream:
            writer.write(stream)
        make_pdf(self.papers / "valid.pdf", "Statistical regression estimates relationships.")
        result = PaperIndex(self.papers).search("regression")
        self.assertEqual(result["results"][0]["source"], "valid.pdf")
        self.assertEqual(len(result["warnings"]), 2)

    def test_chunking(self):
        make_pdf(self.papers / "long.pdf", "Repeated scientific measurement improves precision. " * 20)
        index = PaperIndex(self.papers, chunk_size=120, overlap=20)
        results = index.search("precision", top_k=10)["results"]
        self.assertGreater(len(results), 1)
        self.assertTrue(all(len(item["text"]) <= 120 for item in results))


class AgentTests(WorkspaceTest):
    def test_tool_loop_multiple_calls_and_history(self):
        client = FakeClient([reply(calls=[tool_call(), tool_call(arguments='{"expression":"8/2"}', call_id="call_2")]),
                             reply("计算结果：5 和 4"), reply("记得上一轮")])
        agent = ResearchAgent(self.settings, client)
        self.assertEqual(agent.chat("请计算", "one"), "计算结果：5 和 4")
        messages = client.requests[1]["messages"]
        self.assertEqual([m["tool_call_id"] for m in messages if m["role"] == "tool"], ["call_1", "call_2"])
        self.assertEqual(json.loads(messages[-1]["content"])["result"], 4)
        agent.chat("刚才算了什么？", "one")
        self.assertIn("计算结果：5 和 4", [m.get("content") for m in client.requests[2]["messages"]])

    def test_max_tool_loops_forces_final_request(self):
        client = FakeClient([reply(calls=[tool_call()]), reply("基于已有结果回答")])
        agent = ResearchAgent(replace(self.settings, max_tool_loops=1), client)
        self.assertEqual(agent.chat("计算"), "基于已有结果回答")
        self.assertEqual(client.requests[-1]["tool_choice"], "none")
        self.assertEqual(len(client.requests), 2)

    def test_provider_ignoring_limit_does_not_execute_more_tools(self):
        client = FakeClient([reply(calls=[tool_call()]), reply(calls=[tool_call()])])
        agent = ResearchAgent(replace(self.settings, max_tool_loops=1), client)
        self.assertIn("上限", agent.chat("计算"))
        with agent.memory.session("default") as session:
            self.assertEqual(sum(m["role"] == "tool" for m in session.messages()), 1)

    def test_invalid_tool_arguments_are_returned_to_model(self):
        client = FakeClient([reply(calls=[tool_call(arguments="oops")]), reply("参数错误")])
        ResearchAgent(self.settings, client).chat("计算")
        self.assertIn("error", json.loads(client.requests[-1]["messages"][-1]["content"]))

    def test_session_isolation_and_whole_turn_trimming(self):
        memory = ConversationMemory(max_turns=1)
        with memory.session("a") as session:
            session.append([{"role": "user", "content": "old"}])
            session.append([{"role": "user", "content": "new"}, {"role": "assistant", "content": "answer"}])
            self.assertEqual(len(session.messages()), 2)
            self.assertEqual(session.messages()[0]["content"], "new")
        with memory.session("b") as session:
            self.assertEqual(session.messages(), [])

    def test_same_session_concurrent_updates(self):
        memory = ConversationMemory(max_turns=100)

        def append(index):
            with memory.session("same") as session:
                session.append([{"role": "user", "content": str(index)}])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(append, range(30)))
        with memory.session("same") as session:
            self.assertEqual(len(session.messages()), 30)

    def test_memory_capacity(self):
        memory = ConversationMemory(max_sessions=1)
        with memory.session("a"):
            pass
        with self.assertRaises(MemoryCapacityError), memory.session("b"):
            pass

    def test_missing_configuration(self):
        with self.assertRaises(AgentError):
            ResearchAgent(self.settings).chat("hello")

    def test_real_sdk_protocol_with_mock_http_transport(self):
        requests = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append(payload)
            message = {"role": "assistant", "content": "5"}
            if len(requests) == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "sdk_call", "type": "function", "function": {
                        "name": "calculator", "arguments": '{"expression":"2+3"}',
                    },
                }]}
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                                            "model": "test-model", "choices": [{"index": 0,
                                                "finish_reason": "stop", "message": message}]})

        # This is a dummy transport token, never a real API credential.
        client = OpenAI(api_key="offline-test-placeholder", base_url="https://example.invalid/v1",
                        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(client.close)
        agent = ResearchAgent(self.settings, client)
        self.assertEqual(agent.chat("2+3"), "5")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "sdk_call")


class ApiTests(WorkspaceTest):
    def test_endpoints_validation_and_missing_key(self):
        with patch("api.server.Settings.from_env", return_value=self.settings):
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                ui = client.get("/ui")
                self.assertEqual(ui.status_code, 200)
                self.assertIn("text/html", ui.headers["content-type"])
                self.assertIn("Research Agent", ui.text)
                self.assertIn("从一个问题", ui.text)
                self.assertIn("Turn a question", ui.text)
                self.assertFalse(client.get("/").json()["llm_configured"])
                status = client.get("/").json()
                self.assertGreaterEqual(status["agent_count"], 2)
                self.assertEqual({item["id"] for item in status["agents"]}, {"research", "multi"})
                self.assertIn("mcp", status)
                self.assertEqual(client.get("/agents").status_code, 200)
                self.assertEqual(client.get("/mcp").status_code, 200)
                self.assertEqual(client.post("/chat", json={"message": " "}).status_code, 422)
                self.assertEqual(client.post("/chat", json={"message": "hi", "session_id": ""}).status_code, 422)
                self.assertEqual(client.post("/chat", json={"message": "hi"}).status_code, 503)
                app.state.agent._client = FakeClient([reply("研究结论")])
                response = client.post("/chat", json={"message": "你好", "session_id": "demo"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"answer": "研究结论"})

    def test_chat_agent_alias_and_multi_mode(self):
        with patch("api.server.Settings.from_env", return_value=self.settings):
            with TestClient(app) as client:
                app.state.agent._client = FakeClient([reply("单 Agent")])
                response = client.post("/chat", json={"message": "你好", "agent": "research"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["answer"], "单 Agent")
                self.assertEqual(client.post("/chat", json={"message": "你好", "agent_id": "missing"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
