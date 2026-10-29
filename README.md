# Research Agent

[简体中文](./README.md) | [English](./README_EN.md)

A Python 3.11+ research assistant built with the native OpenAI SDK Chat Completions tool-calling interface. It can search local PDFs, run scientific calculations, search the web, and synthesize evidence. No LangChain, embedding API, or manually maintained vector database is required.

The project is also a hands-on introduction to AI agents: follow a question through planning, tool selection, validation, execution, observation, and a final evidence-based answer.

## What is an agent?

An agent is more than a model that generates text. It uses a bounded loop:

1. Read the user’s question and conversation context.
2. Ask the model whether it can answer directly or needs a tool.
3. Validate any requested tool arguments before execution.
4. Return the actual tool result to the model.
5. Continue as needed, then explain the answer, evidence, and limitations.

The model interprets and plans; application code validates inputs, executes tools, enforces limits, and reports failures. A failed tool must never be presented as a successful result.

## Features

- **Single-agent CLI and API** with in-process, session-isolated conversation memory.
- **Multi-agent research** with a planner, researcher, analyst, and synthesizer.
- **Local PDF search (RAG)** with page references, automatic indexing, and no embedding service.
- **Scientific computing** on approved local CSV files using a restricted Python tool.
- **Web search** using Tavily when configured.
- **Optional MCP integration** over stdio or HTTP JSON-RPC.
- **Bilingual research UI** at `/ui`, with single-agent and multi-agent modes.
- **Offline regression tests** that do not require real LLM or Tavily credentials.

## 1. Install

Use Python 3.11 or newer. From the project directory:

### Windows PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Using the virtual-environment interpreter directly avoids activation and shell-policy issues.

## 2. Configure model access

Create a `.env` file in the project root and set:

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=<your-provider-api-key>
OPENAI_MODEL=<a-model-that-supports-tool-calling>
TAVILY_API_KEY=<optional-tavily-api-key>
```

The service must support OpenAI-compatible **Chat Completions with tools/tool_choice**. Use the model ID enabled for your account. `OPENAI_BASE_URL` is normally the service root ending in `/v1`, not a full `/chat/completions` URL.

`.env` is ignored by Git. Never commit or share API keys. Environment variables take precedence over values loaded from `.env`. Tavily is optional; without its key, web search reports that it is unavailable instead of crashing. The agent can run offline tests without real credentials.

Optional limits and defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_TOOL_LOOPS` | `8` | Maximum tool batches per turn (1–16) |
| `MEMORY_TURNS` | `10` | Recent complete turns kept per session |
| `MAX_SESSIONS` | `1000` | In-process session capacity |
| `LLM_TIMEOUT_SECONDS` | `60` | Model request timeout |
| `PYTHON_TIMEOUT_SECONDS` | `30` | Scientific Python subprocess timeout |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## 3. Start the command-line agent

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe main.py --session-id experiment-1
.\.venv\Scripts\python.exe main.py --mode multi
```

### macOS / Linux

```bash
.venv/bin/python main.py
.venv/bin/python main.py --session-id experiment-1
.venv/bin/python main.py --mode multi
```

Enter a question and press Enter. Use `/clear` to clear the current session and `/exit` to quit. Conversation memory is isolated by session ID, exists only in the current process, and is cleared when the process restarts.

Try asking:

```text
Read example.csv and calculate the mean, sample standard deviation, and median of x.
```

## 4. Start the bilingual web UI and API

From the project root, run:

```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open:

- **Research workspace:** <http://127.0.0.1:8000/ui>
- **Interactive API docs:** <http://127.0.0.1:8000/docs>
- **Health check:** <http://127.0.0.1:8000/health>

The `/ui` workspace supports Chinese and English, suggested research questions, persistent session IDs in browser storage, single-agent or multi-agent mode, and an MCP toggle. The root endpoint `/` remains a JSON status endpoint for API clients.

`POST /chat` accepts:

```json
{
  "message": "Read example.csv and calculate the mean of x",
  "session_id": "demo",
  "agent_id": "research",
  "mode": "single",
  "use_mcp": true
}
```

The response is:

```json
{"answer": "..."}
```

Set `mode` to `multi` to run the Planner/Researcher/Analyst/Synthesizer workflow. Other status routes include `GET /agents` and `GET /mcp`. A session ID is not an authentication mechanism. The API has no built-in user authentication; do not expose it publicly without adding authentication, rate limiting, and isolation.

## 5. Search local papers

Put text-based PDFs that you are allowed to use in:

```text
data/papers/
```

The first `paper_search` automatically extracts text, splits it into page-local chunks, and builds an in-memory TF-IDF index. New, changed, or deleted files trigger an index rebuild. The index is recreated after a restart.

- Results include the relative source path and page number.
- Character n-gram TF-IDF supports Chinese and English without an embedding service.
- Scanned PDFs without a text layer are not OCR-processed.
- A non-zero similarity score does not prove that a passage supports a claim.
- PDF files are excluded from Git by default. Only publish documents you have permission to share.

Ask the agent:

```text
Search the local papers for transformer attention. Explain the method and cite the source filename and page number.
```

## 6. Analyze the example data

`data/example.csv` is provided for repeatable offline calculations. Ask the agent:

```text
Use python_calculate to read example.csv. Calculate the mean, sample standard deviation,
and median of x, then explain the limitations of this small sample.
```

The `python_calculate` tool runs in a separate process, validates a restricted Python syntax, and accepts data files only from `data/`. Every call uses a fresh process, so variables do not carry over between calls.

This is a constrained scientific-computing environment, **not a security-audited operating-system sandbox**. Do not treat it as a safe environment for executing malicious code or expose it as such. Public deployments require stronger process/container isolation and resource limits.

## 7. Search the web

Configure `TAVILY_API_KEY`, restart the process, and ask:

```text
Search for recent research on scientific applications of large language models.
Include the source URLs and distinguish retrieved facts from your analysis.
```

Web results are summaries, not guaranteed full-page content or authoritative conclusions. Verify source dates and claims.

## 8. Understand the project structure

```text
research-agent/
├── main.py                 # Interactive CLI
├── agent.py                # Model requests and the single-agent tool loop
├── multi_agent.py          # Planner, specialists, and synthesis
├── config.py               # Environment configuration and logging
├── memory.py               # Session-isolated conversation memory
├── rag.py                  # Local PDF parsing and TF-IDF retrieval
├── api/
│   ├── server.py           # FastAPI routes and application lifecycle
│   └── static/index.html   # Bilingual research workspace
├── tools/
│   ├── __init__.py         # Tool schemas, validation, and dispatch
│   ├── calculator.py       # Restricted arithmetic
│   ├── python_tool.py      # Restricted scientific Python subprocess
│   ├── file_tool.py        # Reads allowed files under data/
│   ├── web_search.py       # Tavily web search
│   └── mcp_bridge.py       # Optional MCP JSON-RPC bridge
├── skills/                 # Reusable research workflow guidance
├── data/
│   ├── example.csv         # Reproducible sample data
│   └── papers/             # Local PDFs (ignored by Git)
└── tests/
    └── test_project.py     # Offline regression suite
```

The main single-agent flow is:

```text
Question → session history → model → validate and execute tool calls
         → return tool results → model continues or answers → save the turn
```

The default allows up to eight tool batches, with at most 16 calls per batch, followed by a final synthesis request with tools disabled. The loop is bounded; tool failures are returned as errors rather than treated as success.

Suggested reading order:

```text
main.py → agent.py → tools/__init__.py → individual tools
                     ├→ memory.py
                     └→ config.py
Then explore multi_agent.py and api/server.py.
```

## 9. Run offline tests

From the project root:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall .
```

### macOS / Linux

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall .
```

Tests use local fixtures and a mocked HTTP transport; they do not make real LLM or Tavily requests.

## 10. MCP and extension points

MCP is optional. Configure one of the following in `.env`, then restart the API:

```dotenv
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=-y @modelcontextprotocol/server-filesystem /path/to/safe-dir
```

Or use an HTTP JSON-RPC endpoint:

```dotenv
MCP_SERVER_URL=http://127.0.0.1:8080/mcp
```

The bridge discovers tools and exposes them through the existing validated tool registry. If MCP is not configured or is unavailable, built-in tools remain usable. Only connect trusted servers; add explicit authorization for tools that write, delete, or send data externally. Never put credentials in documentation, source code, or commits.

The `skills/` directory contains reusable guidance for literature reviews, data analysis, and web evidence. Beginners can start by inspecting `agent.py`, then trace a `tool_calls` request through `tools/__init__.py` to a tool implementation and its tests.
