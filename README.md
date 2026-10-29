# Research Agent

一个 Python 3.11+ 科研助手：通过原生 OpenAI SDK 的 Chat Completions 工具调用接口，检索本地 PDF、执行科学计算、搜索互联网，并综合证据回答。无需 LangChain、embedding API 或手动构建向量数据库。

## 1. 安装（Windows PowerShell）

以下命令由你在终端执行。先进入项目目录：

```powershell
cd C:\Users\win10\Desktop\agent\research-agent
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 必须为 3.11 或更高版本。建议使用虚拟环境；全文直接指定虚拟环境解释器，无需激活，也无需修改 PowerShell 执行策略。

Linux / macOS 对应命令：

```bash
cd research-agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

之后将教程中的 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python` 即可。

## 2. 配置 .env

```powershell
Copy-Item .env.example .env
notepad .env
```

如果已有 `.env`，直接编辑，避免覆盖。填写以下字段，替换尖括号中的说明文字：

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=<你的模型提供方密钥>
OPENAI_MODEL=<提供方支持工具调用的模型名称>
TAVILY_API_KEY=<你的 Tavily 密钥，可留空>
```

- 可以使用任何实现 OpenAI-compatible **Chat Completions + tools/tool_choice** 的提供方。普通文本聊天接口不够。
- `OPENAI_BASE_URL` 填服务根地址，通常以 `/v1` 结尾；不要填完整的 `/chat/completions` 路径。自建兼容服务请按提供方要求填写。
- `OPENAI_MODEL` 必须填写你账号实际可用的模型 ID，不在代码中固定型号。
- 所有真实密钥仅从环境变量读取；`python-dotenv` 将项目根目录的 `.env` 加载到环境中。已有系统环境变量优先于 `.env`。
- `.env` 已加入 `.gitignore`，不要提交或分享。修改配置后重启 CLI / API。
- 没有模型密钥时 `/`、`/health` 和本地工具测试仍可使用；`/chat` 会返回明确的 503 配置提示。
- 没有 Tavily 密钥时 `web_search` 返回说明，不会让程序崩溃。

可选环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MAX_TOOL_LOOPS` | `8` | 单轮最多执行的工具批次数，范围 1–16 |
| `MEMORY_TURNS` | `10` | 每个 session 保存的完整最近问答轮数 |
| `MAX_SESSIONS` | `1000` | 进程内会话数量上限 |
| `LLM_TIMEOUT_SECONDS` | `60` | 模型单次请求超时，SDK 最多重试一次 |
| `PYTHON_TIMEOUT_SECONDS` | `30` | Python 子进程总超时（含依赖导入） |
| `LOG_LEVEL` | `INFO` | 日志等级 |

## 3. 启动 CLI

```powershell
.\.venv\Scripts\python.exe main.py
# 使用独立会话
.\.venv\Scripts\python.exe main.py --session-id experiment-1
```

输入问题并回车。`/clear` 清空当前会话历史；`/exit` 退出。每轮可以多次调用工具后再回答。

例如先问：`请读取 example.csv，计算 x 的均值、样本标准差和中位数。`

然后问：`用同一份数据计算 x 和 y 的 Pearson 相关性，说明样本量的限制。`

Memory 保存在当前进程内，按 session 隔离。同一 session 的并发请求会串行执行，防止历史交错。保存完整问答轮次及工具结果，不截断工具调用配对；程序重启后记忆消失，不同 CLI 进程不共享历史。会话达到上限时拒绝新会话，不自动丢弃旧会话。

## 4. 启动 FastAPI

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000/docs`，可直接试用接口。按 `Ctrl+C` 停止服务。
打开 `http://127.0.0.1:8000/ui` 可使用中英双语研究工作台；右上角可切换语言，支持
单 Agent / 多 Agent、会话 ID 和 MCP 开关。根路径 `/` 仍返回 JSON 服务状态。

另外打开一个 PowerShell 窗口：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
    message = '读取 example.csv 并计算 x 的均值和样本标准差'
    session_id = 'demo'
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

POST `/chat` 请求：

```json
{"message": "请总结本地论文的方法", "session_id": "default", "agent_id": "research", "mode": "single", "use_mcp": true}
```

成功响应：

```json
{"answer": "……"}
```

`session_id` 可省略，默认 `default`。不同用户应使用不同 ID；ID 本身不是认证机制。

| 接口 | 行为 |
| --- | --- |
| `GET /` | 运行状态、Agent 列表、LLM / Tavily / MCP 状态和工具循环上限；不返回密钥 |
| `GET /agents` | 列出已注册 Agent、角色和工具（默认 `research`、`multi`） |
| `GET /mcp` | 返回 MCP 启用、服务器和工具数量等状态；不返回密钥 |
| `GET /health` | 进程存活检查，返回 `{"status":"ok"}`，不测试外部服务 |
| `POST /chat` | 根据 `mode` 运行单 Agent 或多 Agent 流程，返回 `answer` |

`POST /chat` 默认使用 `research` 单 Agent。`agent` 是 `agent_id` 的兼容别名；将
`mode` 设置为 `multi` 会运行 Planner、Researcher、Analyst 和 Synthesizer，再由最终
综合 Agent 汇总。多 Agent 会复用相同的 OpenAI-compatible 配置和工具校验，并为每个
specialist 使用独立 session 分支。例如：

```powershell
$body = @{ message = '比较本地论文和最新网页中的方法'; session_id = 'demo'; mode = 'multi'; use_mcp = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json' -Body $body
```

`agent_id` 可以填写 `/agents` 返回的其它注册 ID。未知 ID 返回 404；空消息、空 ID
仍返回 422。`use_mcp=false` 将本次请求标记为不使用 MCP，扩展编排器可据此跳过外部工具。

空白问题、超长输入等返回 422；模型不可用或缺配置返回 503；会话容量达到上限返回 429。工具执行失败通常会作为结构化工具结果交给模型，由模型解释或修正输入。

默认用于本机使用，尚未加入登录认证。对外部署前请添加身份认证、限流和隔离。使用一个 Uvicorn worker；多 worker 会各自持有不同记忆，后续可改用 Redis。`--reload` 重载也会清空记忆。

## 5. 放入 PDF 与自动 RAG

将论文复制到：

```text
research-agent/data/papers/
```

支持子目录及 `.pdf` / `.PDF`。无需运行导入命令：第一次 `paper_search` 时自动发现、提取文本、按页分块并建立 TF-IDF 索引。以后每次检索检查文件路径、大小和纳秒级修改时间；新增、修改或删除 PDF 后自动重建。索引只在内存保存，重启后第一次查询重新构建。

- 使用 pypdf 提取文本，默认 chunk 为 1000 字符，重叠 150 字符，不跨页。
- 使用字符级 1–3 gram TF-IDF，兼容中英文；归一化向量点积作为相似度，无需 embedding 服务。
- `top_k` 默认 5，范围 1–10。每段带 `source`（相对 papers/ 的文件路径）、`page`、`chunk_id`、`score` 和 `text`。
- 损坏、加密或无可提取文字的 PDF 返回提示。扫描图片 PDF 暂不支持 OCR，需先转为可搜索 PDF。
- 单个 PDF 限制 50 MiB。当前适合小型本地文献集，海量文献可后续替换持久化向量索引。
- TF-IDF 是词面检索，不进行跨语言翻译；英文论文优先使用英文关键术语。非零相关度不保证片段足以支持结论。

## 6. 测试 RAG

先放入至少一篇文字型 PDF，再在 CLI 或 `/chat` 中提问：

```text
请检索本地论文中关于 transformer attention 的内容，给出相关方法、原文依据、文件名和页码。
```

也可以在项目根目录直接测试检索，不需要任何 API key：

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from rag import PaperIndex; import json; r=PaperIndex(Path('data/papers')).search('transformer attention', top_k=3); print(json.dumps(r, ensure_ascii=False, indent=2))"
```

将关键词换成自己论文里的术语。确认 `results` 中含文本、`source` 和 `page`；没有 PDF 时应返回友好提示而不是异常。

## 7. 测试 Python Tool

项目自带 `data/example.csv`，其中 `x=[1,2,3,4,5]`、`y=[2,4,5,8,11]`。

在 CLI 中输入：

```text
用 python_calculate 读取 example.csv：计算 x、y 的均值、样本标准差和中位数；计算 Pearson r 及 p 值；用 x 拟合一元线性回归预测 y，并计算训练集 R²。明确区分计算结果与分析，并说明样本很少、训练集 R² 不代表泛化能力。
```

工具可执行的示例代码：

```python
print(df[['x', 'y']].agg(['mean', 'std', 'median']))
pearson = stats.pearsonr(df['x'], df['y'])
print('Pearson r:', pearson.statistic, 'p-value:', pearson.pvalue)
model = linear_model.LinearRegression()
model.fit(df[['x']], df['y'])
prediction = model.predict(df[['x']])
print('R2:', metrics.r2_score(df['y'], prediction))
```

调用时 `data_file="example.csv"` 会自动提供 `df`。预期 x 均值为 3、中位数为 3、样本标准差约 1.5811；Pearson r 约 0.9839，线性回归训练集 R² 为 0.968。p 值等精确结果以实际工具输出为准。

直接离线测试：

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from tools.python_tool import python_calculate; print(python_calculate('print(df.describe())', Path('data'), data_file='example.csv'))"
```

也可自行构造 `pd.DataFrame` 或使用 `np.array`。提供 `np`、`pd`、`stats`、`metrics`、`linear_model`、`math`、`statistics`，允许从 numpy / pandas / scipy / sklearn / math / statistics 的指定模块导入白名单成员。使用 `print(...)` 输出结果；仅写最后一个表达式不会自动展示。每次工具调用均为新进程，变量不保留。

### Python 执行边界

- AST 语法与成员访问白名单，禁用 `os`、`subprocess`、`shutil`、`socket`、`open`、`eval`、`exec`、`getattr`、私有属性和任意导入。
- 禁止文件删除、shell、网络、任意读写；不开放 `pd.read_csv`、`to_csv`、pickle、pandas `eval/query/apply` 等间接执行入口。
- CSV 必须由 `data_file` 提供，父进程验证其在 data/ 内、格式和 2 MiB 大小限制，再复制到临时目录；科学计算代码只能访问预加载的 DataFrame。
- 使用独立子进程、隔离启动选项、临时工作目录、最小环境变量；不传入 API key。父进程超过设定时间会终止子进程。输出限制 12000 字符，代码限制 12000 字符。
- 这是受限科学计算环境，**不是经过安全审计的操作系统沙箱**。科学计算原生库仍可能大量分配内存；Windows 没有硬内存配额。公开服务或恶意代码执行场景应使用禁网容器/虚拟机、只读挂载和 CPU/内存配额，不应仅依赖 AST 校验。

## 8. 测试 Web Search

配置 `TAVILY_API_KEY` 后重启，再输入：

```text
请用 web_search 搜索本月关于大语言模型科研应用的研究进展，给出来源链接，区分检索到的事实和你的分析。
```

也可直接测试缺少密钥时的降级行为（不会发起网络请求）：

```powershell
.\.venv\Scripts\python.exe -c "from tools.web_search import web_search; print(web_search('recent research', api_key=''))"
```

返回说明缺少 `TAVILY_API_KEY`，`results` 为空。实际联网需要有效 Tavily 账号额度和网络；超时、认证失败等返回结构化错误。搜索只返回网页摘要，没有自动读取整篇网页，也不会保证结果按日期排序；涉及最新结论时需核实来源发布日期。

## 9. 项目结构与运行流程

```text
research-agent/
├── main.py                 # 交互式 CLI
├── agent.py                # System Prompt、模型请求、多轮工具循环
├── multi_agent.py          # Planner/Researcher/Analyst/Synthesizer 编排
├── config.py               # .env 和设置校验、logging
├── memory.py               # 按 session 存储完整轮次与并发锁
├── rag.py                  # PDF 发现、解析、chunk、TF-IDF 检索
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── tools/
│   ├── __init__.py          # 工具 JSON schema、参数校验、注册与分发
│   ├── calculator.py       # 安全算术表达式
│   ├── python_tool.py      # AST 白名单、科学计算工作进程
│   ├── file_tool.py        # data/ 内文件列表与 txt/csv/json 读取
│   └── web_search.py       # Tavily 搜索与错误处理
│   └── mcp_bridge.py        # 可选 MCP stdio/HTTP JSON-RPC 工具桥接
├── data/
│   ├── example.csv         # 可复现统计示例
│   └── papers/
├── api/
│   ├── __init__.py
│   └── server.py           # FastAPI 路由与生命周期
└── tests/
    └── test_project.py     # 无真实外部 API 的回归检查
```

流程：用户问题 → 加载 session 历史 → LLM → 判断 `tool_calls` → 参数校验并执行工具 → 以匹配的 `tool_call_id` 返回结果 → LLM 继续调用或给出最终回答 → 保存完整本轮记录。

默认最多 8 批工具循环，每批最多 16 个调用；达到上限后再发送一次 `tool_choice="none"` 的综合请求。因此模型请求最多 9 次（不含 SDK 网络重试）。提供方即使忽略禁用工具指令，也不会继续执行工具。工具结果包含来源并限制长度；异常不会伪装成计算成功。

### 代码自检

无需真实 API key：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall .
```

测试会生成临时 PDF 和 CSV，覆盖自动索引更新、来源、扫描/损坏 PDF、统计计算、危险操作拦截、执行超时、Tool Calling 协议、多工具与循环上限、会话隔离，以及 FastAPI 请求。真实 OpenAI SDK 使用 MockTransport 验证消息格式；不会产生实际模型或 Tavily 费用。完整外部服务连通性由你填好密钥后测试。

常见问题：

- `ModuleNotFoundError`：确认安装和运行使用同一个 `.venv` 解释器。
- 模型请求失败：检查 base URL、模型 ID、key、账户额度和提供方是否支持 tools；普通聊天可用不代表工具调用可用。
- 本地检索为空：检查文件路径、论文是否有文字层，以及关键词是否与原文一致。
- Python 首次执行超时：依赖导入也计入时间，可适当提高 `PYTHON_TIMEOUT_SECONDS`。
- 控制台中文乱码：在 PowerShell 中执行 `$env:PYTHONUTF8='1'` 后重新启动 Python。

## 10. MCP 与后续扩展

项目已内置可选 MCP 桥接：`tools/mcp_bridge.py` 会读取 `MCP_SERVER_COMMAND`/`MCP_SERVER_ARGS`（stdio JSON-RPC）或 `MCP_SERVER_URL`（HTTP JSON-RPC），执行 `initialize`、`tools/list` 和 `tools/call`。发现的工具会以 `mcp_` 前缀转换为 OpenAI function schema，并自动加入现有 `ToolRegistry`。未配置、启动失败或超时会记录日志并继续使用内置工具。

MCP 是可选能力，不配置时程序仍可启动，`GET /mcp` 会返回 `enabled=false` 或
`configured=false`。配置示例仅使用本地命令或地址；不要把 API key 写入 README、代码
或提交记录。配置后重启 FastAPI，并用以下接口检查发现结果：

stdio 示例：

```dotenv
MCP_SERVER_COMMAND=npx
MCP_SERVER_ARGS=-y @modelcontextprotocol/server-filesystem C:\\path\\to\\safe-dir
```

HTTP 示例：

```dotenv
MCP_SERVER_URL=http://127.0.0.1:8080/mcp
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/mcp
Invoke-RestMethod http://127.0.0.1:8000/agents
```

调试顺序建议是先运行 `GET /health`，再查看 `GET /` 的 `agents` 与 `mcp` 字段，最后
使用 `/chat` 的 `mode=multi`。MCP server 启动失败时，内置计算、论文、文件和网页工具
仍可用；`/mcp` 的 `error` 字段只包含错误摘要，不包含环境变量或请求内容。

离线 smoke test：

```powershell
.\.venv\Scripts\python.exe -c "from tools.mcp_bridge import MCPBridge; b=MCPBridge.from_tools([{'name':'ping','inputSchema':{'type':'object','properties':{}}}]); print(b.status(), b.definitions)"
```

工具接入点统一在 `tools/__init__.py` 的 `ToolRegistry`。未来可在此基础上扩展多服务器路由、认证、资源和 prompts。

建议为外部工具增加前缀避免重名，并保留 schema 校验、超时、输出限长和来源字段。不要将 MCP 工具返回的文本当成系统指令。只注册你认可的服务器和工具；涉及写入、删除、对外发送的工具应增加独立授权。未来扩展还可以包括 Redis 会话存储、OCR、混合检索和容器化 Python 执行。

## 11. 新手如何阅读这个项目

先把项目看成四层：

1. `main.py` 是命令行入口。你在这里启动程序、输入问题；`--mode single` 使用一个 Agent，`--mode multi` 使用多个 Agent。
2. `agent.py` 是单 Agent 的“大脑”。它读取会话历史，向 OpenAI-compatible 模型发送消息，接收 `tool_calls`，执行工具，再把结果交回模型。
3. `tools/` 是“手脚”：`calculator.py` 算数学，`python_tool.py` 做受限数据分析，`file_tool.py` 读本地文件，`web_search.py` 搜索 Tavily，`mcp_bridge.py` 接入 MCP 工具。
4. `api/server.py` 是 HTTP 门面。浏览器或其他程序通过 `/chat` 调用 Agent，通过 `/agents` 和 `/mcp` 查看运行状态。

建议按这个顺序阅读：

```text
main.py
  -> agent.py
      -> tools/__init__.py
          -> tools/calculator.py / python_tool.py / file_tool.py / web_search.py
      -> memory.py
      -> config.py
  -> multi_agent.py       # 需要多 Agent 时再看
  -> api/server.py         # 需要 HTTP 接口时再看
```

看代码时重点找三个词：`chat()` 表示一次对话，`execute()` 表示执行一个工具，`tool_calls` 表示模型请求工具。先不必理解全部类型标注；只要能沿着这三个词走一遍，就能掌握主流程。

推荐的新手练习：

- 把 `data/example.csv` 复制成 `data/my_data.csv`，让 CLI 读取它并计算均值。
- 在 `tools/calculator.py` 中增加一个安全的数学函数，然后为它补一个测试。
- 将一篇可复制文本的 PDF 放到 `data/papers/`，询问“请给出 source 和 page”。
- 运行 `GET /agents`，再把 `/chat` 请求中的 `mode` 改成 `multi`，比较单 Agent 和多 Agent 的返回。
- 运行 `from skills import list_skills; print(list_skills())`，阅读 `skills/` 下对应的工作流说明。

## 12. 项目内置 Skills

`skills/` 保存可复用的科研工作流约束，帮助 Planner、Researcher、Analyst 和 Synthesizer 使用一致的方法：

- `skills/literature-review/SKILL.md`：PDF 文献检索、source/page 证据和方法比较。
- `skills/data-analysis/SKILL.md`：CSV 检查、统计公式、样本量和局限。
- `skills/web-evidence/SKILL.md`：Tavily 检索、URL 保存、事实与分析分开。

这些文件是项目文档和提示词资源，不会自动执行代码，也不包含任何 API key。新增 skill 时保持一个清晰职责，写明触发场景、工具顺序、证据要求和失败时的处理方式。
