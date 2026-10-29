# 项目 Skills

这里的 skills 是 Research Agent 的科研工作流说明，不是 Python 插件。它们让不同 Agent 使用一致的方法，并帮助新手理解“什么时候检索、什么时候计算、怎样报告证据”。

当前包括：

- `literature-review`：本地 PDF 文献综述、方法比较、source/page 引用。
- `data-analysis`：CSV 数据检查、统计计算、公式和局限说明。
- `web-evidence`：Tavily 最新信息检索、URL 保存和事实/分析区分。

查看 skills：

```powershell
.\.venv\Scripts\python.exe -c "from skills import list_skills; print(list_skills())"
```
