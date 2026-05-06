# VeritasMed

**自校验医学文献智能问答系统**

基于 LangGraph 的主动式 RAG 系统，集成混合向量检索、自主查询重写、忠实度校验与安全 MCP 接口。语料覆盖 PubMed 摘要与 PMC 开放全文，本地运行，无外部 API 依赖。

---

## 核心特性

**主动自校验推理环路**  
检索质量不达标时自动重写查询（最多 2 次）；生成答案后由独立节点逐项审计忠实度，不通过则重新生成。整个流程由 LangGraph StateGraph 编排，SqliteSaver 持久化每一步状态用于崩溃恢复和多轮会话。

**混合双向量检索**  
BGE-M3 单次前向传播同时输出稠密（1024-d cosine）与稀疏（SPLADE dot product）向量，经 RRF 融合后送入 BGE-Reranker 交叉编码器精排。50 题评测集 Recall@5 = 100%，MRR@20 = 1.000。

**双 LLM 策略**  
语义判断节点（grade / rewrite / check）使用 `thinking=ON` 进行扩展推理；生成与路由节点使用 `thinking=OFF` 保持低延迟。同一模型（Qwen3-8B）按节点切换模式，无需部署多个实例。

**安全 MCP 接口**  
5 层中间件：token 认证 → 令牌桶限流 → 注入检测 → PII 脱敏 → JSON-Lines 审计日志。检索文档以 XML 边界标签隔离，从结构层面阻止语料投毒。

---

## 评估结果

**检索（50 题 golden dataset）**

| 管道 | Recall@5 | MRR@20 | 延迟 |
|------|----------|--------|------|
| P1 Dense | 98.0% | 0.963 | 0.48s |
| **P2 Hybrid** | **100.0%** | **1.000** | 0.55s |
| P3 Hybrid+Reranker | 100.0% | 1.000 | 64.7s |
| P4 HyDE | 88.0% | 0.810 | 8.97s |
| P5 Multi-Query | 96.0% | 0.936 | 8.09s |

**答案质量（MiMo-V2.5-Pro 评判）**

| 管道 | Faithfulness | Relevance | Correctness | Composite |
|------|-------------|-----------|-------------|-----------|
| P3 Static | 0.405 | 0.996 | 0.916 | 0.772 |
| **P4-Agentic** | 0.401 | **1.000** | **0.920** | **0.774** |

---

## 快速开始

**环境要求**
- Python 3.12（conda）
- Ollama（运行 Qwen3-8B）
- Docker Desktop（运行 Qdrant）
- NVIDIA GPU ≥ 8 GB VRAM（LLM 使用 GPU，嵌入与重排序在 CPU）

**安装**

```bash
git clone <repo-url>
cd veritas-med
conda env create -f environment.yml
conda activate medrag
pip install -e .
```

**启动服务**

```bash
# 拉取模型
ollama pull qwen3:8b

# 启动 Qdrant（PowerShell）
docker run -d --name qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v "${PWD}/qdrant_storage:/qdrant/storage" `
  qdrant/qdrant:latest
```

**构建语料索引（一次性，约 45–120 分钟）**

```bash
python scripts/01_ingest_pubmed.py   # 摄取 PubMed 摘要
python scripts/02_ingest_pmc.py      # 摄取 PMC 全文
python scripts/04_build_index.py     # 嵌入 + 写入 Qdrant
```

**调用智能体**

```python
from medrag.agent.graph import app

config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke({
    "query": "What are the contraindications of warfarin in elderly patients?",
    "rewritten_queries": [], "retrieved_chunks": [],
    "relevance_score": 0.0, "grade_reason": "", "rewrite_hint": "",
    "iterations": 0, "answer": "", "citations": [], "confidence": 0.0,
    "faithful": False, "faithfulness_issues": "", "regen_count": 0,
    "history": [], "summary": "",
}, config=config)

print(result["answer"])
# Citations: ['PMID:...', 'PMC:...']
# Faithful: True / False
```

**启动 MCP 服务器**

```bash
# 开发模式（无需 token）
mcp dev src/medrag/mcp_server/server.py

# 生产模式（设置认证 token）
$env:MEDRAG_LOCAL_TOKEN = python -c "import secrets; print(secrets.token_hex(32))"
mcp dev src/medrag/mcp_server/server.py
```

---

## MCP 工具

| 工具 | 说明 |
|------|------|
| `ask_agent` | 完整推理环路：检索 → 评分/重写 → 生成 → 忠实度校验，支持多轮会话 |
| `search_literature` | 快速混合检索（P2/P3），返回文档片段 |
| `evaluate_query` | 对给定上下文块评分，判断能否回答问题 |
| `search_visual` | 图表/影像检索（预留接口，待实现） |

---

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | Qwen3-8B via Ollama（Q4_K_M，~5.2 GB VRAM） |
| 主动推理框架 | LangGraph 0.2（StateGraph + SqliteSaver） |
| MCP 服务器 | FastMCP 2.x（stdio transport） |
| 嵌入模型 | BAAI/bge-m3（CPU，dense 1024-d + sparse） |
| 重排序模型 | BAAI/bge-reranker-v2-m3（CPU，cross-encoder） |
| 向量数据库 | Qdrant（Docker，单节点） |
| 语料 | PubMed abstracts + PMC OA full-text（~186k chunks） |

---

## 测试

```bash
pytest tests/ -v
# 49 passed
#   test_agent.py:          19 tests（图拓扑 / 路由逻辑 / 节点变换 / 记忆机制）
#   test_mcp_security.py:   30 tests（注入防护 / 限流 / 认证 / 审计 / PII脱敏）
```

---

## 文档

- [`docs/architecture.md`](docs/architecture.md) — 系统架构图、节点职责、数据流
- [`docs/mcp_security.md`](docs/mcp_security.md) — 威胁模型、5 层安全中间件
- [`docs/project_spec.md`](docs/project_spec.md) — 完整项目说明书

---

## License

Apache 2.0
