# VeritasMed

**安全优先、Agentic 容错的医学文献 RAG MCP 参考架构**

> 面向 Claude Desktop / Claude Code 的本地化医学 RAG MCP 服务器。  
> 5 层安全中间件 · LangGraph 自校验推理环路 · 引用感知生成 · 端到端 < 8 s

---

## 为什么 VeritasMed 与众不同

### 第一层 — 安全优先的 MCP 参考架构（最稀缺）

绝大多数 MCP server 在安全上裸奔。VeritasMed 在用户请求进入 LangGraph 推理环路前设置了 **5 层串行安全中间件**：

```
用户请求
   ▼
┌──────────────────────────────────────────┐
│  Layer 1: HMAC Token 认证 (auth)          │  防未授权访问，常数时间比较防时序攻击
│  Layer 2: 令牌桶限流 (rate_limit)         │  30 rpm global / 10 rpm LLM，防 DoS
│  Layer 3: 注入检测 (injection_guard)      │  11 条正则 + 特殊 Token 中性化 + XML 隔离
│  Layer 4: PII 脱敏 (pii)                 │  6 类 PHI 自动脱敏，满足 HIPAA/GDPR
│  Layer 5: 审计日志 (audit)               │  SHA-256 哈希查询，JSON-Lines，含 token 用量
└──────────────────────────────────────────┘
   ▼
LangGraph 推理环路
```

**30 个安全单元测试，覆盖 5 类威胁面，全部通过。** 详见 [`docs/security_test_report.md`](docs/security_test_report.md)。

### 第二层 — Agentic 容错设计

静态 RAG 管道遇到低质量检索只能交出一个糟糕的答案。VeritasMed 使用 **grade → rewrite → regen 三级自校验**：

```
hybrid_retrieve → rerank → grade_relevance
                                │
                    score < 0.6 │ (触发重写)
                                ▼
                          rewrite_query ──(最多 2 次)──▶ hybrid_retrieve
                                │ score ≥ 0.6
                                ▼
                        generate_answer ──validate_citations──▶
                        check_faithfulness
                                │ unfaithful
                    (最多 1 次) │
                                ▼
                          increment_regen ──▶ generate_answer
```

- **Citation-Grounded Generation**：每个 claim 必须携带至少一个来自当前检索结果的 cite key，幻觉 citation 在生成后即被过滤丢弃
- **Faithfulness 目标 ≥ 0.70**（Stage 2 优化后，基线 0.40）

### 第三层 — 工程完整度

| 能力 | 实现 |
|------|------|
| 双 LLM 后端 | `LLM_BACKEND=mimo\|ollama`；思考节点用 Pro/thinking=ON，生成节点用 fast |
| 混合检索 | BGE-M3 dense(1024-d) + sparse(SPLADE)，RRF 融合，Recall@5 = 100% |
| GPU 重排序 | `RERANKER_DEVICE=auto\|cuda\|cpu`；RTX 4060 上 20 对 ~200 ms（vs CPU 20 s）|
| 两级记忆 | L1: SqliteSaver 逐步持久化；L2: 超过 10 轮后滚动摘要压缩 |
| Demo UI | FastAPI WebSocket + React/Zustand，流式节点事件 + 彩色引用角标 |
| token 成本审计 | MCP audit 日志记录每次调用的 prompt/completion tokens |

---

## 评估结果

**检索质量（50 题 golden dataset）**

| 管道 | Recall@5 | MRR@20 | P50 延迟 |
|------|----------|--------|---------|
| P1 Dense only | 98.0% | 0.963 | 0.5 s |
| **P2 Hybrid RRF** | **100.0%** | **1.000** | 0.6 s |
| P3 Hybrid + Reranker | 100.0% | 1.000 | 64.7 s → **< 1 s** ¹ |

¹ Stage 3 后 Reranker 迁移至 GPU，延迟从 20 s 降至 ~200 ms。

**答案质量（MiMo-V2.5-Pro judge）**

| 管道 | Faithfulness | Correctness | Composite |
|------|-------------|-------------|-----------|
| P3 Static (baseline) | 0.40 | 0.916 | 0.772 |
| P4-Agentic + Stage 2 | ≥ 0.70 ¹ | ≥ 0.916 | — |

¹ Stage 2 Citation-Grounded Generation 后，Faithfulness 目标 ≥ 0.70（Hard Set 结果待补充）。

---

## 快速开始

**依赖**

```
Python 3.12 · Docker（Qdrant）· NVIDIA GPU ≥ 4 GB VRAM（GPU reranker 可选）
MiMo API key（或 Ollama + Qwen3-8B 本地推理）
```

**安装**

```bash
git clone https://github.com/lijingshan-6/medrag-agent.git
cd medrag-agent
conda env create -f environment.yml
conda activate medrag
pip install -e .
```

**配置 `.env`**

```ini
# Generator LLM（MiMo OpenAI-compatible API）
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-mimo-endpoint/v1
LLM_BACKEND=mimo          # mimo | ollama

# Reranker（auto 自动检测 CUDA）
RERANKER_DEVICE=auto       # auto | cuda | cpu

# 独立 judge（填入后 eval 脚本使用第三方，避免 self-eval bias）
# JUDGE_API_KEY=
# JUDGE_BASE_URL=
# JUDGE_MODEL=
```

**启动 Qdrant**

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v "${PWD}/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

**构建索引（一次性，约 45–120 分钟）**

```bash
python scripts/01_ingest_pubmed.py
python scripts/02_ingest_pmc.py
python scripts/04_build_index.py
```

**Demo UI**

```powershell
.\start_demo.ps1          # 同时启动 FastAPI backend + Vite dev server
# 访问 http://localhost:5173
```

**MCP 服务器**

```bash
mcp dev src/medrag/mcp_server/server.py   # 开发模式（token 认证可选）
```

---

## MCP 工具

| 工具 | 说明 |
|------|------|
| `ask_agent` | 完整推理环路：hybrid retrieval → grade/rewrite → citation-grounded generate → faithfulness check |
| `search_literature` | 快速混合检索 P2/P3，返回 ranked 文档片段 |
| `evaluate_query` | 对给定上下文块打分，判断能否回答问题 |
| `search_visual` | 图表检索预留接口（stub） |

所有工具共享同一套 5 层安全中间件。

---

## 测试

```bash
pytest tests/ -v
# 49 passed
#   tests/test_agent.py:        19 tests（图拓扑 / 节点逻辑 / 记忆机制）
#   tests/test_mcp_security.py: 30 tests（注入防护 / 限流 / 认证 / 审计 / PII）
```

---

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | MiMo-V2.5 / V2.5-Pro（API）或 Qwen3-8B via Ollama（本地 fallback） |
| 推理框架 | LangGraph 0.2（StateGraph + SqliteSaver 检查点） |
| MCP 服务器 | FastMCP 2.x（stdio transport） |
| 嵌入模型 | BAAI/bge-m3（dense 1024-d + sparse SPLADE，CPU） |
| 重排序模型 | BAAI/bge-reranker-v2-m3（GPU fp16 / CPU 自动切换） |
| 向量数据库 | Qdrant（Docker，单节点） |
| Demo 前端 | React + Zustand + Tailwind CSS v4 + Vite |
| Demo 后端 | FastAPI + WebSocket（asyncio.Queue 桥接 LangGraph） |
| 语料 | PubMed abstracts + PMC OA full-text（~186k chunks） |

---

## 文档

| 文件 | 内容 |
|------|------|
| [`docs/architecture.md`](docs/architecture.md) | 系统架构、节点职责、数据流 |
| [`docs/security_test_report.md`](docs/security_test_report.md) | 5 类威胁面 · 30 安全单元测试报告 |
| [`docs/latency_report.md`](docs/latency_report.md) | Stage 3 前后延迟对比 |
| [`docs/mcp_security.md`](docs/mcp_security.md) | 威胁模型与中间件设计 |
| [`docs/project_spec.md`](docs/project_spec.md) | 完整项目说明书 |

---

## License

Apache 2.0
