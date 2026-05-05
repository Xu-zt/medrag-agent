# Week 3 开发计划：Query Enhancement + Pipeline 对比评测 + MCP Server

| 字段 | 内容 |
|---|---|
| 时间预算 | ~18-22 h / 7 天 |
| 上周交付 | P1/P2/P3 全部跑通 · 44768 points (dense+sparse) · Streamlit UI |
| 本周交付 | P4 HyDE · P5 Multi-Query · Pipeline 对比脚本 · MCP Server |
| 核心难点 | HyDE prompt 设计 · Multi-Query RRF 去重 · MCP 工具注册与测试 |
| Golden Dataset | 继续搁置，评测脚本留接口等待填完再跑 |

---

## 目录

1. [Week 3 全局目标](#1-week-3-全局目标)
2. [Day 1：Pipeline 对比评测脚本](#2-day-1pipeline-对比评测脚本)
3. [Day 2：HyDE 检索器（P4）](#3-day-2hyde-检索器p4)
4. [Day 3：Multi-Query 检索器（P5）](#4-day-3multi-query-检索器p5)
5. [Day 4：更新 UI 与 quick_demo](#5-day-4更新-ui-与-quick_demo)
6. [Day 5-6：MCP Server](#6-day-5-6mcp-server)
7. [Day 7：Week 3 Tutorial 文档](#7-day-7week-3-tutorial-文档)
8. [技术背景：为什么还需要 P4/P5？](#8-技术背景为什么还需要-p4p5)
9. [完整 Pipeline 演进图](#9-完整-pipeline-演进图)
10. [可裁剪项](#10-可裁剪项)

---

## 1. Week 3 全局目标

### 本周新增的两条 Pipeline

到 Week 3 末，项目拥有 5 条可切换的检索管道：

| ID | 名称 | 核心技术 | 相比上一级的改进 |
|----|------|---------|----------------|
| P1 | Dense Only | BGE-M3 cosine | 基线 |
| P2 | Hybrid RRF | dense + neural sparse | 精确词汇命中↑ |
| P3 | Hybrid + Reranker | P2 + cross-encoder | 精排质量↑ |
| **P4** | **HyDE** | 假设文档嵌入 | **问题空间→答案空间，复杂问题召回↑** |
| **P5** | **Multi-Query** | 多查询 RRF | **单一表述的局限性↓，覆盖率↑** |

### 本周交付清单

| 交付物 | 位置 | 验收标准 |
|---|---|---|
| `scripts/06_compare_pipelines.py` | scripts/ | 输出 P1~P5 五列对比表，含延迟 |
| `src/medrag/retrieval/hyde.py` | src/medrag/retrieval/ | `HyDERetriever.retrieve()` 返回 RetrievedChunk |
| `src/medrag/retrieval/multi_query.py` | src/medrag/retrieval/ | `MultiQueryRetriever.retrieve()` 返回 RRF 融合结果 |
| UI 更新（P4/P5 选项） | src/medrag/ui/app.py | Streamlit 能选 P4/P5 |
| `src/medrag/mcp_server/server.py` | src/medrag/mcp_server/ | `mcp dev` 能列出工具，`ask` 工具能返回答案 |
| `docs/week3_tutorial.md` | docs/ | 覆盖本周所有模块 |

---

## 2. Day 1：Pipeline 对比评测脚本

**文件**：`scripts/06_compare_pipelines.py`

### 设计目标

Week 2 计划里遗留的任务。这个脚本不依赖 Golden Dataset，直接用一批固定的 sample query 跑五条管道，输出对比表。核心价值：

1. 可以直观看到 P1→P2→P3→P4→P5 的 retrieved docs 差异
2. 记录每条管道的延迟（latency），了解速度-质量权衡
3. 为面试展示提供"消融实验"的可视化证据

### 实现要点

```python
# scripts/06_compare_pipelines.py

SAMPLE_QUERIES = [
    "What is the typical spatial resolution of 3T MRI?",
    "What is the mechanism of action of PARP inhibitors in BRCA-mutated cancers?",
    "How do PI-RADS v2.1 criteria distinguish between score 3 and 4?",
    "What are the contraindications for thrombolytic therapy in ischemic stroke?",
    "Describe the role of EGFR T790M mutation in NSCLC treatment resistance.",
]

PIPELINES = {
    "P1": lambda: DenseRetriever(qdrant, embedder),
    "P2": lambda: HybridRetriever(qdrant, embedder),
    "P3": lambda: RerankerPipeline(qdrant, embedder, reranker),  # 封装 P2+rerank
    "P4": lambda: HyDERetriever(qdrant, embedder, llm),
    "P5": lambda: MultiQueryRetriever(qdrant, embedder, llm),
}
```

对每个 query × pipeline 组合：
1. 记录 `time.perf_counter()` 开始/结束时间
2. 收集 top-5 的 `citation` 和 `score`
3. 可选：调用 `generate_answer()` 得到最终答案

**输出格式**（Markdown 表格 + JSON）：

```
Query: "What is the mechanism of PARP inhibitors..."
┌──────┬──────────┬────────────────────────────────────────┐
│      │ Latency  │ Top-3 Retrieved Docs                   │
├──────┼──────────┼────────────────────────────────────────┤
│  P1  │  0.24s   │ PMC:doc6 (0.82), PMC:doc12 (0.79)...  │
│  P2  │  0.31s   │ PMC:doc6 (0.031), PMID:123 (0.029)... │
│  P3  │  4.2s    │ PMC:doc6 (1.24), PMC:doc8 (0.91)...   │
│  P4  │  5.8s    │ PMC:doc8 (0.84), PMC:doc6 (0.82)...   │
│  P5  │  12.1s   │ PMC:doc6 (0.031), PMC:doc3 (0.028)... │
└──────┴──────────┴────────────────────────────────────────┘
```

结果同时保存到 `data/eval/pipeline_comparison.json` 供后续分析。

---

## 3. Day 2：HyDE 检索器（P4）

**文件**：`src/medrag/retrieval/hyde.py`

### 3.1 什么是 HyDE？

HyDE（Hypothetical Document Embeddings，Gao et al. 2022）的核心洞察：

**传统检索的问题**：

```
Query: "What are the contraindications of thrombolysis?"
       ↓ BGE-M3 encode
  [question-space vector] ←── 与文献向量存在"模态差距"
```

医学文献是以**陈述句**写的（"Contraindications include..."），而查询是**疑问句**（"What are..."）。两种文本的 BGE-M3 向量天然不在同一个分布中心。

**HyDE 的解法**：

```
Query: "What are the contraindications of thrombolysis?"
       ↓ LLM 生成一段假设性答案
  "Thrombolysis is contraindicated in patients with recent surgery,
   active bleeding, or history of hemorrhagic stroke..."
       ↓ BGE-M3 encode
  [answer-space vector] ←── 与文献向量分布更接近！
       ↓ Qdrant dense search
  真实文献...
```

这个"假设性答案"不需要正确，只需要在**语义空间**上接近真实答案的位置。即使 LLM 的假设包含错误，向量化后仍然可以找到比原始 query 更相关的文档。

### 3.2 实现设计

```python
# src/medrag/retrieval/hyde.py

class HyDERetriever:
    HYDE_PROMPT = (
        "You are a medical expert. Write a concise 2-3 sentence passage "
        "from a medical research paper that would answer the following question. "
        "Focus on factual, technical content. Do NOT include citations or references.\n\n"
        "Question: {query}\n\n"
        "Hypothetical passage:"
    )

    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: BGEM3Embedder,
        model: str = "qwen3:8b",
        k: int = 5,
    ):
        ...

    def _generate_hypothesis(self, query: str) -> str:
        """用 Qwen3-8B 生成假设性文档片段。"""
        llm = ChatOllama(model=self.model, base_url="http://127.0.0.1:11434",
                         reasoning=False, temperature=0.4)
        resp = llm.invoke([HumanMessage(content=self.HYDE_PROMPT.format(query=query))])
        return strip_thinking(resp.content).strip()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        # 1. LLM 生成假设性文档
        hypothesis = self._generate_hypothesis(query)

        # 2. 用假设文档的 dense 向量检索（而非原始 query）
        enc = self.embedder.encode([hypothesis])
        hyp_vec = enc["dense"][0].tolist()

        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=hyp_vec,
            using="dense",
            limit=k,
            with_payload=True,
        )
        return [RetrievedChunk(...) for h in result.points]
```

**关键设计决策**：

| 决策 | 选择 | 理由 |
|------|------|------|
| 是否用 sparse 向量检索？ | 否，只用 dense | 假设文档已经是"答案风格"，dense 最合适 |
| 生成 1 还是多个假设？ | 1 个（起步） | 简单可行，多个会多倍增加延迟 |
| LLM 温度？ | 0.4 | 略高于 0.2，允许一定创造性但不偏离事实太远 |
| 假设文档是否加入上下文？ | 否 | 只用假设向量，不把假设文本传给最终 generator |

**延迟估算**：
- `_generate_hypothesis()` ≈ 3-5 秒（Qwen3-8B 本地）
- Qdrant 查询 ≈ 0.1 秒
- 总计 P4 ≈ 4-6 秒（比 P3 慢，但质量更高）

### 3.3 P4 的适用场景

HyDE 对以下类型的问题改善最明显：
- 复杂机制类问题（"How does X work?"）
- 多步推理问题（"What happens when A and B both occur?"）
- 文献中以陈述风格写作、但查询以疑问风格提问的情况

对于**精确词汇查询**（"What is the dosage of X?"），HyDE 未必优于 P2/P3，因为稀疏检索对精确词更有优势。

---

## 4. Day 3：Multi-Query 检索器（P5）

**文件**：`src/medrag/retrieval/multi_query.py`

### 4.1 Multi-Query 的动机

单一 query 表述的局限：

```
原始: "What is the role of BRCA1 in DNA repair?"

可能的文献说法：
  - "BRCA1 functions in homologous recombination..."
  - "The tumor suppressor BRCA1 participates in double-strand break repair..."
  - "Germline BRCA1 mutations impair the HR pathway..."
```

三种说法都是正确答案，但 single-query embedding 可能只覆盖其中一种。

**Multi-Query 的解法**：让 LLM 生成同一问题的 N 种表述，对每种表述分别检索，然后 RRF 融合所有结果集。

### 4.2 实现设计

```python
# src/medrag/retrieval/multi_query.py

class MultiQueryRetriever:
    REWRITE_PROMPT = (
        "Generate {n} different phrasings of the following medical question. "
        "Each rephrasing should capture the same information need but use "
        "different vocabulary, focusing on: synonyms, alternative terminologies, "
        "and varied sentence structures. Output ONLY the rephrased questions, "
        "one per line, no numbering or explanations.\n\n"
        "Original question: {query}"
    )

    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: BGEM3Embedder,
        model: str = "qwen3:8b",
        n_queries: int = 3,   # 生成 3 个改写版本（加上原始共 4 个）
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        ...

    def _rewrite_query(self, query: str) -> list[str]:
        """用 LLM 生成 n 个改写版本，返回 [原始, 改写1, 改写2, ...]。"""
        ...
        rewrites = resp.strip().split("\n")
        return [query] + [r.strip() for r in rewrites if r.strip()][:self.n_queries]

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        # 1. 生成改写
        queries = self._rewrite_query(query)

        # 2. 对每个改写分别检索
        all_rankings: list[list[str]] = []
        id_to_point: dict[str, object] = {}

        for q in queries:
            enc = self.embedder.encode([q])
            vec = enc["dense"][0].tolist()
            result = self.qdrant.query_points(..., query=vec, limit=self.candidate_k)
            ranking = []
            for p in result.points:
                cid = p.payload["chunk_id"]
                ranking.append(cid)
                id_to_point.setdefault(cid, p)
            all_rankings.append(ranking)

        # 3. RRF 融合所有排名列表（复用 Week 2 的函数）
        from medrag.retrieval.hybrid import _reciprocal_rank_fusion
        fused = _reciprocal_rank_fusion(all_rankings, k=self.rrf_k)

        # 4. 重建 top-k RetrievedChunk
        return [RetrievedChunk(..., score=rrf_score) for chunk_id, rrf_score in fused[:k]]
```

**关键设计决策**：

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| `n_queries` | 3 | 3 个改写 + 1 原始 = 4 路检索；更多会显著增加延迟 |
| `candidate_k` per query | 10 | 每路检索 10 个，4 路最多 40 个候选，RRF 后取 top-5 |
| 是否用 sparse？ | 否（起步版）| 简化实现；高级版可以对每个改写做混合检索 |

**延迟估算**：
- Query rewriting ≈ 2-4 秒（Qwen3-8B）
- 4 × Qdrant query ≈ 0.4 秒
- 总计 P5 ≈ 3-5 秒（比 P3 快，但比 P4 慢）

### 4.3 Multi-Query vs HyDE：对比与选择

| 维度 | P4 HyDE | P5 Multi-Query |
|------|---------|----------------|
| 核心思想 | 用 LLM "预测"答案的向量 | 用 LLM 扩展查询的表达多样性 |
| 适合场景 | 问题与文献"风格差距"大 | 问题有多种合法表述 |
| 对 LLM 依赖 | 高（需要高质量假设） | 中（只需改写，不需正确答案） |
| 延迟 | 稍高（生成一段话） | 稍低（只生成几行） |
| 可以组合吗？ | P4+P5 = 多假设文档 RRF（Week 4 扩展） | — |

---

## 5. Day 4：更新 UI 与 quick_demo

### 5.1 Streamlit UI 更新

在 `src/medrag/ui/app.py` 的侧边栏加入 P4/P5 选项：

```python
pipeline = st.radio(
    "Retrieval Pipeline",
    options=[
        "P1 · Dense Only",
        "P2 · Hybrid (Dense + Sparse)",
        "P3 · Hybrid + Reranker",
        "P4 · HyDE (Hypothetical Document)",  # 新增
        "P5 · Multi-Query RRF",                # 新增
    ],
    index=2,
)
```

侧边栏说明表格增加两行：

```
| P4 | HyDE：LLM 生成假设文档 → dense 检索 | 复杂机制类问题 |
| P5 | Multi-Query：生成 3 个改写 → RRF 融合 | 多义词/多角度问题 |
```

主查询逻辑增加 P4/P5 分支：

```python
elif mode_key == "P4":
    retriever = HyDERetriever(qdrant, embedder)
    chunks = retriever.retrieve(query, k=top_k)
else:  # P5
    retriever = MultiQueryRetriever(qdrant, embedder)
    chunks = retriever.retrieve(query, k=top_k)
```

**注意**：P4/P5 都用 LLM 生成中间内容，`load_resources()` 不需要额外缓存 retriever（LLM 是 ChatOllama，无状态）。

### 5.2 quick_demo.py 更新

```python
parser.add_argument("--mode", choices=["p1", "p2", "p3", "p4", "p5"], default="p3")
```

增加 p4/p5 分支，输出中额外显示 LLM 生成的 hypothesis / rewritten queries（debug 信息）：

```
Mode  : P4: HyDE
Hypothesis: "BRCA1 plays a central role in homologous recombination,
             the high-fidelity pathway for repairing..."
[1] score=0.84  PMC:doc6 ...
```

---

## 6. Day 5-6：MCP Server

**文件**：`src/medrag/mcp_server/server.py`

### 6.1 什么是 MCP？

Model Context Protocol（MCP）是 Anthropic 发布的开放协议，允许 LLM 客户端（如 Claude Desktop、Claude Code）通过标准化接口调用外部工具。

MCP Server 可以理解为：**把 Python 函数"注册"成一个工具，Claude 可以自动调用它**。

对这个项目的意义：把 MedRAG-Agent 的检索-生成管道封装成一个 MCP 工具，用户在 Claude Desktop 中直接提问，Claude 自动调用 MedRAG 工具检索医学文献并生成回答。

### 6.2 技术选型：fastmcp

```bash
pip install fastmcp
```

`fastmcp` 是 MCP 的 Python 高层封装，语法类似 FastAPI：用 `@mcp.tool()` 装饰器注册工具。

### 6.3 Server 设计

```python
# src/medrag/mcp_server/server.py

from fastmcp import FastMCP
from qdrant_client import QdrantClient

from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker
from medrag.agent.generator import generate_answer

mcp = FastMCP("MedRAG-Agent")

# 懒加载：MCP server 启动时才初始化（避免 import 时就加载大模型）
_resources = None

def _get_resources():
    global _resources
    if _resources is None:
        qdrant = QdrantClient(url="http://localhost:6333")
        embedder = BGEM3Embedder(device="cpu")
        reranker = BGEReranker()
        _resources = (qdrant, embedder, reranker)
    return _resources


@mcp.tool()
def retrieve(
    query: str,
    pipeline: str = "p3",
    k: int = 5,
) -> list[dict]:
    """Retrieve relevant medical literature chunks for a query.

    Args:
        query: The medical question to search for.
        pipeline: One of 'p1' (dense), 'p2' (hybrid), 'p3' (hybrid+reranker).
        k: Number of chunks to return (3-10).

    Returns:
        List of {citation, score, text_preview} dicts.
    """
    qdrant, embedder, reranker = _get_resources()
    if pipeline == "p1":
        retriever = DenseRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
    elif pipeline == "p2":
        retriever = HybridRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
    else:  # p3 default
        retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
        candidates = retriever.retrieve(query, k=20)
        chunks = reranker.rerank(query, candidates, top_k=k)

    return [
        {
            "citation": c.citation,
            "score": round(c.score, 4),
            "text_preview": c.text[:300],
        }
        for c in chunks
    ]


@mcp.tool()
def ask(
    query: str,
    pipeline: str = "p3",
    k: int = 5,
) -> str:
    """Retrieve medical literature and generate an answer using local LLM.

    Args:
        query: The medical question.
        pipeline: One of 'p1', 'p2', 'p3' (default: p3 for best quality).
        k: Number of source documents to use.

    Returns:
        LLM-generated answer with inline citations.
    """
    qdrant, embedder, reranker = _get_resources()
    # ... 同 retrieve() 的检索逻辑 ...
    answer = generate_answer(query, chunks)
    return answer


if __name__ == "__main__":
    mcp.run()
```

### 6.4 启动方式

**开发模式**（带调试 UI）：

```bash
mcp dev src/medrag/mcp_server/server.py
```

浏览器打开 MCP Inspector，可以直接调用工具测试。

**生产模式**（供 Claude Desktop 使用）：

在 Claude Desktop 的 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "medrag": {
      "command": "C:\\Users\\lijingshan\\.conda\\envs\\medrag\\python.exe",
      "args": [
        "D:\\Desktop\\Agent\\medrag-agent\\src\\medrag\\mcp_server\\server.py"
      ]
    }
  }
}
```

### 6.5 启动脚本

```powershell
# scripts/run_mcp_server.ps1
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
Set-Location "D:\Desktop\Agent\medrag-agent"
Write-Host "Starting MedRAG MCP Server..." -ForegroundColor Cyan
& $py src/medrag/mcp_server/server.py
```

---

## 7. Day 7：Week 3 Tutorial 文档

**文件**：`docs/week3_tutorial.md`

涵盖本周所有模块的详细说明，重点：

1. HyDE 论文背景与"向量空间中的问题-答案距离"直觉解释
2. Multi-Query RRF 与 P2 RRF 的异同（P2 融合不同检索器 vs P5 融合不同表述）
3. MCP 协议简介：为什么 MCP 比 REST API 对 AI 应用更友好
4. Pipeline 对比实验结果分析（P1→P5 各自的召回特征）

---

## 8. 技术背景：为什么还需要 P4/P5？

### P2/P3 解决了什么，没解决什么？

P2 引入稀疏检索，解决了"精确词汇不匹配"的问题。P3 引入交叉编码器，解决了"bi-encoder 精排不足"的问题。

**还剩一个问题没解决：query 本身的表述偏差。**

假设用户问：`"Why do beta-blockers worsen asthma?"`

文献里的说法是：
- `"Non-selective beta-adrenergic blockers can precipitate bronchospasm..."`
- `"Beta-blockade causes bronchoconstriction via β2 receptor inhibition..."`

这两句话和原始 query 的 dense 向量距离都不近，因为：
- 文献不说 "worsen"，说 "precipitate"、"cause"
- 文献不说 "asthma"，说 "bronchospasm"、"bronchoconstriction"

P2 的稀疏检索也帮不了多少，因为 "worsen" 和 "precipitate" 是不同的 token。

这就是 P4/P5 要解决的核心问题：**query 与文献的词汇风格差距**。

### HyDE 的理论依据

HyDE 的核心假设（已被实验验证）：

> 在 embedding 空间中，**答案的向量比问题的向量更靠近真实文档的向量**。

为什么？文献是以"陈述"风格写的，而问题是以"疑问"风格写的。LLM 生成的假设答案也是"陈述"风格，因此它的向量和文献向量处于更相似的分布中。

即使 LLM 生成的答案在事实上不准确，只要它在**语义风格**上接近真实文献，就能找到更好的检索结果。

### Multi-Query 的理论依据

Multi-Query 假设：

> **没有任何一种 query 表述是最优的**，同一问题的不同表达会在不同的"语义邻域"命中不同的文档。

通过生成 3-4 种表述并 RRF 融合，我们降低了单一表述的"运气成分"，提升了检索的稳健性（robustness）。

这个技术特别适合医学领域，因为同一概念往往有多个合法表述（通用名/商品名、拉丁语/英文、简称/全称）。

---

## 9. 完整 Pipeline 演进图

```
Week 1                 Week 2                    Week 3
──────                 ──────                    ──────
P1: Dense ──────────→ P2: Hybrid RRF ──────────→ P4: HyDE
    │                      │                         │
    │                      ↓                         ↓
    │                 P3: Hybrid +              P5: Multi-Query
    │                     Reranker                  RRF
    │
    └─────────────────────────────────────────────────────────
                    所有管道共享:
                    - BGE-M3 embedder
                    - Qdrant collection (44768 points, dense+sparse)
                    - Qwen3-8B generator
                    - RetrievedChunk 数据结构
```

**一句话总结每条管道的"专长"**：

- P1：速度最快，适合语义相似度高的问题
- P2：词汇精确命中更好，适合含专有名词/缩写的问题
- P3：精排质量最高，适合需要高精度的场景
- P4：复杂机制问题效果好，query 与文献风格差距大时尤其有用
- P5：覆盖率最高，适合问题本身有多种等价表述的场景

---

## 10. 可裁剪项

如果时间不足（7 天内无法完成所有内容），按优先级裁剪：

| 优先级 | 模块 | 裁剪后影响 |
|--------|------|-----------|
| 🔴 必做 | P4 HyDE | Week 3 的核心新功能 |
| 🔴 必做 | 06_compare_pipelines.py | 面试展示的关键脚本 |
| 🟡 重要 | P5 Multi-Query | 可以 Week 4 补做 |
| 🟡 重要 | MCP Server | 可以 Week 4 补做，UI 已经能展示管道 |
| 🟢 可选 | Week 3 Tutorial | 参考 Week 1/2 格式即可 |

**最小可行交付（如果时间很紧）**：

```
P4 (HyDE) + 06_compare_pipelines.py = Week 3 核心
```

---

*文档创建于 Week 2 完成后，作为 Week 3 开发路线图。*
