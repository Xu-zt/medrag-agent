# Week 3 教程：Query Enhancement · Pipeline 对比 · MCP Server

| 字段 | 内容 |
|---|---|
| 本周新增 | P4 HyDE · P5 Multi-Query · MCP Server · 对比评测脚本 |
| 前置依赖 | Week 2 完成（Qdrant 中已有 44768 个 dense+sparse 向量点）|
| 核心难点 | HyDE prompt 设计 · Multi-Query RRF 去重 · MCP 工具注册 |

---

## 目录

1. [为什么 P1/P2/P3 还不够？](#1-为什么-p1p2p3-还不够)
2. [P4：HyDE 检索器](#2-p4hyde-检索器)
3. [P5：Multi-Query 检索器](#3-p5multi-query-检索器)
4. [Pipeline 对比评测脚本](#4-pipeline-对比评测脚本)
5. [UI 更新：P4/P5 选项与展示](#5-ui-更新p4p5-选项与展示)
6. [MCP Server](#6-mcp-server)
7. [调试经验](#7-调试经验)
8. [完整 Pipeline 演进图](#8-完整-pipeline-演进图)

---

## 1. 为什么 P1/P2/P3 还不够？

P1/P2/P3 的检索策略都假设：**原始问题的措辞就是最佳查询**。
但在医学文献检索中，这个假设经常失效：

### 问题空间 vs. 答案空间的 Embedding Gap

想象一下这个查询：
> "How does pembrolizumab work in lung cancer?"

PubMed 里的相关文献不会这样写，它们更像：
> "Pembrolizumab, an anti-PD-1 monoclonal antibody, demonstrated significant overall survival improvement in PD-L1-positive NSCLC patients in the KEYNOTE-024 trial..."

两段文字的语义非常接近，但措辞截然不同。**BGE-M3 的 dense embedding 能捕捉语义相似性**，但如果问题和答案的表达差距太大（问题很口语，论文很技术），Embedding 空间里的距离会比预期更远。

### 词汇变体问题

同一个概念在医学中有多种表达：
- "心肌梗死" / "myocardial infarction" / "MI" / "heart attack" / "AMI"
- "NSCLC" / "non-small-cell lung cancer" / "non-small cell lung carcinoma"

一个问题只能用其中一种措辞，但文献可能用任何一种。P5 Multi-Query 专门解决这个问题。

### 两种解决方案

| 问题 | 解决方案 | Pipeline |
|---|---|---|
| 问题 vs. 答案的 Embedding 偏移 | HyDE：先生成假设答案，再检索 | P4 |
| 单一措辞的词汇局限 | Multi-Query：生成多种措辞，各自检索后融合 | P5 |

---

## 2. P4：HyDE 检索器

**文件**：`src/medrag/retrieval/hyde.py`

### 核心思想

HyDE（Hypothetical Document Embeddings）来自 Gao et al., 2022 的论文。核心想法极其简单：

```
原始查询（问题）
    │
    ▼ LLM（Qwen3-8B）
    │
    ▼ 假设文档段落（2-3句，像论文摘要的风格）
    │
    ▼ BGE-M3 dense encode
    │
    ▼ Qdrant ANN 搜索
    │
    ▼ Top-k 结果
```

**为什么有效**：假设文档的 embedding 在向量空间里更接近真实的论文文本，而不是"问题"这种格式的文本。

### Prompt 设计

```python
HYDE_SYSTEM = (
    "You are a medical research assistant. Your task is to write a short, "
    "factual passage (2-3 sentences) as if it were excerpted from a real "
    "PubMed abstract or clinical guideline that directly answers the given "
    "medical question. Focus on technical accuracy. Do NOT include citations, "
    "author names, or journal names. Do NOT answer the question conversationally "
    "— write in the style of a paper excerpt."
)
```

**关键设计决策**：
1. **"as if it were excerpted from a real PubMed abstract"** — 告诉 LLM 模拟学术写作风格，不要用口语
2. **"Do NOT include citations"** — 避免生成幻觉引用
3. **不要对话式回答** — 防止生成 "Great question! The answer is..." 这种风格

### 完整实现

```python
class HyDERetriever:
    def __init__(self, qdrant, embedder, collection="medrag_text",
                 llm_model="qwen3:8b", temperature=0.3):
        self.llm = ChatOllama(model=llm_model, ...)

    def _generate_hypothesis(self, query: str) -> str:
        resp = self.llm.invoke([
            SystemMessage(content=HYDE_SYSTEM),
            HumanMessage(content=HYDE_USER_TEMPLATE.format(query=query)),
        ])
        hypothesis = strip_thinking(resp.content).strip()
        self._last_hypothesis = hypothesis   # UI 展示用
        return hypothesis

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        hypothesis = self._generate_hypothesis(query)

        # 注意：encode 的是假设文档，不是原始查询
        enc = self.embedder.encode([hypothesis])
        dense_vec = enc["dense"][0].tolist()

        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=dense_vec,
            using="dense",
            limit=k,
            with_payload=True,
        )
        return [RetrievedChunk(...) for p in result.points]
```

### HyDE 的局限性

| 优点 | 局限 |
|---|---|
| 复杂问题召回率提升 | 多一次 LLM 调用，延迟增加 ~1-3s |
| 对话题型问题效果好 | LLM 可能生成不准确的假设（幻觉） |
| 不需要额外数据集 | 对简单查询效果不如 P3 |

**温度参数**：设为 0.3（略高于 0.2）。HyDE 需要 LLM 生成一些"探索性"内容，太低的温度会让生成结果过于保守。

---

## 3. P5：Multi-Query 检索器

**文件**：`src/medrag/retrieval/multi_query.py`

### 核心思想

```
原始查询
    │
    ▼ LLM 生成 3 种改写
    │
    ▼ [原始查询, 改写1, 改写2, 改写3] — 共 4 个查询
    │
    ▼ 对每个查询：BGE-M3 dense encode → Qdrant 检索 top-10
    │
    ▼ 四个 ranked list → RRF 融合
    │
    ▼ Top-k 结果（去重 + 重新排序）
```

### Prompt 设计

```python
MQ_SYSTEM = (
    "You are a medical information retrieval expert. Given a medical question, "
    "generate 3 alternative phrasings that express the same information need "
    "but use different vocabulary, perspective, or level of technicality. "
    "Output ONLY a numbered list (1. ... 2. ... 3. ...). No explanations."
)
```

**关键点**：`Output ONLY a numbered list` — 防止 LLM 加解释文字，方便 regex 解析。

**改写解析**：

```python
def _rewrite_query(self, query: str) -> list[str]:
    resp = self.llm.invoke([...])
    raw = strip_thinking(resp.content).strip()
    
    # 解析 "1. xxx\n2. xxx\n3. xxx"
    rewrites = re.findall(r"^\d+\.\s*(.+)$", raw, re.MULTILINE)
    rewrites = [r.strip() for r in rewrites if r.strip()][:3]
    
    # 原始查询始终放在第一位
    return [query] + rewrites
```

原始查询始终包含在列表中，确保不会因为改写方向偏差而完全遗漏相关文档。

### RRF 融合（4 个 ranked list）

```python
def _rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

这和 `hybrid.py` 里的 `_reciprocal_rank_fusion` 完全一样的算法，但这里融合 4 个列表（每个改写查询一个），而 P2/P3 融合的是 2 个（dense 和 sparse）。

**RRF 对重复文档的处理**：如果同一个 chunk 在 4 个查询的结果里都出现了，它的 RRF 分数是其他文档的 4 倍左右。这是一个隐式的投票机制——多个查询角度都认为相关的文档，最终排名会大幅提升。

### temperature 设为 0.5

Multi-Query 需要 LLM 生成**多样性**的改写，如果 temperature 太低，4 个改写会很相似，失去改写的意义。0.5 是一个合适的平衡点（不太随机，但有足够变化）。

---

## 4. Pipeline 对比评测脚本

**文件**：`scripts/06_compare_pipelines.py`

### 设计目标

这是整个项目中最重要的**消融实验**脚本。没有对比数据，就没有办法证明 P3 比 P1 好、P4 比 P3 好。这个脚本可以直接生成面试中展示的对比表格。

### 架构：Pipeline Factory 模式

```python
def build_pipelines(qdrant, embedder, reranker, candidate_k, requested):
    available = {}
    
    if "p1" in requested:
        p1 = DenseRetriever(qdrant, embedder)
        available["p1"] = lambda q, k, _r=p1: _r.retrieve(q, k=k)
    
    if "p3" in requested:
        p3_retriever = HybridRetriever(qdrant, embedder, candidate_k=candidate_k)
        available["p3"] = lambda q, k, _r=p3_retriever, _rr=reranker: _rr.rerank(
            q, _r.retrieve(q, k=candidate_k), top_k=k
        )
    
    if "p4" in requested:
        try:
            from medrag.retrieval.hyde import HyDERetriever
            available["p4"] = ...
        except ImportError:
            print("[skip] p4: not found yet")  # 优雅降级
    ...
    return available
```

**lambda + default argument 的 trick**：Python 的闭包在捕获循环变量时有一个经典陷阱。这里用 `_r=p3_retriever` 作为默认参数，强制每个 lambda 捕获当前的对象引用而不是变量名。

**优雅降级**：P4/P5 用 `try/except ImportError` 包裹，这样即使 `hyde.py` 还没实现，`--pipelines p1,p2,p3` 也能正常运行。

### 延迟测量

```python
t0 = time.perf_counter()
chunks = fn(query, k)
latency = time.perf_counter() - t0
```

`time.perf_counter()` 是 Python 里精度最高的计时器（纳秒级），比 `time.time()` 更适合性能测量。

### 运行方式

```bash
# 默认：P1/P2/P3，5 个样本查询，k=5
python scripts/06_compare_pipelines.py

# 包含 P4/P5
python scripts/06_compare_pipelines.py --pipelines p1,p2,p3,p4,p5

# 显示每条命中的详细内容
python scripts/06_compare_pipelines.py --detailed

# 指定输出路径
python scripts/06_compare_pipelines.py --output data/eval/my_run.json
```

### 输出格式

**Markdown 表格**（打印到 stdout）：

```
## Pipeline Comparison Results

| Query | P1 Dense | P2 Hybrid | P3 Hybrid+Reranker |
|---|---|---|---|
| `What is the typical spatial reso...` | 0.312s · PMID:123, PMID:456, PMC:789 | 0.408s · ... | 1.234s · ... |
...

## Latency Summary (avg over queries)

| Pipeline | Avg Latency (s) | Min (s) | Max (s) |
|---|---|---|---|
| P1 Dense | 0.298 | 0.201 | 0.412 |
```

**JSON 输出**（保存到文件）：

```json
{
  "k": 5,
  "candidate_k": 20,
  "results": [
    {
      "query": "...",
      "pipeline": "p1",
      "latency_s": 0.298,
      "error": null,
      "hits": [
        {"rank": 1, "citation": "PMID:123", "score": 0.8231, "snippet": "..."}
      ]
    },
    ...
  ]
}
```

JSON 格式方便后续用 Python/pandas 做进一步统计分析。

---

## 5. UI 更新：P4/P5 选项与展示

**文件**：`src/medrag/ui/app.py`

### 新增两条 Pipeline 选项

```python
pipeline = st.radio(
    "Retrieval Pipeline",
    options=[
        "P1 · Dense Only",
        "P2 · Hybrid (Dense + Sparse)",
        "P3 · Hybrid + Reranker",
        "P4 · HyDE",            # 新增
        "P5 · Multi-Query",     # 新增
    ],
    index=2,
)
```

### 展示 HyDE 假设和 Multi-Query 改写

```python
with st.expander(f"📚 {len(chunks)} source documents  ·  [{mode_key}]"):
    if hypothesis:
        st.info(f"**HyDE hypothesis:** {hypothesis}")
    if sub_queries:
        st.info("**Query expansions:** " + " · ".join(sub_queries))
    ...
```

这个展示很重要：它让用户能**看到 LLM 在检索前做了什么**。对于 HyDE，可以检查生成的假设是否合理；对于 Multi-Query，可以看到查询扩展的方向是否有意义。

### `@st.cache_resource` 的延伸

Week 2 已经解释过 `@st.cache_resource`。Week 3 新增了 HyDE 和 MultiQuery：

```python
@st.cache_resource(show_spinner="Loading models…")
def load_resources():
    qdrant = QdrantClient(...)
    embedder = BGEM3Embedder(device="cpu")
    reranker = BGEReranker()
    hyde = HyDERetriever(qdrant, embedder)       # 新增
    multi_query = MultiQueryRetriever(qdrant, embedder)  # 新增
    return qdrant, embedder, reranker, hyde, multi_query
```

HyDE 和 MultiQuery 内部的 LLM（`ChatOllama`）在 `load_resources()` 里初始化时只是创建了连接配置，**不加载任何模型到内存**（Ollama 是独立进程）。所以这里的 cache 主要是避免重复创建对象，实际的模型加载由 Ollama 管理。

---

## 6. MCP Server

**文件**：`src/medrag/mcp_server/server.py`

### 什么是 MCP？

MCP（Model Context Protocol）是 Anthropic 提出的标准，允许 Claude Desktop / Claude Code 调用**外部 Python 函数作为工具**。Claude 看到一个 MCP 工具就像看到 `search_web` 或 `read_file` 一样，可以在对话中自主调用。

MedRAG-Agent 注册了两个工具：
- `retrieve(query, pipeline, k)` → 返回文档片段列表
- `ask(query, pipeline, k)` → 返回带引用的回答

### FastMCP 基础

```python
from fastmcp import FastMCP
mcp = FastMCP("MedRAG-Agent", instructions="...")

@mcp.tool()
def retrieve(query: str, pipeline: str = "p3", k: int = 5) -> list[dict]:
    """Retrieve top-k relevant medical document chunks for a query.
    
    Args:
        query: The medical question or search query.
        pipeline: Retrieval pipeline to use (p1/p2/p3/p4/p5).
        k: Number of documents to return (1-10).
    
    Returns:
        List of dicts with keys: rank, citation, score, snippet, ...
    """
    ...
```

**关键点**：
- 函数的 **docstring 就是工具描述**，Claude 会读这个决定什么时候调用工具
- **类型注解是强制性的**，FastMCP 用它生成 JSON Schema
- **默认值**让 Claude 在不确定时有合理的选择

### 懒加载单例（Lazy-Init Singletons）

MCP Server 在每次工具调用之间可能长时间空闲，**不应该在 import 时就加载模型**：

```python
_qdrant = None
_embedder = None

def _get_resources():
    global _qdrant, _embedder, ...
    if _qdrant is None:           # 第一次调用时初始化
        _qdrant = QdrantClient(...)
        _embedder = BGEM3Embedder(device="cpu")
        ...
    return _qdrant, _embedder, ...
```

这个 "lazy singleton" 模式确保：
1. 首次工具调用时才加载模型（约 5-10 秒）
2. 后续调用直接复用，无延迟
3. Server 进程本身启动极快

### 环境变量支持

```python
qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
_qdrant = QdrantClient(url=qdrant_url, timeout=30)
```

通过环境变量而不是硬编码 URL，方便在不同环境（本地开发、Docker、远程服务器）部署而不需要修改代码。

### 两个工具的设计差异

| 工具 | 返回格式 | 延迟 | 用途 |
|---|---|---|---|
| `retrieve` | `list[dict]` — 文档片段 | 较快 | 当 Claude 需要找资料但自己生成答案 |
| `ask` | `str` — 完整回答 | 较慢（含 LLM） | 当用户直接问医学问题时 |

为什么提供两个工具？Claude 有时需要先 `retrieve` 看看有什么，再决定如何回答；有时直接 `ask` 更高效。两个工具给 Claude 更多灵活性。

### 运行方式

```bash
# 开发调试（开启 MCP Inspector UI）
mcp dev src/medrag/mcp_server/server.py

# 安装到 Claude Desktop（运行一次）
mcp install src/medrag/mcp_server/server.py --name "MedRAG-Agent"
```

安装后，Claude Desktop 的对话里输入 "find me papers about PARP inhibitors" 时，Claude 会自动调用 `retrieve` 工具。

---

## 7. 调试经验

### 7.1 Lambda 闭包陷阱

在 `06_compare_pipelines.py` 的 `build_pipelines()` 里：

```python
# 错误写法：所有 lambda 共享同一个 p 变量引用
for pid in requested:
    available[pid] = lambda q, k: retrievers[pid].retrieve(q, k=k)  # 陷阱！

# 正确写法：用默认参数强制捕获当前值
available["p1"] = lambda q, k, _r=p1_retriever: _r.retrieve(q, k=k)
```

Python 的闭包是"延迟绑定"的：lambda 内的变量引用在**调用时**才查找，不是在**定义时**。如果在循环里创建 lambda，最终所有 lambda 都会用循环结束后的变量值。用默认参数绑定则在定义时就固定了值。

### 7.2 Multi-Query 温度和多样性的权衡

| temperature | 改写多样性 | 问题 |
|---|---|---|
| 0.1 | 低（基本重复原查询） | 没有改写的意义 |
| 0.5 | 中等（词汇和视角变化） | 推荐值 |
| 1.0 | 高（可能偏离原意） | 有时生成不相关的查询 |

### 7.3 `strip_thinking()` 的必要性

Qwen3-8B 有思维链（`<think>...</think>` 块）。如果不过滤，HyDE 生成的"假设文档"会包含大量内部推理文字，干扰 embedding：

```python
# raw response:
# <think>The question is about MRI resolution. I should write a passage about
# the typical spatial resolution values used in clinical practice...</think>
# 3T MRI typically achieves in-plane spatial resolutions of 0.5-1.0 mm...

hypothesis = strip_thinking(resp.content).strip()
# → "3T MRI typically achieves in-plane spatial resolutions of 0.5-1.0 mm..."
```

`strip_thinking` 在 `src/medrag/agent/utils.py` 里用一个简单的正则实现：

```python
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()
```

### 7.4 `last_hypothesis` / `last_queries` 属性

HyDE 和 MultiQuery 都暴露了最后一次调用的内部状态：

```python
class HyDERetriever:
    def _generate_hypothesis(self, query):
        ...
        self._last_hypothesis = hypothesis   # 保存
        return hypothesis
    
    @property
    def last_hypothesis(self) -> str | None:
        return getattr(self, "_last_hypothesis", None)
```

这不是生产代码的好模式（有状态），但对于 demo 和调试非常有用：Streamlit UI 可以直接显示 "LLM 生成的假设是什么"，帮助用户理解 P4 的工作原理。

---

## 8. 完整 Pipeline 演进图

```
Query (用户输入)
│
├─── P1: Dense Only ──────────────────────────────────────────── Top-k
│         BGE-M3 dense encode → Qdrant cosine ANN
│
├─── P2: Hybrid RRF ──────────────────────────────────────────── Top-k
│         BGE-M3 dense encode → Qdrant dense ANN ─┐
│         BGE-M3 sparse encode → Qdrant sparse ANN ┤ RRF → Top-k
│
├─── P3: Hybrid + Reranker ───────────────────────────────────── Top-k
│         P2 top-20 candidates → BGE Reranker cross-encoder → Top-k
│
├─── P4: HyDE ────────────────────────────────────────────────── Top-k
│         Qwen3-8B → Hypothetical passage → BGE-M3 dense → Qdrant ANN
│
└─── P5: Multi-Query ─────────────────────────────────────────── Top-k
          Qwen3-8B → [Q0, Q1, Q2, Q3]
          For each Qi: BGE-M3 dense → Qdrant ANN → ranked list_i
          RRF([list_0, list_1, list_2, list_3]) → Top-k
```

### 速度 vs. 质量权衡

| Pipeline | 典型延迟 | 预期质量 | 推荐场景 |
|---|---|---|---|
| P1 | ~0.3s | ★★☆ | 实时搜索、速度优先 |
| P2 | ~0.5s | ★★★ | 含专业术语的查询 |
| P3 | ~1.5s | ★★★★ | 最常用的生产配置 |
| P4 | ~3s   | ★★★★ | 复杂推理型问题 |
| P5 | ~4s   | ★★★★ | 多种表达都有相关文献时 |

> 延迟数字基于 CPU 推理（BGE-M3 + BGE Reranker 均在 CPU），实际值与机器性能相关。

---

## 附录：本周文件清单

| 文件 | 类型 | 功能 |
|---|---|---|
| `scripts/06_compare_pipelines.py` | 脚本 | P1-P5 对比评测，输出 MD 表格 + JSON |
| `src/medrag/retrieval/hyde.py` | 模块 | HyDERetriever：LLM 假设文档检索 |
| `src/medrag/retrieval/multi_query.py` | 模块 | MultiQueryRetriever：多查询 RRF 融合 |
| `src/medrag/mcp_server/server.py` | 模块 | FastMCP Server：retrieve + ask 工具 |
| `scripts/quick_demo.py` | 脚本 | 更新：支持 --mode p4/p5 |
| `src/medrag/ui/app.py` | 模块 | 更新：P4/P5 UI 选项 + 假设/改写展示 |
