# MedRAG-Agent Week 2 教程：混合检索、重排序与 Streamlit UI

> **目标读者**：已完成 Week 1、理解基本 RAG 流程，现在想深入学习 Week 2 新增的混合检索（Hybrid Retrieval）、交叉编码器重排序（Cross-Encoder Reranking）和 Streamlit Demo UI 的构建思路与实现细节。

---

## 目录

1. [Week 2 总览：从 P1 到 P3](#1-week-2-总览从-p1-到-p3)
2. [核心概念：Dense vs Sparse 检索](#2-核心概念dense-vs-sparse-检索)
3. [Qdrant 升级：支持稀疏向量](#3-qdrant-升级支持稀疏向量)
4. [索引构建升级：三阶段设计](#4-索引构建升级三阶段设计)
5. [混合检索器（hybrid.py）](#5-混合检索器hybridpy)
6. [RRF 融合算法详解](#6-rrf-融合算法详解)
7. [交叉编码器重排序（reranker.py）](#7-交叉编码器重排序rerankerpy)
8. [Streamlit Demo UI（app.py）](#8-streamlit-demo-uiapppy)
9. [CLI 验证：quick_demo.py](#9-cli-验证quick_demopy)
10. [单元测试（test_hybrid.py）](#10-单元测试test_hybridpy)
11. [踩坑记录：调试过程中的关键发现](#11-踩坑记录调试过程中的关键发现)
12. [完整数据流总结](#12-完整数据流总结)

---

## 1. Week 2 总览：从 P1 到 P3

Week 1 建立了 P1 管道（纯 Dense 检索）：用 BGE-M3 把文本编码成 1024 维的稠密向量，存入 Qdrant，查询时做余弦相似度检索。

Week 2 在此基础上新增两条更强的管道：

| 管道 | 方法 | 优势 | 劣势 |
|------|------|------|------|
| **P1** | BGE-M3 dense cosine | 语义相似度好，通用 | 对精确词汇（缩写、药品名）不敏感 |
| **P2** | Dense + Sparse (RRF 融合) | 同时捕捉语义相似 + 词汇精确匹配 | 比 P1 多一次稀疏查询 |
| **P3** | P2 top-20 → 交叉编码器重排序 | 精度最高，对 (query, doc) 联合建模 | 最慢，需逐对打分 |

**关键洞察**：这三个管道不是相互独立的，而是逐层"精炼"：
- P2 用两种检索方式撒更大的网（召回率更高）
- P3 用重排序模型对 P2 的候选集做精细打分（精度更高）

---

## 2. 核心概念：Dense vs Sparse 检索

### 2.1 Dense（稠密）检索

BGE-M3 将一段文本编码成一个固定长度（1024 维）的浮点向量。这个向量捕捉的是**语义**：意思相近的文本，它们的向量在空间中相近（余弦相似度高）。

**优势**：能做语义泛化，比如 "heart attack" 和 "myocardial infarction" 会有相近的向量。

**劣势**：对精确的词汇匹配不敏感。如果用户查询 "BRCA1"，dense 检索可能返回语义相关的癌症文档，但未必是含 "BRCA1" 这个精确词的文档。

### 2.2 Sparse（稀疏）检索

稀疏向量的思路来自传统 BM25（词频倒排索引）：一篇文档由它包含的词来表示，词的权重代表它在该文档中的重要性。

BGE-M3 的创新之处在于它提供了**神经稀疏**（Neural Sparse，也叫 `lexical_weights`）：

- 仍然是"词 → 权重"的稀疏表示
- 但权重由神经网络计算，而非简单的 TF-IDF
- 向量维度等于整个词汇表大小（约 250,000 维），但绝大多数维度为 0
- 每个文本只有几十到几百个非零维度（"稀疏"名字的由来）

**BGE-M3 的独特性**：它是目前极少数能同时输出 dense、sparse 两种表示的模型，而且一次前向传播就能得到两者，不需要跑两个模型。

```python
# BGE-M3 一次 encode 可以同时返回 dense 和 sparse
out = model.encode(
    texts,
    return_dense=True,   # 1024维浮点向量
    return_sparse=True,  # lexical_weights: {token_id: weight, ...}
)
dense_vec = out["dense_vecs"]       # shape: (N, 1024)
sparse_weights = out["lexical_weights"]  # list of dict
```

**`lexical_weights` 的格式**：

```python
# 一段文本的稀疏表示
{
    "3034": 0.712,   # token id 3034 对应某个词，权重 0.712
    "149357": 0.483,
    "7892": 0.291,
    # ... 通常有 20-200 个非零项
}
```

键是 **字符串形式的 token ID**（XLM-RoBERTa tokenizer 的 vocabulary index），值是该 token 的重要性权重。

---

## 3. Qdrant 升级：支持稀疏向量

### 3.1 为什么需要修改 Qdrant collection？

Week 1 的 collection 只存储了 dense 向量。要支持稀疏检索，必须在创建 collection 时声明一个额外的 **sparse vector 字段**。

Qdrant 支持在同一个 collection 里存多种向量，通过**命名向量（Named Vectors）**区分：

```
collection "medrag_text"
├── 每个 point
│   ├── vectors["dense"]  : float[1024]   （原来就有）
│   ├── vectors["sparse"] : SparseVector  （新增）
│   └── payload           : {chunk_id, text, source, ...}
```

### 3.2 代码实现：`qdrant_setup.py`

```python
client.create_collection(
    collection_name=name,
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False),  # 索引存在内存中，查询更快
        ),
    },
)
```

**关键设计点**：
- `vectors_config` 和 `sparse_vectors_config` 是两个独立的配置项
- Dense 使用 COSINE 距离（适合归一化的嵌入向量）
- Sparse 不需要指定维度（维度由实际数据自动确定）
- `on_disk=False`：稀疏索引放内存，避免磁盘 IO 瓶颈

### 3.3 代码实现：`indexer.py` 中的 SparseVector 格式转换

Qdrant 需要的 `SparseVector` 格式是两个平行数组：

```
SparseVector(
    indices=[3034, 149357, 7892, ...],  # token id 的整数列表
    values=[0.712, 0.483, 0.291, ...],  # 对应权重的浮点列表
)
```

而 BGE-M3 输出的 `lexical_weights` 是 `{"3034": 0.712, ...}` 字典（键为字符串）。

转换函数：

```python
def _to_sparse_vector(weights: dict) -> SparseVector:
    if not weights:
        return SparseVector(indices=[0], values=[0.0])  # Qdrant 不接受空 SparseVector
    return SparseVector(
        indices=[int(k) for k in weights],     # 字符串键 → 整数
        values=[float(v) for v in weights.values()],  # 确保是 Python float
    )
```

**注意事项**：
- 键必须转为 `int`（`int("3034")` → `3034`）
- 值必须转为 Python 原生 `float`（BGE-M3 在 GPU fp16 模式下输出 `numpy.float16`，JSON 无法序列化它，必须显式转换）
- 空字典需要特殊处理（某些极短的文本可能没有任何有效 token）

写入 Qdrant 时，将 dense 和 sparse 打包在一个 `PointStruct` 里：

```python
PointStruct(
    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
    vector={
        "dense": vec.tolist(),           # 1024维列表
        "sparse": _to_sparse_vector(sparse_weights[i]),  # SparseVector 对象
    },
    payload={"chunk_id": ..., "text": ..., "source": ...},
)
```

---

## 4. 索引构建升级：三阶段设计

`scripts/04_build_index.py` 采用三阶段（Phase）设计，每个阶段独立可重跑，中间结果缓存到磁盘。

### 4.1 为什么要三阶段？

Dense 编码（Phase embed）和 Sparse 编码（Phase sparse）都是计算密集型任务，各需要十几到几十分钟。如果某个阶段失败，不应该从头重跑。三阶段设计允许独立重跑：

```
data/index_cache/
├── dense.npy      ← Phase embed 的输出（44768 × 1024 的 float32 矩阵）
├── chunks.jsonl   ← 所有文本块的元数据（chunk_id, text, source...）
└── sparse.jsonl   ← Phase sparse 的输出（每行一个 lexical_weights dict）
```

### 4.2 Phase embed（Week 1 已有）

用 BGE-M3 的 dense 模式对所有文本块编码，保存为 `.npy` 文件。

```python
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
out = model.encode(texts, batch_size=4, return_dense=True, return_sparse=False, ...)
dense = np.array(out["dense_vecs"], dtype=np.float32)
np.save(DENSE_FILE, dense)
```

### 4.3 Phase sparse（Week 2 新增）

用 BGE-M3 的 sparse 模式编码，保存为 `.jsonl` 文件（每行一个 JSON 字典）。

```python
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
model.model = model.model.to("cuda")  # 见调试记录

out = model.encode(texts, batch_size=256, return_dense=False, return_sparse=True, ...)
sparse_weights = out["lexical_weights"]  # list[dict]

with SPARSE_FILE.open("w") as f:
    for w in sparse_weights:
        # 必须将 float16 转为 Python float，否则 json.dumps 报错
        f.write(json.dumps({k: float(v) for k, v in w.items()}) + "\n")
```

**稀疏编码的数据量**：

| 文件 | 内容 | 大小（估算） |
|------|------|-------------|
| `dense.npy` | 44768 × 1024 × 4 bytes | ~183 MB |
| `sparse.jsonl` | 44768 行，每行约 500 bytes | ~22 MB |

### 4.4 Phase index

加载缓存文件，重建 Qdrant collection：

```python
def phase_index(chunks, dense, sparse_weights):
    client = QdrantClient(url="http://localhost:6333", timeout=120)
    create_collection(client, "medrag_text", recreate=True)  # 删除旧的，新建带 sparse 的
    index_chunks(client, chunks, dense, sparse_weights=sparse_weights, batch=256)
    count = client.count(collection_name="medrag_text").count
    print(f"[done] qdrant points: {count}")
```

### 4.5 智能跳过逻辑

`--phase all` 模式会检查缓存文件是否已存在，避免重复计算：

```python
if args.phase == "all" and DENSE_FILE.exists():
    # 直接加载缓存，跳过 embed 阶段
    dense = np.load(DENSE_FILE)
```

---

## 5. 混合检索器（hybrid.py）

**文件**：`src/medrag/retrieval/hybrid.py`

### 5.1 设计思路

`HybridRetriever` 的核心思想：对同一个 query，**同时向 Qdrant 发两个检索请求**（一个 dense，一个 sparse），再用 RRF 把两个排名列表融合成一个。

```
query
├──→ BGE-M3 encode → dense_vec (1024维) ──→ Qdrant dense search → 排名列表 A
└──→ BGE-M3 encode → sparse_weights (字典) → Qdrant sparse search → 排名列表 B
                                                          ↓
                                               RRF 融合 → 最终排名
```

**一次 encode 同时得到两种表示**：BGE-M3 的优势是 dense 和 sparse 是同一次前向传播的两个输出头，不需要运行两次模型：

```python
enc = self.embedder.encode([query], return_sparse=True)
dense_vec = enc["dense"][0].tolist()    # 从同一次 encode 得到
sparse_weights = enc["sparse"][0]        # 从同一次 encode 得到
```

### 5.2 Qdrant 的命名向量查询

查询时，通过 `using=` 参数指定使用哪个向量字段：

```python
# Dense 检索
dense_result = self.qdrant.query_points(
    collection_name=self.collection,
    query=dense_vec,   # 1024维浮点列表
    using="dense",     # 指定使用 dense 向量字段
    limit=self.candidate_k,
)

# Sparse 检索
sparse_result = self.qdrant.query_points(
    collection_name=self.collection,
    query=_to_sparse_vec(sparse_weights),  # SparseVector 对象
    using="sparse",    # 指定使用 sparse 向量字段
    limit=self.candidate_k,
)
```

### 5.3 payload 缓存的必要性

两次检索返回的是不同的 point 集合（各自 top-N），RRF 融合后需要根据 `chunk_id` 查到对应的文本内容。

如果不做缓存，融合后需要再向 Qdrant 发查询请求（额外 IO）。设计中选择在第一次检索时就把 payload 缓存到内存字典中：

```python
id_to_point: dict[str, object] = {}

for p in dense_result.points:
    cid = p.payload["chunk_id"]
    dense_ranking.append(cid)
    id_to_point[cid] = p          # 缓存 dense 命中的 payload

for p in sparse_result.points:
    cid = p.payload["chunk_id"]
    sparse_ranking.append(cid)
    if cid not in id_to_point:
        id_to_point[cid] = p      # 仅 sparse 命中的也缓存（sparse-only hits）
```

注意：sparse-only 命中（只被稀疏检索找到，dense 未找到的文档）同样需要缓存，否则 RRF 融合后找不到它们的 payload。

### 5.4 完整的 retrieve 流程

```python
def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
    # 1. 编码 query（一次 forward pass，同时得到 dense 和 sparse）
    enc = self.embedder.encode([query], return_sparse=True)
    dense_vec = enc["dense"][0].tolist()
    sparse_weights = enc["sparse"][0]

    # 2. 并行向 Qdrant 发两个检索请求
    dense_result = self.qdrant.query_points(..., query=dense_vec, using="dense", ...)
    sparse_result = self.qdrant.query_points(..., query=_to_sparse_vec(sparse_weights), using="sparse", ...)

    # 3. 构建排名列表 + 缓存 payload
    dense_ranking = [p.payload["chunk_id"] for p in dense_result.points]
    sparse_ranking = [p.payload["chunk_id"] for p in sparse_result.points]

    # 4. RRF 融合
    fused = _reciprocal_rank_fusion([dense_ranking, sparse_ranking])

    # 5. 取 top-k，用缓存的 payload 重建 RetrievedChunk
    return [RetrievedChunk(chunk_id, text, score=rrf_score, ...) for chunk_id, rrf_score in fused[:k]]
```

---

## 6. RRF 融合算法详解

**函数**：`_reciprocal_rank_fusion()` 在 `hybrid.py` 中

### 6.1 为什么需要 RRF？

Dense 检索的分数是余弦相似度（范围 0~1），Sparse 检索的分数是稀疏向量点积（无上界，可能是 0~100+）。这两种分数**尺度完全不同，不能直接相加**。

RRF 的核心思想：**不使用原始分数，只使用排名位置**。排名第1的文档获得 1/(60+1) 的分，排名第2的获得 1/(60+2)，以此类推。对所有排名列表求和。

### 6.2 公式

$$\text{score}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d) + 1}$$

其中：
- $d$：文档
- $i$：遍历所有排名列表（本项目中是 dense 列表和 sparse 列表）
- $k = 60$：平滑常数（来自 Cormack et al. 2009 原始论文）
- $\text{rank}_i(d)$：文档 $d$ 在第 $i$ 个列表中的排名（从 0 开始）

**加 1 的原因**：rank 从 0 开始，如果不加 1，排名第0的文档得分为 1/60，而非 1/61，这与 0-indexed 的直觉略有差异，加 1 使第1名的分数为 1/(k+1)。

### 6.3 手工验算示例

假设 dense 排名：`[A, B, C]`，sparse 排名：`[B, A, D]`

```
score(A) = 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
score(B) = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
score(C) = 1/(60+2+1) = 1/63 = 0.01587
score(D) = 1/(60+2+1) = 1/63 = 0.01587
```

A 和 B 分数相同（A 在 dense 第1，sparse 第2；B 在 dense 第2，sparse 第1）。C 和 D 只出现在一个列表中，分数更低。

### 6.4 代码实现

```python
def _reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**代码特点**：
- `defaultdict(float)`：没有出现过的 chunk_id 自动初始化为 0，避免 KeyError
- 支持任意多个排名列表（不仅仅是两个）
- 输出是 `[(chunk_id, score), ...]`，按分数降序排列

### 6.5 为什么 k=60？

这个值来自原始 RRF 论文的经验调参结论，对多数检索任务都表现良好。k 越大，排名靠前和靠后的文档分差越小（更平滑）；k 越小，排名靠前的文档获得更多奖励（更激进）。

---

## 7. 交叉编码器重排序（reranker.py）

**文件**：`src/medrag/retrieval/reranker.py`

### 7.1 Bi-Encoder vs Cross-Encoder

理解重排序，需要先理解两种编码器架构的本质区别：

| 架构 | 工作方式 | 特点 |
|------|---------|------|
| **Bi-Encoder**（双编码器） | query 和 document 分别独立编码，最后计算向量相似度 | 可以预计算 doc 向量（离线索引），查询快 |
| **Cross-Encoder**（交叉编码器） | query 和 document **拼接**后一起输入模型，模型直接输出相关性分数 | 不能预计算，每次查询都要把 query+doc 对跑一次完整的 transformer |

BGE-M3 是 Bi-Encoder：文档向量在索引时就算好了，查询时只需要编码 query，然后做向量检索。

`bge-reranker-v2-m3` 是 Cross-Encoder：它接受 `[query, document]` 对，让 query 和 document 的 token 在 attention 中互相"看"到彼此，因此可以捕捉更精细的相关性信号。

**直觉上的差别**：Bi-Encoder 就像只看标题就决定一本书是否相关；Cross-Encoder 是把书和你的问题一起读，再判断相关性。后者精度更高，但不能预计算，所以只用于精排少量候选（P3 对 P2 的 top-20 重排序）。

### 7.2 重排序的分数含义

Cross-Encoder 输出的分数是**原始 logit**（未归一化的实数）：
- 正数：相关
- 负数：不相关
- 数值越大越相关

这就是为什么 P3 输出中会看到负数分数（例如 `-0.0556`），这是正常的，表示"略微不相关"。

### 7.3 代码实现

```python
class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device="cpu", ...):
        self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)

    def rerank(self, query, chunks, top_k=5):
        # 构建 [query, doc] 对
        pairs = [[query, c.text] for c in chunks]

        # 批量打分（cross-encoder 对每对进行完整 forward pass）
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)

        # 按分数降序排列，取 top_k
        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])

        # 创建新的 RetrievedChunk 对象（不修改原来的 P2 候选列表）
        return [
            RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=float(s), payload=c.payload)
            for s, c in ranked[:top_k]
        ]
```

**设计细节**：
- `pairs = [[query, c.text] for c in chunks]`：FlagReranker 接受 list of [query, doc] 格式
- 创建**新的** `RetrievedChunk` 对象而不是修改原来的 — 避免影响调用方（P2 候选集保持不变）
- `float(s)`：将 numpy 分数转为 Python float，确保可序列化

### 7.4 P3 管道的完整流程

```python
# P3 = P2 top-20 → reranker top-5
retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
candidates = retriever.retrieve(query, k=20)   # 先用 P2 召回 20 个候选
chunks = reranker.rerank(query, candidates, top_k=5)  # 再用重排序精选 5 个
```

为什么用 P2 而不是 P1 做候选召回？P2 的召回集合已经包含了稀疏检索的命中，语义+词汇双覆盖，候选质量比纯 dense 更好，重排序的起点更高。

---

## 8. Streamlit Demo UI（app.py）

**文件**：`src/medrag/ui/app.py`

### 8.1 Streamlit 的工作原理

Streamlit 的核心机制是**每次用户交互都重新执行整个 Python 脚本**。这很简单直观，但也带来一个问题：重型资源（大模型、数据库连接）不能在每次交互时都重新加载。

解决方案：`@st.cache_resource` 装饰器。它让函数只在**第一次被调用时**真正执行，之后的调用直接返回缓存的对象：

```python
@st.cache_resource(show_spinner="Loading models…")
def load_resources():
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")
    reranker = BGEReranker()  # bge-reranker-v2-m3，首次加载需要几十秒
    return qdrant, embedder, reranker

# 全局调用 — 整个 session 期间只加载一次
qdrant, embedder, reranker = load_resources()
```

**为什么不缓存 retriever？**

`DenseRetriever` 和 `HybridRetriever` 是轻量级对象（只包含引用，不加载模型），每次使用时临时创建即可。缓存它们反而会在 pipeline 切换时造成混淆。

### 8.2 Session State：多轮对话历史

Streamlit 的 `st.session_state` 是一个持久化字典，在同一浏览器 session 内跨多次脚本执行保持状态。

```python
if "history" not in st.session_state:
    st.session_state.history = []  # 初始化为空列表

# 每轮对话完成后，追加到历史
st.session_state.history.append({
    "query": query,
    "answer": answer,
    "chunks": chunks,
    "pipeline": mode_key,
})
```

渲染历史时，遍历这个列表展示之前的对话轮次：

```python
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander(f"📚 {len(turn['chunks'])} source documents"):
            ...
```

### 8.3 侧边栏控件

```python
with st.sidebar:
    pipeline = st.radio(
        "Retrieval Pipeline",
        options=["P1 · Dense Only", "P2 · Hybrid (Dense + Sparse)", "P3 · Hybrid + Reranker"],
        index=2,  # 默认选中 P3
    )
    top_k = st.slider("Top-K documents", min_value=3, max_value=10, value=5)
```

从 pipeline 选项字符串提取模式键：

```python
mode_key = pipeline.split("·")[0].strip()  # "P1 · Dense Only" → "P1"
```

### 8.4 主查询流程

```python
query = st.chat_input("Ask a medical question…")

if query:
    with st.spinner(f"Retrieving [{mode_key}]…"):
        if mode_key == "P1":
            retriever = DenseRetriever(qdrant, embedder)
            chunks = retriever.retrieve(query, k=top_k)
        elif mode_key == "P2":
            retriever = HybridRetriever(qdrant, embedder)
            chunks = retriever.retrieve(query, k=top_k)
        else:  # P3
            retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
            candidates = retriever.retrieve(query, k=20)
            chunks = reranker.rerank(query, candidates, top_k=top_k)

    with st.spinner("Generating answer (Qwen3-8B)…"):
        answer = generate_answer(query, chunks)
```

**两个 spinner 分别标注**：分开显示"检索中"和"生成中"，用户可以看到瓶颈在哪个阶段。

### 8.5 来源文档的展示

```python
with st.expander(f"📚 {len(chunks)} source documents  ·  [{mode_key}]"):
    for i, c in enumerate(chunks, 1):
        col_meta, col_text = st.columns([1, 3])
        with col_meta:
            st.metric(label=f"#{i} {c.citation}", value=f"{c.score:.4f}")
        with col_text:
            st.caption(c.text[:350] + ("…" if len(c.text) > 350 else ""))
        st.divider()
```

- `st.columns([1, 3])`：左列宽1份（元信息），右列宽3份（文本预览）
- `st.metric`：用醒目的大字体展示分数
- 文本截断至 350 字符防止页面过长

### 8.6 启动方式

```powershell
# scripts/run_ui.ps1
$streamlit = "C:\Users\lijingshan\.conda\envs\medrag\Scripts\streamlit.exe"
& $streamlit run src/medrag/ui/app.py --server.port 8501
```

访问 `http://localhost:8501` 即可使用。

---

## 9. CLI 验证：quick_demo.py

**文件**：`scripts/quick_demo.py`

在 UI 之外，`quick_demo.py` 提供了更方便的命令行验证入口，支持 `--mode p1/p2/p3` 参数切换管道：

```python
parser.add_argument("--mode", choices=["p1", "p2", "p3"], default="p3")
parser.add_argument("--k", type=int, default=5)
```

运行示例：

```bash
python scripts/quick_demo.py "What is the typical resolution of 3T MRI?" --mode p3

# 输出：
# [1] score=0.7325  PMC:doc4
#     The measurements were performed...
# [2] score=-0.0556  PMC:doc326
#     ...
# ANSWER:
# The typical resolution of 3T MRI in clinical practice is...
```

注意 P3 的分数格式：第1个文档 0.7325（正，明确相关），第2个 -0.0556（负，轻微相关），这是 cross-encoder 的原始 logit，符合预期。

---

## 10. 单元测试（test_hybrid.py）

**文件**：`tests/test_hybrid.py`

RRF 是 P2/P3 管道的核心逻辑，但不依赖任何外部服务（不需要 Qdrant、不需要模型），非常适合单元测试。

### 10.1 测试设计思路

5 个测试覆盖 RRF 的所有关键行为：

```python
def test_rrf_top_item_wins():
    """双列表排名第一的文档应获得最高分"""
    list1 = ["A", "B", "C"]
    list2 = ["A", "C", "D"]
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    assert fused["A"] > fused["B"]  # A 在两个列表都排第1
    assert fused["A"] > fused["D"]

def test_rrf_union_of_lists():
    """融合结果应包含两个列表的并集"""
    list1 = ["A", "B"]
    list2 = ["C", "D"]
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    assert set(fused) == {"A", "B", "C", "D"}

def test_rrf_cross_list_bonus():
    """同时出现在两个列表的文档应优于只出现在一个列表的"""
    list1 = ["A", "B"]
    list2 = ["B", "C"]  # B 同时在两个列表
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    assert fused["B"] > fused["A"]  # B 双重加分
    assert fused["B"] > fused["C"]

def test_rrf_single_list_preserves_order():
    """单列表时，RRF 排名应与原始排名一致"""
    ...

def test_rrf_empty_lists():
    """边界条件：空列表应返回空结果"""
    assert _reciprocal_rank_fusion([]) == []
    assert _reciprocal_rank_fusion([[]]) == []
```

运行：

```bash
pytest tests/test_hybrid.py -v
# 全部 5 个测试通过
```

**测试策略的价值**：这些测试和 Qdrant、BGE-M3 完全解耦，几秒内就能跑完，可以在没有 GPU 的环境（如 CI）中验证 RRF 逻辑正确性。

---

## 11. 踩坑记录：调试过程中的关键发现

Week 2 实现过程中遇到了几个值得记录的坑，理解它们有助于理解代码中一些"奇怪"的写法。

### 11.1 坑1：BGE-M3 的 `device=` 参数被忽略

**现象**：在 `BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")` 初始化后，模型实际运行在 CPU 上（GPU 利用率 0%）。

**根本原因**：`BGEM3FlagModel` 的 `__init__` 接受 `device=` 参数，但内部实现（新版 FlagEmbedding）不把它正确传递给 PyTorch 模型加载。模型默认在 CPU 上初始化。

**解决方案**：初始化后手动将模型搬到 GPU：

```python
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
model.model = model.model.to("cuda")  # model.model 是 EncoderOnlyEmbedderM3ModelForInference
```

**为什么有效**：`BGEM3FlagModel.encode_single_device()` 内部调用 `self.model.to(device)`，但 `device` 取自 `self.target_devices[0]`，默认为 `"cuda:0"`（在有 GPU 的机器上）。手动 `.to("cuda")` 确保参数在 CUDA 上，后续 forward pass 自然走 GPU。

### 11.2 坑2：僵尸进程占用 GPU，导致 OOM 后 batch_size 自动缩减

**现象**：手动 `.to("cuda")` 后，GPU 内存从 134 MiB 升到 1381 MiB（模型已上 GPU），但推理速度仍然很慢（7s/batch），`nvidia-smi` 显示 GPU 利用率 0%。

**根本原因**：
1. 之前有多个 Python 进程（"僵尸进程"）在后台存活，占用了大部分 GPU 显存
2. `encode_single_device` 中有 OOM 自动降级逻辑：如果 batch_size=256 的第一批（最长的文本）触发 OOM，batch_size 会自动降到 3/4 → 反复降，最终降到 4
3. batch_size=4 意味着 44768/4 = 11192 批次，即使是 GPU 也需要数小时

```python
# FlagEmbedding 内部的 OOM 降级逻辑（已阅读源码确认）
while flag is False:
    try:
        outputs = model(batch[:batch_size])
        flag = True
    except torch.cuda.OutOfMemoryError:
        batch_size = batch_size * 3 // 4  # 每次降到 75%
```

**解决方案**：清理所有僵尸进程后重新运行，GPU 显存从 1381 MiB 降回 134 MiB（干净状态），batch_size=256 正常运行，175 批次 × 5-6 秒/批 ≈ 17 分钟完成。

### 11.3 坑3：float16 无法 JSON 序列化

**现象**：sparse 编码完成（`[sparse] encoded 44768 items`），但保存时抛出：

```
TypeError: Object of type float16 is not JSON serializable
```

**原因**：BGE-M3 在 fp16 模式下运行，`lexical_weights` 中的值是 `numpy.float16` 类型，而 Python 标准库的 `json.dumps` 只支持 Python 原生 `float`（即 64 位浮点）。

**解决方案**：保存时显式转换：

```python
# 错误写法
f.write(json.dumps(w) + "\n")

# 正确写法
f.write(json.dumps({k: float(v) for k, v in w.items()}) + "\n")
```

**举一反三**：任何时候把 numpy 数值写入 JSON，都应该先转为 Python 原生类型（`float()`、`int()`）。

### 11.4 坑4：Windows localhost 解析到 IPv6

这个坑在 Week 1 已修复，但值得在这里记录：

**现象**：Ollama 的生成请求返回 `WinError 10049`（地址无法分配）。

**原因**：Windows 上 `localhost` 默认解析到 `::1`（IPv6），但 Ollama 只监听 `127.0.0.1`（IPv4）。

**修复**：

```python
# src/medrag/agent/generator.py
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://127.0.0.1:11434",  # 不用 localhost，强制 IPv4
)
```

---

## 12. 完整数据流总结

### 12.1 索引构建阶段（离线，一次性）

```
原始语料 (PubMed + PMC)
        ↓ chunker.py
44768 个文本块
        ↓
    ┌───────────────────────────────────┐
    │       04_build_index.py           │
    │                                   │
    │  Phase embed (Week 1)             │
    │  BGE-M3 → dense.npy (183MB)       │
    │                                   │
    │  Phase sparse (Week 2)            │
    │  BGE-M3 → sparse.jsonl (22MB)     │
    │    (fp16, GPU, batch=256, ~17min) │
    │                                   │
    │  Phase index                      │
    │  Qdrant upsert(dense+sparse)      │
    └───────────────────────────────────┘
              ↓
    Qdrant collection "medrag_text"
    (44768 points, each with dense+sparse vectors)
```

### 12.2 查询阶段（在线）

```
用户输入 query
     │
     ├─── P1 管道 ──────────────────────────────────────────────────────┐
     │    BGEM3Embedder.encode(query) → dense_vec                        │
     │    Qdrant.query_points(using="dense") → top-K                    │
     │    generate_answer(query, chunks) → LLM 回答                     │
     │                                                                   │
     ├─── P2 管道 ──────────────────────────────────────────────────────┤
     │    BGEM3Embedder.encode(query) → dense_vec + sparse_weights       │
     │    Qdrant.query_points(using="dense") → dense_ranking             │
     │    Qdrant.query_points(using="sparse") → sparse_ranking           │
     │    RRF融合 → top-K                                                │
     │    generate_answer(query, chunks) → LLM 回答                     │
     │                                                                   │
     └─── P3 管道 ──────────────────────────────────────────────────────┘
          P2 top-20 候选
          BGEReranker.rerank(query, candidates) → top-K（cross-encoder）
          generate_answer(query, chunks) → LLM 回答
```

### 12.3 核心文件索引

| 文件 | 职责 |
|------|------|
| `src/medrag/index/qdrant_setup.py` | 创建含 dense+sparse 的 Qdrant collection |
| `src/medrag/index/indexer.py` | 将 Chunk + dense/sparse 向量写入 Qdrant |
| `scripts/04_build_index.py` | 三阶段索引构建脚本（embed/sparse/index） |
| `src/medrag/retrieval/hybrid.py` | HybridRetriever（P2）+ RRF 融合算法 |
| `src/medrag/retrieval/reranker.py` | BGEReranker（P3 重排序） |
| `src/medrag/ui/app.py` | Streamlit Demo UI |
| `scripts/quick_demo.py` | CLI 验证工具（--mode p1/p2/p3） |
| `tests/test_hybrid.py` | RRF 单元测试（5个，全部通过） |

---

*文档生成于 Week 2 完成后，对应 git commit `509193b`。*
