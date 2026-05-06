# Week 2 开发计划：混合检索 + 重排序 + Golden Dataset

| 字段 | 内容 |
|---|---|
| 时间预算 | ~15-18 h / 7 天 |
| 上周交付 | 44768 points (dense only) · P1 baseline 已跑通 |
| 本周交付 | P2 混合检索 + P3 重排序 · 50 题 golden dataset · Streamlit Demo |
| 核心难点 | Qdrant sparse 向量格式转换 · RRF 融合逻辑 · Golden dataset 不能"泄题" |

---

## 目录

1. [Week 2 全局目标与交付物](#1-week-2-全局目标与交付物)
2. [技术背景：为什么要做混合检索](#2-技术背景为什么要做混合检索)
3. [Day 1：稀疏向量索引（Qdrant 升级）](#3-day-1稀疏向量索引qdrant-升级)
4. [Day 2：混合检索 + RRF 融合（P2）](#4-day-2混合检索--rrf-融合p2)
5. [Day 3：Cross-Encoder 重排序（P3）](#5-day-3cross-encoder-重排序p3)
6. [Day 4：Golden Dataset 构建](#6-day-4golden-dataset-构建)
7. [Day 5：Pipeline 对比评测（P1 vs P2 vs P3）](#7-day-5pipeline-对比评测p1-vs-p2-vs-p3)
8. [Day 6：Streamlit Demo UI](#8-day-6streamlit-demo-ui)
9. [Day 7：文档 + 提交](#9-day-7文档--提交)
10. [踩坑预防手册](#10-踩坑预防手册)
11. [可裁剪项](#11-可裁剪项)

---

## 1. Week 2 全局目标与交付物

### 从 P1 到 P3：三个 Pipeline

整个 Week 2 围绕一个主线——在 Week 1 的 dense-only（P1）基础上逐步叠加两个检索增强：

```
P1: 纯 Dense 检索
    Query → BGE-M3 dense 向量 → Qdrant cosine top-K

P2: 混合检索（Dense + Sparse + RRF 融合）
    Query → BGE-M3 dense 向量 ──┐
    Query → BGE-M3 sparse 向量 ─┼→ RRF 融合 → merged top-20
                                 ┘

P3: 混合检索 + 重排序
    P2 结果 top-20 → bge-reranker-v2-m3 cross-encoder → top-5
```

每个 Pipeline 跑同一个 50 题 golden dataset，输出对比表。这是 README 的核心展示物，也是面试里讲"我做了消融实验"的依据。

### 本周交付清单

| 交付物 | 位置 | 验收标准 |
|---|---|---|
| Qdrant collection 升级（dense + sparse）| Qdrant `medrag_text` | `sparse_vectors_config` 存在，point count 不变 |
| `retrieval/hybrid.py` | src/medrag/retrieval/ | RRF 结果与 P1 结果分布不同，sparse 确实在起作用 |
| `retrieval/reranker.py` | src/medrag/retrieval/ | reranker 分数与 dense score 有差异（不是简单复制）|
| `eval/golden_dataset.jsonl` | eval/ | 50 题，含 20 事实 + 20 综合 + 10 多跳，人工审核通过 |
| `scripts/06_compare_pipelines.py` | scripts/ | 输出 P1/P2/P3 三列对比表 |
| `src/medrag/ui/app.py` | src/medrag/ui/ | Streamlit 能跑，能切换 pipeline 模式 |
| `docs/week2_tutorial.md` | docs/ | 本计划中每天的实现细节都写进去 |

---

## 2. 技术背景：为什么要做混合检索

### Dense（密集检索）的局限

Dense 检索用语义向量做相似度匹配，擅长处理语义等价的表达（比如"心脏磁共振"≈ "cardiac MRI"）。但它有一个致命弱点：**对精确词汇的敏感度低**。

举例：医学语料中有大量专有缩写，比如 `TAVR`（经导管主动脉瓣置换术）。如果用户问"TAVR 适应症"，dense 向量会把这个缩写映射到一个语义空间里，但如果训练时 TAVR 出现次数少，它的向量就不准，检索失败。

### Sparse（稀疏检索）的优势

BGE-M3 的 sparse 输出是"神经 BM25"：模型给每个词一个权重，精确匹配时权重高。TAVR → 权重高；语义相近但拼写不同的词 → 权重低。这恰好弥补了 dense 的盲区。

Sparse 的格式是 `lexical_weights`：一个字典，key 是 token ID，value 是浮点权重：

```python
# 示例输出（BGE-M3 sparse）
{"lexical_weights": {
    12345: 0.85,   # token "TAVR" 的权重
    67890: 0.43,   # token "aortic" 的权重
    ...
}}
```

Qdrant 的 SparseVector 格式需要两个并列数组：

```python
SparseVector(
    indices=[12345, 67890, ...],
    values=[0.85, 0.43, ...]
)
```

### RRF（倒数排名融合）

两路检索各自返回一个排名列表，如何合并？RRF 是最简单也最鲁棒的方法：

```python
RRF_score(doc) = sum(1 / (k + rank_in_list_i))
```

其中 `k=60` 是平滑参数（防止第 1 名独大），`rank` 从 0 开始。

**为什么不用加权平均分数？** 因为 dense 和 sparse 的分数尺度完全不同（dense 是余弦相似度 0-1，sparse 是权重乘积，可能 > 1），直接加权会产生 bias。RRF 只依赖排名，天然尺度无关。

### Cross-Encoder 重排序

Dense/Sparse 检索用的是双塔模型（Bi-encoder）：query 和 document 分别编码，做相似度。这很快，但精度有上限——两个向量只能捕捉全局语义，无法建模 query 和 document 之间的细粒度交互。

Cross-Encoder 把 `[query, document]` 拼在一起送进模型，直接输出一个相关性分数。这样 query 里的每个词都能和 document 里的每个词做 attention，准确度高得多。代价是：每个 query 需要和每个候选 document 分别跑一次模型，不能预计算，所以只适合对 top-20 重排序，不适合全量检索。

---

## 3. Day 1：稀疏向量索引（Qdrant 升级）

**目标**：把 sparse 向量加进 Qdrant，升级 `qdrant_setup.py`、`embedder.py`、`indexer.py`。  
**时间预算**：3 h  
**验收**：Qdrant collection 有 `sparse_vectors_config`，重新索引后 point count 仍为 44768。

### Step 1.1：理解现状（15 min）

当前 Qdrant collection 只有 `dense` 向量：

```python
# 现有配置（qdrant_setup.py）
vectors_config={
    "dense": VectorParams(size=1024, distance=Distance.COSINE),
}
# 缺少 sparse_vectors_config
```

稀疏向量需要单独的 `sparse_vectors_config` 字段，不能往现有 collection 里追加。**必须重建 collection**。

好消息：Week 1 的 dense.npy 缓存（`data/index_cache/dense.npy`）还在，dense 部分不用重算。只需新增一轮 sparse 编码。

### Step 1.2：升级 `qdrant_setup.py`

文件路径：`src/medrag/index/qdrant_setup.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    SparseVectorParams, SparseIndexParams,
)


def create_collection(
    client: QdrantClient,
    name: str = "medrag_text",
    recreate: bool = False,
) -> None:
    if client.collection_exists(name):
        if not recreate:
            print(f"[qdrant] '{name}' already exists, skipping")
            return
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)  # 全量放内存，检索更快
            ),
        },
    )
    print(f"[qdrant] created collection '{name}' (dense + sparse)")
```

**关键变化**：加了 `sparse_vectors_config`，使用 `SparseVectorParams`。`on_disk=False` 表示 sparse 倒排索引放内存——44768 条数据量不大，内存够用，换取检索速度。

### Step 1.3：升级 `embedder.py`（理解 sparse 输出格式）

文件路径：`src/medrag/index/embedder.py`

```python
import numpy as np
from typing import Literal
from FlagEmbedding import BGEM3FlagModel


class BGEM3Embedder:
    def __init__(self, device: Literal["cuda", "cpu"] = "cpu", use_fp16: bool = False):
        try:
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, devices=device)
        except TypeError:
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
        return_sparse: bool = False,   # 新增 sparse 开关
    ) -> dict:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
            max_length=512,
        )
        result = {"dense": np.array(out["dense_vecs"], dtype=np.float32)}
        if return_sparse:
            # lexical_weights 是 list[dict[str, float]]
            # 每个 dict 的 key 是 token ID（字符串），value 是权重
            result["sparse"] = out["lexical_weights"]
        return result
```

`lexical_weights` 的原始格式长这样：

```python
[
    {"12345": 0.85, "67890": 0.43, ...},  # 第 1 条文本的稀疏权重
    {"99999": 0.72, ...},                  # 第 2 条文本
]
```

注意：key 是**字符串**形式的 token ID，转 Qdrant 格式时需要转成 `int`。

### Step 1.4：升级 `indexer.py`（处理 sparse 向量格式转换）

文件路径：`src/medrag/index/indexer.py`

```python
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from medrag.ingest.chunker import Chunk


def _to_sparse_vector(weights: dict) -> SparseVector:
    """把 BGE-M3 的 lexical_weights dict 转成 Qdrant SparseVector 格式。"""
    if not weights:
        # 空文本或 OOV 文本会返回空 dict，给一个占位值避免 Qdrant 报错
        return SparseVector(indices=[0], values=[0.0])
    # key 是字符串 token ID，转 int；value 是浮点权重
    indices = [int(k) for k in weights.keys()]
    values = [float(v) for v in weights.values()]
    return SparseVector(indices=indices, values=values)


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vecs: np.ndarray,
    sparse_weights: list[dict] | None = None,  # None = 纯 dense 模式（向后兼容）
    collection: str = "medrag_text",
    batch: int = 256,
) -> None:
    points = []
    for i, (c, vec) in enumerate(zip(chunks, dense_vecs)):
        vector_payload: dict = {"dense": vec.tolist()}
        if sparse_weights is not None:
            vector_payload["sparse"] = _to_sparse_vector(sparse_weights[i])

        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector=vector_payload,
            payload={
                "chunk_id": c.chunk_id,
                "source": c.source,
                "doc_id": c.doc_id,
                "text": c.text,
                **c.metadata,
            },
        ))
        if len(points) >= batch:
            client.upsert(collection_name=collection, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection, points=points)
```

**关键设计**：`sparse_weights=None` 时走纯 dense 模式，向后兼容 Week 1 的调用。`_to_sparse_vector` 处理空字典边缘情况，避免 Qdrant 因空向量报错。

### Step 1.5：新增三阶段索引脚本 `scripts/04_build_index.py` 升级

在原有 `--phase embed/index/all` 基础上，增加 sparse 编码阶段。设计思路：

```
Phase 1 (--phase embed):     密集编码 → dense.npy（已有缓存，可跳过）
Phase 2 (--phase sparse):    稀疏编码 → sparse.jsonl（新增，~40 min on CPU）
Phase 3 (--phase index):     dense.npy + sparse.jsonl → Qdrant
--phase all:                 按顺序跑，已有缓存自动跳过
```

**为什么 sparse 要单独一个阶段？**

sparse 编码（`return_sparse=True`）比 dense 慢约 1.5-2x，因为需要额外做词汇权重计算。把它和 dense 编码分开，失败时只需重跑 sparse 阶段，不影响已有的 dense.npy 缓存。

`sparse.jsonl` 格式（每行是一条文本的稀疏权重字典）：

```jsonl
{"12345": 0.85, "67890": 0.43}
{"99999": 0.72}
```

### Step 1.6：运行并验证

```powershell
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
cd "D:\Desktop\Agent\medrag-agent"

# 只跑 sparse 编码 + index（dense 已有缓存）
& $py scripts\04_build_index.py --phase sparse
& $py scripts\04_build_index.py --phase index
```

验证：

```powershell
# 检查 collection 配置
curl http://localhost:6333/collections/medrag_text
# 期望：同时看到 "vectors" (dense) 和 "sparse_vectors" (sparse) 两个配置段

# 检查 point 数量不变
# 期望：points_count = 44768
```

---

## 4. Day 2：混合检索 + RRF 融合（P2）

**目标**：实现 `retrieval/hybrid.py`，把 dense 和 sparse 两路结果用 RRF 融合，成为 P2 pipeline。  
**时间预算**：3 h  
**验收**：同一查询，P2 结果列表与 P1 不同（顺序或成员有变化），说明 sparse 在起作用。

### Step 2.1：理解 Qdrant 的 sparse 查询 API

Dense 查询：把 query 向量和所有 document 向量做余弦相似度，返回 top-K。

Sparse 查询：Qdrant 内置了倒排索引，给定 query 的稀疏向量（哪些 token 权重高），找到包含这些 token 的 document，用内积打分。API：

```python
from qdrant_client.models import SparseVector

result = qdrant.query_points(
    collection_name="medrag_text",
    query=SparseVector(indices=[12345, 67890], values=[0.85, 0.43]),
    using="sparse",   # 指定用 sparse 向量
    limit=20,
    with_payload=True,
)
```

### Step 2.2：实现 `retrieval/hybrid.py`

文件路径：`src/medrag/retrieval/hybrid.py`

```python
"""混合检索：BGE-M3 dense + sparse，RRF 融合。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import RetrievedChunk  # 复用 Week 1 的数据类


def _reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    rankings: 多个排名列表，每个列表是 chunk_id 的有序序列（第 0 位最相关）
    k:        平滑参数，防止第 1 名权重过大；k=60 是 TREC 论文推荐值
    
    返回：按 RRF 分数排序的 (chunk_id, rrf_score) 列表
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _to_sparse_vec(weights: dict) -> SparseVector:
    """把 BGE-M3 的 lexical_weights 转成 Qdrant 查询格式。"""
    if not weights:
        return SparseVector(indices=[0], values=[0.0])
    return SparseVector(
        indices=[int(k) for k in weights],
        values=[float(v) for v in weights.values()],
    )


class HybridRetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: BGEM3Embedder,
        collection: str = "medrag_text",
        rrf_k: int = 60,
        candidate_k: int = 20,  # 两路各取 top-20，再融合
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        # 1. 同时编码 dense + sparse
        enc = self.embedder.encode([query], return_sparse=True)
        dense_vec = enc["dense"][0].tolist()
        sparse_weights = enc["sparse"][0]  # dict[str, float]

        # 2. Dense 检索 top-20
        dense_result = self.qdrant.query_points(
            collection_name=self.collection,
            query=dense_vec,
            using="dense",
            limit=self.candidate_k,
            with_payload=True,
        )
        # 注意：这里需要把 payload 缓存起来，后面 RRF 时只有 chunk_id，
        # 还需要根据 chunk_id 找回 payload
        id_to_point: dict[str, object] = {}
        dense_ranking: list[str] = []
        for p in dense_result.points:
            cid = p.payload["chunk_id"]
            dense_ranking.append(cid)
            id_to_point[cid] = p

        # 3. Sparse 检索 top-20
        sparse_result = self.qdrant.query_points(
            collection_name=self.collection,
            query=_to_sparse_vec(sparse_weights),
            using="sparse",
            limit=self.candidate_k,
            with_payload=True,
        )
        sparse_ranking: list[str] = []
        for p in sparse_result.points:
            cid = p.payload["chunk_id"]
            sparse_ranking.append(cid)
            if cid not in id_to_point:
                id_to_point[cid] = p  # sparse 找到了 dense 没找到的，也保留

        # 4. RRF 融合
        fused = _reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=self.rrf_k)

        # 5. 取 top-k，从缓存中恢复完整信息
        results: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused[:k]:
            p = id_to_point.get(chunk_id)
            if p is None:
                continue
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=p.payload["text"],
                score=rrf_score,   # 用 RRF 分数替代原始相似度分数
                payload=p.payload,
            ))
        return results


__all__ = ["HybridRetriever", "_reciprocal_rank_fusion"]
```

**关键设计点**：

1. `id_to_point` 字典缓存：RRF 之后只有 chunk_id，需要反查 payload（text、source 等）。这里把两路检索的结果都缓存在一个 dict 里，sparse 路找到而 dense 路没找到的也保留。

2. `candidate_k=20`：每路候选取 20，融合后取 top-5。如果只取 top-5 再融合，sparse 和 dense 可能重叠度太高，融合失去意义。

3. `score=rrf_score`：最终的 score 是 RRF 分数（0-1 之间的小值，如 0.03），不再是余弦相似度。展示时注意解释含义变化。

### Step 2.3：单元测试

文件路径：`tests/test_hybrid.py`

```python
from medrag.retrieval.hybrid import _reciprocal_rank_fusion

def test_rrf_basic():
    """同一 doc 出现在两个列表的头部，应该得到最高 RRF 分。"""
    list1 = ["A", "B", "C"]
    list2 = ["A", "C", "D"]
    fused = dict(_reciprocal_rank_fusion([list1, list2]))
    assert fused["A"] > fused["B"]   # A 在两个列表都排第一
    assert fused["A"] > fused["D"]   # D 只在一个列表里
    assert fused["C"] > fused["D"]   # C 在两个列表都出现

def test_rrf_single_list():
    """只有一个列表时，RRF 退化为按原始排名排序。"""
    ranking = ["X", "Y", "Z"]
    fused = dict(_reciprocal_rank_fusion([ranking]))
    scores = [fused[k] for k in ["X", "Y", "Z"]]
    assert scores[0] > scores[1] > scores[2]
```

### Step 2.4：快速对比测试

在 PowerShell 里跑一条对比查询，直观感受 P1 vs P2 的差异：

```powershell
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
cd "D:\Desktop\Agent\medrag-agent"

# P1: dense-only
& $py -c "
from qdrant_client import QdrantClient
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever
q = 'TAVR procedural outcomes'
r = DenseRetriever(QdrantClient('http://localhost:6333'), BGEM3Embedder())
for c in r.retrieve(q): print(f'P1 {c.score:.3f} {c.citation} {c.text[:80]}')
"

# P2: hybrid
& $py -c "
from qdrant_client import QdrantClient
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.hybrid import HybridRetriever
q = 'TAVR procedural outcomes'
r = HybridRetriever(QdrantClient('http://localhost:6333'), BGEM3Embedder())
for c in r.retrieve(q): print(f'P2 {c.score:.4f} {c.citation} {c.text[:80]}')
"
```

期望：对于包含精确医学缩写（TAVR）的查询，P2 的结果与 P1 有差异（新的 chunk 出现，或排名变化）。

---

## 5. Day 3：Cross-Encoder 重排序（P3）

**目标**：实现 `retrieval/reranker.py`，用 bge-reranker-v2-m3 对 P2 的 top-20 重排序，得到更精准的 top-5。  
**时间预算**：2 h  
**验收**：reranker 对同一批候选给出的排序与原始 RRF 排序有明显差异（至少 2-3 个名次变化）。

### Step 3.1：了解 FlagReranker API

`bge-reranker-v2-m3` 通过 `FlagEmbedding` 的 `FlagReranker` 类调用：

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False, device='cpu')

# compute_score 接受 [query, passage] 对的列表，返回分数列表
scores = reranker.compute_score([
    ["What is 3T MRI resolution?", "3T MRI offers sub-millimeter spatial resolution..."],
    ["What is 3T MRI resolution?", "CT scan uses X-rays to create cross-sectional images..."],
])
# 期望：第一对分数高，第二对分数低
```

### Step 3.2：实现 `retrieval/reranker.py`

文件路径：`src/medrag/retrieval/reranker.py`

```python
"""Cross-encoder reranker using bge-reranker-v2-m3 (CPU, Plan B)."""
from __future__ import annotations

from FlagEmbedding import FlagReranker
from medrag.retrieval.retriever import RetrievedChunk


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 8,
    ):
        # Plan B：CPU 运行，不占显存
        try:
            self.model = FlagReranker(model_name, use_fp16=use_fp16, devices=device)
        except TypeError:
            self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        # 构建 [query, passage] 对
        pairs = [[query, c.text] for c in chunks]

        # compute_score 返回分数列表，顺序与 pairs 一一对应
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)

        # 给每个 chunk 附上 reranker 分数，排序
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: -x[0],
        )

        # 返回 top-k，更新 score 为 reranker 分数
        results = []
        for score, chunk in scored[:top_k]:
            # 创建新的 RetrievedChunk 而不是 mutate 原对象
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),   # 用 reranker 分数覆盖 RRF 分数
                payload=chunk.payload,
            ))
        return results


__all__ = ["BGEReranker"]
```

**设计要点**：

1. 同样的 `try/except` 兼容不同版本的 FlagReranker 参数名。

2. `compute_score` 的分数是 logit（无界浮点数，一般在 -10 到 10 之间），不是 0-1 的概率。直接用来排序没问题，但展示时要注意说明。

3. 创建新的 `RetrievedChunk` 而不是修改原对象——避免副作用，便于在同一个流程里比较 P2 和 P3 的结果。

### Step 3.3：更新 `quick_demo.py`，加 `--mode` 参数

更新后的 `quick_demo.py` 支持三种模式，方便对比：

```python
# scripts/quick_demo.py（更新版）
import pyarrow.dataset
import sys
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from medrag.agent.generator import generate_answer
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker


def main(query: str, k: int = 5, mode: str = "p3") -> None:
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")

    if mode == "p1":
        retriever = DenseRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
        print(f"[mode=P1: dense-only]")
    elif mode == "p2":
        retriever = HybridRetriever(qdrant, embedder)
        chunks = retriever.retrieve(query, k=k)
        print(f"[mode=P2: hybrid RRF]")
    else:  # p3（默认）
        retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
        candidates = retriever.retrieve(query, k=20)  # 先取 20 个候选
        reranker = BGEReranker()
        chunks = reranker.rerank(query, candidates, top_k=k)
        print(f"[mode=P3: hybrid + reranker]")

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c.score:.3f}  {c.citation}")
        print(f"    {c.text[:200]}...")
    print("=" * 60)
    print("ANSWER:\n")
    print(generate_answer(query, chunks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    parser.add_argument("--mode", choices=["p1", "p2", "p3"], default="p3")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    q = " ".join(args.query) or "What is the typical resolution of 3T MRI?"
    main(q, k=args.k, mode=args.mode)
```

运行对比：

```powershell
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
& $py scripts\quick_demo.py "TAVR procedural outcomes" --mode p1
& $py scripts\quick_demo.py "TAVR procedural outcomes" --mode p2
& $py scripts\quick_demo.py "TAVR procedural outcomes" --mode p3
```

---

## 6. Day 4：Golden Dataset 构建

**目标**：用 Qwen3-8B（本地，免费）半自动生成 50 道医学文献问答题，人工审核后冻结。  
**时间预算**：4 h（2h 脚本 + 2h 人工审核）  
**验收**：`eval/golden_dataset.jsonl` 有 50 行，题型分布合理，没有泄题（问题不包含答案的大段原文）。

### Step 4.1：Golden Dataset 的设计原则

**题型分布（50 题）**：

| 类型 | 数量 | 特征 | 用途 |
|------|------|------|------|
| 事实型 (factual) | 20 | 可从单一 chunk 回答；有具体数值/名称 | 测基础检索精度 |
| 综合型 (synthesis) | 20 | 需要综合 2-3 个同主题 chunk | 测多文档综合能力 |
| 多跳型 (multihop) | 10 | 必须从 ≥ 2 篇**不同** PMID 的 chunk 组合回答 | 测 agentic loop（Week 3 用）|

**防泄题原则**：

问题里不能包含答案文本的大段原文（比如问题直接引用了那段文字）。这在评测时相当于"开卷考试"，评分无效。

验证方法：计算问题字符串与来源 chunk 的 n-gram 重叠，超过阈值就丢弃。

**每道题的字段**：

```json
{
  "id": "f001",
  "type": "factual",
  "query": "What spatial resolution is typically reported for 3T brain MRI?",
  "expected_pmids": ["41916137"],
  "expected_answer_summary": "0.9-1.0 mm isotropic voxels in clinical practice",
  "rubric": "Must mention sub-millimeter or 0.9-1.0 mm voxel size and 3T field strength"
}
```

`rubric` 是给评测 judge（Week 5 的 LLM-as-judge）用的评分标准，描述"一个正确答案必须包含的关键要素"。

### Step 4.2：构建脚本 `scripts/05_build_golden.py`

整体思路：

1. 从 Qdrant 随机抽取种子 chunks（按主题分层）
2. 把 chunk 内容喂给 Qwen3-8B（thinking ON），让它生成问题
3. 自动健康检查（过滤太短、泄题、重复的题）
4. 输出候选题供人工审核

```python
# scripts/05_build_golden.py
"""
半自动构建 golden dataset。

使用本地 Qwen3-8B（thinking ON）生成问题，避免 GPT-4o-mini 费用。
生成质量略低于 GPT-4o，但对医学影像领域够用，且完全离线。
"""
import pyarrow.dataset
import json
import random
import re
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

QDRANT_URL = "http://localhost:6333"
COLLECTION = "medrag_text"
OUT = Path("eval/golden_dataset_candidates.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 使用 thinking ON：让 Qwen3 认真思考如何生成好问题
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://127.0.0.1:11434",
    reasoning=True,   # thinking ON
    temperature=0.7,  # 稍高温度增加多样性
    num_ctx=4096,
)

FACTUAL_PROMPT = """You are building a medical literature QA benchmark.

Given this medical text chunk (source: {source}, id: {chunk_id}):
<chunk>
{text}
</chunk>

Generate ONE specific factual question whose answer comes DIRECTLY from this chunk.
Requirements:
- The question must be answerable from this chunk alone
- Avoid yes/no questions
- The question should NOT quote more than 5 words verbatim from the chunk
- Output ONLY valid JSON, no extra text:
{{"query": "...", "expected_answer_summary": "...", "rubric": "key facts that must appear in a correct answer"}}"""

MULTIHOP_PROMPT = """You are building a medical literature QA benchmark.

Given these TWO medical text chunks from DIFFERENT papers:

Chunk A (id: {chunk_id_a}):
<chunk_a>
{text_a}
</chunk_a>

Chunk B (id: {chunk_id_b}):
<chunk_b>
{text_b}
</chunk_b>

Generate ONE question whose answer requires synthesizing BOTH chunks.
The question must be UNANSWERABLE from either chunk alone.
Output ONLY valid JSON:
{{"query": "...", "expected_answer_summary": "...", "required_chunk_ids": ["{chunk_id_a}", "{chunk_id_b}"], "rubric": "..."}}"""


from medrag.agent.utils import strip_thinking


def _generate_question(prompt: str) -> dict | None:
    """调用 LLM 生成问题，返回解析后的 dict 或 None（解析失败）。"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = strip_thinking(resp.content)
        # 提取 JSON（LLM 有时会在 JSON 前后加说明文字）
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group())
    except Exception as e:
        print(f"[warn] LLM generation failed: {e}")
        return None


def _is_leaking(query: str, chunk_text: str, threshold: int = 8) -> bool:
    """检测问题是否泄露了 chunk 的原文（n-gram 重叠检测）。"""
    query_words = query.lower().split()
    chunk_words = chunk_text.lower().split()
    if len(query_words) < threshold:
        return False
    # 检查 query 中是否有连续 threshold 个词在 chunk 中出现
    chunk_ngrams = set(
        " ".join(chunk_words[i:i + threshold])
        for i in range(len(chunk_words) - threshold + 1)
    )
    for i in range(len(query_words) - threshold + 1):
        ngram = " ".join(query_words[i:i + threshold])
        if ngram in chunk_ngrams:
            return True
    return False


def sample_chunks(qdrant: QdrantClient, n: int = 80) -> list[dict]:
    """从 Qdrant 随机抽取 chunks（按 source 分层：pubmed 和 pmc 各一半）。"""
    # Qdrant 没有直接的 random sample API，用 scroll + 随机偏移模拟
    all_chunks = []
    offset = None
    while len(all_chunks) < n * 3:  # 多抽一些，之后过滤
        result = qdrant.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        if not result[0]:
            break
        all_chunks.extend([p.payload for p in result[0]])
        offset = result[1]
        if offset is None:
            break

    random.shuffle(all_chunks)
    # 分层采样：各取 n//2 个 pubmed 和 pmc
    pubmed = [c for c in all_chunks if c.get("source") == "pubmed"][:n // 2]
    pmc = [c for c in all_chunks if c.get("source") == "pmc"][:n // 2]
    return pubmed + pmc


def main():
    qdrant = QdrantClient(url=QDRANT_URL)
    chunks = sample_chunks(qdrant, n=80)
    print(f"[sample] got {len(chunks)} seed chunks")

    candidates = []

    # 生成 30 道事实题（候选，过滤后留 ~20）
    print("[gen] generating factual questions...")
    for chunk in chunks[:40]:
        prompt = FACTUAL_PROMPT.format(
            source=chunk.get("source", ""),
            chunk_id=chunk.get("chunk_id", ""),
            text=chunk.get("text", "")[:1000],  # 截断避免 context 太长
        )
        q = _generate_question(prompt)
        if q is None:
            continue
        query = q.get("query", "")
        if len(query.split()) < 8 or len(query.split()) > 50:
            continue  # 太短或太长
        if _is_leaking(query, chunk.get("text", "")):
            print(f"[filter] leaking question: {query[:60]}...")
            continue
        candidates.append({
            "type": "factual",
            "query": query,
            "expected_pmids": [chunk.get("doc_id", "")],
            "expected_answer_summary": q.get("expected_answer_summary", ""),
            "rubric": q.get("rubric", ""),
            "source_chunk_id": chunk.get("chunk_id", ""),
        })
        print(f"  + factual [{len(candidates)}]: {query[:60]}")

    # 生成 20 道多跳题（候选，过滤后留 ~10）
    print("[gen] generating multihop questions...")
    pmc_chunks = [c for c in chunks if c.get("source") == "pmc"]
    for i in range(0, min(len(pmc_chunks) - 1, 30), 2):
        ca, cb = pmc_chunks[i], pmc_chunks[i + 1]
        if ca.get("doc_id") == cb.get("doc_id"):
            continue  # 必须来自不同文章
        prompt = MULTIHOP_PROMPT.format(
            chunk_id_a=ca.get("chunk_id", ""),
            text_a=ca.get("text", "")[:600],
            chunk_id_b=cb.get("chunk_id", ""),
            text_b=cb.get("text", "")[:600],
        )
        q = _generate_question(prompt)
        if q is None:
            continue
        query = q.get("query", "")
        if len(query.split()) < 10:
            continue
        if _is_leaking(query, ca.get("text", "")) or _is_leaking(query, cb.get("text", "")):
            continue
        candidates.append({
            "type": "multihop",
            "query": query,
            "expected_pmids": [ca.get("doc_id", ""), cb.get("doc_id", "")],
            "expected_answer_summary": q.get("expected_answer_summary", ""),
            "rubric": q.get("rubric", ""),
            "source_chunk_ids": [ca.get("chunk_id", ""), cb.get("chunk_id", "")],
        })
        print(f"  + multihop [{len(candidates)}]: {query[:60]}")

    # 写出候选文件
    with OUT.open("w", encoding="utf-8") as f:
        for i, q in enumerate(candidates):
            q["id"] = f"{q['type'][0]}{i:03d}"
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(candidates)} candidates -> {OUT}")
    print("Next: manually review, add synthesis questions, finalize to eval/golden_dataset.jsonl")


if __name__ == "__main__":
    main()
```

### Step 4.3：人工审核流程（2 h）

脚本跑完后会生成 `eval/golden_dataset_candidates.jsonl`。逐条审核：

**审核标准**：

1. **问题清晰**：一个非专家能理解问题在问什么
2. **期望 PMID 正确**：`expected_pmids` 里的文章确实包含答案
3. **rubric 可操作**：rubric 描述了 1-3 个具体的"必须包含的关键要素"，不要写模糊的"答案必须正确"
4. **多跳题真的需要多文章**：自己只看一篇能不能回答？不能 = 合格的多跳题

**人工补题**：

综合型问题（synthesis，20 道）在上面的脚本里没有自动生成。需要手工补：选 2-3 个同主题的 chunk，写一道"比较……和……的区别"或"综合以上文献，……的最佳实践是什么"类型的问题。

最终冻结到 `eval/golden_dataset.jsonl`（50 道），并计算哈希：

```powershell
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
& $py -c "
import hashlib, json
data = open('eval/golden_dataset.jsonl', 'rb').read()
h = hashlib.sha256(data).hexdigest()
open('eval/golden_dataset.sha256', 'w').write(h)
print(f'SHA256: {h}')
"
```

---

## 7. Day 5：Pipeline 对比评测（P1 vs P2 vs P3）

**目标**：用 50 题 golden dataset 跑 P1/P2/P3 三个 pipeline，输出对比表。  
**时间预算**：3 h（1h 写脚本 + 2h 跑评测）  
**验收**：输出一个 Markdown 对比表，P3 ≥ P2 ≥ P1（至少在多数指标上成立）。

### Step 5.1：评测指标设计

Week 5 才用完整的 RAGAS。Week 2 先用轻量级的代理指标：

| 指标 | 计算方法 | 含义 |
|------|----------|------|
| **Citation Hit Rate** | 答案中是否包含 `expected_pmids` 中的至少一个 PMID | 引用正确率 |
| **Answer Non-Empty Rate** | 答案是否不是"The retrieved documents do not provide..." | 有效回答率 |
| **Top-1 PMID Hit** | 检索结果第 1 名的 doc_id 是否在 expected_pmids 里 | 检索精度 |
| **Latency (s)** | 端到端时间（embed + retrieve + generate）| 响应速度 |

这几个指标不需要 LLM judge，可以纯靠字符串匹配自动计算，速度快。

### Step 5.2：实现 `scripts/06_compare_pipelines.py`

```python
# scripts/06_compare_pipelines.py
"""
在 golden dataset 上运行 P1/P2/P3，输出对比表。
预计运行时间：50 题 × 3 pipeline × ~20s/题 ≈ 50 分钟。
建议先用 --limit 5 快速验证，再跑全量。
"""
import pyarrow.dataset
import json
import time
import sys
import io
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker
from medrag.agent.generator import generate_answer

GOLDEN = Path("eval/golden_dataset.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_golden(limit: int | None = None) -> list[dict]:
    items = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    return items[:limit] if limit else items


def citation_hit(answer: str, expected_pmids: list[str]) -> bool:
    """检查答案文本中是否包含 expected_pmids 中的至少一个。"""
    for pmid in expected_pmids:
        if pmid in answer:
            return True
    return False


def top1_hit(chunks, expected_pmids: list[str]) -> bool:
    if not chunks:
        return False
    return chunks[0].payload.get("doc_id", "") in expected_pmids


def run_pipeline(pipeline_name: str, questions: list[dict], qdrant, embedder) -> list[dict]:
    reranker = BGEReranker() if pipeline_name == "p3" else None
    results = []

    for q in questions:
        t0 = time.time()

        if pipeline_name == "p1":
            retriever = DenseRetriever(qdrant, embedder)
            chunks = retriever.retrieve(q["query"], k=5)
        elif pipeline_name == "p2":
            retriever = HybridRetriever(qdrant, embedder)
            chunks = retriever.retrieve(q["query"], k=5)
        else:  # p3
            retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
            candidates = retriever.retrieve(q["query"], k=20)
            chunks = reranker.rerank(q["query"], candidates, top_k=5)

        answer = generate_answer(q["query"], chunks)
        latency = time.time() - t0

        results.append({
            "id": q["id"],
            "type": q["type"],
            "query": q["query"],
            "pipeline": pipeline_name,
            "answer": answer,
            "latency": round(latency, 1),
            "citation_hit": citation_hit(answer, q["expected_pmids"]),
            "top1_hit": top1_hit(chunks, q["expected_pmids"]),
            "answer_non_empty": "do not provide" not in answer.lower(),
            "retrieved_pmids": [c.payload.get("doc_id", "") for c in chunks],
        })
        print(f"  [{pipeline_name.upper()}] {q['id']} done ({latency:.1f}s) "
              f"citation={'✓' if results[-1]['citation_hit'] else '✗'}")

    return results


def print_comparison_table(all_results: dict[str, list[dict]]):
    """输出 Markdown 格式的对比表。"""
    print("\n## Pipeline Comparison Results\n")
    print("| Metric | P1 (Dense) | P2 (Hybrid) | P3 (Hybrid+Reranker) |")
    print("|--------|-----------|-------------|----------------------|")

    for metric_name, key, fmt in [
        ("Citation Hit Rate", "citation_hit", ".1%"),
        ("Top-1 PMID Hit", "top1_hit", ".1%"),
        ("Answer Non-Empty", "answer_non_empty", ".1%"),
        ("Avg Latency (s)", "latency", ".1f"),
    ]:
        row = [metric_name]
        for pipeline in ["p1", "p2", "p3"]:
            vals = [r[key] for r in all_results[pipeline]]
            avg = sum(vals) / len(vals)
            row.append(f"{avg:{fmt}}")
        print(f"| {' | '.join(row)} |")

    # 按题型分组
    print("\n### By Question Type\n")
    print("| Type | P1 Citation | P2 Citation | P3 Citation |")
    print("|------|-------------|-------------|-------------|")
    for qtype in ["factual", "synthesis", "multihop"]:
        row = [qtype]
        for pipeline in ["p1", "p2", "p3"]:
            subset = [r for r in all_results[pipeline] if r["type"] == qtype]
            if not subset:
                row.append("N/A")
                continue
            rate = sum(r["citation_hit"] for r in subset) / len(subset)
            row.append(f"{rate:.1%} ({len(subset)} q)")
        print(f"| {' | '.join(row)} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="题目数量上限（调试用）")
    parser.add_argument("--pipelines", default="p1,p2,p3")
    args = parser.parse_args()

    questions = load_golden(limit=args.limit)
    pipelines = args.pipelines.split(",")
    print(f"[eval] running {len(questions)} questions on pipelines: {pipelines}")

    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")

    all_results = {}
    for pipeline in pipelines:
        print(f"\n[{pipeline.upper()}] starting...")
        results = run_pipeline(pipeline, questions, qdrant, embedder)
        all_results[pipeline] = results
        # 保存中间结果，防止崩溃丢失
        out_file = RESULTS_DIR / f"{pipeline}_results.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{pipeline.upper()}] saved to {out_file}")

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
```

运行（先跑 5 题验证，再跑全量）：

```powershell
# 快速验证（5 题，~3 分钟）
& $py scripts\06_compare_pipelines.py --limit 5

# 全量（50 题，~50 分钟）
& $py scripts\06_compare_pipelines.py
```

---

## 8. Day 6：Streamlit Demo UI

**目标**：做一个简单的 Streamlit 网页 demo，展示检索结果和回答，支持切换 pipeline。  
**时间预算**：2 h  
**验收**：`streamlit run src/medrag/ui/app.py` 能跑，浏览器里能输入问题、看到结果。

### Step 8.1：安装 Streamlit

```powershell
conda activate medrag
pip install streamlit>=1.40
```

### Step 8.2：实现 `src/medrag/ui/app.py`

Streamlit 的核心思路：每次用户操作（改输入框、点按钮），Python 脚本从头到尾重跑一次，用 `st.session_state` 在多次运行之间保存状态。

```python
# src/medrag/ui/app.py
"""
MedRAG-Agent Streamlit Demo (Week 2)

功能:
- 输入医学问题，选择 pipeline (P1/P2/P3)
- 展示检索到的文档片段（含来源、相关度分数）
- 展示 Qwen3-8B 生成的带引用答案
- 对话历史保存（同一会话内）
"""
import pyarrow.dataset  # Windows AV fix

import streamlit as st
from qdrant_client import QdrantClient

from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import BGEReranker
from medrag.agent.generator import generate_answer

# ─── 页面配置 ───────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG-Agent",
    page_icon="🏥",
    layout="wide",
)

# ─── 缓存重量级资源（只在第一次加载）──────────────────────
@st.cache_resource
def load_resources():
    qdrant = QdrantClient(url="http://localhost:6333")
    embedder = BGEM3Embedder(device="cpu")
    reranker = BGEReranker()
    return qdrant, embedder, reranker


qdrant, embedder, reranker = load_resources()

# ─── 侧边栏：配置 ───────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 配置")
    pipeline = st.radio(
        "检索 Pipeline",
        options=["P1: Dense Only", "P2: Hybrid (Dense + Sparse)", "P3: Hybrid + Reranker"],
        index=2,
    )
    top_k = st.slider("返回文档数 (K)", min_value=3, max_value=10, value=5)
    st.divider()
    st.markdown("""
    **Pipeline 说明**
    - **P1**: BGE-M3 dense 向量，纯语义检索
    - **P2**: P1 + sparse 向量，RRF 融合，擅长精确词匹配
    - **P3**: P2 候选集 + Cross-Encoder 重排序，最高精度
    """)

# ─── 主界面 ─────────────────────────────────────────────
st.title("🏥 MedRAG-Agent")
st.caption("Local medical literature QA · PubMed + PMC · Qwen3-8B · BGE-M3")

# 对话历史
if "history" not in st.session_state:
    st.session_state.history = []

# 展示历史对话
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["query"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander(f"📚 {len(item['chunks'])} retrieved documents"):
            for i, c in enumerate(item["chunks"], 1):
                st.markdown(f"**[{i}] {c.citation}** · score={c.score:.3f}")
                st.caption(c.text[:300] + "...")

# 输入框
query = st.chat_input("输入医学问题（英文效果最佳）...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            # 执行检索
            mode = pipeline.split(":")[0].strip().lower()
            if mode == "p1":
                retriever = DenseRetriever(qdrant, embedder)
                chunks = retriever.retrieve(query, k=top_k)
            elif mode == "p2":
                retriever = HybridRetriever(qdrant, embedder)
                chunks = retriever.retrieve(query, k=top_k)
            else:  # p3
                retriever = HybridRetriever(qdrant, embedder, candidate_k=20)
                candidates = retriever.retrieve(query, k=20)
                chunks = reranker.rerank(query, candidates, top_k=top_k)

        with st.spinner("Qwen3-8B 生成回答中..."):
            answer = generate_answer(query, chunks)

        # 展示回答
        st.write(answer)

        # 展示来源
        with st.expander(f"📚 {len(chunks)} retrieved documents"):
            for i, c in enumerate(chunks, 1):
                col1, col2 = st.columns([1, 4])
                col1.metric(f"#{i}", f"{c.citation}", f"score={c.score:.3f}")
                col2.caption(c.text[:400] + "...")

        # 保存到历史
        st.session_state.history.append({
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "pipeline": mode,
        })
```

运行：

```powershell
$env:PYTHONIOENCODING = "utf-8"
conda activate medrag
cd "D:\Desktop\Agent\medrag-agent"
streamlit run src/medrag/ui/app.py
# 浏览器访问 http://localhost:8501
```

### Step 8.3：验证关键功能

在浏览器里测试以下场景：

1. **正常问题**：`"What is the spatial resolution of 3T brain MRI?"` → 应返回 5 个文档 + 有引用的答案
2. **域外问题**：`"What is the boiling point of water?"` → 应回答"The retrieved documents do not provide..."
3. **切换 pipeline**：同一问题用 P1/P2/P3 分别问，观察结果是否有差异
4. **多轮对话**：问第一个问题后，再问跟进问题，历史对话应显示

---

## 9. Day 7：文档 + 提交

**目标**：更新 README，写 Week 2 tutorial，git commit。  
**时间预算**：1 h

### Step 9.1：更新 README

在 README 的 Status 部分更新：

```markdown
- [x] Week 1: Basic dense RAG — 44768 chunks, Qwen3-8B, BGE-M3 ✅
- [x] Week 2: Hybrid retrieval (dense+sparse RRF) + reranker + 50-Q golden dataset ✅
- [ ] Week 3: LangGraph agentic loop with self-correction + session memory
```

在 README 里加一个对比结果表（从 `06_compare_pipelines.py` 的输出里复制）：

```markdown
## Retrieval Pipeline Comparison (50-Q Golden Dataset)

| Metric | P1 Dense | P2 Hybrid | P3 +Reranker |
|--------|----------|-----------|--------------|
| Citation Hit Rate | X% | X% | X% |
| Top-1 PMID Hit | X% | X% | X% |
| Avg Latency | Xs | Xs | Xs |
```

### Step 9.2：git commit

```powershell
cd "D:\Desktop\Agent\medrag-agent"
git add src/ scripts/ eval/ docs/
git commit -m "$(cat <<'EOF'
feat(week2): hybrid retrieval + reranker + 50-Q golden dataset

- Qdrant collection upgraded with sparse vectors (BGE-M3 lexical weights)
- HybridRetriever: dense + sparse, RRF fusion (k=60, candidate_k=20)
- BGEReranker: bge-reranker-v2-m3 cross-encoder, CPU, Plan B
- quick_demo.py: --mode p1/p2/p3 flag for pipeline comparison
- scripts/05_build_golden.py: Qwen3-8B (thinking ON) semi-auto question gen
- eval/golden_dataset.jsonl: 50 questions (20 factual + 20 synthesis + 10 multihop)
- scripts/06_compare_pipelines.py: P1/P2/P3 evaluation with citation hit rate
- Streamlit demo: pipeline toggle, retrieval sources, chat history

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## 10. 踩坑预防手册

### 坑 1：Qdrant SparseVector 格式错误

**现象**：`upsert` 报错 `ValidationError: sparse vector must have equal length indices and values`

**原因**：`lexical_weights` 的 key 是字符串而非整数，不能直接传给 Qdrant。

**预防**：用 `_to_sparse_vector` 明确转换：`indices=[int(k) for k in weights]`。

---

### 坑 2：Sparse 查询时 Qdrant 报 "Collection does not have sparse vector"

**现象**：P2 hybrid 查询时报 `CollectionNotFound` 或 `No sparse vector config`

**原因**：collection 用旧配置创建的（只有 dense），没有 `sparse_vectors_config`。

**预防**：运行 Day 1 时用 `recreate=True` 强制重建 collection，不要复用旧的。

---

### 坑 3：FlagReranker 下载模型很慢

**现象**：第一次 `FlagReranker('BAAI/bge-reranker-v2-m3', ...)` 卡住 10+ 分钟

**原因**：从 HuggingFace 下载约 1.1 GB 的 reranker 模型。

**预防**：提前下载，或设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
& $py -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"
```

---

### 坑 4：Golden Dataset 的 expected_pmids 用了 PMC doc ID 而非 PMID

**现象**：评测时 `citation_hit` 全是 False，即使答案里有引用

**原因**：PMC 语料的 `doc_id` 是 `"doc199"` 这种格式，不是真实 PMCID。LLM 生成的答案引用了 `[PMC:doc199]`，但 `expected_pmids` 里写的是真实 PMID。

**预防**：在 golden dataset 里统一用 `chunk.get("doc_id")` 作为期望 ID，在 `citation_hit` 函数里也匹配 `chunk.citation` 格式（`PMID:xxx` 或 `PMC:xxx`）。

---

### 坑 5：Streamlit 的 `@st.cache_resource` 导致修改代码后不更新

**现象**：修改了 `embedder.py` 或 `retriever.py`，但 Streamlit 里行为没变

**原因**：`@st.cache_resource` 把模型实例缓存在内存里，不受文件修改影响。

**预防**：开发调试时在浏览器里点右上角 "Always rerun"（或按 `R` 键）清空缓存；模型代码稳定后再用 cache。

---

### 坑 6：评测跑 50 题花 2-3 小时

**现象**：`06_compare_pipelines.py` 跑全量时间太长

**原因**：每道题需要加载 BGE-M3（第一次），调用 Qwen3-8B 生成答案（~10-20s/题）。

**预防**：
1. 先用 `--limit 5` 验证流程正确，再跑全量
2. 如果 BGE-M3 每次都重新加载，检查是否把 `embedder` 实例化放在了循环内部（应该只实例化一次）
3. 全量跑时放后台：`Start-Process python -ArgumentList "scripts\06_compare_pipelines.py"`

---

## 11. 可裁剪项

进度落后时，按优先级从高到低裁减：

| 优先级 | 项目 | 裁减后的影响 |
|--------|------|-------------|
| **不可裁** | Qdrant sparse 升级 + HybridRetriever | P2 是 Week 2 核心 |
| **不可裁** | BGEReranker + P3 pipeline | P3 是 Week 3 agentic loop 的前置 |
| **不可裁** | Golden Dataset（至少 30 题）| Week 5 评测的基础 |
| 可裁减 | Golden Dataset 从 50 → 30 题 | 评测统计意义略降，可接受 |
| 可裁减 | `06_compare_pipelines.py` 完整评测 | 手工跑 5 道验证即可 |
| 可裁减 | Streamlit Demo UI | 改用 CLI demo 展示 |
| 可裁减 | 多跳型 golden 题（10 → 0 题）| Week 3 agentic loop 有示例就够 |

---

## Week 2 总时间预算

| Day | 内容 | 预算 | 关键风险 |
|-----|------|------|---------|
| Day 1 | Qdrant sparse 升级 + re-index | 3 h | sparse 格式转换出错 |
| Day 2 | HybridRetriever + RRF + 单测 | 3 h | Qdrant API 版本差异 |
| Day 3 | BGEReranker + P3 pipeline | 2 h | reranker 模型下载慢 |
| Day 4 | Golden Dataset 生成 + 人工审核 | 4 h | 审核时间最难估计 |
| Day 5 | Pipeline 对比评测 | 3 h | 50题×3pipeline运行时间 |
| Day 6 | Streamlit Demo | 2 h | 首次用 Streamlit 可能有学习曲线 |
| Day 7 | 文档 + README + commit | 1 h | — |
| **总计** | | **18 h** | |
