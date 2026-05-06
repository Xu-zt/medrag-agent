# VeritasMed — 前端设计方案与接口文档

> 版本: v1.0 · 日期: 2026-05-06

---

## 一、设计理念

VeritasMed 的核心价值在于**可验证性**——每一条回答都有文献支撑，每一步推理都可追溯。前端设计围绕这一核心展开，目标是让用户不只是"得到答案"，而是"看到证据链的形成过程"。

**三个设计原则：**

1. **推理过程可视化**：Agentic 循环的每一步（检索 → 评分 → 重写 → 生成 → 校验）实时展示，让用户理解系统为何给出这个答案
2. **证据可穿透**：答案中的每个引用 `[PMID:xxx]` 均可点击展开，直接显示语料库中对应的原文段落及上下文
3. **来源可信度量化**：每个检索结果附带相关性分数、来源类型（PubMed/PMC）、发表信息，用户可自行判断证据质量

---

## 二、系统架构（前后端）

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (React + TypeScript)                                   │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  问答界面    │  │  文献浏览器   │  │  推理过程时间轴       │  │
│  │  AnswerView  │  │  DocExplorer │  │  AgentTimeline       │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                │                      │               │
│         └────────────────┴──────────────────────┘               │
│                          │                                       │
│              REST / WebSocket (SSE)                              │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  FastAPI 后端 (Python)                                           │
│                                                                 │
│  /api/ask         WebSocket — 流式推理事件                       │
│  /api/search      REST     — 混合检索                            │
│  /api/document    REST     — 文档详情                            │
│  /api/chunk       REST     — 块级原文 + 上下文                   │
│  /api/history     REST     — 会话历史                            │
│  /api/corpus      REST     — 语料统计                            │
│                                                                 │
│  ↓↓ 复用现有模块 ↓↓                                              │
│  medrag.agent.graph  ·  medrag.retrieval.*  ·  Qdrant client    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、后端接口规范

### 3.1 技术选型

| 项 | 选型 | 理由 |
|----|------|------|
| Web 框架 | FastAPI 0.111 | 原生 async、自动 OpenAPI 文档、与现有 Python 栈无缝集成 |
| 实时推送 | WebSocket (JSON 事件流) | Agent 循环步骤需要服务端推送；SSE 备选 |
| 序列化 | Pydantic v2 | 类型安全，与 FastAPI 原生集成 |
| 跨域 | CORSMiddleware | 允许前端开发服务器访问 |

---

### 3.2 数据模型

```python
# 检索块（与 RetrievedChunk 对齐）
class ChunkOut(BaseModel):
    chunk_id: str          # "pmc:doc196:112"
    citation: str          # "PMC:doc196" / "PMID:12345"
    source: str            # "pubmed" | "pmc"
    doc_id: str
    title: str
    section: str | None    # PMC 章节（Abstract/Introduction/REF...）
    pmid: str | None
    chunk_idx: int
    total_chunks: int
    text: str              # 块全文
    score: float           # 检索/重排序得分
    highlight_ranges: list[tuple[int,int]]  # 前端高亮范围（关键词位置）

# Agent 事件（WebSocket 流）
class AgentEvent(BaseModel):
    event: Literal[
        "node_start",      # 节点开始执行
        "node_end",        # 节点完成，附带输出
        "chunk_retrieved", # 检索到块（实时推送）
        "answer_token",    # 生成 token（流式）
        "done",            # 完整结果
        "error"
    ]
    node: str | None       # "retrieve" | "grade" | "rewrite" | "generate" | "check"
    data: dict             # 随 event 类型变化

# 完整答案
class AnswerOut(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    faithful: bool
    faithfulness_issues: str
    iterations: int        # 触发了几次重写
    regen_count: int
    rewritten_queries: list[str]
    chunks: list[ChunkOut] # 生成时使用的 top-k 块
    thread_id: str
    latency_ms: float
```

---

### 3.3 接口详情

#### `WS /api/ask`

**用途**：执行完整 Agentic 推理，实时推送每个节点的执行状态。

**建立连接**：
```
ws://localhost:8000/api/ask
```

**客户端发送（JSON）**：
```json
{
  "query": "What are the contraindications of warfarin in elderly patients?",
  "thread_id": "session-abc123",
  "pipeline": "p3"
}
```

**服务端事件流（逐条推送）**：

```jsonc
// 1. 节点开始
{"event": "node_start", "node": "retrieve", "data": {"query": "warfarin contraindications elderly"}}

// 2. 检索结果实时推送（每块一条）
{"event": "chunk_retrieved", "node": "retrieve", "data": {
  "chunk_id": "pubmed:12345:0",
  "citation": "PMID:12345",
  "title": "Warfarin therapy in elderly patients...",
  "score": 0.812,
  "text_snippet": "Absolute contraindications include active bleeding..."
}}

// 3. 评分节点完成
{"event": "node_end", "node": "grade", "data": {
  "relevance_score": 0.75,
  "relevant": true,
  "reason": "Chunks cover contraindication categories but lack quantitative bleeding risk data",
  "rewrite_hint": ""
}}

// 4. 如触发重写
{"event": "node_start", "node": "rewrite", "data": {"iteration": 1}}
{"event": "node_end", "node": "rewrite", "data": {
  "new_query": "warfarin absolute relative contraindications elderly bleeding risk anticoagulation"
}}

// 5. 生成答案（流式 token）
{"event": "answer_token", "node": "generate", "data": {"token": "Warfarin"}}
{"event": "answer_token", "node": "generate", "data": {"token": " is contraindicated"}}
// ...

// 6. 忠实度校验
{"event": "node_end", "node": "check", "data": {
  "faithful": true,
  "issues": ""
}}

// 7. 最终结果
{"event": "done", "node": null, "data": {
  "answer": "Warfarin is contraindicated in patients with... [PMID:12345] [PMC:doc88]",
  "citations": ["PMID:12345", "PMC:doc88"],
  "confidence": 0.92,
  "faithful": true,
  "faithfulness_issues": "",
  "iterations": 0,
  "regen_count": 0,
  "rewritten_queries": [],
  "chunks": [...],
  "thread_id": "session-abc123",
  "latency_ms": 4821.3
}}
```

---

#### `GET /api/search`

**用途**：独立检索，不触发 Agent 推理，用于文献浏览。

**请求**：
```
GET /api/search?q=warfarin+elderly&k=10&pipeline=p2&highlight=true
```

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `q` | string | 必填 | 查询文本 |
| `k` | int | 5 | 返回块数（1–20） |
| `pipeline` | `p1`\|`p2`\|`p3` | `p2` | 检索管道 |
| `highlight` | bool | true | 是否计算关键词高亮位置 |

**响应 200**：
```json
{
  "query": "warfarin elderly",
  "pipeline": "p2",
  "latency_ms": 512.3,
  "chunks": [
    {
      "chunk_id": "pubmed:12345:0",
      "citation": "PMID:12345",
      "source": "pubmed",
      "doc_id": "12345",
      "title": "Anticoagulation management in elderly patients",
      "section": null,
      "pmid": "12345",
      "chunk_idx": 0,
      "total_chunks": 3,
      "text": "Warfarin remains the most widely used oral anticoagulant...",
      "score": 0.8312,
      "highlight_ranges": [[0, 8], [72, 85]]
    }
  ]
}
```

---

#### `GET /api/document/{citation}`

**用途**：获取文档级完整信息（所有块 + 元数据），用于"来源详情"面板。

**请求**：
```
GET /api/document/PMID:12345
GET /api/document/PMC:doc196
```

**响应 200**：
```json
{
  "citation": "PMID:12345",
  "source": "pubmed",
  "doc_id": "12345",
  "title": "Anticoagulation management in elderly patients with atrial fibrillation",
  "pmid": "12345",
  "external_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
  "total_chunks": 3,
  "chunks": [
    {
      "chunk_id": "pubmed:12345:0",
      "chunk_idx": 0,
      "section": null,
      "text": "Warfarin remains the most widely used oral anticoagulant..."
    }
  ]
}
```

---

#### `GET /api/chunk/{chunk_id}`

**用途**：获取单个块的原文及前后上下文块（用于"查看更多上下文"功能）。

**请求**：
```
GET /api/chunk/pubmed:12345:1?context_window=1
```

**参数**：`context_window` — 前后各取 N 块（默认 1）

**响应 200**：
```json
{
  "chunk": {
    "chunk_id": "pubmed:12345:1",
    "text": "Absolute contraindications include...",
    "score": null
  },
  "prev_chunk": {
    "chunk_id": "pubmed:12345:0",
    "text": "Warfarin remains the most widely used..."
  },
  "next_chunk": {
    "chunk_id": "pubmed:12345:2",
    "text": "Relative contraindications and risk stratification..."
  },
  "document": {
    "title": "Anticoagulation management...",
    "citation": "PMID:12345",
    "external_url": "https://pubmed.ncbi.nlm.nih.gov/12345/"
  }
}
```

---

#### `GET /api/history/{thread_id}`

**用途**：获取指定会话的历史问答，用于多轮对话展示。

**响应 200**：
```json
{
  "thread_id": "session-abc123",
  "turns": [
    {
      "query": "What is warfarin?",
      "answer": "Warfarin is a vitamin K antagonist...",
      "citations": ["PMID:12345"],
      "timestamp": "2026-05-06T19:30:00Z"
    }
  ],
  "summary": "Discussion of warfarin pharmacology and elderly patient considerations."
}
```

---

#### `GET /api/corpus/stats`

**用途**：首页 / 侧边栏展示语料统计。

**响应 200**：
```json
{
  "total_chunks": 44768,
  "pubmed_chunks": 32100,
  "pmc_chunks": 12668,
  "collection": "medrag_text",
  "embedding_model": "BAAI/bge-m3"
}
```

---

#### `GET /api/health`

```json
{"status": "ok", "qdrant": "connected", "ollama": "connected"}
```

---

## 四、前端设计方案

### 4.1 技术选型

| 项 | 选型 |
|----|------|
| 框架 | React 18 + TypeScript |
| 样式 | Tailwind CSS + shadcn/ui 组件库 |
| 状态管理 | Zustand（轻量，无 Redux 样板代码） |
| 实时通信 | 原生 WebSocket（封装为 `useAgentStream` hook） |
| 代码高亮 | `react-syntax-highlighter` |
| 图表（推理时间轴） | `@xyflow/react`（ReactFlow）节点图 |
| Markdown 渲染 | `react-markdown` + `remark-gfm` |
| 路由 | React Router v6 |
| HTTP | Axios |

---

### 4.2 页面结构

```
/                   → 主问答界面 (AnswerPage)
/explore            → 文献浏览器 (ExplorerPage)
/document/:citation → 文档详情页 (DocumentPage)
```

---

### 4.3 主问答界面 (AnswerPage)

**布局：三栏自适应**

```
┌──────────────────────────────────────────────────────────────────┐
│  HEADER: VeritasMed logo  +  会话选择器  +  语料统计（44k chunks）│
├─────────────────┬──────────────────────┬──────────────────────────┤
│                 │                      │                          │
│  推理过程        │      答案区域          │     证据面板             │
│  时间轴          │                      │                          │
│  (AgentTimeline)│  (AnswerPanel)        │  (EvidencePanel)         │
│                 │                      │                          │
│  280px 固定宽   │      flex-1           │  360px 固定宽            │
│                 │                      │                          │
├─────────────────┴──────────────────────┴──────────────────────────┤
│  FOOTER: 查询输入框                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

#### 组件 A：推理过程时间轴 `AgentTimeline`

**功能**：实时展示 Agent 循环每个节点的执行状态和输出摘要。

**视觉设计**：垂直时间线，每个节点一张卡片。

```
  ●  route          ✓  factual query
  │
  ●  retrieve       ✓  20 candidates found
  │                    ← 0.55s
  ●  rerank         ✓  top-5 selected
  │
  ●  grade          ✓  score: 0.75  relevant
  │
  ●  generate       ⟳  generating...       ← 动画旋转
  │
  ○  check          ·  waiting
```

**节点状态颜色**：
- 灰色 `○` — 待执行
- 蓝色 `⟳` — 执行中（pulse 动画）
- 绿色 `✓` — 成功
- 橙色 `↺` — 触发重写（附带新查询预览）
- 红色 `✗` — 错误

**重写事件特殊展示**：
```
  ●  grade    ✗  score: 0.32  insufficient
  │
  ↺  rewrite  ↺  iter 1/2
  │           原始: "warfarin elderly"
  │           重写: "warfarin contraindications elderly
  │                  bleeding risk anticoagulation"
  │
  ●  retrieve  ✓  20 candidates (retry)
```

---

#### 组件 B：答案区域 `AnswerPanel`

**功能**：流式渲染答案 Markdown，内联引用可点击。

**内联引用交互**：

```
Warfarin is absolutely contraindicated in patients with
active bleeding¹ or a known hypersensitivity².
Relative contraindications include severe hypertension³...

 ¹[PMID:12345]  ²[PMID:12345]  ③[PMC:doc88]
```

- 引用编号以角标形式渲染，颜色与证据面板中对应块配色一致
- 鼠标悬停 → 弹出 Popover，显示该块的前 200 字 + 文献标题
- 点击 → 证据面板滚动至对应块并高亮

**忠实度标识**：
```
  ┌─ 忠实度校验 ──────────────────────────────────────────┐
  │  ✓ 所有声明均有文献支撑                                │
  │  置信度: 0.92                  重写次数: 0  重生成: 0  │
  └───────────────────────────────────────────────────────┘
```

若不忠实（`faithful=False`）：
```
  ┌─ 忠实度校验 ──────────────────────────────────────────┐
  │  ⚠ 发现未支撑声明                                      │
  │  "The bleeding risk increases 3-fold..."               │
  │  该声明未在检索文档中找到直接依据                       │
  └───────────────────────────────────────────────────────┘
```

---

#### 组件 C：证据面板 `EvidencePanel`

**功能**：展示检索到的 top-k 原始文献块，是系统的核心差异化特性。

**每个块的卡片**：

```
┌────────────────────────────────────────────────────────┐
│  ①  PMID:12345                           分: 0.83  ████│
│  Anticoagulation management in elderly patients...     │
│  PubMed · 全文 3 块 · 查看 →                            │
├────────────────────────────────────────────────────────┤
│  "...Warfarin is absolutely contraindicated in         │
│  patients with ==active bleeding==, recent intracranial│
│  surgery, or known hypersensitivity to the drug..."    │
│                                                        │
│  [查看上下文 ↕]  [在 PubMed 中打开 ↗]                  │
└────────────────────────────────────────────────────────┘
```

- `==高亮==` — 与答案中引用对应的片段，通过简单关键词匹配实现
- **分数条**：彩色进度条，视觉化相关性得分（绿 > 0.7，黄 0.4–0.7，红 < 0.4）
- **[查看上下文]**：展开前后块（调用 `/api/chunk/{id}?context_window=1`）
- **[在 PubMed/PMC 中打开]**：直接跳转到 `https://pubmed.ncbi.nlm.nih.gov/{pmid}/`

**块颜色编码**（与答案中角标颜色一致）：
```
块 ①  →  蓝色角标  →  答案中 ¹[PMID:12345] 蓝色
块 ②  →  绿色角标  →  答案中 ²[PMC:doc88] 绿色
块 ③  →  紫色角标  →  ...
```

---

#### 组件 D：查询输入框 `QueryInput`

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  What are the contraindications of warfarin in elderly...      │
│                                                                │
│  [会话: session-1 ▼]   [管道: P3 ▼]   [清空]   [发送  →]     │
└────────────────────────────────────────────────────────────────┘
```

- **会话选择器**：下拉选择历史 thread_id，实现多轮对话连续性
- **管道选择**：P2（快速）/ P3（高质量），隐藏 P4/P5
- 发送时禁用输入，显示取消按钮（关闭 WebSocket）

---

### 4.4 文献浏览器 `ExplorerPage`

**布局**：左侧过滤器 + 右侧结果列表

```
┌────────────────┬───────────────────────────────────────────────┐
│  过滤器         │  搜索框: [warfarin elderly            🔍]     │
│                │                                               │
│  来源           │  排序: 相关性 ▼        共 847 结果             │
│  ☑ PubMed      │  ─────────────────────────────────────────── │
│  ☑ PMC         │  ┌─────────────────────────────────────────┐ │
│                │  │  PMID:12345                    0.83 ████ │ │
│  章节（PMC）    │  │  Anticoagulation management in elderly  │ │
│  ☑ Abstract    │  │  patients with atrial fibrillation      │ │
│  ☐ Methods     │  │  "...warfarin is contraindicated in..." │ │
│  ☑ Results     │  │                    [查看文档 →]          │ │
│  ☑ Discussion  │  └─────────────────────────────────────────┘ │
│                │  ...                                         │
│  分数范围       │                                               │
│  0.5 ──●── 1.0 │                                               │
└────────────────┴───────────────────────────────────────────────┘
```

---

### 4.5 文档详情页 `DocumentPage`

路由：`/document/PMID:12345`

```
┌──────────────────────────────────────────────────────────────────┐
│  ← 返回        PMID:12345                     [在 PubMed 打开 ↗] │
├──────────────────────────────────────────────────────────────────┤
│  Anticoagulation management in elderly patients with AF         │
│                                                                  │
│  来源: PubMed  ·  共 3 块  ·  在语料库中                         │
├────────────────────────────────────────────────────────────────  │
│                                                                  │
│  [块 0/3]                                                        │
│  Warfarin remains the most widely used oral anticoagulant        │
│  in patients with non-valvular atrial fibrillation...            │
│                                                                  │
│  [块 1/3]  ← 上次检索命中                                         │
│  ██████████████████████████████████████████                      │
│  Absolute contraindications include active bleeding,             │
│  recent intracranial surgery...                                  │
│  ██████████████████████████████████████████                      │
│                                                                  │
│  [块 2/3]                                                        │
│  Relative contraindications and risk stratification...           │
│                                                                  │
│  ───────────────────────────────────────────────────────────     │
│  ⚡ 用此文档提问  →  在主界面以该文档为上下文发起新问题              │
└──────────────────────────────────────────────────────────────────┘
```

---

### 4.6 关键交互流程

**正常查询流程**：

```
用户输入 → WebSocket 建立 → AgentTimeline 开始动画
  → retrieve: 检索块实时出现在 EvidencePanel（逐条推入）
  → grade:    TimelineCard 显示分数 + 相关性判断
  → generate: AnswerPanel 流式渲染文字，引用编号与 EvidencePanel 块配色绑定
  → check:    忠实度标识出现在答案底部
  → done:     WebSocket 关闭，界面进入可交互状态
```

**引用点击流程**：

```
用户点击 ¹[PMID:12345]
  → EvidencePanel 第 1 块高亮 + 滚动至顶
  → Popover 消失
  → "查看上下文"按钮脉冲提示
```

**重写触发流程**：

```
grade 节点: score=0.32, relevant=False
  → AgentTimeline 新增橙色重写卡片，显示新旧查询对比
  → EvidencePanel 出现"重新检索中..."骨架屏
  → 新的块替换旧块（附带"第 2 次检索"标签）
```

---

## 五、后端实现要点

### 5.1 文件位置

```
src/medrag/
└── api/
    ├── __init__.py
    ├── app.py           # FastAPI 应用入口
    ├── routes/
    │   ├── ask.py       # WebSocket /api/ask
    │   ├── search.py    # GET /api/search
    │   ├── document.py  # GET /api/document/{citation}
    │   ├── chunk.py     # GET /api/chunk/{chunk_id}
    │   ├── history.py   # GET /api/history/{thread_id}
    │   └── corpus.py    # GET /api/corpus/stats
    └── models.py        # Pydantic 数据模型
```

### 5.2 WebSocket 事件推送方案

```python
# ask.py 核心逻辑

@router.websocket("/api/ask")
async def ask_ws(websocket: WebSocket):
    await websocket.accept()
    req = AskRequest(**await websocket.receive_json())

    # 将 LangGraph 的同步调用包装为异步 + 事件流
    async for event in stream_agent(req.query, req.thread_id):
        await websocket.send_json(event.model_dump())

    await websocket.close()
```

LangGraph 提供 `app.astream_events()` API，可以直接获取每个节点的开始/结束事件，无需修改 graph.py：

```python
async def stream_agent(query: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state  = build_initial_state(query)

    async for event in app.astream_events(state, config=config, version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and name in NODE_NAMES:
            yield AgentEvent(event="node_start", node=name, data={...})

        elif kind == "on_chain_end" and name in NODE_NAMES:
            output = event["data"].get("output", {})
            yield AgentEvent(event="node_end", node=name, data=output)
```

### 5.3 块级上下文查询

```python
# chunk.py — 通过 Qdrant payload 过滤获取相邻块
def get_chunk_with_context(chunk_id: str, window: int = 1):
    # chunk_id 格式: "pubmed:12345:1" → doc_id="pubmed:12345", idx=1
    parts   = chunk_id.rsplit(":", 1)
    doc_key = parts[0]   # "pubmed:12345"
    idx     = int(parts[1])

    # Qdrant filter: chunk_id in [doc_key:idx-1, doc_key:idx, doc_key:idx+1]
    target_ids = [f"{doc_key}:{i}" for i in range(idx - window, idx + window + 1)]
    results = qdrant.scroll(
        collection_name="medrag_text",
        scroll_filter=Filter(must=[
            FieldCondition(key="chunk_id", match=MatchAny(any=target_ids))
        ]),
        with_payload=True,
    )
    ...
```

### 5.4 外部链接生成

```python
def external_url(source: str, doc_id: str, pmid: str | None) -> str:
    if source == "pubmed" and pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if source == "pmc":
        # doc_id 格式为 "doc196"，需要真实 PMCID
        # 如 payload 中有 pmcid 字段则直接用；否则链接到搜索
        return f"https://www.ncbi.nlm.nih.gov/pmc/search/?term={doc_id}"
    return ""
```

---

## 六、开发优先级

| 优先级 | 模块 | 估时 |
|--------|------|------|
| P0 | FastAPI app.py + WebSocket /api/ask | 4h |
| P0 | React 骨架 + AgentTimeline 组件 | 3h |
| P0 | AnswerPanel 流式渲染 + 引用角标 | 3h |
| P0 | EvidencePanel 块卡片 + 分数可视化 | 3h |
| P1 | GET /api/search + ExplorerPage | 3h |
| P1 | GET /api/document + DocumentPage | 2h |
| P1 | GET /api/chunk (上下文展开) | 2h |
| P2 | 引用-块颜色联动 | 2h |
| P2 | 会话历史 + thread_id 管理 | 2h |
| P2 | 移动端响应式布局 | 2h |

**最小可展示版本（P0，约 13h）**：输入问题 → 实时看到 Agent 各节点执行 → 答案流式出现 + 证据面板同步展示检索块 → 点击引用跳转到块

---

## 七、启动方式（规划）

```bash
# 后端
uvicorn src.medrag.api.app:app --reload --port 8000

# 前端（开发）
cd frontend
npm install
npm run dev   # → http://localhost:5173

# 前端（生产构建）
npm run build
# FastAPI 挂载 dist/ 目录作为静态文件
```

---

*VeritasMed Frontend Design v1.0 — 2026-05-06*
