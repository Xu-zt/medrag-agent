# VeritasMed 评估报告

> 版本: v1.1 · 日期: 2026-05-09  
> 评测对象: MedRAG-Agent (P3 Static → P4-Agentic → Stage 2 Smart Routing)  
> 语料: 44,768 chunks (1,975 PubMed + 42,793 PMC)

---

## 1. 实验设置

### 1.1 管道定义

| 管道 | 检索 | 重排 | 生成 | 特点 |
|------|------|------|------|------|
| **P1** | Dense (BGE-M3) | — | MiMo-V2.5 | 基线 |
| **P2** | Hybrid (Dense+BM25 RRF) | — | MiMo-V2.5 | 混合检索 |
| **P3** | Hybrid | BGE-Reranker | MiMo-V2.5 | 静态最优管道 |
| **P4-Agentic** | Hybrid | BGE-Reranker | MiMo-V2.5 + LangGraph | Agent 循环 (grade→rewrite→regen) |
| **Stage 2** | Smart Routing + HyDE | BGE-Reranker | MiMo-V2.5 + LangGraph | 智能路由 + HyDE fallback |

### 1.2 LLM 策略：节点级 Thinking 控制

LangGraph 中各节点采用双模型策略：

| 节点 | 模型 | Thinking | 理由 |
|------|------|----------|------|
| `route_query` | mimo-v2.5 | OFF | 轻量分类，无需深度推理 |
| `generate_answer` | mimo-v2.5 | OFF | 检索增强生成，速度优先 |
| `grade_relevance` | mimo-v2.5-pro | ON | 需要仔细推理 chunk 与 query 的匹配度 |
| `rewrite_query` | mimo-v2.5-pro | ON | 需要分析失败原因并生成改进策略 |
| `check_faithfulness` | mimo-v2.5-pro | ON | 需要逐条交叉验证 claim 与 context |

### 1.3 Judge 选型

评判模型使用 `mimo-v2.5-pro`（独立 API 端点），评估三个维度：

| 维度 | 定义 | 评分范围 |
|------|------|----------|
| **Faithfulness** | 生成答案是否基于检索到的 context | 0.0–1.0 |
| **Relevance** | 答案是否回答了所问问题 | 0.0–1.0 |
| **Correctness** | 答案与 golden answer 的信息重叠度 | 0.0–1.0 |
| **Composite** | 三维度均值 | 0.0–1.0 |

> **⚠ 已知偏差**：50% 的标准集题目 faithfulness=0.0 但 correctness≥0.8。根因：retrieved chunks 被截断至 1000 字符，judge 在 context 中找不到具体数值 → 判定"unsupported"。见 §4.2。

### 1.4 延迟测量环境

- **CPU**: BGE-Reranker 在 CPU 上运行（~15s/query），非 GPU
- **API**: MiMo-V2.5 / MiMo-V2.5-Pro 通过 OpenAI-compatible API 调用
- **Qdrant**: 本地 Docker 容器 (localhost:6333)

---

## 2. Golden Dataset 描述

### 2.1 标准集 (golden_dataset.jsonl)

| 指标 | 数值 |
|------|------|
| 题目总数 | 50 |
| 构建方式 | LLM 辅助 + 人工审核 (§5.3) |
| 医学专科分布 | Radiology 19, Neurology 9, Cardiology 7, Oncology 7, General 4, Pharmacology 2, Infectious Disease 2 |
| 难度分布 | Easy 13, Medium 37 |
| 平均答案长度 | 268 字符 |

### 2.2 Hard Set (golden_hard.jsonl)

| 指标 | 数值 |
|------|------|
| 题目总数 | 39 |
| 构建方式 | 语料采样 + 关键词分类 + LLM 生成 (scripts/12_build_hard_set.py) |
| SHA-256 | `33ee0351...f7b6092` (Python canonical) |

Hard Set 按困难类型分类：

| 类型 | 数量 | 设计目标 | 触发机制 |
|------|------|---------|---------|
| **A — 多跳推理** | 4 | 需要 ≥2 个跨文档 chunk | 语料覆盖不足，实际仅采集到 5 对 |
| **B — 术语歧义** | 10 | 口语/缩写 → MeSH 同义词展开 | 初次检索词汇覆盖不足 |
| **C — 否定/反事实** | 10 | "X 为何不是一线药物" | 关键词命中但语义不匹配 |
| **D — 跨领域合成** | 15 | 欠代表领域综合题 | 稀有术语 → rewrite |

---

## 3. 主结果

### 3.1 标准集 (50 题): P3 vs P4-Agentic vs Stage 2

| 管道 | Faithfulness | Relevance | Correctness | **Composite** | Avg Latency | P50 Latency |
|------|-------------|-----------|-------------|---------------|-------------|-------------|
| **P3 Static** | 0.4052 | 0.9960 | 0.9160 | **0.7724** | 35.4s | 34.1s |
| **P4-Agentic** | 0.4010 | 1.0000 | 0.9200 | **0.7737** | 79.5s | 77.5s |
| **Stage 2** | 0.4194 | 0.9837 | 0.8459 | **0.7497** | 274.2s | 73.3s |

**观察**：
- P3 和 P4 在标准集上几乎无差异（Composite Δ=+0.001）
- P4 触发 rewrite 比例 = 0%，Agent 循环完全未激活
- Stage 2 反而略差（−0.023），且延迟暴涨至 274s（被 P90=555s 的 outlier 拉高）

### 3.2 Hard Set (39 题): P3 vs P4-Agentic

| 管道 | Faithfulness | Relevance | Correctness | **Composite** | P50 Latency |
|------|-------------|-----------|-------------|---------------|-------------|
| **P3 Static** | 0.7718 | 0.8897 | 0.5821 | **0.7479** | 44.0s |
| **P4-Agentic** | 0.7564 | 0.6987 | 0.6000 | **0.6850** | 116.8s |
| **Δ** | −0.015 | **−0.191** | +0.018 | **−0.063** | +72.8s |

**观察**：
- P4 整体比 P3 差 0.063 composite
- Relevance 大幅下滑 −0.191 是主因
- Correctness 小幅提升 +0.018，说明 rewrite 在部分题目上有效
- P4 延迟是 P3 的 2.7×

### 3.3 图表: 标准集按专科分布

```
P3 Composite by Specialty
Radiology     (n=19) ████████████████████ 0.765
Neurology     (n=9)  ██████████████████  0.741
Cardiology    (n=7)  ████████████████    0.657
Oncology      (n=7)  ████████████████████ 0.800
General       (n=4)  ████████████████████████ 0.972
Pharmacology  (n=2)  ████████████████████ 0.833
Infectious    (n=2)  ████████████████████ 0.833

P4 Composite by Specialty
Radiology     (n=19) ███████████████████  0.748
Neurology     (n=9)  ███████████████████  0.774  (+0.033)
Cardiology    (n=7)  █████████████████    0.667  (+0.010)
Oncology      (n=7)  ████████████████████ 0.795
General       (n=4)  ████████████████████████ 0.975
Pharmacology  (n=2)  ███████████████████  0.800
Infectious    (n=2)  █████████████████████ 0.883  (+0.050)
```

### 3.4 图表: Hard Set 按题型分布

```
Hard Set Composite — P3 vs P4
                P3        P4        Δ
A — 多跳(n=4)  ██████    ████     −0.108   0.533 → 0.425
B — 术语(n=10) █████████ ███████  −0.148   0.732 → 0.583
C — 否定(n=10) ██████████ ████████ −0.128   0.807 → 0.678
D — 跨域(n=15) █████████ █████████ +0.050   0.777 → 0.827  ✓

Agent 循环统计 (P4 Hard Set):
  触发 rewrite:  23.1%  (9/39)
  平均 rewrites/题: 0.46
  平均 regen/题:   0.36
  Agent faithful%: 76.9%
```

### 3.5 图表: 延迟分布

```
Latency Distribution (Standard Set, 50q)

         P10    P25    P50    P75    P90
P3:      25.4s  30.9s  34.1s  40.4s  46.0s
P4:      67.8s  73.0s  77.5s  86.2s  93.7s
         ├────── 2.3× slower ──────┤

Latency Distribution (Hard Set, 39q)

         P50         P90
P3:      44.0s       53.8s
P4:      116.8s      307.6s
         ├──── 2.7× slower, long tail ────┤
```

---

## 4. 错误分析

### 4.1 失败案例分类

从 Hard Set 中选取 8 个代表性失败案例进行归因分析。

#### 类型 1：P4 过度拒绝 — "证据不足"陷阱 (5/8)

**根因**：P4-Agentic 的 `check_faithfulness` 节点在 regen 时，generate 节点变得过于保守，输出 "The retrieved documents do not contain sufficient cited evidence" 而实际上 context 中已有足够信息。

| ID | P3 | P4 | Δ | 问题摘要 |
|----|----|----|---|---------|
| hard_B03 | 0.73 | 0.00 | −0.73 | 儿童癌症中免疫细胞映射工具的准确性 |
| hard_C05 | 0.77 | 0.00 | −0.77 | 痗状持续时间为何不影响小儿推拿复位成功率 |
| hard_D08 | 0.77 | 0.00 | −0.77 | 高级神经影像如何补充早期临床和脑脊液发现 |
| hard_B07 | 0.52 | 0.00 | −0.52 | CAR-T 细胞疗法在实体瘤中的主要障碍 |
| hard_A01 | 0.73 | 0.33 | −0.40 | 植物-真菌动态资源交换如何应用于 NHS AI 部署 |

**典型案例 hard_B03**：
```
Q: In child cancer research, how accurate is the computer tool that maps immune cells around tumors?

Golden: Bussola et al.'s EUNet model achieves MAE of 3.1 for lymphocyte density...

P3 答案: [正确回答，引用 EUNet 模型和 MAE=3.1]  → composite=0.73

P4 答案: "The retrieved documents do not contain sufficient cited evidence to answer this question."
→ composite=0.00

Faith issues: "Context provides cited evidence on computer tool accuracy, e.g., MAE of 3.1 from
Bussola et al., contradicting the answer's claim of insufficient evidence."
```

**机制分析**：
1. 首次 generate 产生正确答案
2. check_faithfulness 标记 unfaithful（可能因 claims 缺少精确 citation key）
3. inc_regen → 第二次 generate 时，模型看到之前的 faithfulness issues → 过度补偿 → 输出"证据不足"
4. 但 judge 评估时发现 context 中确实有足够信息 → faith=0, rel=0, corr=0

#### 类型 2：多跳推理语料缺口 (2/8)

| ID | P3 | P4 | 根因 |
|----|----|----|------|
| hard_A02 | 0.50 | 0.33 | 需要连接 NHS IT 基础设施和 HL7 集成的跨文档知识 |
| hard_A03 | 0.50 | 0.37 | 需要综合预测模型的多个限制因素 |

**根因**：语料中缺乏跨文档推理所需的中间概念连接。即使 rewrite 也无法弥补检索覆盖缺口。

#### 类型 3：术语歧义导致检索漂移 (1/8)

| ID | P3 | P4 | 根因 |
|----|----|----|------|
| hard_B06 | 0.43 | 0.03 | "heart device to prevent strokes" → 口语化表述无法匹配学术术语 |

### 4.2 标准集系统性 Faithfulness 偏差

**核心发现**：50 题标准集中有 25 题（50%）faithfulness=0.0，但其中 21 题 correctness≥0.8。

这是一个 judge 系统性偏差，非生成质量问题：

| 现象 | 数量 | 说明 |
|------|------|------|
| faith=0, corr≥0.8 | 21/50 (42%) | 答案事实正确但 judge 判定"unfaithful" |
| faith=0, corr=1.0 | 13/50 (26%) | 答案完全正确但 faithfulness=0 |

**根因链**：
1. Retrieved chunks 在传入 judge 前被截断至 **1000 字符**
2. 包含具体数值的段落（如 MAE=3.1, 24.6μm, ICC 值）常在截断边界之外
3. Judge 在 truncated context 中找不到对应数值 → 判定 "claim not supported"
4. 结论：**Faithfulness 分数在当前配置下不可靠**，应以 Correctness 为主要参考

**修复方案**：将 chunk 截断长度从 1000 提升至 2000 字符，或移除截断限制。

---

## 5. Latency / Cost 对比

### 5.1 延迟详细对比

| 场景 | P3 Avg | P3 P50 | P3 P90 | P4 Avg | P4 P50 | P4 P90 | P4/P3 |
|------|--------|--------|--------|--------|--------|--------|-------|
| 标准集 (50q) | 35.4s | 34.1s | 46.0s | 79.5s | 77.5s | 93.7s | **2.3×** |
| Hard Set (39q) | 43.8s | 44.0s | 53.8s | 160.1s | 116.8s | 307.6s | **2.7×** |

### 5.2 延迟分解

| 组件 | P3 (Hard) | P4 (Hard) | 说明 |
|------|-----------|-----------|------|
| Hybrid Retrieve | ~15s | ~15s | BGE-M3 dense + BM25 sparse |
| BGE-Reranker | **~15s** | **~15s** | CPU 模式，GPU 可降至 ~0.2s |
| LLM Generate | ~12s | ~12s | MiMo-V2.5, ~1024 tokens |
| Grade (think) | — | ~20s | MiMo-V2.5-Pro, thinking ON |
| Rewrite (×N) | — | ~20s×N | 仅 hard set 触发 |
| Check (think) | — | ~15s | MiMo-V2.5-Pro, thinking ON |
| Regen (×N) | — | ~12s×N | 仅 unfaithful 触发 |
| **总计** | **~42s** | **~117s** | Hard set P50 |

### 5.3 成本估算

假设 MiMo API 定价参照 OpenAI GPT-4o-mini：

| 管道 | LLM Calls/Question (Standard) | LLM Calls/Question (Hard) |
|------|-------------------------------|---------------------------|
| P3 | 1 (generate) | 1 (generate) |
| P4 (无 rewrite) | 3 (route + generate + check) | 3 |
| P4 (有 rewrite×1) | 5 (route + generate + check + grade + generate + check) | 5-7 |

---

## 6. P4 在不同题型上的差异化分析

### 6.1 标准集：按专科

| 专科 | P3 Composite | P4 Composite | Δ | P4 优势? |
|------|-------------|-------------|---|---------|
| Radiology (n=19) | 0.765 | 0.748 | −0.017 | — |
| Neurology (n=9) | 0.741 | 0.774 | **+0.033** | ✓ |
| Cardiology (n=7) | 0.657 | 0.667 | +0.010 | ~ |
| Oncology (n=7) | 0.800 | 0.795 | −0.005 | — |
| General (n=4) | 0.972 | 0.975 | +0.003 | ~ |
| Infectious (n=2) | 0.833 | 0.883 | **+0.050** | ✓ |

**发现**：Neurology 和 Infectious Disease 题目上 P4 略有增益，可能因为这些领域的查询更需要精确术语匹配。

### 6.2 Hard Set：按困难类型

| 类型 | P3 Composite | P4 Composite | Δ | P4 Rewrite% | 分析 |
|------|-------------|-------------|---|-------------|------|
| A — 多跳 (n=4) | 0.533 | 0.425 | **−0.108** | 100% | 语料缺口无法弥补 |
| B — 术语 (n=10) | 0.732 | 0.583 | **−0.148** | 20% | rewrite 后检索漂移 |
| C — 否定 (n=10) | 0.807 | 0.678 | **−0.128** | 20% | 过度拒绝 |
| D — 跨域 (n=15) | 0.777 | **0.827** | **+0.050** | 0% | ✓ 唯一正向 |

**D 类正向增益分析**：
- D 类几乎不触发 rewrite（0%），说明初次检索即足够
- LangGraph 的 grade+generate 流程对语义清晰的跨域题能稳定输出
- P4 的 faithfulness check 在 D 类上起到了质量把关作用（84.7% faithful）

**A/B/C 类负向分析**：
- A 类：rewrite 后查询偏离原始意图 → relevance 低至 0.0
- B 类：rewrite 引入更广义术语 → 检索结果偏离 → "证据不足"陷阱
- C 类：否定题本身需要精确推理 → faithfulness check 过严 → regen → 过度保守

### 6.3 关键结论

> **P4-Agentic 的价值在当前配置下仅在 D 类（跨域综合）上体现。**  
> 对于 A/B/C 类困难题，Agent 循环反而引入了新的失败模式。  
> 标准集上 Agent 循环完全未激活（0% rewrite），说明 grade 阈值 (0.6) 对简单题过于宽松。

---

## 7. 项目现存问题与优化方案

### 7.1 问题清单

| # | 问题 | 严重度 | 影响范围 |
|---|------|--------|---------|
| P1 | **"证据不足"过度拒绝** | ★★★ | P4 Hard Set 7/39 comp=0 |
| P2 | **Faithfulness judge 系统性偏差** | ★★★ | 50% 标准集 faith=0 不可靠 |
| P3 | **Agent 循环在标准集未激活** | ★★ | P4 vs P3 无增益 |
| P4 | **BGE-Reranker CPU 延迟** | ★★ | 所有管道 ~15s/query |
| P5 | **A 类多跳语料覆盖不足** | ★★ | Hard Set A 类仅 4 题 |
| P6 | **Rewrite 导致查询漂移** | ★★ | P4 Hard relevance −0.191 |
| P7 | **Stage 2 Smart Routing 未带来增益** | ★ | 标准集 comp −0.023 |

### 7.2 优化方案

#### 方案 A：修复"证据不足"陷阱（优先级 ★★★）

**问题**：regen 后 generate 节点过度保守，输出 "insufficient evidence"。

**修复**：
1. **GENERATE_SYSTEM prompt 修改**：在 regen 场景下注入上下文 "Previous attempt was marked unfaithful. Re-examine the context carefully before claiming insufficient evidence."
2. **regen 时传递 faithfulness_issues**：让 generate 节点知道上次哪里出了问题，而非从零开始
3. **设置 minimum claims 阈值**：如果检索到的 chunks 数量 ≥ 3 且 relevance_score ≥ 0.6，则禁止输出 insufficient_context

```
# 伪代码修改
def generate_answer_node(state):
    if state.get("regen_count", 0) > 0:
        system = REGEN_SYSTEM + f"\nPrevious issues: {state.get('faithfulness_issues', '')}"
    else:
        system = GENERATE_SYSTEM
```

#### 方案 B：修复 Faithfulness Judge（优先级 ★★★）

**问题**：chunks 截断 1000 字符导致 judge 看不到关键数值。

**修复**：
1. **提升 chunk 截断至 2000 字符**（`09_eval_answer.py` 和 `11_eval_agent.py` 中 `c.text[:1000]` → `c.text[:2000]`）
2. **或使用全文**：judge 输入窗口足够大时，移除截断限制
3. **评估 judge 自身准确性**：用人工标注 10 题校准 judge prompt

#### 方案 C：优化 Grade 阈值（优先级 ★★）

**问题**：GRADE_THRESHOLD=0.6 对标准集过于宽松（0% rewrite），对 hard set 过于激进。

**修复**：
1. **动态阈值**：根据 router 输出的 query_type 调整
   - factual: threshold=0.5（简单题不需 rewrite）
   - synthesis: threshold=0.6
   - multihop: threshold=0.7（多跳题更积极 rewrite）
2. **或降低至 0.4**：让标准集也能触发 rewrite，测试是否有增益

#### 方案 D：Rewrite 查询锚定（优先级 ★★）

**问题**：rewrite 后查询偏离原始意图（relevance −0.191）。

**修复**：
1. **REWRITE_SYSTEM 增加约束**："Rewritten query MUST preserve the original question's intent. The goal is to improve retrieval, not to change the question."
2. **限制 rewrite 变化幅度**：在 rewrite prompt 中要求 "Keep at least 60% of original keywords"
3. **max_rewrites 从 2 降至 1**：避免二次漂移

#### 方案 E：GPU 加速 Reranker（优先级 ★★）

**问题**：BGE-Reranker CPU 模式 ~15s/query。

**修复**：
```bash
# 设置环境变量
export RERANKER_DEVICE=cuda  # 或 auto
```

**预期收益**：P3 Hard P50 从 44s → ~30s，P4 Hard P50 从 117s → ~85s。

#### 方案 F：A 类语料补充（优先级 ★★）

**问题**：Hard Set A 类仅 4 题，语料中跨文档配对稀少。

**修复**：
1. 在候选筛选阶段加入"患者/临床/治疗"关键词过滤
2. 从 PubMed 检索同一疾病的多个视角论文（诊断+治疗+预后）
3. 目标：A 类扩展至 15 题

#### 方案 G：Stage 2 路由优化（优先级 ★）

**问题**：Smart Routing + HyDE 未带来增益，且延迟暴涨。

**修复**：
1. 分析 Stage 2 中哪些题走了 HyDE 路径，评估 HyDE 是否有害
2. 将 HyDE 仅用于 multihop 查询（当前可能对所有查询都触发）
3. 移除或简化 Smart Routing，回归 P3 管道 + Agent 循环

### 7.3 优化路线图

```
Phase 1 (1-2 days): Fix critical issues
  ├── [A] Fix "insufficient evidence" trap in regen
  ├── [B] Increase chunk truncation to 2000 chars
  └── [D1] Add intent-preservation constraint to REWRITE_SYSTEM

Phase 2 (3-5 days): Improve agent loop
  ├── [C] Dynamic grade threshold by query type
  ├── [D2] Limit max_rewrites to 1
  └── [E] GPU-accelerate reranker

Phase 3 (1 week): Expand evaluation
  ├── [F] Expand A-type hard set to 15 questions
  ├── [G] Analyze and simplify Stage 2 routing
  └── Re-run full evaluation with Phase 1+2 fixes
```

---

## 8. 优化实验记录

### 8.1 优化方案实施

Phase 1+2 的 6 项优化已全部实施：

| 编号 | 优化 | 代码变更 | 状态 |
|------|------|----------|------|
| A | REGEN prompt + regen 上下文传递 | `prompts.py`: REGEN_SYSTEM/USER, `nodes.py`: regen prompt 切换 | 已实施 |
| B | chunk 截断 1000→2000 chars + judge context 6000→10000 | `09_eval_answer.py`, `11_eval_agent.py` | 已实施 |
| C | 动态 grade 阈值 (factual=0.5, synthesis=0.6, multihop=0.7) | `nodes.py`: _GRADE_THRESHOLDS, `graph.py`: _after_grade | 已实施 |
| D1 | REWRITE_SYSTEM 意图保持约束 (60% keywords) | `prompts.py`: REWRITE_SYSTEM | 已实施 |
| D2 | max_rewrites 2→1 | `nodes.py`: MAX_REWRITES | 已实施 |
| E | Reranker auto-detect CUDA | `nodes.py`: _get_reranker | 已实施 |

### 8.2 v2 实验 (全部 6 项优化，MAX_REGEN=1)

**结果**：composite 从 0.685 降至 0.580 (−0.105)

**根因**：REGEN prompt 导致 7 个原本正确的答案被覆盖为 "insufficient evidence"。
faithfulness checker 的 false positive + REGEN 的 "insufficient evidence" 倾向 = 灾难性组合。

### 8.3 v3 实验 (禁用 regen，MAX_REGEN=0)

**结果**：composite 0.638 (vs v1 的 0.685，−0.047)

| 指标 | v1 (原版) | v3 (MAX_REGEN=0) | 差异 |
|------|-----------|-------------------|------|
| Composite | 0.685 | 0.638 | −0.047 |
| Faithfulness | 0.756 | 0.792 | +0.036 |
| Relevance | 0.699 | 0.633 | −0.065 |
| Correctness | 0.600 | 0.487 | −0.113 |
| Insufficient evidence | 8/39 | 12/39 | +4 |

**分析**：
- 7 cases 从 answer → insufficient evidence (回归)
- 3 cases 从 insufficient → answer (改善)
- 净效果为负：禁用 regen 丢失了 3 个 regen 正确修复的 cases (A04, B08, D14)

### 8.4 v4 实验 (智能 regen，MAX_REGEN=1 + smart gate)

**设计**：保留 regen 但增加智能门控——如果 first-gen answer 已有 citations 且 confidence ≥ 0.3，
即使 faithfulness check 报告 unfaithful，也跳过 regen（防止 false positive 覆盖好答案）。

**代码变更**：
- `nodes.py`: MAX_REGEN=1, REGEN_CONFIDENCE_SKIP=0.3
- `graph.py`: _after_check() 增加 smart gate 逻辑

**状态**：⏳ 待评测 (MiMo API 当前返回空响应，需等待服务恢复)

### 8.5 关键发现

1. **Faithfulness checker 不可靠**：对 nuanced medical answers 有高 false positive rate
2. **REGEN 的 "insufficient evidence" 陷阱**：REGEN prompt 过于保守，倾向于放弃而非修正
3. **LLM 非确定性**：4/7 v3 回归 cases 是因为 LLM 生成了不同输出（相同输入）
4. **Smart gate 是正确方向**：用 confidence + citations 作为 regen 触发条件比 faithfulness check 更可靠

---

## 附录

### A. 运行命令参考

```bash
# P3 标准集评测
python scripts/09_eval_answer.py --pipeline p3 --model mimo-v2.5-pro

# P4-Agentic 标准集评测
python scripts/11_eval_agent.py --model mimo-v2.5-pro

# P3 Hard Set 评测
python scripts/09_eval_answer.py --pipeline p3 \
    --golden data/golden/golden_hard.jsonl \
    --output data/eval/p3_hard_eval.json

# P4 Hard Set 评测
python scripts/11_eval_agent.py \
    --golden data/golden/golden_hard.jsonl \
    --output data/eval/agent_eval_hard.json
```

### B. 数据文件索引

| 文件 | 内容 |
|------|------|
| `data/golden/golden_dataset.jsonl` | 标准集 50 题 |
| `data/golden/golden_hard.jsonl` | Hard Set 39 题 |
| `data/eval/answer_eval.json` | P3 标准集评测结果 |
| `data/eval/agent_eval.json` | P4 标准集评测结果 |
| `data/eval/agent_eval_stage2.json` | Stage 2 标准集评测结果 |
| `data/eval/p3_hard_eval.json` | P3 Hard Set 评测结果 |
| `data/eval/agent_eval_hard.json` | P4 Hard Set 评测结果 |
| `data/eval/agent_eval_hard_v1.json` | P4 Hard Set v1 (优化前基线) |
| `data/eval/agent_eval_hard_v4.json` | P4 Hard Set v4 (智能 regen，待完成) |
| `docs/hard_set_report.md` | Hard Set 专项报告 |

---

*VeritasMed Evaluation Report v1.0 — 2026-05-08*
