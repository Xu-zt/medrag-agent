# VeritasMed 端到端延迟报告

> 版本: v2.0 · 日期: 2026-05-08  
> 对比基准: Stage 3 优化前（CPU-only, Ollama Qwen3-8B）vs 优化后（MiMo API + GPU Reranker）

---

## 1. 测量方法

```python
# scripts/11_eval_agent.py 中的 agent_latency 字段即单次 wall-clock 时间
t0 = time.perf_counter()
result = app.invoke(initial_state, config=config)
agent_latency = time.perf_counter() - t0
```

每阶段耗时通过日志 `[retrieve]`、`[rerank]`、`[grade]`、`[generate]`、`[check]` 前缀的时间戳差分获得。

---

## 2. Stage 3 优化前基线（Ollama Qwen3-8B on CPU）

| 阶段 | P50 延迟 | 备注 |
|------|---------|------|
| BGE-M3 编码（dense+sparse） | ~0.3 s | CPU，已较快 |
| Qdrant 向量检索 | ~0.1 s | 本地 in-memory |
| BGE-Reranker（20 对）| **~20 s** | CPU 瓶颈 |
| route_query（LLM, thinking=OFF）| ~5 s | Qwen3-8B local |
| grade_relevance（LLM, thinking=ON）| ~15 s | 含 thinking block |
| generate_answer（LLM, thinking=OFF）| ~10 s | Qwen3-8B local |
| check_faithfulness（LLM, thinking=ON）| ~12 s | 含 thinking block |
| **端到端 P50（单次检索，无 rewrite）** | **~64 s** | eval 脚本实测均值 |

---

## 3. Stage 3 优化后目标（MiMo API + GPU Reranker）

| 阶段 | 预期延迟 | 改动 |
|------|---------|------|
| BGE-M3 编码 | ~0.3 s | 不变 |
| Qdrant 向量检索 | ~0.1 s | 不变 |
| BGE-Reranker（20 对，CUDA fp16） | **~0.2 s** | `RERANKER_DEVICE=cuda` |
| route_query（MiMo-V2.5） | ~0.5–1 s | API 调用 |
| grade_relevance（MiMo-V2.5-Pro） | ~1–2 s | API 调用，Pro 含 thinking |
| generate_answer（MiMo-V2.5） | ~1–2 s | API 调用 |
| check_faithfulness（MiMo-V2.5-Pro）| ~1–2 s | API 调用 |
| **端到端 P50（预估）** | **< 8 s** | 目标 |

---

## 4. 实测结果

> ⚠️ 此节待填入：切换到 MiMo API 后，运行 `python scripts/11_eval_agent.py` 并记录 `agent_latency_s` 字段分布。

| 指标 | 数值 |
|------|------|
| P50 延迟（50 题均值） | *待测* |
| P90 延迟 | *待测* |
| 最快单题 | *待测* |
| 最慢单题（含 rewrite） | *待测* |
| BGE-Reranker 实测（GPU/CPU） | *待测* |

运行方式：
```bash
# 确保 .env 中 LLM_BACKEND=mimo, RERANKER_DEVICE=cuda
python scripts/11_eval_agent.py --output data/eval/agent_eval_mimo.json
# 查看延迟分布
python -c "
import json, statistics
data = json.load(open('data/eval/agent_eval_mimo.json'))
lats = [r['agent_latency_s'] for r in data['results'] if 'agent_latency_s' in r]
print(f'P50={statistics.median(lats):.1f}s  P90={sorted(lats)[int(len(lats)*0.9)]:.1f}s  mean={statistics.mean(lats):.1f}s')
"
```

---

## 5. 延迟优化路径总结

```
Stage 3 前: GPU → Qwen3-8B LLM (5.2 GB VRAM 占满)
            CPU → BGE-M3 + BGE-Reranker (主瓶颈)

Stage 3 后: GPU → BGE-Reranker-v2-m3 (~2 GB VRAM, fp16, <200ms/20对)
            API → MiMo-V2.5 / V2.5-Pro  (网络延迟 ~1-3s/调用)
            CPU → BGE-M3 embedding (轻量，0.3s)
```

RTX 4060 8 GB VRAM 分配：
- Reranker fp16: ~2 GB
- 剩余 ~6 GB 可用于未来扩展（图像模型、更大 embedding 模型）

---

*VeritasMed Latency Report v2.0 — 2026-05-08*
