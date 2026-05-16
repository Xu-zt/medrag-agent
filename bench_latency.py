"""
Latency benchmark — measures each pipeline component independently.

Usage:
    python bench_latency.py [--skip-llm] [--skip-qdrant]

Components tested:
  1. Embedder  — dense encode, dense+sparse encode (warmup + 3 runs)
  2. Reranker  — 20 pairs (matches production usage)
  3. LLM API   — route/generate (thinking=OFF) + grade/check (thinking=ON)
  4. Qdrant    — dense query, sparse query (if Qdrant is running)
  5. Summary   — projected end-to-end for P3 and P4
"""
import sys, os, time, argparse, statistics
sys.path.insert(0, "src")

import sentence_transformers  # torch must load before qdrant_client

from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--skip-llm",    action="store_true")
parser.add_argument("--skip-qdrant", action="store_true")
args = parser.parse_args()

QUERY = "What are the contraindications of warfarin in elderly patients?"
FAKE_CHUNKS = [
    f"Warfarin is an anticoagulant used to prevent blood clots. Contraindication {i}: "
    f"Active bleeding, severe hypertension, pregnancy, recent surgery. "
    f"Elderly patients require careful dose adjustment due to increased sensitivity." * 3
    for i in range(20)
]

results: dict[str, float] = {}

def timeit(label, fn, runs=3, warmup=1):
    """Run fn() warmup+runs times, return median of timed runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    results[label] = med
    print(f"  {label:<42} {med*1000:>8.0f} ms   (runs={runs}, min={min(times)*1000:.0f}, max={max(times)*1000:.0f})")
    return med


# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Embedder (BGEM3Embedder via M3Embedder) ──────────────────────────")
from medrag.index.embedder import BGEM3Embedder
import torch

device = "auto"
emb = BGEM3Embedder(device=device)
actual_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  device: {actual_device}")

timeit("encode dense ×1",           lambda: emb.encode([QUERY], return_sparse=False))
timeit("encode dense+sparse ×1",    lambda: emb.encode([QUERY], return_sparse=True))
timeit("encode dense+sparse ×8",    lambda: emb.encode([QUERY]*8, return_sparse=True))


# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Reranker (BGEReranker CrossEncoder) ──────────────────────────────")
from medrag.retrieval.retriever import RetrievedChunk
from medrag.retrieval.reranker import BGEReranker

reranker = BGEReranker()  # auto device
fake_retrieved = [
    RetrievedChunk(chunk_id=f"c{i}", text=FAKE_CHUNKS[i], score=0.9-i*0.01, payload={})
    for i in range(20)
]
# Note: timeit warmup=1 includes one full run first
timeit("rerank 20 pairs → top-5",   lambda: reranker.rerank(QUERY, fake_retrieved, top_k=5), runs=3, warmup=1)
timeit("rerank 5 pairs → top-5",    lambda: reranker.rerank(QUERY, fake_retrieved[:5], top_k=5), runs=3, warmup=1)


# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. LLM API calls ────────────────────────────────────────────────────")
if args.skip_llm:
    print("  SKIPPED (--skip-llm)")
else:
    from medrag.agent.llms import make_llm_fast, make_llm_think
    from langchain_core.messages import HumanMessage, SystemMessage

    fast_model  = os.getenv("MIMO_MODEL_FAST",  "mimo-v2.5")
    think_model = os.getenv("MIMO_MODEL_THINK", "mimo-v2.5-pro")
    backend     = os.getenv("LLM_BACKEND", "mimo")
    print(f"  backend={backend}  fast_model={fast_model}  think_model={think_model}")

    llm_fast  = make_llm_fast()
    llm_think = make_llm_think()

    short_sys = "You are a medical assistant. Be concise."
    route_msgs = [
        SystemMessage(content="Classify the query as: factual, synthesis, or multihop. Reply with one word only."),
        HumanMessage(content=QUERY),
    ]
    gen_sys = (
        "Answer the medical question using ONLY the documents below. "
        "Cite sources as [PMID:xxx]. Be concise (2-3 sentences)."
    )
    gen_msgs = [
        SystemMessage(content=gen_sys),
        HumanMessage(content=f"Q: {QUERY}\n\nDoc: {FAKE_CHUNKS[0][:400]}\n\nAnswer:"),
    ]
    grade_msgs = [
        SystemMessage(content="Score 0.0-1.0 how well the document answers the query. Reply JSON: {\"score\": 0.8}"),
        HumanMessage(content=f"Query: {QUERY}\nDoc: {FAKE_CHUNKS[0][:400]}"),
    ]

    print(f"\n  thinking=OFF ({fast_model}):")
    timeit("  route call (1 token out)",   lambda: llm_fast.invoke(route_msgs), runs=3, warmup=0)
    timeit("  generate call (~200 tok out)", lambda: llm_fast.invoke(gen_msgs),  runs=3, warmup=0)

    print(f"\n  thinking=ON ({think_model}):")
    timeit("  grade call (thinking=ON)",   lambda: llm_think.invoke(grade_msgs), runs=3, warmup=0)


# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. Qdrant retrieval ─────────────────────────────────────────────────")
if args.skip_qdrant:
    print("  SKIPPED (--skip-qdrant)")
else:
    try:
        from qdrant_client import QdrantClient
        from medrag.retrieval.hybrid import HybridRetriever

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url, timeout=10)
        client.get_collections()
        print(f"  Qdrant UP at {qdrant_url}")

        retriever = HybridRetriever(client, emb, candidate_k=20)
        timeit("hybrid retrieve (dense+sparse →20)", lambda: retriever.retrieve(QUERY, k=20), runs=3, warmup=1)
        timeit("hybrid retrieve (dense+sparse →5)",  lambda: retriever.retrieve(QUERY, k=5),  runs=3, warmup=0)

    except Exception as e:
        print(f"  SKIPPED — Qdrant not reachable ({e})")
        results["Qdrant hybrid retrieve"] = None


# ══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════════════════════════")
print("LATENCY SUMMARY (median, ms)")
print("══════════════════════════════════════════════════════════════════════════")
for k, v in results.items():
    if v is not None:
        print(f"  {k:<42}  {v*1000:>8.0f} ms")

# Projected totals
print()
embed   = results.get("encode dense+sparse ×1", 0)
rerank  = results.get("rerank 20 pairs → top-5", 0)
qdrant  = results.get("hybrid retrieve (dense+sparse →20)", embed + 0.05)
route   = results.get("  route call (1 token out)", 0)
gen     = results.get("  generate call (~200 tok out)", 0)
grade   = results.get("  grade call (thinking=ON)", 0)

p3_est = embed + qdrant + rerank + route + gen
p4_est = embed + qdrant + rerank + route + grade + gen + grade  # grade + check ≈ 2× grade

if p3_est > 0:
    print(f"  {'── Projected P3 (route+retrieve+rerank+generate)':<42}  {p3_est*1000:>8.0f} ms")
if p4_est > 0:
    print(f"  {'── Projected P4 (+ grade + check, no rewrite)':<42}  {p4_est*1000:>8.0f} ms")

print()
# Diagnose bottleneck
bottlenecks = sorted([(v, k) for k, v in results.items() if v is not None], reverse=True)
print("TOP BOTTLENECKS:")
for v, k in bottlenecks[:4]:
    print(f"  {k:<42}  {v*1000:>8.0f} ms")
