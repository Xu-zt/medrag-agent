"""Measure grade + check call latency in isolation (no queue pressure)."""
import sys, os, time, statistics
sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv()

from medrag.agent.llms import make_llm_think, make_llm_fast
from langchain_core.messages import HumanMessage, SystemMessage

QUERY = "What are the contraindications of warfarin in elderly patients?"
CHUNKS_CTX = """[PMID:1234] (score=0.91):
Warfarin is an anticoagulant widely prescribed to prevent blood clots. In elderly patients, contraindications include active bleeding, severe hypertension, recent intracranial hemorrhage, and known hypersensitivity. Dose adjustment is essential due to increased sensitivity.

[PMID:5678] (score=0.88):
Age-related pharmacokinetic changes increase warfarin sensitivity in elderly patients. Concurrent NSAID use, uncontrolled hypertension, and renal impairment are major contraindications. Regular INR monitoring is mandatory."""

ANSWER_TEXT = (
    "Warfarin contraindications in elderly patients include active bleeding [PMID:1234], "
    "severe hypertension [PMID:1234], and renal impairment [PMID:5678]. "
    "Close INR monitoring is required due to increased sensitivity [PMID:5678]."
)

grade_sys = (
    'Score 0.0-1.0 how well the provided document chunks answer the query. '
    'Reply JSON: {"relevant": true, "score": 0.8, "reason": "...", "rewrite_hint": ""}'
)
check_sys = (
    "You are a medical fact-checker. Verify every factual claim in the answer is "
    "directly supported by the context documents. Reply JSON: "
    '{"faithful": true, "issues": ""}'
)

fast_model = os.getenv("MIMO_MODEL_FAST", "mimo-v2.5")
think_model = os.getenv("MIMO_MODEL_THINK", "mimo-v2.5-pro")
print(f"fast_model={fast_model}  think_model={think_model}\n")

llm_think = make_llm_think()
llm_fast  = make_llm_fast()


def timeit_isolated(label, llm, msgs, runs=3):
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        r = llm.invoke(msgs)
        t = time.perf_counter() - t0
        times.append(t)
        print(f"    run {i+1}: {t*1000:.0f}ms  content={repr(r.content[:60])}")
    med = statistics.median(times)
    print(f"  {label:<40} median={med*1000:.0f}ms  min={min(times)*1000:.0f}  max={max(times)*1000:.0f}\n")
    return med


print("── grade call (think model, isolated) ──────────────────────────────────")
grade_med = timeit_isolated(
    "grade",
    llm_think,
    [
        SystemMessage(content=grade_sys),
        HumanMessage(content=f"Query: {QUERY}\n\n{CHUNKS_CTX}"),
    ],
)

print("── check call (think model, isolated) ──────────────────────────────────")
check_med = timeit_isolated(
    "check",
    llm_think,
    [
        SystemMessage(content=check_sys),
        HumanMessage(content=f"Context:\n{CHUNKS_CTX}\n\nAnswer: {ANSWER_TEXT}"),
    ],
)

print("── route call (fast model, isolated) ───────────────────────────────────")
route_med = timeit_isolated(
    "route",
    llm_fast,
    [
        SystemMessage(content="Classify as: factual, synthesis, or multihop. Reply with one word only."),
        HumanMessage(content=QUERY),
    ],
)

print("── generate call (fast model, isolated) ────────────────────────────────")
generate_med = timeit_isolated(
    "generate",
    llm_fast,
    [
        SystemMessage(content=(
            "Answer the medical question using ONLY the documents below. "
            "Cite sources. Reply JSON: {\"claims\": [{\"text\": \"...\", \"cite\": [\"PMID:1234\"]}], \"confidence\": 0.9}"
        )),
        HumanMessage(content=f"Q: {QUERY}\n\nContext:\n{CHUNKS_CTX}"),
    ],
)

print("══════════════════════════════════════════════════════════════════════════")
print("SUMMARY (isolated calls, no consecutive-call rate limiting)")
print(f"  route           {route_med*1000:>6.0f} ms")
print(f"  grade           {grade_med*1000:>6.0f} ms")
print(f"  generate        {generate_med*1000:>6.0f} ms")
print(f"  check           {check_med*1000:>6.0f} ms")

retrieve_est = 500
rerank_est = 132
p3 = route_med + retrieve_est/1000 + rerank_est/1000 + grade_med + generate_med + check_med
print(f"\n  Projected P3 happy-path (no rewrite):")
print(f"  route({route_med*1000:.0f}) + retrieve(~500) + rerank({rerank_est}) + grade({grade_med*1000:.0f}) + generate({generate_med*1000:.0f}) + check({check_med*1000:.0f})")
print(f"  = {p3*1000:.0f} ms  ({p3:.1f}s)")
