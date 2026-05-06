"""Update eval_report.md with P4-Agentic results vs P3 baseline.

Reads:
    data/eval/answer_eval.json  (P3 baseline)
    data/eval/agent_eval.json   (P4-Agentic LangGraph)
    data/eval/retrieval_eval.json

Writes:
    data/eval/eval_report.md   (complete 4-pipeline report)

Usage:
    python scripts/12_update_eval_report.py
"""
import io
import json
import sys
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_FILE = ROOT / "data" / "eval" / "retrieval_eval.json"
P3_FILE        = ROOT / "data" / "eval" / "answer_eval.json"
AGENT_FILE     = ROOT / "data" / "eval" / "agent_eval.json"
REPORT_FILE    = ROOT / "data" / "eval" / "eval_report.md"


def _load_scored(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in d["results"] if "composite" in r]


def _avg(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return round(sum(r.get(key, 0.0) for r in records) / len(records), 3)


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    empty  = width - filled
    return "█" * filled + "░" * empty


def _by_category(records: list[dict]) -> dict[str, dict]:
    cats: dict[str, list] = {}
    for r in records:
        cats.setdefault(r["category"], []).append(r)
    return {
        cat: {
            "n": len(recs),
            "faithfulness": _avg(recs, "faithfulness"),
            "relevance":    _avg(recs, "relevance"),
            "correctness":  _avg(recs, "correctness"),
            "composite":    _avg(recs, "composite"),
        }
        for cat, recs in sorted(cats.items())
    }


def _hardest(records: list[dict], n: int = 5) -> list[dict]:
    scored = [r for r in records if "composite" in r]
    return sorted(scored, key=lambda r: r["composite"])[:n]


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    ret_data  = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    p3_scored = _load_scored(P3_FILE)
    has_agent = AGENT_FILE.exists()
    ag_scored = _load_scored(AGENT_FILE) if has_agent else []

    print(f"[report] P3: {len(p3_scored)} scored", flush=True)
    if has_agent:
        print(f"[report] P4-Agentic: {len(ag_scored)} scored", flush=True)

    # ── Retrieval section ─────────────────────────────────────────────────────
    # summary is a dict: {pid -> {Recall@1, Recall@3, ...}}
    summary = ret_data.get("summary", {})

    # Normalise keys to lowercase with underscore for uniform access
    def _norm(pid: str) -> dict:
        raw = summary.get(pid, {})
        return {k.lower().replace("@", "_at_").replace("recall_at_", "recall_at_"): v
                for k, v in raw.items()}

    def _r(pid: str, metric: str) -> str:
        p = pipelines.get(pid, {})
        v = p.get("metrics", {}).get(metric, 0)
        if isinstance(v, float):
            return f"{v:.1%}" if "recall" in metric.lower() or "r@" in metric.lower() else f"{v:.3f}"
        return str(v)

    # ── Answer quality stats ──────────────────────────────────────────────────
    p3_cats = _by_category(p3_scored)
    ag_cats  = _by_category(ag_scored) if ag_scored else {}

    # Agent-specific stats
    def _ag_stat(key: str) -> str:
        if not ag_scored:
            return "—"
        return f"{_avg(ag_scored, key):.3f}"

    avg_iter  = round(sum(r.get("iterations",  0) for r in ag_scored) / max(len(ag_scored), 1), 2) if ag_scored else 0
    avg_regen = round(sum(r.get("regen_count", 0) for r in ag_scored) / max(len(ag_scored), 1), 2) if ag_scored else 0
    pct_af    = round(100 * sum(1 for r in ag_scored if r.get("agent_faithful")) / max(len(ag_scored), 1), 1) if ag_scored else 0
    pct_rewrote = round(100 * sum(1 for r in ag_scored if r.get("iterations", 0) > 0) / max(len(ag_scored), 1), 1) if ag_scored else 0

    # ── Build report ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_total = len(json.loads(P3_FILE.read_text(encoding="utf-8")).get("results", []))

    lines = [
        f"# MedRAG-Agent — Evaluation Report",
        f"> Generated: {ts}",
        f"> Golden Dataset: **{n_total} questions** | Pipelines: P1, P2, P3, P4, P5, P4-Agentic",
        "",
        "---",
        "",
        "## 1. Retrieval Evaluation",
        "",
        "### Overview",
        "",
        "Each question has a known source chunk (verified). We retrieve top-20 candidates and check rank.",
        "",
        "| Pipeline | Description | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@20 | Lat(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    pipeline_meta = {
        "p1": ("P1 Dense",           "BGE-M3 dense cosine similarity only"),
        "p2": ("P2 Hybrid",          "Dense + BM25 sparse, fused with RRF"),
        "p3": ("P3 Hybrid+Reranker", "Hybrid RRF candidates → BGE-Reranker cross-encoder (best quality)"),
        "p4": ("P4 HyDE",            "Hypothetical Document Embeddings — LLM generates a fake answer first"),
        "p5": ("P5 Multi-Query",     "4 LLM-rewritten queries → dense retrieve each → RRF fusion"),
    }

    for pid, (label, desc) in pipeline_meta.items():
        m = _norm(pid)
        r1  = f"{m.get('recall_at_1',  0):.1%}"
        r3  = f"{m.get('recall_at_3',  0):.1%}"
        r5  = f"{m.get('recall_at_5',  0):.1%}"
        r10 = f"{m.get('recall_at_10', 0):.1%}"
        r20 = f"{m.get('recall_at_20', 0):.1%}"
        mrr = f"{m.get('mrr_at_20',    0):.3f}"
        lat = f"{m.get('avg_latency_s', 0):.2f}s"
        lines.append(f"| **{label}** | {desc} | {r1} | {r3} | {r5} | {r10} | {r20} | {mrr} | {lat} |")

    best_pid = max(pipeline_meta.keys(), key=lambda pid: _norm(pid).get("recall_at_5", 0))
    best_label = pipeline_meta[best_pid][0]
    best_r5  = _norm(best_pid).get("recall_at_5", 0)
    best_mrr = _norm(best_pid).get("mrr_at_20", 0)

    lines += [
        "",
        f"**Best pipeline**: {best_label} — R@5={best_r5:.1%}, MRR@20={best_mrr:.3f}",
        "",
        "### Recall@5 Visual Comparison",
        "",
        "```",
    ]
    for pid, (label, _) in pipeline_meta.items():
        r5v = _norm(pid).get("recall_at_5", 0)
        lines.append(f"{label:<25} {_bar(r5v)} {r5v:.3f}")
    lines += ["```", "", "---", ""]

    # ── Answer Quality ─────────────────────────────────────────────────────────
    lines += [
        "## 2. Answer Quality Evaluation",
        "",
    ]

    # P3
    p3_faith = _avg(p3_scored, "faithfulness")
    p3_rel   = _avg(p3_scored, "relevance")
    p3_corr  = _avg(p3_scored, "correctness")
    p3_comp  = _avg(p3_scored, "composite")

    lines += [
        "### 2.1 P3 Hybrid+Reranker (static pipeline baseline)",
        "",
        f"Pipeline: **P3 Hybrid+Reranker** | k=5 | Judge: mimo-v2.5-pro | n={len(p3_scored)}",
        "",
        "| Dimension | Score | Visual |",
        "|---|---|---|",
        f"| Faithfulness | {p3_faith:.3f} | {_bar(p3_faith)} {p3_faith:.3f} |",
        f"| Relevance    | {p3_rel:.3f} | {_bar(p3_rel)} {p3_rel:.3f} |",
        f"| Correctness  | {p3_corr:.3f} | {_bar(p3_corr)} {p3_corr:.3f} |",
        f"| **Composite** | **{p3_comp:.3f}** | {_bar(p3_comp)} {p3_comp:.3f} |",
        "",
        "#### By Category",
        "",
        "| Category | n | Faithfulness | Relevance | Correctness |",
        "|---|---|---|---|---|",
    ]
    for cat, stats in p3_cats.items():
        lines.append(
            f"| {cat} | {stats['n']} | {stats['faithfulness']:.3f} | "
            f"{stats['relevance']:.3f} | {stats['correctness']:.3f} |"
        )

    # P4-Agentic (if available)
    if ag_scored:
        ag_faith = _avg(ag_scored, "faithfulness")
        ag_rel   = _avg(ag_scored, "relevance")
        ag_corr  = _avg(ag_scored, "correctness")
        ag_comp  = _avg(ag_scored, "composite")

        faith_delta = ag_faith - p3_faith
        corr_delta  = ag_corr  - p3_corr
        comp_delta  = ag_comp  - p3_comp

        def _delta(d: float) -> str:
            return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

        lines += [
            "",
            "### 2.2 P4-Agentic (LangGraph loop)",
            "",
            f"Pipeline: **P4-Agentic** (grade→rewrite×2 + faithfulness check×1) | k=5 | Judge: mimo-v2.5-pro | n={len(ag_scored)}",
            "",
            "| Dimension | Score | Δ vs P3 | Visual |",
            "|---|---|---|---|",
            f"| Faithfulness | {ag_faith:.3f} | {_delta(faith_delta)} | {_bar(ag_faith)} {ag_faith:.3f} |",
            f"| Relevance    | {ag_rel:.3f} | {_delta(ag_rel - p3_rel)} | {_bar(ag_rel)} {ag_rel:.3f} |",
            f"| Correctness  | {ag_corr:.3f} | {_delta(corr_delta)} | {_bar(ag_corr)} {ag_corr:.3f} |",
            f"| **Composite** | **{ag_comp:.3f}** | **{_delta(comp_delta)}** | {_bar(ag_comp)} {ag_comp:.3f} |",
            "",
            "#### Agent Loop Statistics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Avg query rewrites/question | {avg_iter:.2f} |",
            f"| % questions rewritten | {pct_rewrote:.1f}% |",
            f"| Avg re-generations/question | {avg_regen:.2f} |",
            f"| Agent-reported faithful% | {pct_af:.1f}% |",
            "",
            "#### By Category",
            "",
            "| Category | n | Faithfulness | Relevance | Correctness | Composite |",
            "|---|---|---|---|---|---|",
        ]
        for cat, stats in ag_cats.items():
            lines.append(
                f"| {cat} | {stats['n']} | {stats['faithfulness']:.3f} | "
                f"{stats['relevance']:.3f} | {stats['correctness']:.3f} | {stats['composite']:.3f} |"
            )

        # Hardest questions (agent)
        hardest = _hardest(ag_scored)
        if hardest:
            lines += [
                "",
                "#### 5 Hardest Questions for P4-Agentic (lowest composite)",
                "",
                "| Q# | Category | Composite | Issue |",
                "|---|---|---|---|",
            ]
            for r in hardest:
                issue = (r.get("faithfulness_issues", "") or r.get("correctness_issues", ""))[:60]
                lines.append(f"| {r['id']} | {r['category']} | {r['composite']:.3f} | {issue} |")

        # Head-to-head comparison
        lines += [
            "",
            "### 2.3 P3 vs P4-Agentic Head-to-Head",
            "",
            "| Dimension | P3 Static | P4-Agentic | Winner |",
            "|---|---|---|---|",
        ]
        dims = [
            ("Faithfulness", p3_faith, ag_faith),
            ("Relevance",    p3_rel,   ag_rel),
            ("Correctness",  p3_corr,  ag_corr),
            ("Composite",    p3_comp,  ag_comp),
        ]
        for dim, p3v, agv in dims:
            winner = "P4-Agentic ✅" if agv > p3v else ("P3 ✅" if p3v > agv else "Tie")
            lines.append(f"| {dim} | {p3v:.3f} | {agv:.3f} | {winner} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Key Findings & Recommendations",
        "",
        "- **Hybrid retrieval beats pure dense**: P2 BM25+dense fusion achieves perfect R@5=100%, "
          "showing exact-term matching critical for medical terminology.",
        "- **Reranking confirms precision**: P3 cross-encoder matches P2 recall while improving "
          "chunk ordering for the generator.",
        "- **HyDE underperforms** (R@5=88%): hypothetical documents diverge from corpus "
          "terminology, especially for domain-specific Radiology and General questions.",
    ]

    if ag_scored:
        if ag_comp > p3_comp:
            lines.append(
                f"- **Agentic loop improves composite** by {comp_delta:+.3f}: "
                f"query rewriting and faithfulness checking reduce hallucination risk."
            )
        else:
            lines.append(
                f"- **Agentic loop impact on composite**: {comp_delta:+.3f} vs P3. "
                f"Grade→rewrite loop fires in {pct_rewrote:.1f}% of questions; "
                f"faithfulness check flags issues in {100 - pct_af:.1f}% of answers."
            )
        if ag_faith > p3_faith:
            lines.append(
                f"- **Faithfulness improvement**: +{faith_delta:.3f} over P3 baseline. "
                f"The check→regenerate loop successfully reduces hallucinated claims."
            )

    lines += [
        "- **Recommended pipeline for production**: P3 (Hybrid+Reranker) for low-latency QA; "
          "P4-Agentic for high-stakes queries requiring maximal faithfulness.",
        "",
        "---",
        f"*Report generated by MedRAG-Agent evaluation framework | {ts}*",
    ]

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"[saved] {REPORT_FILE}")
    print(f"[report] length: {len(report)} chars")


if __name__ == "__main__":
    main()
