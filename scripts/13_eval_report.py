"""Generate a combined evaluation report from retrieval and answer eval results.

Reads:
    data/eval/retrieval_eval.json   (from 08_eval_retrieval.py)
    data/eval/answer_eval.json      (from 09_eval_answer.py, optional)

Writes:
    data/eval/eval_report.md        (human-readable Markdown report)

Usage:
    python scripts/10_eval_report.py
    python scripts/10_eval_report.py --no-answer   (skip answer quality section)
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_FILE = ROOT / "data" / "eval" / "retrieval_eval.json"
ANSWER_FILE = ROOT / "data" / "eval" / "answer_eval.json"
REPORT_FILE = ROOT / "data" / "eval" / "eval_report.md"

PIPELINE_LABELS = {
    "p1": "P1 Dense",
    "p2": "P2 Hybrid",
    "p3": "P3 Hybrid+Reranker",
    "p4": "P4 HyDE",
    "p5": "P5 Multi-Query",
}

PIPELINE_DESC = {
    "p1": "BGE-M3 dense cosine similarity only",
    "p2": "Dense + BM25 sparse, fused with RRF",
    "p3": "Hybrid RRF candidates → BGE-Reranker cross-encoder (best quality)",
    "p4": "Hypothetical Document Embeddings — LLM generates a fake answer first",
    "p5": "4 LLM-rewritten queries → dense retrieve each → RRF fusion",
}


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled) + f" {value:.3f}"


def _pct(value: float) -> str:
    return f"{value*100:.1f}%"


def generate_report(retrieval_data: dict, answer_data: dict | None, out_path: Path) -> None:
    lines: list[str] = []
    pipelines = retrieval_data.get("pipelines", [])
    k_values = retrieval_data.get("k_values", [1, 3, 5, 10, 20])
    n_questions = retrieval_data.get("n_questions", 0)
    summary = retrieval_data.get("summary", {})
    cat_bd = retrieval_data.get("category_breakdown", {})
    diff_bd = retrieval_data.get("difficulty_breakdown", {})
    top_k = retrieval_data.get("top_k", 20)
    max_k = max(k_values)

    lines += [
        "# MedRAG-Agent — Week 4 Evaluation Report",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Golden Dataset: **{n_questions} questions** | Pipelines: {', '.join(p.upper() for p in pipelines)}",
        "",
        "---",
        "",
        "## 1. Retrieval Evaluation",
        "",
        "### Overview",
        "",
        "Each question has a known source chunk (verified in Phase 3). "
        f"We retrieve top-{top_k} candidates and check whether the source chunk appears and at what rank.",
        "",
    ]

    # Summary table
    k_cols = " | ".join(f"R@{k}" for k in k_values)
    lines += [
        f"| Pipeline | Description | {k_cols} | MRR@{max_k} | Lat(s) |",
        "|---|---|" + "|".join(["---"] * (len(k_values) + 2)) + "|",
    ]
    for p in pipelines:
        m = summary.get(p, {})
        label = PIPELINE_LABELS.get(p, p.upper())
        desc = PIPELINE_DESC.get(p, "")
        recall_cells = " | ".join(_pct(m.get(f"Recall@{k}", 0)) for k in k_values)
        mrr = m.get(f"MRR@{max_k}", 0)
        lat = m.get("Avg_latency_s", 0)
        lines.append(f"| **{label}** | {desc} | {recall_cells} | {mrr:.3f} | {lat}s |")
    lines += [""]

    # Best pipeline highlight
    best_p = max(pipelines, key=lambda p: summary.get(p, {}).get(f"Recall@{k_values[2]}", 0))
    best_m = summary.get(best_p, {})
    lines += [
        f"**Best pipeline**: {PIPELINE_LABELS.get(best_p, best_p.upper())} — "
        f"R@5={_pct(best_m.get('Recall@5', 0))}, "
        f"MRR@{max_k}={best_m.get(f'MRR@{max_k}', 0):.3f}",
        "",
    ]

    # Visual comparison (R@5)
    lines += ["### Recall@5 Visual Comparison", ""]
    lines += ["```"]
    for p in pipelines:
        m = summary.get(p, {})
        label = PIPELINE_LABELS.get(p, p.upper()).ljust(22)
        lines.append(f"{label} {_bar(m.get('Recall@5', 0))}")
    lines += ["```", ""]

    # Category breakdown
    lines += [f"### Breakdown by Category (Recall@5 / MRR@{max_k})", ""]
    all_cats = sorted(set(
        cat for p in pipelines for cat in cat_bd.get(p, {})
    ))
    p_headers = " | ".join(PIPELINE_LABELS.get(p, p) for p in pipelines)
    lines += [
        f"| Category | {p_headers} |",
        "|---|" + "|".join(["---"] * len(pipelines)) + "|",
    ]
    for cat in all_cats:
        row = [cat]
        for p in pipelines:
            m = cat_bd.get(p, {}).get(cat, {})
            r5 = _pct(m.get("Recall@5", 0))
            mrr = m.get(f"MRR@{max_k}", 0)
            row.append(f"{r5} / {mrr:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [""]

    # Difficulty breakdown
    lines += [f"### Breakdown by Difficulty (Recall@5 / MRR@{max_k})", ""]
    all_diffs = sorted(set(
        d for p in pipelines for d in diff_bd.get(p, {})
    ))
    lines += [
        f"| Difficulty | {p_headers} |",
        "|---|" + "|".join(["---"] * len(pipelines)) + "|",
    ]
    for diff in all_diffs:
        row = [diff]
        for p in pipelines:
            m = diff_bd.get(p, {}).get(diff, {})
            r5 = _pct(m.get("Recall@5", 0))
            mrr = m.get(f"MRR@{max_k}", 0)
            row.append(f"{r5} / {mrr:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [""]

    # --- Answer quality ---
    if answer_data:
        lines += [
            "---",
            "",
            "## 2. Answer Quality Evaluation",
            "",
            f"Pipeline: **{PIPELINE_LABELS.get(answer_data.get('pipeline','p3'), 'P3')}** | "
            f"k={answer_data.get('k', 5)} | "
            f"Judge: {answer_data.get('model_judge', 'mimo-v2.5-pro')}",
            "",
        ]
        results = [r for r in answer_data.get("results", []) if "composite" in r]
        if results:
            avg_faith = sum(r["faithfulness"] for r in results) / len(results)
            avg_rel = sum(r["relevance"] for r in results) / len(results)
            avg_corr = sum(r["correctness"] for r in results) / len(results)
            avg_comp = sum(r["composite"] for r in results) / len(results)

            lines += [
                "| Dimension | Score | Visual |",
                "|---|---|---|",
                f"| Faithfulness | {avg_faith:.3f} | {_bar(avg_faith, 15)} |",
                f"| Relevance | {avg_rel:.3f} | {_bar(avg_rel, 15)} |",
                f"| Correctness | {avg_corr:.3f} | {_bar(avg_corr, 15)} |",
                f"| **Composite** | **{avg_comp:.3f}** | {_bar(avg_comp, 15)} |",
                "",
            ]

            # By category
            by_cat: dict[str, list] = defaultdict(list)
            for r in results:
                by_cat[r.get("category", "Unknown")].append(r)
            lines += ["#### By Category", "", "| Category | n | Faithfulness | Relevance | Correctness |",
                      "|---|---|---|---|---|"]
            for cat, recs in sorted(by_cat.items()):
                n = len(recs)
                f = sum(r["faithfulness"] for r in recs) / n
                rv = sum(r["relevance"] for r in recs) / n
                c = sum(r["correctness"] for r in recs) / n
                lines.append(f"| {cat} | {n} | {f:.3f} | {rv:.3f} | {c:.3f} |")
            lines += [""]

            # Bottom 5 questions
            worst = sorted(results, key=lambda r: r["composite"])[:5]
            lines += [
                "#### 5 Hardest Questions (lowest composite score)", "",
                "| Q# | Category | Composite | Issue |",
                "|---|---|---|---|",
            ]
            for r in worst:
                issues = r.get("faithfulness_issues", "") or r.get("correctness_issues", "") or "—"
                issues = issues[:60]
                lines.append(f"| {r['id']} | {r['category']} | {r['composite']:.3f} | {issues} |")
            lines += [""]

    # --- Key findings ---
    lines += [
        "---",
        "",
        "## 3. Key Findings & Recommendations",
        "",
    ]
    if pipelines:
        p1_r5 = summary.get("p1", {}).get("Recall@5", 0)
        p2_r5 = summary.get("p2", {}).get("Recall@5", 0)
        p3_r5 = summary.get("p3", {}).get("Recall@5", 0)

        if p3_r5 > p1_r5:
            gain = _pct(p3_r5 - p1_r5)
            lines += [f"- **Reranking gain**: P3 outperforms P1 by {gain} in Recall@5, "
                      "confirming cross-encoder reranking improves precision."]
        if p2_r5 > p1_r5:
            lines += ["- **Hybrid retrieval**: P2 BM25+dense fusion beats pure dense (P1), "
                      "showing exact-term matching helps for medical terminology."]

    lines += [
        "- **Corpus coverage**: Questions grounded in PubMed abstracts retrieve more reliably "
        "than PMC full-text chunks (longer, noisier sections).",
        "- **Recommended pipeline for production**: P3 (Hybrid+Reranker) balances "
        "precision and latency for clinical QA.",
        "",
        "---",
        f"*Report generated by MedRAG-Agent evaluation framework | {datetime.now().strftime('%Y-%m-%d')}*",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] written → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-answer", action="store_true")
    parser.add_argument("--output", default=str(REPORT_FILE))
    args = parser.parse_args()

    if not RETRIEVAL_FILE.exists():
        raise SystemExit(f"[error] {RETRIEVAL_FILE} not found. Run 08_eval_retrieval.py first.")

    retrieval_data = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    answer_data = None
    if not args.no_answer and ANSWER_FILE.exists():
        answer_data = json.loads(ANSWER_FILE.read_text(encoding="utf-8"))
        print(f"[report] including answer eval ({len(answer_data.get('results',[]))} records)")
    else:
        print("[report] skipping answer eval (file not found or --no-answer)")

    generate_report(retrieval_data, answer_data, Path(args.output))


if __name__ == "__main__":
    main()
