"""Hard Eval Set Builder — Stage 1.

Pipeline:
  1. Sample Qdrant content chunks, classify by medical domain
  2. Identify candidate chunk-sets for each question type (A/B/C/D)
  3. Use MiMo API to generate question + golden answer from each candidate set
  4. Save to data/golden/golden_hard.jsonl  +  data/eval/gd_hard_candidates.jsonl
  5. SHA-256 lock the output file

Question types:
  A — Multi-hop reasoning (15 questions): requires ≥2 chunks from different docs
  B — Terminology/ambiguity (10): jargon/abbreviation → MeSH expansion needed
  C — Negation/counterfactual (10): "why is X NOT first-line", negative results
  D — Cross-category synthesis (15): underrepresented domains + cross-domain

Usage:
    python scripts/12_build_hard_set.py
    python scripts/12_build_hard_set.py --limit 10   # quick smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_OUT  = ROOT / "data" / "golden" / "golden_hard.jsonl"
CAND_OUT    = ROOT / "data" / "eval"   / "gd_hard_candidates.jsonl"
SHA_OUT     = ROOT / "data" / "eval"   / "golden_hard.sha256"

random.seed(2026)

# ── Domain keyword classifier ──────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Cardiology":       ["cardiac", "heart", "coronary", "myocardial", "atrial",
                         "ventricular", "arrhythmia", "aortic", "pcsk9", "statin",
                         "lipid", "ldl", "ascvd", "cardiovascular", "angina", "heart failure"],
    "Oncology":         ["cancer", "tumor", "tumour", "carcinoma", "lymphoma",
                         "leukemia", "chemotherapy", "immunotherapy", "checkpoint",
                         "bevacizumab", "her2", "metastasis", "radiation", "radiotherapy",
                         "myeloma", "glioma", "melanoma"],
    "Neurology":        ["brain", "neural", "stroke", "epilepsy", "parkinson",
                         "alzheimer", "dementia", "cerebral", "cortical", "seizure",
                         "migraine", "multiple sclerosis", "dopamine", "tia",
                         "ischemic", "hemorrhagic", "neuropathy"],
    "Pharmacology":     ["pharmacokinetic", "pharmacodynamic", "dosage", "adverse",
                         "side effect", "contraindic", "mechanism", "receptor",
                         "inhibitor", "antibiotic", "antiviral", "warfarin",
                         "anticoagul", "drug interaction", "bioavailability",
                         "half-life", "clearance", "toxicity"],
    "Infectious":       ["infection", "infectious", "bacterial", "viral", "hiv",
                         "covid", "tuberculosis", "sepsis", "antimicrobial",
                         "pathogen", "vaccine", "hepatitis", "pneumonia", "fungal",
                         "antibiotic resistance", "mrsa"],
    "Endocrinology":    ["diabetes", "insulin", "thyroid", "hormonal", "glucose",
                         "hba1c", "obesity", "metabolic syndrome", "endocrine",
                         "pancreas", "glucagon", "type 2 diabetes", "hypoglycemia"],
    "Gastroenterology": ["gastric", "colon", "liver", "hepatic", "gastrointestinal",
                         "inflammatory bowel", "crohn", "colitis", "cirrhosis",
                         "hepatocellular", "pancreatitis", "esophageal"],
    "Pulmonology":      ["lung", "pulmonary", "respiratory", "copd", "asthma",
                         "pneumonia", "bronchial", "fibrosis", "pleural",
                         "emphysema", "ventilation", "sputum"],
    "Nephrology":       ["renal", "kidney", "nephrology", "glomerulo", "dialysis",
                         "proteinuria", "creatinine", "acute kidney", "chronic kidney"],
    "Radiology":        ["imaging", "radiology", "mri", "ct scan", "ultrasound",
                         "computed tomography", "radiograph", "diffusion", "contrast",
                         "positron", "pet", "sonograph"],
}

NEGATION_PATTERNS = [
    re.compile(r"not\s+(?:effective|recommended|superior|significant|associated|approved)", re.I),
    re.compile(r"fail(?:ed)?\s+to\s+(?:show|demonstrate|improve|reduce|prevent)", re.I),
    re.compile(r"no\s+(?:significant|statistically|clinically)\s+(?:difference|benefit|improvement|effect)", re.I),
    re.compile(r"contraindicated|contraindication", re.I),
    re.compile(r"did\s+not\s+(?:improve|reduce|show|demonstrate|meet)", re.I),
    re.compile(r"(?:lack|absence)\s+of\s+(?:evidence|efficacy|benefit|effect)", re.I),
    re.compile(r"not\s+(?:first.?line|standard\s+of\s+care|guideline.?recommended)", re.I),
    re.compile(r"inferior\s+to|worse\s+than|no\s+better\s+than", re.I),
]

JARGON_PATTERNS = [
    re.compile(r"\b(?:TIA|STEMI|NSTEMI|ACS|COPD|HbA1c|HER2|BRCA|EGFR|ALK|PD-L1|CAR-T|HSCT)\b"),
    re.compile(r"\b(?:PCI|CABG|TAVR|ICD|CRT|BiVAD|LVAD|SVT|AF|VF|VT)\b"),
    re.compile(r"\b(?:eGFR|CKD|ESRD|AKI|ATN|FSGS|IgA\s*nephropathy)\b"),
    re.compile(r"\b(?:TNF|IL-6|JAK|VEGF|mTOR|PI3K|KRAS|BRAF|MEK)\b"),
    re.compile(r"\b(?:NNT|NNH|RRR|ARR|OR|HR|RR)\s*=?\s*\d"),
    re.compile(r"\b(?:ICU|ARDS|SIRS|qSOFA|SOFA|DIC)\b"),
    re.compile(r"\b(?:tPA|rtPA|alteplase|thromboly|thrombolysis)\b"),
]


def classify_domain(text: str, title: str) -> str:
    combined = (title + " " + text[:400]).lower()
    scores = {d: sum(1 for kw in kws if kw in combined)
              for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def has_negation(text: str) -> bool:
    return any(p.search(text) for p in NEGATION_PATTERNS)


def has_jargon(text: str) -> bool:
    return any(p.search(text) for p in JARGON_PATTERNS)


def chunk_citation(payload: dict) -> str:
    src = payload.get("source", "")
    if src == "pubmed":
        pmid = payload.get("pmid")
        return f"PMID:{pmid}" if pmid else payload["chunk_id"]
    return f"PMC:{payload['doc_id']}"


# ── Qdrant sampling ────────────────────────────────────────────────────────────

def load_corpus_chunks() -> list[dict]:
    """Load content chunks from Qdrant. Returns list of payload dicts."""
    sys.path.insert(0, str(ROOT / "src"))
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    q = QdrantClient(url="http://localhost:6333", timeout=30)

    good_sections = ["ABSTRACT", "INTRO", "RESULTS", "DISCUSS", "CONCL", "CONCLUSION"]
    chunks = []
    next_offset = None

    logger.info("Scrolling PMC content sections …")
    while len(chunks) < 5000:
        pts, next_offset = q.scroll(
            "medrag_text",
            scroll_filter=Filter(must=[
                FieldCondition(key="source", match=MatchValue(value="pmc")),
                FieldCondition(key="section", match=MatchAny(any=good_sections)),
            ]),
            limit=200, offset=next_offset,
            with_payload=True, with_vectors=False,
        )
        for p in pts:
            chunks.append({**p.payload, "_id": str(p.id)})
        if not next_offset:
            break

    logger.info("Scrolling PubMed abstracts …")
    pts2, _ = q.scroll(
        "medrag_text",
        scroll_filter=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value="pubmed")),
        ]),
        limit=1975, with_payload=True, with_vectors=False,
    )
    for p in pts2:
        chunks.append({**p.payload, "_id": str(p.id)})

    logger.info("Total content chunks loaded: %d", len(chunks))
    return chunks


# ── Candidate selection ────────────────────────────────────────────────────────

def select_candidates(chunks: list[dict]) -> dict[str, list]:
    """Return candidate sets for each question type."""

    # Annotate each chunk
    for c in chunks:
        c["domain"]      = classify_domain(c.get("text", ""), c.get("title", ""))
        c["has_neg"]     = has_negation(c.get("text", ""))
        c["has_jargon"]  = has_jargon(c.get("text", ""))
        c["citation"]    = chunk_citation(c)
        c["word_count"]  = len(c.get("text", "").split())

    # Only keep chunks with enough text
    chunks = [c for c in chunks if c["word_count"] >= 80]

    # ── Type A — Multi-hop: find doc_id groups with ≥2 substantive chunks ──────
    by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        if c.get("source") == "pmc":
            by_doc[c["doc_id"]].append(c)

    type_a_candidates: list[dict] = []
    # Also look for cross-doc multi-hop: group by domain, pick pairs from DIFFERENT docs
    domain_docs: dict[str, list] = defaultdict(set)
    for c in chunks:
        domain_docs[c["domain"]].add(c.get("doc_id", c["chunk_id"]))

    for domain in ["Cardiology", "Oncology", "Neurology", "Pharmacology",
                   "Infectious", "Endocrinology", "Gastroenterology"]:
        doc_ids = list(domain_docs[domain])
        if len(doc_ids) < 2:
            continue
        # Pick chunk pairs from 2 different documents in same domain
        for _ in range(10):
            d1, d2 = random.sample(doc_ids, 2)
            chunks_d1 = [c for c in chunks if c.get("doc_id") == d1
                         and c.get("section", "") in ("RESULTS", "DISCUSS", "ABSTRACT", "CONCL")]
            chunks_d2 = [c for c in chunks if c.get("doc_id") == d2
                         and c.get("section", "") in ("RESULTS", "DISCUSS", "ABSTRACT", "CONCL")]
            if chunks_d1 and chunks_d2:
                type_a_candidates.append({
                    "type": "A",
                    "domain": domain,
                    "chunks": [
                        random.choice(chunks_d1),
                        random.choice(chunks_d2),
                    ],
                })

    random.shuffle(type_a_candidates)
    type_a_sel = type_a_candidates[:30]  # keep 30, generate 15

    # ── Type B — Terminology/jargon ───────────────────────────────────────────
    jargon_chunks = [c for c in chunks if c["has_jargon"] and c["word_count"] >= 100]
    random.shuffle(jargon_chunks)
    type_b_sel = [{"type": "B", "domain": c["domain"], "chunks": [c]}
                  for c in jargon_chunks[:25]]

    # ── Type C — Negation/counterfactual ─────────────────────────────────────
    neg_chunks = [c for c in chunks if c["has_neg"] and c["word_count"] >= 80]
    random.shuffle(neg_chunks)
    type_c_sel = [{"type": "C", "domain": c["domain"], "chunks": [c]}
                  for c in neg_chunks[:25]]

    # ── Type D — Cross-category / underrepresented domains ───────────────────
    # Underrepresented: Infectious, Pharmacology, Endocrinology, Gastroenterology, Nephrology
    priority_domains = ["Infectious", "Pharmacology", "Endocrinology",
                        "Gastroenterology", "Nephrology", "Pulmonology"]
    type_d_candidates = []
    for domain in priority_domains:
        dc = [c for c in chunks if c["domain"] == domain and c["word_count"] >= 100]
        random.shuffle(dc)
        for chunk in dc[:8]:
            type_d_candidates.append({"type": "D", "domain": domain, "chunks": [chunk]})

    # Cross-domain pairs: e.g., Cardiology + Pharmacology, Oncology + Infectious
    cross_pairs = [
        ("Cardiology", "Pharmacology"),
        ("Oncology", "Pharmacology"),
        ("Neurology", "Pharmacology"),
        ("Infectious", "Pharmacology"),
        ("Oncology", "Endocrinology"),
        ("Cardiology", "Endocrinology"),
    ]
    for d1, d2 in cross_pairs:
        c1_list = [c for c in chunks if c["domain"] == d1 and c["word_count"] >= 100]
        c2_list = [c for c in chunks if c["domain"] == d2 and c["word_count"] >= 100]
        if c1_list and c2_list:
            type_d_candidates.append({
                "type": "D",
                "domain": f"{d1}+{d2}",
                "chunks": [random.choice(c1_list), random.choice(c2_list)],
            })

    random.shuffle(type_d_candidates)
    type_d_sel = type_d_candidates[:30]

    logger.info("Candidates: A=%d B=%d C=%d D=%d",
                len(type_a_sel), len(type_b_sel), len(type_c_sel), len(type_d_sel))
    return {"A": type_a_sel, "B": type_b_sel, "C": type_c_sel, "D": type_d_sel}


# ── LLM question generation ────────────────────────────────────────────────────

def make_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "")
    url = os.environ.get("OPENAI_BASE_URL", "")
    if not key or not url:
        raise SystemExit("[error] OPENAI_API_KEY / OPENAI_BASE_URL missing")
    return OpenAI(api_key=key, base_url=url)


TYPE_INSTRUCTIONS = {
    "A": (
        "You are constructing a MULTI-HOP medical question that REQUIRES combining information "
        "from BOTH provided document chunks to answer completely. "
        "Neither chunk alone can answer the question — the answer must synthesize BOTH."
    ),
    "B": (
        "You are constructing a TERMINOLOGY question where the query uses colloquial or "
        "abbreviated phrasing (e.g., 'mini-stroke', 'sugar disease', 'water pill') "
        "that must be rewritten using proper medical terminology (MeSH terms, ICD codes) "
        "to retrieve the relevant literature. "
        "The question MUST use lay/abbreviated language that differs from the document's terminology."
    ),
    "C": (
        "You are constructing a NEGATION / COUNTERFACTUAL question based on a document "
        "that contains a negative result, contraindication, or treatment limitation. "
        "The question should ask WHY something is NOT effective, NOT recommended, or NOT first-line. "
        "The answer must come DIRECTLY from the provided document's negative finding."
    ),
    "D": (
        "You are constructing a CROSS-CATEGORY SYNTHESIS question that spans multiple medical "
        "specialties (e.g., oncology + pharmacology, cardiology + endocrinology). "
        "The question should require integrating clinical knowledge from the provided chunks "
        "across different disease areas."
    ),
}

GEN_SYSTEM = """You are a medical exam question writer building a RAG benchmark evaluation set.

Your task: read the document chunk(s) provided by the user, then write ONE hard clinical question and its answer.

STRICT RULES:
1. The question must be answerable ONLY from the provided chunks — no general medical knowledge.
2. The answer must cite specific facts found in the chunks (numbers, drug names, outcomes, etc.).
3. Output ONLY a single valid JSON object — no preamble, no explanation, no markdown.

JSON FIELDS (all required):
- question: a complete, specific clinical question string (not a placeholder)
- golden_answer: 2-4 complete sentences with the answer, citing specific data from the chunks
- golden_chunk_ids: list of chunk_id strings that are needed to answer (copied from the chunk headers)
- difficulty_reason: one sentence explaining why a keyword-search RAG would struggle

EXAMPLE (for formatting reference only — do NOT use this content):
{"question": "What LDL-C reduction was observed in diabetic ASCVD patients receiving evolocumab versus placebo?",
 "golden_answer": "In diabetic ASCVD patients, evolocumab reduced LDL-C by 59% compared to placebo at 48 weeks. The absolute reduction was 56 mg/dL. Cardiovascular event risk was reduced by 17% in this subgroup.",
 "golden_chunk_ids": ["pmc:doc42:5", "pmc:doc42:8"],
 "difficulty_reason": "Requires combining subgroup efficacy data from two separate results sections that are not co-located in a single chunk."}"""

GEN_USER = """{type_instruction}

Document chunk(s) — read these carefully and base your question entirely on their content:

{context}

Now write a hard clinical question + golden answer grounded in the above chunks.
Return ONLY the JSON object."""


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        cid = c["chunk_id"]
        title = c.get("title", "")[:80]
        section = c.get("section", "")
        text = c.get("text", "")[:600]
        parts.append(f"[chunk_id: {cid}]\nTitle: {title}\nSection: {section}\nText: {text}")
    return "\n\n---\n\n".join(parts)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _extract_content(resp) -> str:
    """Extract usable text from an API response.

    MiMo thinking mode emits reasoning in `reasoning_content` and the
    final answer in `content`. When the model hits a token limit before
    finishing, `content` can be empty even if `reasoning_content` is full.

    Strategy:
      1. Use `content` if present and non-empty.
      2. Otherwise scan `reasoning_content` for the last JSON object.
      3. Strip <think>…</think> blocks in either case.
    """
    msg = resp.choices[0].message
    raw = (getattr(msg, "content", None) or "").strip()

    # Strip Qwen/MiMo thinking blocks
    raw = _THINK_RE.sub("", raw).strip()

    if not raw:
        # Fallback: scan reasoning_content for a JSON object
        rc = (getattr(msg, "reasoning_content", None) or "").strip()
        rc = _THINK_RE.sub("", rc).strip()
        # Find the last {...} block in the reasoning output
        m = None
        for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", rc, re.DOTALL):
            pass
        if m:
            raw = m.group()

    return raw


def generate_question(client: OpenAI, model: str, cand: dict,
                      retries: int = 3) -> dict | None:
    context = format_chunks_for_prompt(cand["chunks"])
    user_msg = GEN_USER.format(
        type_instruction=TYPE_INSTRUCTIONS[cand["type"]],
        context=context,
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=2000,  # reasoning tokens + JSON answer
            )
            raw = _extract_content(resp)
            if not raw:
                raise ValueError("Empty response from API")
            # Strip markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            # Extract JSON object if extra text is present
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group()
            data = json.loads(raw)
            return data
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("Attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=os.environ.get("OPENAI_MODEL", "mimo-v2.5"))
    parser.add_argument("--limit",      type=int, default=0,
                        help="Max questions per type (0=full: A=15, B=10, C=10, D=15)")
    parser.add_argument("--sleep",      type=float, default=0.4)
    args = parser.parse_args()

    targets = {"A": 15, "B": 10, "C": 10, "D": 15}
    if args.limit:
        targets = {k: min(args.limit, v) for k, v in targets.items()}

    GOLDEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    CAND_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Load corpus ────────────────────────────────────────────────────────────
    chunks = load_corpus_chunks()
    candidates = select_candidates(chunks)

    # Save candidates for inspection
    all_cands = [c for cs in candidates.values() for c in cs]
    CAND_OUT.write_text(
        "\n".join(json.dumps({
            "type": c["type"], "domain": c["domain"],
            "chunk_ids": [ch["chunk_id"] for ch in c["chunks"]],
        }, ensure_ascii=False) for c in all_cands),
        encoding="utf-8",
    )
    logger.info("Saved %d candidates → %s", len(all_cands), CAND_OUT)

    # ── Generate questions ─────────────────────────────────────────────────────
    client = make_client()
    results: list[dict] = []
    q_idx = 0

    type_order = [("A", targets["A"]), ("B", targets["B"]),
                  ("C", targets["C"]), ("D", targets["D"])]

    for qtype, n_target in type_order:
        generated = 0
        cands = candidates[qtype]
        logger.info("── Type %s: need %d questions from %d candidates ──",
                    qtype, n_target, len(cands))

        for cand in cands:
            if generated >= n_target:
                break
            q_idx += 1
            qid = f"hard_{qtype}{generated + 1:02d}"
            logger.info("[%d] Generating %s (%s) …", q_idx, qid, cand["domain"])

            result = generate_question(client, args.model, cand)
            if not result or not result.get("question") or not result.get("golden_answer"):
                logger.warning("  [skip] bad LLM output")
                continue

            record = {
                "id":             qid,
                "type":           qtype,
                "category":       cand["domain"],
                "difficulty":     "hard",
                "question":       result["question"],
                "answer":         result["golden_answer"],
                "golden_chunk_ids": result.get("golden_chunk_ids", [
                    ch["chunk_id"] for ch in cand["chunks"]
                ]),
                "difficulty_reason": result.get("difficulty_reason", ""),
                "source_chunks":  [
                    {
                        "chunk_id":  ch["chunk_id"],
                        "citation":  ch["citation"],
                        "title":     ch.get("title", "")[:100],
                        "section":   ch.get("section", ""),
                        "text":      ch.get("text", "")[:500],
                    }
                    for ch in cand["chunks"]
                ],
            }
            results.append(record)
            generated += 1

            # Incremental save
            GOLDEN_OUT.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
                encoding="utf-8",
            )
            logger.info("  ✓ [%s] %s", qid, result["question"][:80])
            time.sleep(args.sleep)

    # ── SHA-256 lock ──────────────────────────────────────────────────────────
    content = GOLDEN_OUT.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode()).hexdigest()
    SHA_OUT.write_text(f"{sha}  golden_hard.jsonl\n", encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[done] Generated {len(results)} hard questions → {GOLDEN_OUT}")
    from collections import Counter
    type_counts = Counter(r["type"] for r in results)
    domain_counts = Counter(r["category"] for r in results)
    print("Type distribution:", dict(type_counts))
    print("Domain distribution:", dict(domain_counts))
    print(f"SHA-256: {sha[:16]}…  → {SHA_OUT}")


if __name__ == "__main__":
    main()
