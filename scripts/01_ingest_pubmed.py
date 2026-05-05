"""Ingest PubMed abstracts into data/raw/pubmed/abstracts.jsonl."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from medrag.ingest.pubmed import fetch_pubmed_abstracts

load_dotenv()

KEYWORDS = [
    "radiology", "medical imaging", "MRI", "magnetic resonance imaging",
    "CT scan", "computed tomography", "ultrasound", "cardiac imaging",
    "tomography", "echocardiography",
]
OUT = Path("data/raw/pubmed/abstracts.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

email = os.getenv("NCBI_EMAIL", "").strip()
api_key = os.getenv("NCBI_API_KEY") or None

seen: set[str] = set()
with OUT.open("w", encoding="utf-8") as f:
    for rec in tqdm(fetch_pubmed_abstracts(
        keywords=KEYWORDS,
        year_from=2020, year_to=2026,
        max_results=2000,
        email=email,
        api_key=api_key,
    ), desc="pubmed"):
        if rec.pmid in seen:
            continue
        seen.add(rec.pmid)
        f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

print(f"[done] {len(seen)} records -> {OUT}")
