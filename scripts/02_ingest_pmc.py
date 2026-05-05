"""Ingest PMC OA full texts into data/raw/pmc/full_texts.jsonl."""
import json
import os
from pathlib import Path

from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

from medrag.ingest.pmc import fetch_pmc_records

load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL", "").strip()
Entrez.api_key = os.getenv("NCBI_API_KEY") or None

QUERY = (
    '("radiology"[Title/Abstract] OR "MRI"[Title/Abstract] '
    'OR "CT"[Title/Abstract] OR "ultrasound"[Title/Abstract]) '
    'AND open access[filter] AND ("2020"[dp]:"2026"[dp])'
)
TARGET = 400

OUT = Path("data/raw/pmc/full_texts.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("[pmc] esearch...")
h = Entrez.esearch(db="pmc", term=QUERY, retmax=TARGET, sort="pub_date")
ids = Entrez.read(h)["IdList"]
pmcids = [f"PMC{i}" for i in ids]
print(f"[pmc] got {len(pmcids)} pmcids")

with OUT.open("w", encoding="utf-8") as f:
    n_saved = 0
    for rec in tqdm(fetch_pmc_records(pmcids, delay=0.4), total=len(pmcids), desc="pmc"):
        f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        n_saved += 1

print(f"[done] {n_saved} full-texts -> {OUT}")
