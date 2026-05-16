import requests, os, sys
sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv()

print("=== Environment ===")
print(f"LLM_BACKEND       : {os.getenv('LLM_BACKEND', 'mimo (default)')}")
print(f"OPENAI_BASE_URL   : {os.getenv('OPENAI_BASE_URL', 'NOT SET')}")
print(f"MIMO_MODEL_FAST   : {os.getenv('MIMO_MODEL_FAST', 'mimo-v2.5 (default)')}")
print(f"MIMO_MODEL_THINK  : {os.getenv('MIMO_MODEL_THINK', 'mimo-v2.5-pro (default)')}")
print(f"RERANKER_DEVICE   : {os.getenv('RERANKER_DEVICE', 'auto (default)')}")
print(f"EMBEDDER_DEVICE   : {os.getenv('EMBEDDER_DEVICE', 'auto (default)')}")

print("\n=== Services ===")
try:
    r = requests.get("http://localhost:6333/collections", timeout=3)
    cols = [c["name"] for c in r.json().get("result", {}).get("collections", [])]
    print(f"Qdrant            : UP  collections={cols}")
except Exception as e:
    print(f"Qdrant            : DOWN  ({e})")

try:
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    models = [m.get("name") for m in r.json().get("models", [])]
    print(f"Ollama            : UP  models={models}")
except Exception:
    print("Ollama            : DOWN")

try:
    import torch
    print(f"\n=== Hardware ===")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
        print(f"VRAM total        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception as e:
    print(f"torch check failed: {e}")
