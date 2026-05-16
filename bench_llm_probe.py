"""
Probe MiMo API directly (raw openai client) to diagnose:
  1. Why mimo-v2.5 returns empty responses
  2. Whether explicit thinking=disabled helps
  3. Actual latency per model + thinking config
"""
import os, time, json
sys_import = __import__("sys")
sys_import.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

import openai

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key  = os.environ["OPENAI_API_KEY"]
client   = openai.OpenAI(base_url=base_url, api_key=api_key)

SYSTEM = "Classify the query as: factual, synthesis, or multihop. Reply with one word only."
USER   = "What are the contraindications of warfarin in elderly patients?"

def probe(model, extra_body=None, label=""):
    body = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER},
        ],
        max_tokens=16,
        temperature=0.2,
    )
    if extra_body:
        body["extra_body"] = extra_body

    t0 = time.perf_counter()
    resp = client.chat.completions.create(**body)
    elapsed = time.perf_counter() - t0

    choice   = resp.choices[0]
    content  = choice.message.content
    finish   = choice.finish_reason
    # Some APIs return thinking in additional fields
    raw_msg  = choice.message.model_dump()

    print(f"\n{'─'*60}")
    print(f"  model      : {model}")
    print(f"  label      : {label}")
    print(f"  latency    : {elapsed*1000:.0f} ms")
    print(f"  content    : {repr(content)}")
    print(f"  finish     : {finish}")
    # Print any extra fields (thinking tokens etc.)
    extra_keys = {k: v for k, v in raw_msg.items()
                  if k not in ("content", "role", "refusal", "annotations", "audio", "function_call", "tool_calls")
                  and v is not None}
    if extra_keys:
        print(f"  extra_keys : {json.dumps(extra_keys, ensure_ascii=False)[:300]}")
    usage = resp.usage
    if usage:
        print(f"  tokens     : in={usage.prompt_tokens} out={usage.completion_tokens}", end="")
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            d = usage.completion_tokens_details
            print(f"  reasoning={getattr(d, 'reasoning_tokens', '?')}", end="")
        print()
    return content, elapsed

print("=" * 60)
print("MiMo API probe — testing thinking configs")
print(f"base_url: {base_url}")
print("=" * 60)

# Test 1: mimo-v2.5-pro, no extra_body (current production config)
probe("mimo-v2.5-pro", label="current production (no extra_body, temp=0.2)")

# Test 2: mimo-v2.5-pro, explicit thinking disabled
probe("mimo-v2.5-pro",
      extra_body={"enable_thinking": False},
      label="pro + enable_thinking=False")

# Test 3: mimo-v2.5 (non-pro), no extra_body
probe("mimo-v2.5", label="v2.5 baseline (no extra_body)")

# Test 4: mimo-v2.5, explicit thinking disabled
probe("mimo-v2.5",
      extra_body={"enable_thinking": False},
      label="v2.5 + enable_thinking=False")

# Test 5: mimo-v2.5, thinking disabled via budget_tokens=0 (some providers)
probe("mimo-v2.5",
      extra_body={"thinking": {"type": "disabled"}},
      label="v2.5 + thinking.type=disabled")

print("\n" + "=" * 60)
print("Probe complete.")
