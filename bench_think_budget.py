"""
Probe thinking budget options for mimo-v2.5-pro to find the latency/quality tradeoff.
Tests: no budget (full reasoning), budget_tokens=1024, budget_tokens=512,
       thinking disabled (v2.5), and thinking disabled (v2.5-pro).
"""
import os, sys, time, json
sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv()

import openai

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key  = os.environ["OPENAI_API_KEY"]
client   = openai.OpenAI(base_url=base_url, api_key=api_key)

GRADE_SYS = (
    'Score 0.0-1.0 how well the document chunks answer the query. '
    'Reply JSON: {"relevant": true, "score": 0.8, "reason": "...", "rewrite_hint": ""}'
)
GRADE_USER = """Query: What are the contraindications of warfarin in elderly patients?

[PMID:1234] (score=0.91):
Warfarin is contraindicated in active bleeding, severe uncontrolled hypertension, recent intracranial hemorrhage, and known hypersensitivity. Elderly patients require dose adjustment due to increased sensitivity.

[PMID:5678] (score=0.88):
Age-related pharmacokinetic changes increase warfarin sensitivity. NSAID use, uncontrolled hypertension, and renal impairment are major contraindications. Regular INR monitoring is mandatory."""


def probe(model, extra_body, label):
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GRADE_SYS},
                {"role": "user",   "content": GRADE_USER},
            ],
            max_tokens=512,
            temperature=0.6,
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - t0
        choice = resp.choices[0]
        content = choice.message.content or ""
        msg_dump = choice.message.model_dump()
        reasoning = msg_dump.get("reasoning_content", "")
        usage = resp.usage
        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        print(f"\n  {label}")
        print(f"    latency        : {elapsed*1000:.0f} ms")
        print(f"    tokens in/out  : {usage.prompt_tokens}/{usage.completion_tokens}  reasoning={reasoning_tokens}")
        print(f"    content        : {repr(content[:100])}")
        if reasoning:
            print(f"    reasoning[0:80]: {repr(reasoning[:80])}")
        return elapsed, content
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n  {label}  → ERROR after {elapsed*1000:.0f}ms: {e}")
        return elapsed, None


print("=" * 68)
print("Thinking budget probe — grade prompt, single call each")
print("=" * 68)

# Baseline: no extra_body (current production config for think nodes)
probe("mimo-v2.5-pro", {}, "mimo-v2.5-pro, no extra_body (current production)")

# Budget-limited thinking
probe("mimo-v2.5-pro",
      {"thinking": {"type": "enabled", "budget_tokens": 512}},
      "mimo-v2.5-pro, budget_tokens=512")

probe("mimo-v2.5-pro",
      {"thinking": {"type": "enabled", "budget_tokens": 1024}},
      "mimo-v2.5-pro, budget_tokens=1024")

probe("mimo-v2.5-pro",
      {"thinking": {"type": "enabled", "budget_tokens": 2048}},
      "mimo-v2.5-pro, budget_tokens=2048")

# Fully disabled thinking on pro
probe("mimo-v2.5-pro",
      {"thinking": {"type": "disabled"}},
      "mimo-v2.5-pro, thinking disabled")

# v2.5 thinking disabled (already tested, baseline for comparison)
probe("mimo-v2.5",
      {"thinking": {"type": "disabled"}},
      "mimo-v2.5, thinking disabled")

print("\n" + "=" * 68)
