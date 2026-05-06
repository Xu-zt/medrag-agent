import json
from pathlib import Path
d = json.loads(Path('data/eval/answer_eval.json').read_text(encoding='utf-8'))
scored = [r for r in d['results'] if 'composite' in r]
n = len(scored)
print(f'P3 baseline: n={n}')
keys = ['faithfulness', 'relevance', 'correctness', 'composite']
for k in keys:
    avg = sum(r[k] for r in scored) / n
    print(f'  {k}={avg:.3f}')
