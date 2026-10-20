"""Compare two OoD score vectors for the same test maps (no labels needed).

  uv run python scripts/compare.py <result.json> [reference: result.json or a result zip; default reference/]

The reference default is reference/v3-16b-public-result.json, the exact file whose public leaderboard score was
0.7037. If the new run covers only the first n maps, only those are compared. Reports rank agreement and overlap
of the highest-scored maps, which is what the 0.1-5% FPR metric reads.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            return np.asarray(json.loads(z.read('result.json'))['ood_scores'], dtype=np.float64)
    return np.asarray(json.load(open(path))['ood_scores'], dtype=np.float64)


new = load(sys.argv[1])
ref = load(sys.argv[2] if len(sys.argv) > 2 else ROOT / 'reference' / 'v3-16b-public-result.json')[:len(new)]
print(f'n {len(new)} finite {bool(np.isfinite(new).all())}')
print(f'spearman {spearmanr(new, ref)[0]:.6f}')
d = np.abs(new - ref)
print(f'|diff| median {np.median(d):.2e}  99% {np.quantile(d, .99):.2e}  max {d.max():.2e}')
for k in (10, 50, 100, 500, 1000, 3000, 5000):
    if k < len(new):
        a, b = set(np.argsort(-ref)[:k]), set(np.argsort(-new)[:k])
        print(f'top-{k} overlap {len(a & b)}/{k}')
