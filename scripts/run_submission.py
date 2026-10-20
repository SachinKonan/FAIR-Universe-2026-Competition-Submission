"""Run submission/model.py the way the organizers' ingestion program does, without Docker.

  uv run python scripts/run_submission.py --data <dir or .npy> [--n 64] [--submission submission] [--out runs/local]

Mirrors ingestion_program/ingestion.py: the test file is opened as a read-only float16 memmap, Model() is built once
and predict(test_data) is called once; the scores must be one finite number per map. Writes <out>/result.json
({"ood_scores": [...]}, the format the scoring program reads) and <out>/timing.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TEST_FILE = 'WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy'
ROOT = Path(__file__).resolve().parents[1]


def validate(scores, n):
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError(f'predict must return a 1-D array, got shape {scores.shape}')
    if len(scores) != n:
        raise ValueError(f'predict must return {n} scores, got {len(scores)}')
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(scores.dtype, np.complexfloating):
        raise TypeError(f'predict must return real numbers, got dtype {scores.dtype}')
    scores = scores.astype(np.float64, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError('predict returned NaN or infinite scores')
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help=f'directory holding {TEST_FILE}, or any (N, 132019) float16 .npy')
    ap.add_argument('--submission', default=str(ROOT / 'submission'))
    ap.add_argument('--n', type=int, default=0, help='score only the first n maps (0 = all)')
    ap.add_argument('--out', default=str(ROOT / 'runs' / 'local'))
    args = ap.parse_args()

    data = Path(args.data)
    if data.is_dir():
        data = data / TEST_FILE
    test = np.load(data, mmap_mode='r', allow_pickle=False)
    if args.n:
        test = test[:args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.append(os.path.abspath(args.submission))
    t0 = time.time()
    from model import Model
    model = Model()
    t1 = time.time()
    scores = validate(model.predict(test), len(test))
    t2 = time.time()

    (out / 'result.json').write_text(json.dumps({'ood_scores': scores.tolist()}))
    timing = dict(n=len(test), init_s=round(t1 - t0, 1), predict_s=round(t2 - t1, 1), total_s=round(t2 - t0, 1))
    (out / 'timing.json').write_text(json.dumps(timing, indent=1))
    print(f'[run_submission] {timing} -> {out / "result.json"}')


if __name__ == '__main__':
    main()
