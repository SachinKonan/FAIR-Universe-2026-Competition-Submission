"""Why condition on the parameters? Measurements on the 466 summary statistics (S).

  uv run --group research python writeup/figures/intuition.py

1. How much of a statistic's variance across normal maps is explained by which cosmology made the map
   (between-cosmology share of variance, fit maps, one noise view).
2. Detection without the parameters: a Gaussian-copula density of the statistics fitted on fit maps, with no
   parameter input (same copula, OAS shrinkage and eigenvalue floor as our likelihood). Compared with the stored
   misfit at the true and at the fitted parameters (results.json, research/oracle.py).
Writes intuition.json next to this file.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import BLUE, ORANGE, RED, INK, INK2, AXIS  # noqa: E402  (shared style)

V3 = Path('fair-data/v3/stats')
SHIFTS = ['ia0.2', 'bary9.0', 'bary6.5', 'baryDMO', 'supp0.15', 'psfpat', 'peakcut0.3', 'poisson120', 'mult+0.05']


def phase2_score(normal, anomalous):
    from sklearn.metrics import roc_curve
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))]
    fpr, tpr, _ = roc_curve(y, np.r_[normal, anomalous])
    return float(np.mean(np.interp(np.logspace(-3, math.log10(0.05), 100), fpr, tpr)))


def load(name, view=0):
    d = np.load(V3 / f'{name}.npz')
    return d['feats'][:, view].astype(np.float64), d['index'], d['labels']


def oas_corr(g, floor=0.02):
    n, p = g.shape
    s = np.corrcoef(g, rowvar=False)
    tr2, trs2 = p, (s ** 2).sum()
    rho = min(1.0, ((1 - 2 / p) * trs2 + tr2 ** 2) / ((n + 1 - 2 / p) * (trs2 - tr2 ** 2 / p)))
    c = (1 - rho) * s + rho * np.eye(p)
    w, v = np.linalg.eigh(c)
    return (v * np.maximum(w, floor)) @ v.T


def main():
    fit, idx, lab = load('fit')
    out = {}
    # 1. between-cosmology share of variance
    cos = idx[:, 0]
    mu = np.zeros((101, fit.shape[1]))
    for c in range(101):
        mu[c] = fit[cos == c].mean(0)
    ok = fit.var(0) > 1e-12
    share = mu[cos].var(0)[ok] / fit.var(0)[ok]
    out['between_cosmology_share'] = dict(median=float(np.median(share)), q25=float(np.quantile(share, .25)),
                                          q75=float(np.quantile(share, .75)))

    # 2. unconditional copula density (no parameters)
    m, s = fit.mean(0), fit.std(0) + 1e-12
    u = (fit - m) / s
    qs = np.sort(u, 0)

    def gscore(x):
        r = np.stack([np.searchsorted(qs[:, j], x[:, j]) for j in range(x.shape[1])], 1)
        return ndtri((r + 0.5) / (len(qs) + 1))

    rc = oas_corr(gscore(u))
    ri = np.linalg.inv(rc)

    def energy(x):
        g = gscore((x - m) / s)
        return np.einsum('ij,jk,ik->i', g, ri, g)

    a = energy(load('audit')[0])
    prof = json.loads((HERE / 'results.json').read_text())['profiling']['S']
    res = {}
    for sh in SHIFTS:
        res[sh] = dict(no_params=round(phase2_score(a, energy(load('shift_' + sh)[0])), 3))
    z = np.load('fair-data/v3/runs/r2/z_S.npz')
    for sh in SHIFTS:
        res[sh]['fitted'] = round(phase2_score(z['audit'], z['shift_' + sh]), 3)
        if sh in prof:
            res[sh]['true'] = prof[sh][0]
    out['detection_S'] = res
    print(json.dumps(out, indent=1))

    (HERE / 'intuition.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
