"""Figures for the write-up.

  uv run --group research python writeup/figures/make_figures.py [--maps]

fig_profiling.pdf and fig_results.pdf come from results.json and intuition.json. fig_maps.pdf (--maps) reads the
challenge training maps through research/data.py (FAIR_DATA env var, default fair-data/).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RED, TEAL = '#d94f2b', '#1a7f6b'
INK, MUTED, GREY, GRID = '#222222', '#6b6b6b', '#8c8c8c', '#e6e6e6'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False, 'hatch.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
R = json.loads((HERE / 'results.json').read_text())


def save(fig, name):
    fig.savefig(HERE / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    (HERE / 'preview').mkdir(exist_ok=True)
    fig.savefig(HERE / 'preview' / f'{name}.png', dpi=150, bbox_inches='tight', pad_inches=0.02)


def fig_profiling():
    """Detection with no parameters, with fitted parameters, and with the true parameters, per shift."""
    names = {'ia0.2': 'IA-like\nmixing', 'bary9.0': 'strong\nfeedback', 'bary6.5': 'weak\nfeedback',
             'baryDMO': 'no\nbaryons', 'supp0.15': 'damped\nsmall\nscales', 'psfpat': 'PSF\npattern'}
    nop = json.loads((HERE / 'intuition.json').read_text())['detection_S']
    chance = float(np.mean(np.logspace(-3, np.log10(0.05), 100)))
    styles = {'none': dict(facecolor=GREY, alpha=0.45, hatch='////', edgecolor=INK, linewidth=0.4),
              'fit': dict(facecolor=RED, edgecolor=INK, linewidth=0.4),
              'true': dict(facecolor=TEAL, edgecolor=INK, linewidth=0.4)}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, [('S', '(a) summary statistics'), ('C', '(b) CNN embedding')]):
        d = R['profiling'][key]
        series = ([('none', [nop[s]['no_params'] for s in names])] if key == 'S' else []) + \
            [('fit', [d[s][1] for s in names]), ('true', [d[s][0] for s in names])]
        n, w = len(series), 0.84 / 3
        x = np.arange(len(names))
        for j, (kind, vals) in enumerate(series):
            ax.bar(x + (j - (n - 1) / 2) * w, vals, width=w * 0.92, zorder=3, **styles[kind])
        ax.axhline(chance, color=MUTED, lw=0.8, ls=':', zorder=4)
        ax.set_xticks(x, list(names.values()))
        ax.tick_params(axis='x', length=0, labelsize=6.6)
        ax.set_xlim(-0.55, len(names) - 0.45)
        ax.set_ylim(0, 0.9)
        ax.grid(axis='x', visible=False)
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('detection score')
    axes[0].legend(handles=[Patch(label='no parameters', **styles['none']),
                            Patch(label='fitted parameters', **styles['fit']),
                            Patch(label='true parameters', **styles['true']),
                            Line2D([0], [0], color=MUTED, lw=0.8, ls=':', label='chance')],
                   loc='upper right', handlelength=1.4)
    save(fig, 'fig_profiling')


def fig_results():
    """(a) Public leaderboard per submission; (b) leave-cosmology-out proxy vs. leaderboard."""
    rows = R['leaderboard']
    side = {'v3-15g', 'v3-16c'}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True,
                               gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(rows))
    lb = [r['lb'] for r in rows]
    main = [i for i, r in enumerate(rows) if r['name'] not in side]
    a.plot(main, [lb[i] for i in main], color=RED, lw=1.4, marker='o', ms=3.5, mfc=RED, mec=RED, mew=0.9, zorder=3)
    a.plot([i for i in x if rows[i]['name'] in side], [lb[i] for i in x if rows[i]['name'] in side],
           ls='none', marker='o', ms=3.5, mfc='white', mec=RED, mew=0.9, zorder=3)
    for i, r in enumerate(rows):
        a.annotate(f'{lb[i]:.3f}', (i, lb[i]), textcoords='offset points',
                   xytext=(0, -10 if r['name'] in side else 5), ha='center', fontsize=6.8,
                   color=MUTED if r['name'] in side else INK)
    a.set_xticks(x, [r['name'] for r in rows], rotation=35, ha='right')
    a.tick_params(axis='x', length=0)
    a.set_ylim(0.5, 0.73)
    a.set_ylabel('public leaderboard score')
    a.grid(axis='x', visible=False)
    a.set_title('(a) leaderboard, in submission order', loc='left')
    a.legend(handles=[Line2D([0], [0], color=RED, lw=1.4, marker='o', ms=3.5, label='kept'),
                      Line2D([0], [0], ls='none', marker='o', ms=3.5, mfc='white', mec=RED, mew=0.9,
                             label='variant, not kept')], loc='upper left')

    pts = [r for r in rows if r['lco'] is not None]
    b.plot([r['lco'] for r in pts], [r['lb'] for r in pts], ls='none', marker='o', ms=3.5, mfc=RED, mec=RED, zorder=3)
    off = {'v3-01': (5, -2), 'v3-02': (5, -3), 'v3-04': (5, -3), 'v3-15b': (-30, 3), 'v3-15d': (4, 3),
           'v3-15g': (5, -3), 'v3-16b': (-30, 2), 'v3-16c': (4, 2)}
    for r in pts:
        b.annotate(r['name'], (r['lco'], r['lb']), textcoords='offset points', xytext=off[r['name']],
                   fontsize=6.8, color=INK)
    b.annotate('Ol weight 3 $\\to$ 6:\nproxy up, leaderboard down', (0.5258, 0.6993), xytext=(0.535, 0.585),
               fontsize=6.8, color=MUTED, ha='right',
               arrowprops=dict(arrowstyle='-', color=GREY, lw=0.7, shrinkA=2, shrinkB=3))
    b.set_xlabel('leave-cosmology-out proxy')
    b.set_ylabel('public leaderboard score')
    b.set_xlim(0.36, 0.54)
    b.set_ylim(0.5, 0.73)
    b.set_title('(b) proxy vs. leaderboard', loc='left')
    save(fig, 'fig_results')


def fig_maps():
    """A normal map, and 128x128-pixel crops of clean maps: three normal universes and four distortions of one."""
    import torch
    sys.path.insert(0, str(HERE.parents[1] / 'research'))
    import data
    train = np.load(data.TRAIN, mmap_mode='r')
    lab = data.labels()
    m = data.mask().astype(bool)
    mask_t = torch.as_tensor(m, dtype=torch.float32)
    om = lab[:, 0, 0]
    c_fid = int(np.argmin(np.abs(om - 0.3) + np.abs(lab[:, 0, 1] - 0.8)))
    c_lo, c_hi = int(np.argmin(om)), int(np.argmax(om))
    r = 7

    def full(c):
        x = torch.zeros(m.shape)
        x[torch.as_tensor(m)] = torch.as_tensor(np.asarray(train[c, r], dtype=np.float32))
        return x[None]

    clean = full(c_fid)
    noisy = data.add_noise(clean, mask_t, torch.Generator().manual_seed(0))[0]
    views = [('$\\Omega_m$=%.2f\n$S_8$=%.2f' % (om[c], lab[c, 0, 1]), full(c)[0]) for c in (c_fid, c_lo, c_hi)]
    for name, title in [('ia0.2', 'IA-like\nmixing'), ('aniso0.7', 'anisotropic\nsmoothing'),
                        ('peakcut0.3', 'flattened\npeaks'), ('psfpat', 'PSF\npattern')]:
        views.append((title, data.apply_shift(name, clean, mask_t, torch.Generator().manual_seed(1))[0][0]))

    cmap = LinearSegmentedColormap.from_list('div', [TEAL, '#f7f7f7', RED])
    cmap.set_bad('white')
    y0, x0, S = 600, 24, 128
    v = float(np.quantile(np.abs(clean[0][torch.as_tensor(m)].numpy()), 0.995))
    fig = plt.figure(figsize=(6.5, 2.6))
    gs = fig.add_gridspec(2, 7, height_ratios=[0.62, 1], hspace=0.05, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])
    vv = 3 * data.NOISE_SIGMA
    ax.imshow(np.where(m, noisy.numpy(), np.nan).T, cmap=cmap, vmin=-vv, vmax=vv, interpolation='nearest')
    ax.add_patch(plt.Rectangle((y0, x0), S, S, fill=False, ec=INK, lw=0.7))
    ax.set_title('a normal map as observed: 1424 $\\times$ 176 pixels of 2 arcmin, survey mask, shape noise',
                 loc='left')
    ax.axis('off')
    for j, (title, img) in enumerate(views):
        ax = fig.add_subplot(gs[1, j])
        crop = np.where(m, img.numpy(), np.nan)[y0:y0 + S, x0:x0 + S]
        ax.imshow(crop, cmap=cmap, vmin=-v, vmax=v, interpolation='nearest')
        ax.set_title(title, fontsize=7, color=INK if j < 3 else RED)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GREY); s.set_linewidth(0.5)
    fig.canvas.draw()
    crops = fig.axes[1:]
    y = crops[0].get_position().y0 - 0.035
    for lo, hi, text, col in [(0, 2, 'three normal universes', MUTED), (3, 6, 'the first map, distorted', RED)]:
        xc = (crops[lo].get_position().x0 + crops[hi].get_position().x1) / 2
        fig.text(xc, y, text, fontsize=7.5, color=col, ha='center', va='top')
    save(fig, 'fig_maps')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', action='store_true')
    args = ap.parse_args()
    fig_profiling()
    fig_results()
    if args.maps:
        fig_maps()
