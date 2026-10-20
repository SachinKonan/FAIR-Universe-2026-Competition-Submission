"""Figures for the write-up.

  uv run --group research python writeup/figures/make_figures.py [--maps]

fig_profiling.pdf and fig_results.pdf come from results.json. fig_maps.pdf (--maps) reads the challenge training
maps through research/data.py (FAIR_DATA env var, default fair-data/).
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

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, RED, GRAY = '#2a78d6', '#eb6834', '#e34948', '#898781'
INK, INK2, GRID, AXIS, MID = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#f0efec'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
R = json.loads((HERE / 'results.json').read_text())


def fig_profiling():
    """Detection with no parameters, with fitted parameters, and with the true parameters, per shift."""
    names = {'ia0.2': 'IA-like mixing', 'bary9.0': 'baryons, strong feedback', 'bary6.5': 'baryons, weak feedback',
             'baryDMO': 'no baryons (DMO)', 'supp0.15': 'small-scale suppression', 'psfpat': 'PSF pattern'}
    nop = json.loads((HERE / 'intuition.json').read_text())['detection_S']
    chance = float(np.mean(np.logspace(-3, np.log10(0.05), 100)))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.0), sharey=True)
    for ax, (key, title) in zip(axes, [('S', '(a) summary statistics (466)'), ('C', '(b) CNN embedding')]):
        d = R['profiling'][key]
        ax.axvline(chance, color=AXIS, lw=0.8, ls=':', zorder=0)
        for i, s in enumerate(names):
            tru, fit = d[s]
            pts = [tru, fit] + ([nop[s]['no_params']] if key == 'S' else [])
            ax.plot([min(pts), max(pts)], [i, i], color=GRID, lw=1.6, zorder=1, solid_capstyle='round')
            if key == 'S':
                ax.scatter([nop[s]['no_params']], [i], s=22, color=GRAY, zorder=2, edgecolor='white', linewidth=0.8,
                           label='no parameters' if i == 0 else None)
            ax.scatter([fit], [i], s=22, color=ORANGE, zorder=3, edgecolor='white', linewidth=0.8,
                       label='fitted parameters' if i == 0 else None)
            ax.scatter([tru], [i], s=22, color=BLUE, zorder=4, edgecolor='white', linewidth=0.8,
                       label='true parameters' if i == 0 else None)
        ax.set_yticks(range(len(names)), list(names.values()))
        ax.invert_yaxis()
        ax.set_xlim(0, 0.9)
        ax.grid(axis='y', visible=False)
        ax.set_title(title, color=INK, loc='left')
        ax.set_xlabel('detection score (challenge metric)')
    axes[0].text(chance + 0.012, 0.02, 'chance', fontsize=6.3, color=INK2, va='bottom',
                 transform=axes[0].get_xaxis_transform())
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=7, handletextpad=0.2,
               bbox_to_anchor=(0.6, 1.08))
    fig.tight_layout(w_pad=1.0)
    fig.savefig(HERE / 'fig_profiling.pdf'); (HERE / 'preview').mkdir(exist_ok=True); fig.savefig(HERE / 'preview' / 'fig_profiling.png', dpi=130)


def fig_results():
    """(a) Public leaderboard per submission; (b) leave-cosmology-out proxy vs. leaderboard."""
    rows = R['leaderboard']
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.35), gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(rows))
    lb = [r['lb'] for r in rows]
    main = [i for i, r in enumerate(rows) if r['name'] not in ('v3-15g', 'v3-16c')]
    a.plot(main, [lb[i] for i in main], color=BLUE, lw=1.4, zorder=1)
    for i, r in enumerate(rows):
        side = r['name'] in ('v3-15g', 'v3-16c')
        a.scatter([i], [lb[i]], s=20, zorder=2, color='white' if side else BLUE, edgecolor=BLUE, linewidth=1.1)
        a.annotate(f"{lb[i]:.3f}", (i, lb[i]), textcoords='offset points', xytext=(0, -10 if side else 5),
                   ha='center', fontsize=6.3, color=INK2 if side else INK)
    a.set_xticks(x, [r['name'] for r in rows], rotation=35, ha='right')
    a.set_ylim(0.5, 0.73)
    a.set_ylabel('public leaderboard score')
    a.grid(axis='x', visible=False)
    a.set_title('(a) leaderboard, in submission order', color=INK, loc='left')
    a.text(0.02, 0.97, 'open marker: variant not kept', transform=a.transAxes, fontsize=6.3, color=INK2, va='top')

    pts = [r for r in rows if r['lco'] is not None]
    b.scatter([r['lco'] for r in pts], [r['lb'] for r in pts], s=20, color=BLUE, edgecolor='white', linewidth=0.8)
    off = {'v3-01': (4, -2), 'v3-02': (4, -3), 'v3-04': (4, -3), 'v3-15b': (-28, 3), 'v3-15d': (4, 3),
           'v3-15g': (4, -3), 'v3-16b': (-30, 2), 'v3-16c': (4, 2)}
    for r in pts:
        b.annotate(r['name'], (r['lco'], r['lb']), textcoords='offset points', xytext=off[r['name']],
                   fontsize=6.3, color=INK2)
    b.annotate('OE logit weight 3 $\\to$ 6:\nproxy up, leaderboard down', (0.5258, 0.6993), xytext=(0.535, 0.585),
               fontsize=6.3, color=RED, ha='right',
               arrowprops=dict(arrowstyle='->', color=RED, lw=0.7, shrinkA=2, shrinkB=3))
    b.set_xlabel('leave-cosmology-out proxy')
    b.set_ylabel('public leaderboard score')
    b.set_xlim(0.36, 0.54)
    b.set_ylim(0.5, 0.73)
    b.set_title('(b) proxy vs. leaderboard', color=INK, loc='left')
    fig.tight_layout(w_pad=1.5)
    fig.savefig(HERE / 'fig_results.pdf'); (HERE / 'preview').mkdir(exist_ok=True); fig.savefig(HERE / 'preview' / 'fig_results.png', dpi=130)


def fig_maps():
    """A normal map, and 128x128-pixel crops of clean maps: two normal universes and four distortions of one."""
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

    gen = torch.Generator().manual_seed(0)
    clean = full(c_fid)
    noisy = data.add_noise(clean, mask_t, gen)[0]
    views = [('$\\Omega_m$=%.2f\n$S_8$=%.2f' % (om[c], lab[c, 0, 1]), full(c)[0]) for c in (c_fid, c_lo, c_hi)]
    for name, title in [('ia0.2', 'IA-like\nmixing'), ('aniso0.7', 'anisotropic\nsmoothing'),
                        ('peakcut0.3', 'flattened\npeaks'), ('psfpat', 'PSF\npattern')]:
        views.append((title, data.apply_shift(name, clean, mask_t, torch.Generator().manual_seed(1))[0][0]))

    cmap = LinearSegmentedColormap.from_list('div', [BLUE, MID, RED])
    cmap.set_bad('white')
    y0, x0, S = 600, 24, 128
    v = float(np.quantile(np.abs(clean[0][torch.as_tensor(m)].numpy()), 0.995))
    fig = plt.figure(figsize=(6.5, 2.6))
    gs = fig.add_gridspec(2, 7, height_ratios=[0.62, 1], hspace=0.05, wspace=0.06)
    for row, (img, title, vv) in enumerate([(noisy, 'observed (with shape noise)', 3 * data.NOISE_SIGMA)]):
        ax = fig.add_subplot(gs[0, :])
        im = np.where(m, img.numpy(), np.nan).T
        ax.imshow(im, cmap=cmap, vmin=-vv, vmax=vv, interpolation='nearest', aspect='equal')
        ax.add_patch(plt.Rectangle((y0, x0), S, S, fill=False, ec=INK, lw=0.7))
        ax.set_title('a normal map as observed: 1424 $\\times$ 176 pixels of 2 arcmin, survey mask, shape noise',
                     loc='left', color=INK)
        ax.axis('off')
    for j, (title, img) in enumerate(views):
        ax = fig.add_subplot(gs[1, j])
        crop = np.where(m, img.numpy(), np.nan)[y0:y0 + S, x0:x0 + S]
        ax.imshow(crop, cmap=cmap, vmin=-v, vmax=v, interpolation='nearest')
        ax.set_title(title, fontsize=6.3, color=INK if j < 3 else RED)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(AXIS)
    fig.canvas.draw()
    crops = fig.axes[1:]
    y = crops[0].get_position().y0 - 0.035
    for lo, hi, text, col in [(0, 2, 'three normal universes', INK2), (3, 6, 'the first map, distorted', RED)]:
        x = (crops[lo].get_position().x0 + crops[hi].get_position().x1) / 2
        fig.text(x, y, text, fontsize=6.8, color=col, ha='center', va='top')
    fig.savefig(HERE / 'fig_maps.pdf'); (HERE / 'preview').mkdir(exist_ok=True); fig.savefig(HERE / 'preview' / 'fig_maps.png', dpi=130)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', action='store_true')
    args = ap.parse_args()
    fig_profiling()
    fig_results()
    if args.maps:
        fig_maps()
