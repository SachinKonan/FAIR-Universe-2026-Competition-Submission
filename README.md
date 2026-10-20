# FAIR Universe 2026: Weak Lensing OoD Detection, Final Submission

This is my final submission to the FAIR Universe Weak Lensing ML Uncertainty Challenge, Phase 2 (out-of-distribution
detection). The model scored **0.7037** on the public leaderboard. This repository holds the model submission, the
scripts I use to run and check it, my write-up, and the research code that produced the model.

| | |
|---|---|
| Public leaderboard score | 0.7037 (result file `v3-16b-pkg-v315d-Ol3.zip`, 2026-10-11) |
| Submission | `submission/` (43 files, `model.py` at the root), zipped by `scripts/make_zip.py` |
| Runtime | one GPU, PyTorch only; 306 s for the 10,000 public test maps in the organizers' image on one A100 (4 CPUs, peak GPU memory 7.6 GiB) |
| Write-up | `writeup/writeup.pdf` (source `writeup/writeup.tex`, figures in `writeup/figures/`) |
| Full report | `docs/FAIR_PHASE2_REPORT.md`; leaderboard log in `docs/PROGRESS.md` |

## The idea

Normal maps differ mostly because they come from different universes. A detector that ignores the five generating
parameters (Omega_m, S8, T_AGN, f0, Delta z) barely beats chance on my synthetic distortions, while the same
statistics, given the true parameters, detect them well. So for each test map I ask: *can some normal universe
explain this map?* Three tests answer it:

- **Misfit** (members Sd, Dr, N2, PsBh, X). For each feature family I learn p(features | parameters): an MLP gives
  the mean and spread, and a Gaussian copula with shrinkage and an eigenvalue floor gives the correlations. I fit
  the parameters to the map and score what the best fit still fails to explain.
- **Consistency** (xSL, consP8, consRplus). Fitting absorbs part of every distortion, but different views of a map
  (scales, statistic families, networks) absorb it differently. I fit the parameters per view and score their
  disagreement.
- **Outlier exposure** (Oh, Ol). A CNN trained to tell normal maps from synthetically distorted ones. Ol is its OoD
  logit, and Oh is the misfit of its outputs.

Each member becomes a z-score on held-out normal maps, and the final score is their weighted mean: Sd 0.5, Dr 2.5,
xSL 1, consP8 0.5, consRplus 3, N2 0.5, PsBh 0.5, X 2, Oh 1.5, Ol 3 (`submission/config.json`). Every score depends
only on its own map; nothing is fit, calibrated, normalized or ranked on test data.

## Layout

```
submission/        the model submission: model.py (class Model, predict), helper modules fwl_*.py,
                   frozen network weights (*.pt), per-family likelihoods (fam_*.pt), config.json, survey mask
SUBMISSION_SHA256SUMS  sha256 of every file in submission/
reference/         v3-16b-public-result.json: the scores that got 0.7037 (for comparison only)
scripts/           run_submission.py   run model.py the way the organizers' ingestion program does, no Docker
                   compare.py          compare a run's scores with reference/
                   make_zip.py         build dist/FAIR-Universe-2026-v3-16b.zip (checks SUBMISSION_SHA256SUMS)
                   run_official.sh     run a ZIP through the organizers' ingestion program in their image (Apptainer)
                   della_official.sbatch  the same as a Slurm job with the Codabench worker's 4 CPUs and 32 GiB
writeup/           writeup.tex, writeup.pdf, figures/ (figure PDFs, the numbers behind them, and the scripts
                   make_figures.py / intuition.py that regenerate them)
docs/              full report and leaderboard log
research/          the research code that trained and assembled the model
pyproject.toml     uv environment matching the organizers' image: Python 3.12, PyTorch 2.13.0 (CUDA 12.6), NumPy 2.5.3
```

## Setup

```bash
uv sync                      # Python 3.12, torch 2.13.0+cu126, numpy 2.5.3, scipy
```

`uv.lock` pins the exact versions. Add `--group research` for the packages the research code and figures use.

All commands run from the repository root. Put the challenge data (and, for the research code, the features derived
from it) in `fair-data/`, as a directory or a symlink. The public test file is
`fair-data/WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`, shape (10000, 132019), float16.

## Run the submission

**Locally, no Docker.** A GPU is used if present; CPU works but is slow.

```bash
uv run python scripts/run_submission.py --data fair-data            # all 10,000 maps
uv run python scripts/run_submission.py --data fair-data --n 64     # quick check
uv run python scripts/compare.py runs/local/result.json             # vs. the 0.7037 scores
```

`run_submission.py` mirrors the organizers' `ingestion_program/ingestion.py`. It opens the test file as a read-only
float16 memmap, builds `Model()` once, calls `predict` once, applies the same output checks (1-D, one finite real
score per map), and writes `runs/local/result.json` and `timing.json`.

**In the organizers' runtime image** (`ayaga3540/fair_universe:v1.0`, with their ingestion program, under Apptainer):

```bash
uv run python scripts/make_zip.py                                   # -> dist/FAIR-Universe-2026-v3-16b.zip
scripts/run_official.sh dist/FAIR-Universe-2026-v3-16b.zip fair-data
uv run python scripts/compare.py runs/official_*/output/result.json
```

The first run needs internet: it clones `FAIR-Universe/Cosmology_Challenge` for `ingestion_program/` and pulls the
image into `third_party/`. Later runs work offline, so on a cluster I run it once on a login node and then submit

```bash
sbatch scripts/della_official.sbatch dist/FAIR-Universe-2026-v3-16b.zip fair-data
```

which uses one 40 GB A100, 4 CPUs and 32 GiB, and reports wall time and peak GPU memory. With Docker, mount the same
paths:

```bash
docker run --rm --gpus all -v $PWD/unzipped_submission:/app/ingested_program:ro \
  -v $PWD/third_party/Cosmology_Challenge/ingestion_program:/app/program:ro -v $PWD/out:/app/output \
  -v $PWD/fair-data:/app/input_data:ro -w /app/program ayaga3540/fair_universe:v1.0 \
  python3 run_ingestion.py --codabench
```

**The Codabench ZIP.** `scripts/make_zip.py` checks every file against `SUBMISSION_SHA256SUMS` and writes a
reproducible ZIP with `model.py` at the root.

**Figures.** `uv run --group research python writeup/figures/intuition.py` measures how much of each statistic's
variance the cosmology explains and how well the statistics detect distortions without the parameters.
`uv run --group research python writeup/figures/make_figures.py --maps` draws the figures.

## Research code

`research/` is the development code as I used it, kept for provenance. Its paths are relative to the repository
root: data and derived features under `fair-data/`, code under `research/`. The main stages:

1. **Data, splits and noise:** `data.py`. Map-level fit / calib / audit splits (80/10/10); shape noise drawn per map.
2. **Features:**
   - summary statistics: `extract.py`, `stats.py`;
   - CNN embeddings: `cnn.py`, `pipeline_v3.py`, and `tpu/` (JAX CNNs trained on TPU, converted to PyTorch);
   - U-Net denoiser features: `components/denoiser/`;
   - noise, bispectrum and other families: `components/statistics/`, `components/consistency/`;
   - diffusion features (not used in the final weights): `components/diffusion*/`;
   - power-spectrum anisotropy (`dirpower`): `factory/make_source.py`.
3. **Conditional likelihoods and fits:** `model.py`, `fit_v3.py`, `components/consistency/fit_families.py`,
   `factory/fit_family.py`.
4. **Consistency members:** `components/consistency/` (e.g. `scalecons.py`, `probes.py`, `export3.py`) and
   `components/consistency2/`.
5. **Outlier exposure:** distortion generators `components/outlier/gens.py`, CNN training
   `components/outlier/train_oe.py`, Oh / Ol members `components/outlier2/` and `factory/build_v316.py`.
6. **Validation:** the leave-cosmology-out proxy, 3 disjoint sets of 20 held-out cosmologies
   (`components/consistency/holdout3.py`, `lco3_analyze.py`, `factory/*_lco.py`).
7. **Packaging:** `package_full.py` builds the code package from `pack_full/`; `make_pkg_variants.py` makes weight
   variants; `factory/build_v316.py` built `v3-16b`; `verify_full.py` checks parity, per-map independence and runtime.

`factory/grade.py` scores a candidate feature family by its held-out gain; `factory/search_A/` and `search_B/` log the
candidates I scored with it. It contributed one family (`dirpower`).
