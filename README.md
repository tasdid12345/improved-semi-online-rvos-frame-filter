# Simulated referring-video frame filtering

Compares a change-threshold keyframe filter (baseline) with a class-balanced,
CV-tuned LightGBM frame scorer (improved) under the same per-video frame budget
(15 of 150 frames) on simulated videos. Main metric: recall of referred-object
frames on a held-out test set (different simulation seed).

## Run

    pip install numpy pandas lightgbm scikit-learn matplotlib
    python analysis.py

Writes `results.json` and `figures/recall_bar.png`. Seed 42; LightGBM is also run with seeds 42, 43, 44.

## Fix applied

The crash came from the scene descriptor: `np.column_stack` produces 7 columns
(2 mean-position + 4 mean-appearance + 1 detection rate), but `DESC` only named 6,
so assignment failed. `DESC` now has 7 entries (`d0`–`d6`), which keeps the
detection-rate descriptor. The baseline, test set, budget and evaluation are otherwise unchanged.

## Results (latest run)

On a simulated referring-video dataset (60 test videos), a standard change-threshold keyframe filter reached 7.4% recall of referred-object frames, while my class-balanced LightGBM frame scorer reached 31.9% under the same per-video frame budget.

| Metric | Value |
|---|---|
| baseline_score | 0.07422 |
| improved_score | 0.3185 |
| improvement | 0.2443 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| baseline_precision | 0.26 |
| improved_precision | 0.9967 |
| baseline_f1 | 0.1155 |
| improved_f1 | 0.4828 |
| baseline_kept_per_video | 13.4 |
| improved_kept_per_video | 15 |
| baseline_threshold | 0.8135 |
| improved_cv_recall | 0.3236 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulation, not the paper's data or model; the paper's filter is unspecified so the baseline is a stand-in. 200 train and 60 test videos of 150 frames; n_train/n_test count frames. Baseline is deterministic, so its std is 0; improved std is over 3 LightGBM seeds (population std). Recall is bounded by budget / referred-frame count, so absolute values depend on the simulation settings. Fix: scene descriptor DESC now has 7 dims (2 position + 4 appearance + detection rate) to match the stacked array.

## Reference

This project is motivated by the following paper:

> X Hu, H Neumann, J Lang (2025). *A Filtering Framework for Semi-online Referring Video Object Segmentation*. Proceedings of the 33rd ACM International. https://dl.acm.org/doi/abs/10.1145/3746027.3755018

Paper author: Prof. Heiko Neumann, Ulm University (heiko.neumann@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
