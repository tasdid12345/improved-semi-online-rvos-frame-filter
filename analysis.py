"""Simulated referring-video frame filtering.
Baseline: change-threshold keyframe filter (threshold tuned on train videos).
Improved: class-balanced, CV-tuned LightGBM frame scorer, top-k per video.
Both share the same simulated data, budget, and evaluation function."""
import json
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 42
SEEDS = [42, 43, 44]          # LightGBM is randomized -> run all three seeds
T = 150                       # frames per video
BUDGET = 15                   # frames kept per video (shared by both filters)
N_TRAIN, N_TEST = 200, 60     # simulated videos; test uses a different seed
# scene descriptor used by the baseline: 2 mean-position dims + 4 mean-appearance dims + detection rate
DESC = [f"d{i}" for i in range(7)]
FEATS = ["t", "n_det", "text_max", "text_2nd", "text_mean", "speed_mean", "app_mean",
         "text_roll5", "text_roll10max", "text_diff1", "speed_roll5", "app_roll5"]


def simulate_video(rng, vid):
    """One video: K moving objects; one text-referred target is visible in a random window."""
    K = int(rng.integers(3, 9))
    tgt = int(rng.integers(K))
    L = int(rng.integers(25, 71))
    s = int(rng.integers(10, T - L - 10))
    label = np.zeros(T, int)
    label[s:s + L] = 1
    pos = np.zeros((T, K, 2)); app = np.zeros((T, K, 4))
    spd = np.zeros((T, K)); dapp = np.zeros((T, K)); txt = np.zeros((T, K))
    det = rng.random((T, K)) > 0.1                 # 10% missed detections
    p, v = rng.uniform(0, 1, (K, 2)), rng.normal(0, .01, (K, 2))
    a = rng.normal(0, 1, (K, 4))
    for t in range(T):                             # reflected random-walk tracks
        v = v + rng.normal(0, .002, (K, 2))
        p_new = 1 - np.abs(1 - np.abs(p + v))      # reflect into [0, 1]
        spd[t] = np.linalg.norm(p_new - p, axis=1); p = p_new; pos[t] = p
        a_new = a + rng.normal(0, .05, (K, 4))
        dapp[t] = np.linalg.norm(a_new - a, axis=1); a = a_new; app[t] = a
        # noisy text-match score: target scores higher only while it is in the window
        mu = np.full(K, .4); mu[tgt] = .75 if label[t] else .45
        txt[t] = np.clip(mu + rng.normal(0, .12, K), 0, 1)
    cnt = np.maximum(det.sum(1), 1)
    tm = np.where(det, txt, 0.0)
    mpos = (pos * det[..., None]).sum(1) / cnt[:, None]
    mapp = (app * det[..., None]).sum(1) / cnt[:, None]
    df = pd.DataFrame({
        "vid": vid, "t": np.arange(T) / T, "label": label, "n_det": det.sum(1),
        "text_max": tm.max(1), "text_2nd": np.sort(tm, 1)[:, -2],
        "text_mean": tm.sum(1) / cnt, "speed_mean": np.where(det, spd, 0).sum(1) / cnt,
        "app_mean": np.where(det, dapp, 0).sum(1) / cnt})
    # causal (past-only) rolling features, so the scorer is usable online
    df["text_roll5"] = df.text_max.rolling(5, min_periods=1).mean()
    df["text_roll10max"] = df.text_max.rolling(10, min_periods=1).max()
    df["text_diff1"] = df.text_max.diff().fillna(0)
    df["speed_roll5"] = df.speed_mean.rolling(5, min_periods=1).mean()
    df["app_roll5"] = df.app_mean.rolling(5, min_periods=1).mean()
    desc = np.column_stack([mpos, mapp, det.sum(1) / K])  # shape (T, 7) == len(DESC)
    for j, c in enumerate(DESC):
        df[c] = desc[:, j]
    return df


def make_split(n, seed, vid0):
    rng = np.random.default_rng(seed)
    return pd.concat([simulate_video(rng, vid0 + i) for i in range(n)], ignore_index=True)


def score(df, keep):
    """Shared evaluation: recall of referred-object frames (main), precision, F1, frames kept/video."""
    k = np.asarray(keep, bool); y = df.label.to_numpy() == 1
    tp = float((k & y).sum())
    rec = tp / max(y.sum(), 1); prec = tp / max(k.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(recall=rec, precision=prec, f1=f1, kept_per_video=k.sum() / df.vid.nunique())


def change_keep(df, thr):
    """Baseline: keep a frame if its scene descriptor moved more than thr from the last kept frame; cap at BUDGET."""
    D = df[DESC].to_numpy(); keep = np.zeros(len(df), bool)
    for idx in df.groupby("vid").indices.values():
        last, n = D[idx[0]], 1
        keep[idx[0]] = True
        for i in idx[1:]:
            if n < BUDGET and np.linalg.norm(D[i] - last) > thr:
                keep[i] = True; last = D[i]; n += 1
    return keep


def topk_keep(vid, scores):
    """Improved: keep the BUDGET highest-scoring frames within each video."""
    r = pd.Series(scores).groupby(np.asarray(vid)).rank(ascending=False, method="first")
    return r.to_numpy() <= BUDGET


def make_lgb(params, seed):
    # class-balanced LightGBM; subsampling makes the model seed-dependent
    return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                              verbose=-1, **params)


def main():
    print("Simulating train and test videos (different seeds)...")
    tr = make_split(N_TRAIN, SEED, 0)
    te = make_split(N_TEST, 2024, 10000)
    y_tr = tr.label.to_numpy()
    print(f"train frames={len(tr)}  test frames={len(te)}  referred frames (test)={int(te.label.sum())}")
    if te.label.sum() == 0 or not np.isfinite(tr[FEATS].to_numpy()).all():
        sys.exit("no meaningful test signal")

    # ---- baseline: threshold chosen on TRAIN recall under the budget cap ----
    print("Baseline: tuning change threshold on train...")
    D = tr[DESC].to_numpy()
    dists = np.linalg.norm(np.diff(D, axis=0), axis=1)  # includes a few cross-video diffs (negligible)
    cands = np.quantile(dists, np.linspace(0.5, 0.995, 30))
    thr = float(cands[int(np.argmax([score(tr, change_keep(tr, c))["recall"] for c in cands]))])
    base = score(te, change_keep(te, thr))  # deterministic -> same value for every seed
    print(f"  threshold={thr:.4f} test recall={base['recall']:.4f}")

    # ---- improved: small grid, 5-fold GroupKFold by video on TRAIN only ----
    print("Improved: CV over small LightGBM grid on train...")
    grid = [dict(num_leaves=nl, n_estimators=ne) for nl in (7, 31) for ne in (100, 300)]
    gkf = GroupKFold(n_splits=5)
    cv = {}
    for i, p in enumerate(grid):
        rec = []
        for a, b in gkf.split(tr[FEATS], y_tr, tr.vid):
            m = make_lgb(p, SEED).fit(tr[FEATS].iloc[a], y_tr[a])
            vb = tr.iloc[b]
            keep = topk_keep(vb.vid, m.predict_proba(vb[FEATS])[:, 1])
            rec.append(score(vb, keep)["recall"])
        cv[i] = float(np.mean(rec))
        print(f"  {p} cv_recall={cv[i]:.4f}")
    best = grid[max(cv, key=cv.get)]
    print(f"  best params={best}")

    # ---- final fit on all train, one test evaluation per seed ----
    runs = []
    for s in SEEDS:
        m = make_lgb(best, s).fit(tr[FEATS], y_tr)
        runs.append(score(te, topk_keep(te.vid, m.predict_proba(te[FEATS])[:, 1])))
        print(f"  seed={s} test recall={runs[-1]['recall']:.4f}")
    imp = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    imp_std = float(np.std([r["recall"] for r in runs]))
    base_runs = [base] * len(SEEDS)  # baseline is deterministic
    base_std = float(np.std([r["recall"] for r in base_runs]))

    if not all(np.isfinite([imp["recall"], base["recall"]])):
        sys.exit("non-finite metrics")

    metrics = {
        "baseline_score": base["recall"], "improved_score": imp["recall"],
        "improvement": imp["recall"] - base["recall"],
        "baseline_score_std": base_std, "improved_score_std": imp_std,
        "baseline_precision": base["precision"], "improved_precision": imp["precision"],
        "baseline_f1": base["f1"], "improved_f1": imp["f1"],
        "baseline_kept_per_video": base["kept_per_video"], "improved_kept_per_video": imp["kept_per_video"],
        "baseline_threshold": thr, "improved_cv_recall": cv[grid.index(best)],
    }
    results = {
        "question": "Can a class-balanced LightGBM frame filter keep more frames where a text-referred object is visible than a standard change-threshold keyframe filter under the same frame budget?",
        "method": "Simulated videos with moving objects and one text-referred target in a random window; per-frame features from tracks with noise. Both filters keep at most the same per-video budget; the test set uses a different simulation seed. Main metric: recall of referred-object frames on test.",
        "dataset": {"name": "Simulated referring-video tracks (numpy, generated in code)",
                    "source_url": "https://lightgbm.readthedocs.io/",
                    "n_rows": int(len(tr) + len(te)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "recall of referred-object frames kept (test)",
        "higher_is_better": True,
        "baseline": "Change-threshold keyframe filter: keep a frame when its scene descriptor changes more than a threshold tuned on train recall, capped at the per-video budget.",
        "improved": "Class-balanced LightGBM per-frame scorer with causal rolling features, hyper-parameters chosen by 5-fold GroupKFold on train; keeps top-scoring frames per video within the same budget.",
        "metrics": metrics,
        "notes": (f"Simulation, not the paper's data or model; the paper's filter is unspecified so the baseline is a stand-in. "
                  f"{N_TRAIN} train and {N_TEST} test videos of {T} frames; n_train/n_test count frames. "
                  "Baseline is deterministic, so its std is 0; improved std is over 3 LightGBM seeds (population std). "
                  "Recall is bounded by budget / referred-frame count, so absolute values depend on the simulation settings. "
                  "Fix: scene descriptor DESC now has 7 dims (2 position + 4 appearance + detection rate) to match the stacked array."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # optional chart
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["baseline", "improved"], [base["recall"], imp["recall"]],
           yerr=[base_std, imp_std], color=["grey", "steelblue"], capsize=4)
    ax.set_ylabel("referred-frame recall (test)")
    fig.tight_layout(); fig.savefig("figures/recall_bar.png", dpi=120)
    print("Wrote results.json and figures/recall_bar.png")


if __name__ == "__main__":
    main()
