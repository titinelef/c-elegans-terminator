import time
import numpy as np
import pandas as pd
import os

from collections import defaultdict

from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone


# ============================================================
# 1) DATA LOADING / SHAPING
# ============================================================

def load_segments_featureless(files, raw_cols, target_len=900):
    """
    Load preprocessed worm trajectory segments and construct a fixed-length
    featureless representation using raw time-series signals.

    Each segment is padded or truncated to a fixed temporal length to ensure
    a homogeneous tensor shape suitable for machine learning models.

    Parameters
    ----------
    files : list of str
        Paths to CSV files, each corresponding to one worm segment.
    raw_cols : list of str
        Names of raw signal columns to extract (e.g., speed, turning angle).
    target_len : int, default=900
        Target temporal length (number of frames) per segment.

    Returns
    -------
    X : np.ndarray of shape (N, target_len, C)
        Featureless input tensor, where N is the number of segments and
        C the number of selected raw signals.
    worm_ids : np.ndarray of shape (N,)
        Worm identifier associated with each segment.
    seg_idx : np.ndarray of shape (N,)
        Segment index within the worm lifetime.
    info : dict
        Summary statistics including number of segments, skipped files,
        segment length range, and number of unique worms.
    """
    X_list, worm_ids, seg_idx = [], [], []
    lengths = []
    skipped = 0

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            skipped += 1
            continue

        if "worm_id" not in df.columns:
            skipped += 1
            continue

        if "Segment_index" in df.columns:
            sidx = df["Segment_index"].iloc[0]
        elif "Segment" in df.columns:
            sidx = df["Segment"].iloc[0]
        else:
            skipped += 1
            continue

        Xraw = df[raw_cols].to_numpy(dtype=float)
        Xraw = np.nan_to_num(Xraw, nan=0.0, posinf=0.0, neginf=0.0)

        L = Xraw.shape[0]
        lengths.append(L)

        C = Xraw.shape[1]
        arr = np.zeros((target_len, C), dtype=float)
        if L >= target_len:
            arr[:] = Xraw[:target_len, :]
        else:
            arr[:L, :] = Xraw

        X_list.append(arr)
        worm_ids.append(df["worm_id"].iloc[0])
        seg_idx.append(int(float(sidx)))

    X = np.stack(X_list, axis=0)
    worm_ids = np.array(worm_ids)
    seg_idx = np.array(seg_idx, dtype=int)

    info = {
        "N": len(X_list),
        "skipped": skipped,
        "min_len": int(np.min(lengths)) if lengths else None,
        "max_len": int(np.max(lengths)) if lengths else None,
        "n_worms": int(len(np.unique(worm_ids)))
    }
    return X, worm_ids, seg_idx, info


# ============================================================
# 2) LABEL ENGINEERING (NEAR-DEATH)
# ============================================================

def compute_segments_from_end(worm_ids, segment_indices):
    """
    Compute, for each segment, how many segments remain until the end
    of the worm's life (based on last observed segment per worm).

    Parameters
    ----------
    worm_ids : np.ndarray of shape (N,)
        Worm identifier for each segment.
    segment_indices : np.ndarray of shape (N,)
        Segment index within the worm lifetime.

    Returns
    -------
    segments_from_end : np.ndarray of shape (N,)
        Number of segments remaining until the last observed segment
        of the corresponding worm.
    """
    seg_from_end = np.zeros_like(segment_indices, dtype=int)
    for wid in np.unique(worm_ids):
        m = (worm_ids == wid)
        idxs = segment_indices[m]
        seg_from_end[m] = idxs.max() - idxs
    return seg_from_end


def make_y_near_death(segments_from_end, N_last):
    """
    Construct binary near-death labels based on proximity to end-of-life.

    Parameters
    ----------
    segments_from_end : np.ndarray of shape (N,)
        Number of segments remaining until end-of-life.
    N_last : int
        Threshold defining near-death proximity.

    Returns
    -------
    y : np.ndarray of shape (N,)
        Binary labels: 1 for near-death, 0 otherwise.
    """
    return (segments_from_end <= N_last).astype(int)


# ============================================================
# 3) EVALUATION (GROUPED CV)
# ============================================================

def grouped_cv_eval(model, X, y, groups, thr=0.5, n_splits=5, random_state=42):
    """
    Evaluate a classification model using grouped cross-validation.

    Notes
    -----
    AUC is only computed for folds where both classes are present in the test fold.
    Otherwise, that fold's AUC is skipped.

    Returns
    -------
    metrics : dict
        Mean/std of ACC, F1, AUC (nan if no valid AUC folds), and mean fit time.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s, aucs, times = [], [], [], []

    for tr, te in cv.split(X, y, groups=groups):
        m = clone(model)

        t0 = time.time()
        m.fit(X[tr], y[tr])
        times.append(time.time() - t0)

        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= thr).astype(int)

        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))

        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], proba))

    return {
        "ACC_mean": float(np.mean(accs)), "ACC_std": float(np.std(accs)),
        "F1_mean":  float(np.mean(f1s)),  "F1_std":  float(np.std(f1s)),
        "AUC_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "AUC_std":  float(np.std(aucs))  if len(aucs) else np.nan,
        "fit_s_mean": float(np.mean(times)),
    }


def grouped_cv_fixed_threshold(model, X, y, groups, thr, n_splits=5, random_state=42):
    """
    Evaluate a classifier with grouped CV using a fixed decision threshold.

    AUC is only computed for folds where both classes are present in the test fold.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s, aucs = [], [], []

    for tr, te in cv.split(X, y, groups=groups):
        m = clone(model)
        m.fit(X[tr], y[tr])

        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= thr).astype(int)

        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))

        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], proba))

    return {
        "ACC_mean": float(np.mean(accs)), "ACC_std": float(np.std(accs)),
        "F1_mean":  float(np.mean(f1s)),  "F1_std":  float(np.std(f1s)),
        "AUC_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "AUC_std":  float(np.std(aucs))  if len(aucs) else np.nan,
    }


# ============================================================
# 4) HYPERPARAMETER TUNING + THRESHOLD TUNING
# ============================================================

def tune_model(name, model, params, X_train, y_train, g_train, cv_inner, n_iter=25, random_state=42):
    """
    Randomized hyperparameter search with grouped CV on the training set.
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring="f1",
        cv=cv_inner,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True,
        return_train_score=False
    )

    t0 = time.time()
    search.fit(X_train, y_train, groups=g_train)
    dt = time.time() - t0

    best = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1 = search.best_score_

    print(f"\n=== {name} DONE ===")
    print("Best CV F1:", round(best_cv_f1, 4))
    print("Best params:", best_params)
    print(f"Time: {dt/60:.1f} min")

    return search, best, best_params, best_cv_f1, dt


def tune_threshold(model, X_val, y_val, n_steps=19):
    """
    Optimize the decision threshold on a validation set to maximize F1-score.

    Returns best_thr, best_f1, curve DataFrame(['thr','F1']).
    """
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, n_steps)

    best_thr, best_f1 = 0.5, -1
    rows = []

    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_val, pred)
        rows.append((thr, f1))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    curve = pd.DataFrame(rows, columns=["thr", "F1"])
    return float(best_thr), float(best_f1), curve


# ============================================================
# 5) ANALYSES: PROXIMITY SWEEP, ABLATION, LIFE STAGES
# ============================================================

def cv_eval_for_N(model, X, segments_from_end, worm_ids, N_last, thr=0.5, n_splits=5, random_state=42):
    """
    Evaluate model performance for a given near-death definition (last N segments)
    using grouped cross-validation.

    AUC is only computed for folds where both classes are present in the test fold.
    """
    yN = make_y_near_death(segments_from_end, N_last)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs, f1s, aucs = [], [], []

    for tr, te in cv.split(X, yN, groups=worm_ids):
        m = clone(model)
        m.fit(X[tr], yN[tr])

        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= thr).astype(int)

        accs.append(accuracy_score(yN[te], pred))
        f1s.append(f1_score(yN[te], pred))

        if len(np.unique(yN[te])) == 2:
            aucs.append(roc_auc_score(yN[te], proba))

    return {
        "pos_rate": float(yN.mean()),
        "ACC_mean": float(np.mean(accs)), "ACC_std": float(np.std(accs)),
        "F1_mean":  float(np.mean(f1s)),  "F1_std":  float(np.std(f1s)),
        "AUC_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "AUC_std":  float(np.std(aucs))  if len(aucs) else np.nan,
    }


def build_flat_from_channels(X_raw, raw_cols, cols_to_keep):
    """
    Build a flattened feature matrix using only a subset of raw channels.

    Returns X_flat of shape (N, T*k).
    """
    col_to_channel = {c: i for i, c in enumerate(raw_cols)}
    ch_idx = [col_to_channel[c] for c in cols_to_keep]
    X_sub = X_raw[:, :, ch_idx]
    return X_sub.reshape(X_sub.shape[0], -1)


def grouped_cv_f1_for_subset(model, X_raw, y, groups, raw_cols, cols_to_keep,
                             thr=0.5, n_splits=5, random_state=42):
    """
    Evaluate F1 (mean ± std) in grouped CV using only a subset of raw channels.
    """
    X_sub = build_flat_from_channels(X_raw, raw_cols, cols_to_keep)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    f1s = []
    for tr, te in cv.split(X_sub, y, groups=groups):
        m = clone(model)
        m.fit(X_sub[tr], y[tr])
        proba = m.predict_proba(X_sub[te])[:, 1]
        pred = (proba >= thr).astype(int)
        f1s.append(f1_score(y[te], pred))

    return float(np.mean(f1s)), float(np.std(f1s))


def compute_normalized_age(worm_ids, segment_indices):
    """
    Compute normalized age per segment: segment_index / max_segment_index per worm.
    """
    norm_age = np.zeros_like(segment_indices, dtype=float)
    for wid in np.unique(worm_ids):
        m = (worm_ids == wid)
        idxs = segment_indices[m]
        mx = idxs.max()
        norm_age[m] = 0.0 if mx == 0 else idxs / mx
    return norm_age


def stage_metrics_grouped_cv(model, X, y, worm_ids, stages, thr=0.5, n_splits=5, random_state=42):
    """
    Compute classification performance per life stage using grouped CV.

    Returns DataFrame with mean±std per stage.
    AUC is computed only when both classes are present within the stage subset.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_stage = defaultdict(lambda: {"ACC": [], "F1": [], "AUC": [], "n": []})

    for tr, te in cv.split(X, y, groups=worm_ids):
        m = clone(model)
        m.fit(X[tr], y[tr])

        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= thr).astype(int)

        for st in np.unique(stages):
            mask = (stages[te] == st)
            if mask.sum() < 5:
                continue

            y_true = y[te][mask]
            y_pred = pred[mask]
            y_proba = proba[mask]

            per_stage[st]["n"].append(int(mask.sum()))
            per_stage[st]["ACC"].append(accuracy_score(y_true, y_pred))
            per_stage[st]["F1"].append(f1_score(y_true, y_pred))

            if len(np.unique(y_true)) == 2:
                per_stage[st]["AUC"].append(roc_auc_score(y_true, y_proba))

    rows = []
    for st, d in per_stage.items():
        rows.append({
            "stage": st,
            "n_mean": float(np.mean(d["n"])),
            "ACC_mean": float(np.mean(d["ACC"])), "ACC_std": float(np.std(d["ACC"])),
            "F1_mean":  float(np.mean(d["F1"])),  "F1_std":  float(np.std(d["F1"])),
            "AUC_mean": float(np.mean(d["AUC"])) if len(d["AUC"]) else np.nan,
            "AUC_std":  float(np.std(d["AUC"]))  if len(d["AUC"]) else np.nan,
        })

    return pd.DataFrame(rows)

# ============================
# Figure saving utility
# ============================

FIG_DIR = "figures_near_death"
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name, dpi=300):
    """
    Save current matplotlib figure with consistent settings.

    Parameters
    ----------
    name : str
        File name without extension.
    dpi : int
        Figure resolution (300 recommended for LaTeX).
    """
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved figure -> {path}")
