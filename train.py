"""Reproduce the full model-development run.

    python train.py                 # full run (~1-2 h on 2 CPU cores)
    python train.py --quick         # smoke run on a subsample, 3 folds, 1 config each

Steps
 1. Load the raw CSV and verify its checksum, columns and label mapping.
 2. Hold out a stratified 20% test set FIRST, grouped by SHA1 so the same file
    can never appear on both sides. Nothing below looks at the test set.
 3. De-duplicate each partition separately (one row per file; drop files with
    conflicting labels).
 4. Profile the training partition only.
 5. Feature-selection ablation (10-fold CV, LR + LightGBM).
 6. For every model: tuning screen on folds 1-3, then all 10 folds for the
    chosen setting. All preprocessing is refit inside each training fold.
 7. Select the production model by mean CV AUC (accuracy / speed break ties),
    freeze it with threshold 0.5, refit on the full training set and save the
    artifact. Hold-out evaluation is done separately by eval.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from malware_ml import schema
from malware_ml.features import build_preprocessor, deduplicate
from malware_ml.models import ADDITIONAL, ALL_MODELS, BASELINES, GRIDS, SEED, make_estimator

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "brazilian-malware.csv"
SPLIT_DIR = ROOT / "data" / "splits"
ART = ROOT / "artifacts"
REP = ROOT / "reports"
N_FOLDS = 10
SCREEN_FOLDS = 3
THRESHOLD = 0.5
MODEL_VERSION = "1.0.0"

# Chosen after the ablation in step 5 (see reports/feature_ablation.csv).
DEFAULT_PREPROC = {"n_ident": 50, "n_dll": 300, "n_sym": 1000}
ABLATION = {
    "numeric_only": {"n_ident": 0, "n_dll": 0, "n_sym": 0},
    "numeric+identify+dlls": {"n_ident": 50, "n_dll": 300, "n_sym": 0},
    "all_text_small_vocab": {"n_ident": 50, "n_dll": 300, "n_sym": 1000},
    "all_text_large_vocab": {"n_ident": 100, "n_dll": 1000, "n_sym": 5000},
}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
        torch.set_num_threads(2)
    except ImportError:
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- data
def load_and_split(quick: bool):
    digest = sha256(RAW)
    if digest != schema.DATASET["csv_sha256"]:
        log(f"WARNING: CSV checksum {digest} differs from recorded {schema.DATASET['csv_sha256']}")
    df = pd.read_csv(RAW, low_memory=False)
    expected = set(schema.MODEL_FEATURES) | set(schema.EXCLUDED_COLUMNS)
    assert set(df.columns) == expected, f"schema mismatch: {set(df.columns) ^ expected}"
    assert set(df[schema.LABEL_COL].unique()) == {0, 1}, "label must be 0/1"
    assert df.columns.get_loc(schema.LABEL_COL) == 11, "Label expected in column L"

    # Step 2: grouped, stratified 80/20 split BEFORE any preprocessing.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(df, df[schema.LABEL_COL], groups=df[schema.ID_COL]))
    train_raw, test_raw = df.iloc[train_idx], df.iloc[test_idx]
    assert not set(train_raw[schema.ID_COL]) & set(test_raw[schema.ID_COL])

    # Step 3: deterministic de-duplication per partition.
    train, train_stats = deduplicate(train_raw)
    test, test_stats = deduplicate(test_raw)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    test.to_csv(SPLIT_DIR / "holdout_test.csv", index=False)
    train.to_csv(SPLIT_DIR / "train.csv", index=False)

    split_info = {
        "method": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), first fold = test; groups=SHA1",
        "raw_train_rows": int(len(train_raw)),
        "raw_test_rows": int(len(test_raw)),
        "raw_test_fraction": round(len(test_raw) / len(df), 4),
        "train_dedup": train_stats,
        "test_dedup": test_stats,
        "train_class_counts": {int(k): int(v) for k, v in train[schema.LABEL_COL].value_counts().sort_index().items()},
        "test_class_counts": {int(k): int(v) for k, v in test[schema.LABEL_COL].value_counts().sort_index().items()},
        "csv_sha256_observed": digest,
    }
    if quick:
        train = train.sample(4000, random_state=SEED).reset_index(drop=True)
    return train, split_info


def profile(train: pd.DataFrame) -> dict:
    num = train[schema.NUMERIC_FEATURES]
    return {
        "rows": int(len(train)),
        "class_balance": {schema.LABEL_MAPPING[int(k)]: round(float(v), 4) for k, v in train[schema.LABEL_COL].value_counts(normalize=True).items()},
        "missing_values": {c: int(v) for c, v in train.isna().sum().items() if v},
        "constant_columns": [c for c in schema.NUMERIC_FEATURES if num[c].nunique() == 1],
        "heavy_tailed_columns(max/median>1000)": [
            c for c in schema.NUMERIC_FEATURES if num[c].median() > 0 and num[c].max() / num[c].median() > 1000
        ],
        "token_counts_median": {c: float(train[c].fillna("").str.split().str.len().median()) for c in schema.TEXT_FEATURES},
        "single_feature_auc(top10, leakage screen)": dict(
            sorted(
                {
                    c: round(max(roc_auc_score(train[schema.LABEL_COL], num[c]), 1 - roc_auc_score(train[schema.LABEL_COL], num[c])), 4)
                    for c in schema.NUMERIC_FEATURES
                    if num[c].nunique() > 1
                }.items(),
                key=lambda kv: -kv[1],
            )[:10]
        ),
        "describe": num.describe().T.round(3).to_dict(orient="index"),
    }


# --------------------------------------------------------------------------- CV machinery
def make_folds(y, n=N_FOLDS):
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def transform_folds(X: pd.DataFrame, y: np.ndarray, folds, preproc_cfg: dict, fold_ids):
    """Fit the preprocessor on each training fold only, transform its validation fold."""
    out = {}
    for k in fold_ids:
        tr, va = folds[k]
        pre = build_preprocessor(**preproc_cfg)
        Xtr = pre.fit_transform(X.iloc[tr])
        Xva = pre.transform(X.iloc[va])
        out[k] = (sp.csr_matrix(Xtr, dtype=np.float32), y[tr], sp.csr_matrix(Xva, dtype=np.float32), y[va])
    return out


def cv_score(model_name, params, mats, fold_ids):
    aucs, accs, fit_s, pred_ms = [], [], [], []
    for k in fold_ids:
        Xtr, ytr, Xva, yva = mats[k]
        est = make_estimator(model_name, params)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        fit_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        p = est.predict_proba(Xva)[:, 1]
        pred_ms.append((time.perf_counter() - t0) * 1000 / Xva.shape[0] * 1000)  # ms per 1000 rows
        aucs.append(roc_auc_score(yva, p))  # AUC from continuous P(malware)
        accs.append(accuracy_score(yva, (p >= THRESHOLD).astype(int)))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "fit_seconds_mean": float(np.mean(fit_s)),
        "predict_ms_per_1000_rows": float(np.mean(pred_ms)),
        "fold_aucs": [round(a, 6) for a in aucs],
        "fold_accs": [round(a, 6) for a in accs],
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    args = ap.parse_args()
    set_seeds()
    ART.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    log("loading data and creating hold-out split")
    train, split_info = load_and_split(args.quick)
    (REP / "split_info.json").write_text(json.dumps(split_info, indent=2))
    (REP / "data_profile_train.json").write_text(json.dumps(profile(train), indent=2, default=str))
    log(json.dumps({k: v for k, v in split_info.items() if k != "csv_sha256_observed"}))

    X = train[schema.MODEL_FEATURES]
    y = train[schema.LABEL_COL].to_numpy()
    n_folds = 3 if args.quick else N_FOLDS
    folds = make_folds(y, n_folds)
    all_ids = list(range(n_folds))
    screen_ids = all_ids[:SCREEN_FOLDS]

    # Step 5: feature-selection ablation.
    if not args.skip_ablation and not args.quick:
        rows = []
        for cfg_name, cfg in ABLATION.items():
            log(f"ablation: {cfg_name}")
            mats = transform_folds(X, y, folds, cfg, all_ids)
            n_feat = mats[0][0].shape[1]
            for m, p in [("logistic_regression", {"C": 1.0}), ("lightgbm", GRIDS["lightgbm"][1])]:
                r = cv_score(m, p, mats, all_ids)
                rows.append({"feature_set": cfg_name, **cfg, "n_features_fold0": n_feat, "model": m,
                             "auc_mean": r["auc_mean"], "auc_std": r["auc_std"], "acc_mean": r["acc_mean"], "acc_std": r["acc_std"]})
                log(f"  {m}: AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f} acc {r['acc_mean']:.4f}")
            del mats
        pd.DataFrame(rows).to_csv(REP / "feature_ablation.csv", index=False)

    # Step 6: tuning screen + full CV, all on the same leak-free fold matrices.
    log(f"building fold matrices with {DEFAULT_PREPROC}")
    mats = transform_folds(X, y, folds, DEFAULT_PREPROC, all_ids)
    tuning_rows, cv_rows = [], []
    for m in args.models:
        grid = GRIDS[m][:1] if args.quick else GRIDS[m]
        best, best_auc = None, -1
        for params in grid:
            r = cv_score(m, params, mats, screen_ids)
            tuning_rows.append({"model": m, "params": json.dumps(params), "screen_auc": r["auc_mean"], "screen_acc": r["acc_mean"], "fit_seconds": r["fit_seconds_mean"]})
            log(f"tune {m} {params}: AUC {r['auc_mean']:.4f} acc {r['acc_mean']:.4f} ({r['fit_seconds_mean']:.1f}s/fit)")
            if r["auc_mean"] > best_auc:
                best, best_auc = params, r["auc_mean"]
        r = cv_score(m, best, mats, all_ids)
        group = "baseline" if m in BASELINES else "additional"
        cv_rows.append({"model": m, "group": group, "selected_params": json.dumps(best), "grid_size": len(grid), **r})
        log(f"CV {m}: AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f}  acc {r['acc_mean']:.4f}±{r['acc_std']:.4f}")
        pd.DataFrame(tuning_rows).to_csv(REP / "tuning_results.csv", index=False)
        pd.DataFrame(cv_rows).to_csv(REP / "cv_results.csv", index=False)
    del mats

    # Step 7: selection. Highest mean AUC; if within 0.001 of the best, prefer
    # higher accuracy, then faster inference.
    cv = pd.DataFrame(cv_rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    top = cv.iloc[0]["auc_mean"]
    close = cv[cv["auc_mean"] >= top - 0.001].sort_values(["acc_mean", "predict_ms_per_1000_rows"], ascending=[False, True])
    chosen = close.iloc[0]
    log(f"selected production model: {chosen['model']} (close contenders: {list(close['model'])})")

    params = json.loads(chosen["selected_params"])
    if "hidden_sizes" in params:
        params["hidden_sizes"] = tuple(params["hidden_sizes"])
    pipe = Pipeline([("preprocess", build_preprocessor(**DEFAULT_PREPROC)), ("model", make_estimator(chosen["model"], params))])
    t0 = time.perf_counter()
    pipe.fit(X, y)
    log(f"refit on full training set ({len(X)} rows) in {time.perf_counter() - t0:.1f}s")

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": chosen["model"],
        "hyperparameters": params,
        "preprocessing": DEFAULT_PREPROC,
        "classification_threshold": THRESHOLD,
        "threshold_note": "predict malware when P(malware) >= threshold; scores are not calibrated probabilities",
        "label_mapping": schema.LABEL_MAPPING,
        "positive_class": schema.POSITIVE_CLASS,
        "feature_schema": {
            "model_features": schema.MODEL_FEATURES,
            "types": schema.FEATURE_TYPES,
            "nullable": sorted(schema.NULLABLE_FEATURES),
            "ignored_if_present": list(schema.EXCLUDED_COLUMNS),
        },
        "dataset": schema.DATASET,
        "split": split_info,
        "cv": {"folds": n_folds, "auc_mean": float(chosen["auc_mean"]), "auc_std": float(chosen["auc_std"]),
               "acc_mean": float(chosen["acc_mean"]), "acc_std": float(chosen["acc_std"])},
        "selection_rule": "max mean 10-fold CV AUC; models within 0.001 AUC ranked by accuracy then inference speed",
        "seed": SEED,
        "trained_at": started,
        "quick_run": args.quick,
        "environment": {"python": platform.python_version(), **{m: __import__(m).__version__ for m in ("sklearn", "xgboost", "lightgbm", "catboost", "torch", "pandas", "numpy")}},
    }
    joblib.dump(pipe, ART / "model.joblib", compress=3)
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    log("saved artifacts/model.joblib and artifacts/metadata.json — now run: python eval.py")


if __name__ == "__main__":
    main()
