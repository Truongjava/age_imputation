"""Buoc 4: huan luyen LightGBM + danh gia cross-validation + dien age thieu.

Target encoding (article/department/category) duoc tinh lai trong moi fold chi tu
du lieu train cua fold do => khong ro ri nhan. Bao cao dung out-of-fold MAE.
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src import config as C
from src import dataset as D
from src.target_encoding import TargetEncoder


def age_bucket(age: np.ndarray) -> np.ndarray:
    return np.digitize(age, C.AGE_BINS[1:-1])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, C.AGE_MIN, C.AGE_MAX)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bucket_acc": float((age_bucket(y_true) == age_bucket(y_pred)).mean()),
        "within_5y": float((np.abs(y_true - y_pred) <= 5).mean()),
        "within_10y": float((np.abs(y_true - y_pred) <= 10).mean()),
    }


def oof_postal_baseline(y: np.ndarray, postal: np.ndarray) -> dict:
    """Baseline postal-code median nhung tinh out-of-fold cho cong bang."""
    labeled = np.flatnonzero(~np.isnan(y))
    pred = np.full(y.size, np.nan, dtype=np.float32)
    prior = float(np.nanmedian(y))
    kf = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)
    for tr_pos, va_pos in kf.split(labeled):
        tr, va = labeled[tr_pos], labeled[va_pos]
        sums = np.bincount(postal[tr], weights=y[tr], minlength=postal.max() + 1)
        cnts = np.bincount(postal[tr], minlength=postal.max() + 1)
        enc = (sums + prior * 50.0) / (cnts + 50.0)
        pred[va] = enc[postal[va]]
    out = evaluate(y[labeled], pred[labeled])
    out["n"] = int(labeled.size)
    return out


def make_matrix(X: np.ndarray, idx: np.ndarray, te: np.ndarray) -> np.ndarray:
    n_feat = X.shape[1]
    out = np.empty((idx.size, n_feat + te.shape[1]), dtype=np.float32)
    step = 200_000
    for s in range(0, idx.size, step):
        chunk = idx[s : s + step]
        out[s : s + step, :n_feat] = X[chunk]
        out[s : s + step, n_feat:] = te[chunk]
    return out


def main() -> None:
    t0 = time.time()
    X, y, base_names = D.load()
    encoder = TargetEncoder(y.size)
    names = base_names + encoder.names

    labeled = np.flatnonzero(~np.isnan(y))
    print(f"labeled={labeled.size:,} missing={y.size - labeled.size:,}", flush=True)

    oof = np.full(y.size, np.nan, dtype=np.float32)
    fold_metrics, best_iters = [], []
    kf = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)

    for fold, (tr_pos, va_pos) in enumerate(kf.split(labeled), 1):
        tr_idx, va_idx = labeled[tr_pos], labeled[va_pos]
        if C.TRAIN_SAMPLE and C.TRAIN_SAMPLE < tr_idx.size:
            rng = np.random.default_rng(C.RANDOM_STATE + fold)
            fit_idx = np.sort(rng.choice(tr_idx, C.TRAIN_SAMPLE, replace=False))
        else:
            fit_idx = tr_idx

        train_mask = np.zeros(y.size, bool)
        train_mask[tr_idx] = True
        te_fold = encoder.transform(train_mask, y)

        X_tr = make_matrix(X, fit_idx, te_fold)
        X_va = make_matrix(X, va_idx, te_fold)

        dtrain = lgb.Dataset(X_tr, label=y[fit_idx], feature_name=names, free_raw_data=True)
        dvalid = lgb.Dataset(X_va, label=y[va_idx], reference=dtrain, free_raw_data=True)
        model = lgb.train(
            C.LGB_PARAMS,
            dtrain,
            num_boost_round=C.NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(C.EARLY_STOPPING, verbose=False)],
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof[va_idx] = pred
        m = evaluate(y[va_idx], np.clip(pred, C.AGE_MIN, C.AGE_MAX))
        m.update(fold=fold, best_iter=model.best_iteration)
        fold_metrics.append(m)
        best_iters.append(model.best_iteration)
        print(
            f"fold {fold}: MAE={m['mae']:.4f} R2={m['r2']:.4f} iter={m['best_iter']} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del X_tr, X_va, dtrain, dvalid, model

    cv = evaluate(y[labeled], oof[labeled])
    print("CV:", {k: round(v, 4) for k, v in cv.items()}, flush=True)

    postal = (
        pl.read_parquet(C.ARTIFACTS / "customers_base.parquet", columns=["cust_int", "postal_code"])
        .sort("cust_int")
        .get_column("postal_code")
        .cast(pl.Categorical)
        .to_physical()
        .to_numpy()
        .astype(np.int64)
    )
    baseline_global = evaluate(y[labeled], np.full(labeled.size, np.nanmedian(y)))
    baseline_postal = oof_postal_baseline(y, postal)

    # --- Model cuoi cung: encoding tren toan bo du lieu co nhan ---
    te_full = encoder.transform(np.ones(y.size, bool), y)
    n_rounds = int(np.mean(best_iters))
    if C.TRAIN_SAMPLE and C.TRAIN_SAMPLE < labeled.size:
        rng = np.random.default_rng(C.RANDOM_STATE)
        fit_idx = np.sort(rng.choice(labeled, C.TRAIN_SAMPLE, replace=False))
    else:
        fit_idx = labeled
    X_fit = make_matrix(X, fit_idx, te_full)
    final = lgb.train(
        C.LGB_PARAMS,
        lgb.Dataset(X_fit, label=y[fit_idx], feature_name=names),
        num_boost_round=max(n_rounds, 100),
    )
    final.save_model(str(C.ARTIFACTS / "lgbm_age.txt"))

    missing = np.flatnonzero(np.isnan(y))
    X_miss = make_matrix(X, missing, te_full)
    pred_missing = np.clip(final.predict(X_miss), C.AGE_MIN, C.AGE_MAX)

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.save(C.OUTPUTS / "oof_pred.npy", oof)
    np.save(C.OUTPUTS / "pred_missing.npy", pred_missing)
    np.save(C.OUTPUTS / "missing_rows.npy", missing)

    importance = sorted(zip(names, final.feature_importance("gain")), key=lambda t: -t[1])[:25]

    metrics = {
        "cv": cv,
        "folds": fold_metrics,
        "baselines": {"global_median": baseline_global, "postal_median_oof": baseline_postal},
        "n_customers": int(y.size),
        "n_labeled": int(labeled.size),
        "n_missing": int(missing.size),
        "best_iterations": best_iters,
        "final_num_trees": int(final.num_trees()),
        "n_features": len(names),
        "features": names,
        "lgb_params": C.LGB_PARAMS,
        "top_importance": [{"feature": f, "gain": float(g)} for f, g in importance if g > 0],
        "imputed_stats": {
            "mean": float(pred_missing.mean()),
            "std": float(pred_missing.std()),
            "median": float(np.median(pred_missing)),
            "min": float(pred_missing.min()),
            "max": float(pred_missing.max()),
        },
        "observed_stats": {
            "mean": float(np.nanmean(y)),
            "std": float(np.nanstd(y)),
            "median": float(np.nanmedian(y)),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"done in {metrics['runtime_sec']}s -> {C.METRICS_JSON}")


if __name__ == "__main__":
    main()
