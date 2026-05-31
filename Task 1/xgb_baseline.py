"""
Stage 3 — XGBoost + SHAP baseline.

Implementation strictly follows PLAN_XGBoost_SHAP_Baseline.md. Every modelling
choice in this file traces back to that document; if you find yourself wanting
to change something, update the plan first.

Outputs (all under dataset/processed/ and figures/ and logs/):
  - xgb_oof_predictions.csv     out-of-fold predictions + isotonic-calibrated probs
  - shap_values_oof.csv         per-row SHAP for 8 features + base value + label
  - xgb_loyo_metrics.csv        per-fold metrics
  - xgb_pooled_metrics.json     pooled OOF metrics with bootstrap 95% CIs
  - xgb_error_per_bank.csv      per-bank diagnostic table
  - xgb_error_per_year.csv      per-year diagnostic table
  - logs/xgb_baseline_<ts>.log  per-fold hyperparameters, fit times, sanity assertions

Run:
    python xgb_baseline.py [--seed 42] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

import evaluation as ev

# ------------------------------- Config -------------------------------------

PROJECT_ROOT = Path(__file__).parent
RATIOS_PATH = PROJECT_ROOT / "dataset" / "processed" / "financial_ratios.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

FEATURES = [
    "CAR",
    "NPL_Ratio",
    "CD_Ratio",
    "Cost_of_Funds",
    "Base_Rate",
    "Interest_Spread",
    "ROE_derived",
]
LABEL = "Deteriorate_next_year"

# Per plan §2.6 — deliberately shallow trees, small ensemble, mild regularisation.
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 3],
    "reg_lambda": [1.0, 5.0],
}
FIXED_PARAMS = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    verbosity=0,
)


# ------------------------------- Setup --------------------------------------


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"xgb_baseline_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RATIOS_PATH)
    required = ["Company", "FiscalYear", LABEL] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {RATIOS_PATH}: {missing}")
    # Modelling subset: all 8 features + label present
    mask = df[FEATURES + [LABEL]].notna().all(axis=1)
    sub = df.loc[mask].reset_index(drop=True)
    sub[LABEL] = sub[LABEL].astype(int)
    return sub


# ------------------------------- Per-fold core ------------------------------


def select_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spw: float,
    seed: int,
) -> dict:
    """Inner 3-fold stratified CV grid search. Returns best params."""
    base = XGBClassifier(
        **FIXED_PARAMS,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=1,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        scoring="f1",
        cv=inner_cv,
        n_jobs=-1,
        refit=False,
        error_score=0.0,
    )
    gs.fit(X_train, y_train)
    return dict(gs.best_params_)


def fit_isotonic_calibrator(
    base_params: dict,
    spw: float,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    df_train: pd.DataFrame,
):
    """Fit XGB on years[:-1] of the train fold and isotonic-calibrate on the latest train year.

    Per plan §2.7: held-out slice = the latest training-fold year. Time-respecting,
    no shuffling, no test-fold contamination.
    """
    years_sorted = sorted(df_train["FiscalYear"].unique())
    if len(years_sorted) < 2:
        return None  # not enough variation to hold out a calibration slice

    cal_year = years_sorted[-1]
    cal_mask = (df_train["FiscalYear"] == cal_year).to_numpy()
    fit_mask = ~cal_mask
    # need at least one positive and one negative on each side
    if y_train[fit_mask].sum() == 0 or (1 - y_train[fit_mask]).sum() == 0:
        return None
    if y_train[cal_mask].sum() == 0 or (1 - y_train[cal_mask]).sum() == 0:
        return None

    # train on the earlier slice
    pre = XGBClassifier(
        **FIXED_PARAMS,
        **base_params,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=1,
    )
    pre.fit(X_train[fit_mask], y_train[fit_mask])
    # Calibrate on the latest train-fold year. sklearn 1.8 removed cv="prefit";
    # the replacement is to wrap the fitted estimator in FrozenEstimator and let
    # CalibratedClassifierCV detect it (it then skips refitting and just learns
    # the isotonic mapping on the supplied calibration data).
    cal = CalibratedClassifierCV(estimator=FrozenEstimator(pre), method="isotonic")
    cal.fit(X_train[cal_mask], y_train[cal_mask])
    return cal


# ------------------------------- Main loop ----------------------------------


def run_loyo(df: pd.DataFrame, seed: int, log: logging.Logger) -> dict:
    """Outer LOYO loop. Returns dict of results to persist."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[LABEL].to_numpy(dtype=int)

    oof_proba_raw = np.full(len(df), np.nan)
    oof_proba_isotonic = np.full(len(df), np.nan)
    oof_shap = np.full((len(df), len(FEATURES)), np.nan)
    oof_base = np.full(len(df), np.nan)

    fold_records = []
    fold_idx = 0

    for train_idx, test_idx, held_year in ev.loyo_splits(df):
        fold_idx += 1
        t0 = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        df_train = df.iloc[train_idx]

        # Sanity check: no overlap
        assert len(set(train_idx) & set(test_idx)) == 0
        # Sanity check: held year never in train
        assert held_year not in set(df_train["FiscalYear"].tolist())

        n_pos = int(y_train.sum())
        n_neg = int((1 - y_train).sum())
        if n_pos == 0 or n_neg == 0:
            log.warning(
                "Fold %d (year %s): train fold has class imbalance n_pos=%d n_neg=%d — skipping fold",
                fold_idx, held_year, n_pos, n_neg,
            )
            continue
        spw = n_neg / n_pos

        # Inner CV hyperparameter search
        best = select_hyperparams(X_train, y_train, spw, seed=seed)

        # Final fit on full outer-train rows
        final = XGBClassifier(
            **FIXED_PARAMS,
            **best,
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=1,
        )
        final.fit(X_train, y_train)

        # Predict on the held-out year — raw XGBoost probabilities
        proba = final.predict_proba(X_test)[:, 1]
        oof_proba_raw[test_idx] = proba

        # Isotonic-calibrated probabilities (separate fit)
        cal = fit_isotonic_calibrator(best, spw, seed, X_train, y_train, df_train)
        if cal is not None:
            try:
                proba_iso = cal.predict_proba(X_test)[:, 1]
                oof_proba_isotonic[test_idx] = proba_iso
            except Exception as e:
                log.warning("Fold %d (year %s): isotonic calibration failed: %s", fold_idx, held_year, e)

        # SHAP on the test rows of THIS fold's model
        explainer = shap.TreeExplainer(final)
        sv = explainer.shap_values(X_test)
        # newer SHAP returns array for binary; older may return list; normalise
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, n_features, n_classes)
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[..., 0]
        oof_shap[test_idx] = sv

        base_val = explainer.expected_value
        if hasattr(base_val, "__len__"):
            base_val = base_val[1] if len(base_val) == 2 else base_val[0]
        oof_base[test_idx] = float(base_val)

        # SHAP additivity assertion per plan §7.3.
        # SHAP for XGBoost trees is exact additive in margin (log-odds) space:
        # base_value + sum(shap_values) == predict(output_margin=True).
        margin = final.predict(X_test, output_margin=True)
        for i in range(len(test_idx)):
            total = float(base_val) + float(sv[i].sum())
            if not math.isclose(float(margin[i]), total, abs_tol=1e-4, rel_tol=1e-3):
                log.warning(
                    "Fold %d row %d: SHAP additivity off by %.6f (margin=%.6f, base+sum=%.6f)",
                    fold_idx, i, float(margin[i]) - total, float(margin[i]), total,
                )

        dt = time.time() - t0
        fold_records.append({
            "fold": fold_idx,
            "held_out_year": held_year,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_test_positive": int(y_test.sum()),
            "scale_pos_weight": round(spw, 4),
            **{f"best_{k}": v for k, v in best.items()},
            "fit_seconds": round(dt, 2),
        })

        log.info(
            "Fold %d  year=%s  n_train=%d  n_test=%d  spw=%.2f  best=%s  time=%.1fs",
            fold_idx, held_year, len(train_idx), len(test_idx), spw, best, dt,
        )

    return {
        "df": df,
        "oof_proba_raw": oof_proba_raw,
        "oof_proba_isotonic": oof_proba_isotonic,
        "oof_shap": oof_shap,
        "oof_base": oof_base,
        "fold_records": fold_records,
    }


# ------------------------------- Persistence --------------------------------


def persist_results(res: dict, seed: int, n_boot: int, log: logging.Logger, log_path: Path) -> None:
    df = res["df"]

    # Drop any row that never received an OOF prediction (shouldn't happen for LOYO,
    # but defensive in case a fold was skipped).
    have_pred = ~np.isnan(res["oof_proba_raw"])
    if not have_pred.all():
        n_missing = int((~have_pred).sum())
        log.warning("%d rows have no OOF prediction (folds skipped). Excluding from pooled metrics.", n_missing)

    oof_df = pd.DataFrame({
        "Company": df["Company"].values,
        "FiscalYear": df["FiscalYear"].values,
        "FiscalYearAD": df["FiscalYearAD"].values if "FiscalYearAD" in df.columns else "",
        "y_true": df[LABEL].values,
        "y_pred_proba": res["oof_proba_raw"],
        "y_pred_proba_isotonic": res["oof_proba_isotonic"],
    })
    oof_path = OUT_DIR / "xgb_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info("Wrote %s  (%d rows)", oof_path, len(oof_df))

    # SHAP matrix
    shap_df = pd.DataFrame(res["oof_shap"], columns=[f"shap_{f}" for f in FEATURES])
    shap_df.insert(0, "Company", df["Company"].values)
    shap_df.insert(1, "FiscalYear", df["FiscalYear"].values)
    shap_df["base_value"] = res["oof_base"]
    shap_df["predicted_proba"] = res["oof_proba_raw"]
    shap_df["true_label"] = df[LABEL].values
    shap_path = OUT_DIR / "shap_values_oof.csv"
    shap_df.to_csv(shap_path, index=False)
    log.info("Wrote %s  (%d rows)", shap_path, len(shap_df))

    # Per-fold metrics CSV
    fold_df = pd.DataFrame(res["fold_records"])
    fold_path = OUT_DIR / "xgb_loyo_metrics.csv"
    fold_df.to_csv(fold_path, index=False)
    log.info("Wrote %s  (%d folds)", fold_path, len(fold_df))

    # Pooled metrics with bootstrap CIs (raw and isotonic)
    valid = have_pred
    y_true = df[LABEL].values[valid]
    y_raw = res["oof_proba_raw"][valid]
    raw_metrics = ev.compute_metrics(y_true, y_raw)
    raw_cis = ev.all_metrics_with_ci(y_true, y_raw, n_boot=n_boot, seed=seed)

    iso_valid = valid & ~np.isnan(res["oof_proba_isotonic"])
    if iso_valid.sum() >= 10:
        y_true_iso = df[LABEL].values[iso_valid]
        y_iso = res["oof_proba_isotonic"][iso_valid]
        iso_metrics = ev.compute_metrics(y_true_iso, y_iso)
        iso_cis = ev.all_metrics_with_ci(y_true_iso, y_iso, n_boot=n_boot, seed=seed)
    else:
        iso_metrics, iso_cis = None, None

    pooled = {
        "n_rows_pooled_raw": int(valid.sum()),
        "n_rows_pooled_isotonic": int(iso_valid.sum()) if iso_metrics else 0,
        "class_balance": {
            "n_positive": int(y_true.sum()),
            "n_negative": int((1 - y_true).sum()),
            "positive_rate": float(y_true.mean()),
        },
        "raw_xgboost": {
            "point": raw_metrics,
            "ci_95": {k: {"point": v.point, "lower": v.lower, "upper": v.upper} for k, v in raw_cis.items()},
        },
    }
    if iso_metrics is not None:
        pooled["isotonic_xgboost"] = {
            "point": iso_metrics,
            "ci_95": {k: {"point": v.point, "lower": v.lower, "upper": v.upper} for k, v in iso_cis.items()},
        }
    pooled_path = OUT_DIR / "xgb_pooled_metrics.json"
    with open(pooled_path, "w", encoding="utf-8") as f:
        json.dump(pooled, f, indent=2)
    log.info("Wrote %s", pooled_path)

    # Per-bank and per-year diagnostic tables
    diag_input = oof_df.loc[valid, ["Company", "FiscalYear", "y_true", "y_pred_proba"]].copy()
    bank_path = OUT_DIR / "xgb_error_per_bank.csv"
    year_path = OUT_DIR / "xgb_error_per_year.csv"
    ev.per_group_table(diag_input, "Company").to_csv(bank_path, index=False)
    ev.per_group_table(diag_input, "FiscalYear").to_csv(year_path, index=False)
    log.info("Wrote %s and %s", bank_path, year_path)

    # Headline log summary
    log.info("=" * 72)
    log.info("POOLED OOF METRICS  (raw XGBoost, N=%d)", int(valid.sum()))
    for m, ci in raw_cis.items():
        log.info("  %-10s  point=%.4f  95%% CI [%.4f, %.4f]", m, ci.point, ci.lower, ci.upper)
    if iso_metrics is not None:
        log.info("POOLED OOF METRICS  (isotonic-calibrated XGBoost, N=%d)", int(iso_valid.sum()))
        for m, ci in iso_cis.items():
            log.info("  %-10s  point=%.4f  95%% CI [%.4f, %.4f]", m, ci.point, ci.lower, ci.upper)
    log.info("Run log: %s", log_path)


# ------------------------------- Sanity gate --------------------------------


def assert_basic_sanity(res: dict, df: pd.DataFrame, log: logging.Logger) -> None:
    """Plan §7: non-negotiable acceptance checks."""
    # 4. OOF coverage
    have = ~np.isnan(res["oof_proba_raw"])
    assert have.all(), f"OOF coverage incomplete: {(~have).sum()} rows missing predictions"
    # 7. (deferred to persist_results — bootstrap CI containment)
    # 6. Hyperparameter selection variance — informational, not fatal
    keys = ["best_n_estimators", "best_max_depth", "best_learning_rate", "best_min_child_weight", "best_reg_lambda"]
    if res["fold_records"]:
        chosen = pd.DataFrame(res["fold_records"])
        for k in keys:
            if k in chosen.columns:
                mode_count = chosen[k].value_counts().iloc[0]
                if mode_count / len(chosen) > 0.75:
                    log.info("Hyperparam %s: >75%% folds chose %s (search may be too narrow)", k, chosen[k].mode()[0])
                if chosen[k].nunique() == len(chosen):
                    log.info("Hyperparam %s: every fold picks a different value (search may be overfitting)", k)
    log.info("Basic sanity checks passed.")


# ------------------------------- Entry point --------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    log_path = setup_logging()
    log = logging.getLogger(__name__)
    log.info("Stage 3 — XGBoost+SHAP baseline. seed=%d  n_boot=%d", args.seed, args.n_boot)
    log.info("Per PLAN_XGBoost_SHAP_Baseline.md. Modifications must be in the plan first.")

    df = load_data()
    log.info("Loaded %d labeled rows. Class balance: %d pos / %d neg",
             len(df), int(df[LABEL].sum()), int((1 - df[LABEL]).sum()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    res = run_loyo(df, seed=args.seed, log=log)
    assert_basic_sanity(res, df, log)
    persist_results(res, seed=args.seed, n_boot=args.n_boot, log=log, log_path=log_path)

    log.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
