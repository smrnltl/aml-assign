"""
Shared evaluation harness for all modelling stages.

Built once during Stage 3 (XGBoost+SHAP baseline) per PLAN_XGBoost_SHAP_Baseline.md.
Reused unchanged by Stage 4 (GP regression), Stage 5 (GP classification), and
Stage 6 (fuzzy layer) per PLAN_OVERALL_Project.md.

Design rules (locked, do not change without updating the plan):
  - Leave-One-Year-Out (LOYO) is the primary CV splitter.
  - Leave-One-Bank-Out (LOBO) is the secondary robustness splitter.
  - No random K-fold anywhere.
  - Bootstrap CIs are stratified by class label.
  - Calibration uses quantile-spaced bins, never equal-width, given N=80.
  - All randomness is seeded; default seed is 42.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ----------------------------- CV splitters ---------------------------------


def loyo_splits(df: pd.DataFrame, year_col: str = "FiscalYear") -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield (train_idx, test_idx, held_out_year) for each fiscal year in df.

    For each unique year, the test set is all rows from that year, train is the rest.
    Skips years with zero rows (defensive; shouldn't happen on a clean panel).
    """
    years = sorted(df[year_col].unique())
    idx_all = np.arange(len(df))
    for y in years:
        test_mask = (df[year_col] == y).to_numpy()
        train_idx = idx_all[~test_mask]
        test_idx = idx_all[test_mask]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield train_idx, test_idx, y


def lobo_splits(df: pd.DataFrame, bank_col: str = "Company") -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield (train_idx, test_idx, held_out_bank) for each bank in df."""
    banks = sorted(df[bank_col].unique())
    idx_all = np.arange(len(df))
    for b in banks:
        test_mask = (df[bank_col] == b).to_numpy()
        train_idx = idx_all[~test_mask]
        test_idx = idx_all[test_mask]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield train_idx, test_idx, b


# ----------------------------- Metrics --------------------------------------


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 5) -> float:
    """Expected Calibration Error with quantile-spaced bins.

    Equal-width binning leaves bins empty at N=80; quantile bins put roughly
    equal sample counts per bin, which is the honest choice for a small panel.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_proba)
    if n == 0:
        return float("nan")
    # quantile edges; clip to [0, 1]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_proba, quantiles))
    if len(edges) < 2:
        # all predictions identical; ECE = |proba - mean(y)|
        return float(abs(y_proba.mean() - y_true.mean()))
    # ensure last edge is strictly > max(y_proba) so digitize is well-behaved
    edges[-1] = max(edges[-1], y_proba.max()) + 1e-12
    bin_ids = np.digitize(y_proba, edges[1:-1])  # 0..n_bins-1 effectively
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_proba = y_proba[mask].mean()
        bin_truth = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_proba - bin_truth)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 5):
    """Return (mean_predicted, fraction_positive, bin_sizes) for a reliability diagram.

    Bins are quantile-spaced. Bin sizes returned so they can be annotated on the plot.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_proba, quantiles))
    if len(edges) < 2:
        return np.array([y_proba.mean()]), np.array([y_true.mean()]), np.array([len(y_proba)])
    edges[-1] = max(edges[-1], y_proba.max()) + 1e-12
    bin_ids = np.digitize(y_proba, edges[1:-1])
    out_p, out_t, out_n = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        out_p.append(float(y_proba[mask].mean()))
        out_t.append(float(y_true[mask].mean()))
        out_n.append(int(mask.sum()))
    return np.array(out_p), np.array(out_t), np.array(out_n)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    ece_bins: int = 5,
) -> dict[str, float]:
    """Eight headline metrics. NaN for any metric that's undefined on the inputs."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    out: dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))

    # F1, precision, recall on positive class; undefined if no positives in either set
    if y_true.sum() == 0 or y_pred.sum() == 0:
        out["f1"] = float("nan") if y_true.sum() == 0 else float(f1_score(y_true, y_pred, zero_division=0))
        out["precision"] = float("nan") if y_pred.sum() == 0 else float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float("nan") if y_true.sum() == 0 else float(recall_score(y_true, y_pred, zero_division=0))
    else:
        out["f1"] = float(f1_score(y_true, y_pred))
        out["precision"] = float(precision_score(y_true, y_pred))
        out["recall"] = float(recall_score(y_true, y_pred))

    # ROC-AUC and PR-AUC require both classes present
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))

    out["brier"] = float(brier_score_loss(y_true, y_proba))
    out["ece"] = expected_calibration_error(y_true, y_proba, n_bins=ece_bins)
    return out


# ----------------------- Bootstrap confidence intervals ----------------------


@dataclass
class CI:
    point: float
    lower: float
    upper: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.lower, self.upper)


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CI:
    """95% CI for a single metric, computed by stratified bootstrap.

    Stratification preserves the class balance of each resample so F1 etc.
    remain defined. Returns the point estimate (computed on the full data, not
    a resample) and the (alpha/2, 1-alpha/2) percentiles across n_boot resamples.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(seed)

    point = float(metric_fn(y_true, y_proba))

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return CI(point=point, lower=float("nan"), upper=float("nan"))

    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pos_resample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_resample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_resample, neg_resample])
        try:
            samples[b] = float(metric_fn(y_true[idx], y_proba[idx]))
        except Exception:
            samples[b] = float("nan")

    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return CI(point=point, lower=float("nan"), upper=float("nan"))
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return CI(point=point, lower=lo, upper=hi)


def all_metrics_with_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    ece_bins: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, CI]:
    """Run stratified bootstrap CI for every metric in compute_metrics()."""

    def metric_fns():
        return {
            "accuracy": lambda yt, yp: accuracy_score(yt, (yp >= threshold).astype(int)),
            "f1": lambda yt, yp: f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
                if yt.sum() > 0 and (yp >= threshold).sum() > 0 else float("nan"),
            "precision": lambda yt, yp: precision_score(yt, (yp >= threshold).astype(int), zero_division=0)
                if (yp >= threshold).sum() > 0 else float("nan"),
            "recall": lambda yt, yp: recall_score(yt, (yp >= threshold).astype(int), zero_division=0)
                if yt.sum() > 0 else float("nan"),
            "roc_auc": lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else float("nan"),
            "pr_auc": lambda yt, yp: average_precision_score(yt, yp) if len(np.unique(yt)) == 2 else float("nan"),
            "brier": lambda yt, yp: brier_score_loss(yt, yp),
            "ece": lambda yt, yp: expected_calibration_error(yt, yp, n_bins=ece_bins),
        }

    fns = metric_fns()
    out: dict[str, CI] = {}
    for name, fn in fns.items():
        out[name] = stratified_bootstrap_ci(y_true, y_proba, fn, n_boot=n_boot, seed=seed)
    return out


# ------------------------ Per-bank / per-year diagnostics --------------------


def per_group_table(
    df_oof: pd.DataFrame,
    group_col: str,
    y_true_col: str = "y_true",
    y_proba_col: str = "y_pred_proba",
) -> pd.DataFrame:
    """Mean predicted probability vs true rate, per group. Surfaces systematic bias."""
    rows = []
    for g, sub in df_oof.groupby(group_col):
        rows.append({
            group_col: g,
            "n": len(sub),
            "n_positive": int(sub[y_true_col].sum()),
            "true_rate": float(sub[y_true_col].mean()),
            "mean_pred_proba": float(sub[y_proba_col].mean()),
            "bias": float(sub[y_proba_col].mean() - sub[y_true_col].mean()),
        })
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


# ------------------------ Fold-aware scaler (for GP) -------------------------


@dataclass
class FoldScaler:
    """z-scaler fit on training fold only. Used by Stage 4 (GP) per the plan.

    Lives here in evaluation.py rather than gp_model.py because it's a
    cross-stage reusable utility.
    """
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FoldScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        # avoid division by zero on constant features
        self.scale_ = np.where(s > 1e-12, s, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FoldScaler must be fit before transform")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ------------------------ Serialisation helpers ------------------------------


def ci_dict_to_json(metrics_with_ci: dict[str, CI]) -> str:
    """Serialise a {metric: CI} dict to a JSON string for persistence."""
    return json.dumps(
        {k: {"point": v.point, "lower": v.lower, "upper": v.upper} for k, v in metrics_with_ci.items()},
        indent=2,
    )


# ------------------------ Module self-test ----------------------------------


def _self_test() -> None:
    """Quick smoke test runnable via `python evaluation.py`.

    Not a full pytest suite, but enough to catch obvious breakage and to
    make 'does the harness still work after I edit it' a one-command check.
    """
    rng = np.random.default_rng(0)
    # Synthetic toy: 60 rows, 5 banks x 12 years, mild signal
    rows = []
    for bank in ["A", "B", "C", "D", "E"]:
        for year in range(2010, 2022):
            rows.append({"Company": bank, "FiscalYear": str(year)})
    df = pd.DataFrame(rows)
    y_true = rng.binomial(1, 0.4, size=len(df))
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, len(df)), 0.01, 0.99)
    df["y_true"], df["y_pred_proba"] = y_true, y_proba

    # 1. Splitters
    loyo_count = sum(1 for _ in loyo_splits(df))
    lobo_count = sum(1 for _ in lobo_splits(df))
    assert loyo_count == 12, f"expected 12 LOYO folds, got {loyo_count}"
    assert lobo_count == 5, f"expected 5 LOBO folds, got {lobo_count}"
    # No train-test overlap, ever
    for tr, te, _ in loyo_splits(df):
        assert len(set(tr) & set(te)) == 0

    # 2. Metrics on identity baseline (perfect probabilities)
    perfect = compute_metrics(y_true, y_true.astype(float))
    assert perfect["accuracy"] == 1.0
    assert perfect["brier"] == 0.0
    # ECE should be ~0 when probabilities equal labels
    assert perfect["ece"] < 1e-9, f"perfect ECE should be ~0, got {perfect['ece']}"

    # 3. Bootstrap CI: point estimate inside its own CI
    cis = all_metrics_with_ci(y_true, y_proba, n_boot=200, seed=1)
    for name, ci in cis.items():
        if math.isnan(ci.lower) or math.isnan(ci.upper):
            continue
        assert ci.lower <= ci.point <= ci.upper, f"{name}: point {ci.point} outside CI [{ci.lower}, {ci.upper}]"

    # 4. FoldScaler round-trip
    X = rng.normal(5, 3, size=(20, 4))
    sc = FoldScaler().fit(X)
    Xs = sc.transform(X)
    assert abs(Xs.mean(axis=0)).max() < 1e-9
    assert abs(Xs.std(axis=0) - 1).max() < 1e-9

    # 5. per_group_table shape sanity
    tab = per_group_table(df, "Company")
    assert len(tab) == 5
    assert set(tab.columns) == {"Company", "n", "n_positive", "true_rate", "mean_pred_proba", "bias"}

    print("evaluation.py self-test: OK")


if __name__ == "__main__":
    _self_test()
