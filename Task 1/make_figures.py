"""
Regenerate all paper figures from persisted CSVs.

This script is the *only* path that touches matplotlib. It reads from
dataset/processed/*.csv and never re-runs any model. The contract from
PLAN_OVERALL_Project.md §8: paper figures must be regenerable from the
persisted CSVs alone.

Outputs:
  figures/xgb_reliability.png      Stage 3 reliability diagram (raw + isotonic)
  figures/shap_beeswarm.png        Global SHAP feature attribution (beeswarm)
  figures/shap_bar.png             Global mean(|SHAP|) per feature
  figures/xgb_per_year_bias.png    Per-year predicted vs true rate

Run:
    python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; safe for Windows + no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import evaluation as ev

ROOT = Path(__file__).parent
OOF_PATH = ROOT / "dataset" / "processed" / "xgb_oof_predictions.csv"
SHAP_PATH = ROOT / "dataset" / "processed" / "shap_values_oof.csv"
RATIOS_PATH = ROOT / "dataset" / "processed" / "financial_ratios.csv"
PER_YEAR_PATH = ROOT / "dataset" / "processed" / "xgb_error_per_year.csv"
GP_REG_PATH = ROOT / "dataset" / "processed" / "gp_oof_regression.csv"
GP_CLF_PATH = ROOT / "dataset" / "processed" / "gp_oof_classification.csv"
GP_ARD_PATH = ROOT / "dataset" / "processed" / "gp_ard_lengthscales.csv"
FUZZY_OOF_PATH = ROOT / "dataset" / "processed" / "fuzzy_oof_outputs.csv"
FUZZY_MF_PATH = ROOT / "dataset" / "processed" / "fuzzy_mf_params.json"
FIG_DIR = ROOT / "figures"

FEATURES = [
    "CAR", "NPL_Ratio", "CD_Ratio", "Cost_of_Funds",
    "Base_Rate", "Interest_Spread", "ROE_derived",
]


# ----------------------------- Reliability diagram --------------------------


def plot_reliability(oof: pd.DataFrame, out_path: Path, extra: list | None = None) -> None:
    """Reliability diagram. `extra` is a list of (proba_series, label, marker) tuples
    to overlay on top of the XGBoost curves."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    series = [
        (oof["y_pred_proba"], oof["y_true"], "XGBoost (raw)", "o"),
        (oof["y_pred_proba_isotonic"], oof["y_true"], "XGBoost (isotonic)", "s"),
    ]
    if extra is not None:
        for s in extra:
            series.append(s)

    for proba_s, truth_s, label, marker in series:
        sub_mask = proba_s.notna() & truth_s.notna()
        proba = proba_s[sub_mask].to_numpy()
        truth = truth_s[sub_mask].to_numpy()
        if len(proba) == 0:
            continue
        mean_p, frac_pos, ns = ev.reliability_curve(truth, proba, n_bins=5)
        ax.plot(mean_p, frac_pos, marker=marker, lw=1.5, label=label)
        for x, y, n in zip(mean_p, frac_pos, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(6, -2), fontsize=8, color="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Fraction of true deteriorations (per bin)")
    title = "Reliability diagram — forward NPL deterioration"
    if extra is not None:
        title += "  (XGBoost vs GP)"
    title += "\n(5 quantile-spaced bins; N=80 pooled OOF)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- SHAP bar plot (mean |SHAP|) ------------------


def plot_shap_bar(shap_df: pd.DataFrame, out_path: Path) -> None:
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
    mean_abs = shap_df[shap_cols].abs().mean().sort_values(ascending=True)
    feat_labels = [c.replace("shap_", "") for c in mean_abs.index]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(feat_labels, mean_abs.values, color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value|  (log-odds units)")
    ax.set_title("Global feature importance — OOF SHAP attributions\n"
                 "(XGBoost, N=80 out-of-fold predictions)")
    for i, v in enumerate(mean_abs.values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- SHAP beeswarm --------------------------------


def plot_shap_beeswarm(shap_df: pd.DataFrame, ratios_df: pd.DataFrame, out_path: Path) -> None:
    # Match SHAP rows to feature rows by (Company, FiscalYear)
    merged = shap_df.merge(ratios_df, on=["Company", "FiscalYear"], how="inner")
    shap_cols = [f"shap_{f}" for f in FEATURES]
    shap_matrix = merged[shap_cols].to_numpy()
    feat_matrix = merged[FEATURES].to_numpy()

    expl = shap.Explanation(
        values=shap_matrix,
        data=feat_matrix,
        feature_names=FEATURES,
    )

    fig = plt.figure(figsize=(7, 4.5))
    shap.plots.beeswarm(expl, show=False, max_display=len(FEATURES))
    fig = plt.gcf()
    fig.suptitle("SHAP beeswarm — XGBoost OOF attributions for forward NPL deterioration",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Per-year bias --------------------------------


def plot_per_year_bias(per_year: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(per_year))
    width = 0.4
    ax.bar(x - width / 2, per_year["true_rate"], width, label="True deterioration rate",
           color="#dd8452")
    ax.bar(x + width / 2, per_year["mean_pred_proba"], width, label="Mean predicted prob.",
           color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(per_year["FiscalYear"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate / probability")
    ax.set_title("Per-year diagnostic — XGBoost OOF predictions vs ground truth")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- GP figures -----------------------------------


def plot_gp_ard(ard_df: pd.DataFrame, shap_df: pd.DataFrame, out_path: Path) -> None:
    """ARD length scales (shorter = more relevant) side-by-side with SHAP importance."""
    feat_cols = [c for c in ard_df.columns if c != "held_out_year"]
    # row labelled "MEAN" or fallback to numeric mean
    if "MEAN" in ard_df["held_out_year"].astype(str).values:
        mean_row = ard_df[ard_df["held_out_year"].astype(str) == "MEAN"][feat_cols].iloc[0]
    else:
        mean_row = ard_df[feat_cols].mean()

    # SHAP mean |·| (already used for shap_bar.png)
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
    shap_mean = shap_df[shap_cols].abs().mean()
    shap_mean.index = [c.replace("shap_", "") for c in shap_mean.index]

    common = [f for f in feat_cols if f in shap_mean.index]
    if not common:
        return
    ard_vals = mean_row[common].to_numpy(dtype=float)
    # Plot inverse length scale so "taller = more relevant" matches the SHAP plot
    relevance = 1.0 / np.maximum(ard_vals, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    order = np.argsort(relevance)
    labels_ord = [common[i] for i in order]
    axes[0].barh(labels_ord, relevance[order], color="#55a868")
    axes[0].set_xlabel("GP relevance  (1 / mean ARD length scale)")
    axes[0].set_title("GP feature relevance (ARD)")
    for i, v in enumerate(relevance[order]):
        axes[0].text(v, i, f"  {v:.2f}", va="center", fontsize=8)

    shap_ord = shap_mean.reindex(labels_ord).to_numpy()
    axes[1].barh(labels_ord, shap_ord, color="#4c72b0")
    axes[1].set_xlabel("XGBoost mean |SHAP|  (log-odds)")
    axes[1].set_title("XGBoost feature importance (SHAP)")
    for i, v in enumerate(shap_ord):
        axes[1].text(v, i, f"  {v:.3f}", va="center", fontsize=8)
    fig.suptitle("Feature relevance — GP (left) vs XGBoost (right), same ordering",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gp_regression_pred_vs_actual(reg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    yt = reg["y_true_delta"].to_numpy()
    yp = reg["y_pred_mean"].to_numpy()
    ys = reg["y_pred_std"].to_numpy()
    ax.errorbar(yt, yp, yerr=ys, fmt="o", alpha=0.5, capsize=2, lw=0.7)
    lim_lo = min(yt.min(), yp.min()) - 0.5
    lim_hi = max(yt.max(), yp.max()) + 0.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("True ΔNPL_next_year (percentage points)")
    ax.set_ylabel("GP predicted mean ± σ")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_title("GP regression on ΔNPL — predicted (with ±σ) vs actual")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gp_calibration_comparison(out_path: Path) -> None:
    """Headline ECE bars: XGBoost raw / XGBoost isotonic / GP."""
    import json
    xgb_p = ROOT / "dataset" / "processed" / "xgb_pooled_metrics.json"
    gp_p = ROOT / "dataset" / "processed" / "gp_pooled_metrics.json"
    if not xgb_p.exists() or not gp_p.exists():
        return
    xgb_m = json.loads(xgb_p.read_text())
    gp_m = json.loads(gp_p.read_text())

    def ci(m, k):
        v = m["ci_95"][k]
        return v["point"], v["lower"], v["upper"]

    rows = [
        ("XGB raw", *ci(xgb_m["raw_xgboost"], "ece"), *ci(xgb_m["raw_xgboost"], "brier")),
        ("XGB isotonic", *ci(xgb_m["isotonic_xgboost"], "ece"), *ci(xgb_m["isotonic_xgboost"], "brier")),
        ("GP (Student-t)", *ci(gp_m["classification"], "ece"), *ci(gp_m["classification"], "brier")),
    ]
    labels = [r[0] for r in rows]
    ece_pts = [r[1] for r in rows]
    ece_err = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
    brier_pts = [r[4] for r in rows]
    brier_err = [[r[4] - r[5] for r in rows], [r[6] - r[4] for r in rows]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    axes[0].bar(labels, ece_pts, yerr=ece_err, capsize=4, color=colors)
    axes[0].set_ylabel("Expected Calibration Error  (5 quantile bins)")
    axes[0].set_title("ECE — lower is better")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, brier_pts, yerr=brier_err, capsize=4, color=colors)
    axes[1].set_ylabel("Brier score")
    axes[1].set_title("Brier — lower is better")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Calibration comparison: pooled OOF (N=80) with 95% bootstrap CIs",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Fuzzy figures --------------------------------


def plot_fuzzy_mfs(mf_params: dict, fuzzy_df: pd.DataFrame, out_path: Path) -> None:
    """One subplot per input signal showing the 3 Gaussian MFs across observed range."""
    signals = [s for s in mf_params.keys() if s in fuzzy_df.columns or s == "GP_confidence" or s == "GP_mu"
               or s.startswith("SHAP_")]
    n = len(signals)
    cols = 2
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(9, 2.4 * rows_n))
    axes = np.array(axes).reshape(-1)
    state_names = ("Negative", "Neutral", "Positive")
    for ax, sig in zip(axes, signals):
        c_lo, c_mid, c_hi = mf_params[sig]["centres"]
        sigma = mf_params[sig]["width"]
        lo, hi = mf_params[sig]["obs_range"]
        # extend a little for plotting
        pad = 0.1 * (hi - lo + 1e-9)
        x = np.linspace(lo - pad, hi + pad, 200)
        for c, name in zip([c_lo, c_mid, c_hi], state_names):
            y = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
            ax.plot(x, y, lw=1.5, label=name)
        ax.set_title(sig, fontsize=10)
        ax.set_xlabel(sig)
        ax.set_ylabel("Membership")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    # hide any unused axes
    for ax in axes[len(signals):]:
        ax.axis("off")
    fig.suptitle("Gaussian membership functions — tuned widths from grid search",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_outlook_vs_truth(fuzzy_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Two scatter groups: R4 fired (orange) vs not (blue)
    not_fired = fuzzy_df[~fuzzy_df.R4_fired]
    fired = fuzzy_df[fuzzy_df.R4_fired]
    rng = np.random.default_rng(0)
    jitter_no = rng.normal(0, 0.04, len(not_fired))
    jitter_yes = rng.normal(0, 0.04, len(fired))
    ax.scatter(not_fired.Fundamental_Outlook, not_fired.y_true + jitter_no,
               s=24, alpha=0.55, color="#4c72b0", label="R4 not fired")
    ax.scatter(fired.Fundamental_Outlook, fired.y_true + jitter_yes,
               s=30, alpha=0.7, color="#dd8452", label="R4 confidence-override fired")
    ax.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Stable (y_true=0)", "Deteriorate (y_true=1)"])
    ax.set_xlabel("Fuzzy Fundamental_Outlook (post-R4)")
    ax.set_xlim(0, 1)
    ax.set_title("Fuzzy outlook vs true deterioration\n(80 OOF rows; jitter on y-axis for visibility)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_override_demo(fuzzy_df: pd.DataFrame, out_path: Path) -> None:
    """Pre- vs post-override Fundamental_Outlook for rows where R4 fired meaningfully."""
    fired = fuzzy_df[fuzzy_df.R4_fired].copy()
    if len(fired) == 0:
        return
    fired = fired.sort_values("Fundamental_Outlook_raw", ascending=False).reset_index(drop=True)
    # Cap the number of rows shown so the plot stays legible
    max_show = 25
    fired_show = fired.head(max_show)
    labels = [f"{r.Company} {r.FiscalYear}" for _, r in fired_show.iterrows()]
    x = np.arange(len(fired_show))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(x - width / 2, fired_show.Fundamental_Outlook_raw, width,
           color="#4c72b0", label="Pre-R4 (raw fuzzy outlook)")
    ax.bar(x + width / 2, fired_show.Fundamental_Outlook, width,
           color="#dd8452", label="Post-R4 (after confidence-override)")
    ax.axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Fundamental_Outlook")
    ax.set_ylim(0, 1)
    n_total = len(fired)
    extra = f" (top {max_show} of {n_total} R4-fired rows shown)" if n_total > max_show else ""
    ax.set_title(f"Confidence-override (R4) effect on Fundamental_Outlook{extra}")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- Driver ---------------------------------------


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    oof = pd.read_csv(OOF_PATH)
    shap_df = pd.read_csv(SHAP_PATH)
    ratios = pd.read_csv(RATIOS_PATH)
    per_year = pd.read_csv(PER_YEAR_PATH)

    # GP outputs may not exist yet on the very first Stage-3 run
    gp_clf = pd.read_csv(GP_CLF_PATH) if GP_CLF_PATH.exists() else None
    gp_reg = pd.read_csv(GP_REG_PATH) if GP_REG_PATH.exists() else None
    gp_ard = pd.read_csv(GP_ARD_PATH) if GP_ARD_PATH.exists() else None

    # XGBoost-only reliability (kept as a standalone figure)
    plot_reliability(oof, FIG_DIR / "xgb_reliability.png")
    print(f"Wrote {FIG_DIR / 'xgb_reliability.png'}")

    plot_shap_bar(shap_df, FIG_DIR / "shap_bar.png")
    print(f"Wrote {FIG_DIR / 'shap_bar.png'}")

    plot_shap_beeswarm(shap_df, ratios, FIG_DIR / "shap_beeswarm.png")
    print(f"Wrote {FIG_DIR / 'shap_beeswarm.png'}")

    plot_per_year_bias(per_year, FIG_DIR / "xgb_per_year_bias.png")
    print(f"Wrote {FIG_DIR / 'xgb_per_year_bias.png'}")

    if gp_clf is not None:
        # Overlay GP onto the reliability diagram
        plot_reliability(
            oof,
            FIG_DIR / "gp_reliability.png",
            extra=[(gp_clf["y_pred_proba"], gp_clf["y_true"], "GP (Student-t)", "D")],
        )
        print(f"Wrote {FIG_DIR / 'gp_reliability.png'}")

    if gp_reg is not None:
        plot_gp_regression_pred_vs_actual(gp_reg, FIG_DIR / "gp_regression_predicted_vs_actual.png")
        print(f"Wrote {FIG_DIR / 'gp_regression_predicted_vs_actual.png'}")

    if gp_ard is not None:
        plot_gp_ard(gp_ard, shap_df, FIG_DIR / "gp_ard_lengthscales.png")
        print(f"Wrote {FIG_DIR / 'gp_ard_lengthscales.png'}")

    if gp_clf is not None:
        plot_gp_calibration_comparison(FIG_DIR / "gp_calibration_comparison.png")
        print(f"Wrote {FIG_DIR / 'gp_calibration_comparison.png'}")

    # Fuzzy figures
    if FUZZY_OOF_PATH.exists() and FUZZY_MF_PATH.exists():
        fuzzy_df = pd.read_csv(FUZZY_OOF_PATH)
        with open(FUZZY_MF_PATH, "r", encoding="utf-8") as f:
            mf_params = json.load(f)
        plot_fuzzy_mfs(mf_params, fuzzy_df, FIG_DIR / "fuzzy_mfs.png")
        print(f"Wrote {FIG_DIR / 'fuzzy_mfs.png'}")
        plot_outlook_vs_truth(fuzzy_df, FIG_DIR / "fuzzy_outlook_vs_truth.png")
        print(f"Wrote {FIG_DIR / 'fuzzy_outlook_vs_truth.png'}")
        plot_override_demo(fuzzy_df, FIG_DIR / "fuzzy_override_demo.png")
        print(f"Wrote {FIG_DIR / 'fuzzy_override_demo.png'}")


if __name__ == "__main__":
    main()
