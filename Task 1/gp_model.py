"""
Stage 4 + 5 — Gaussian Process regression and classification.

Implementation strictly follows PLAN_GP.md. The GP is fitted as a regression
model over standardised Delta_NPL_next_year using a Matern 5/2 kernel with
ARD (one length scale per feature). Classification probability is derived
from the posterior:

    P(Delta_NPL > 0.5 | x*) = 1 - Phi((0.5 - mu(x*)) / sigma(x*))

Per the plan, the default likelihood is Student-t (nu=4) to handle post-COVID
NPL outliers. The script supports falling back to a Gaussian likelihood
(ExactGP closed-form posterior) if Student-t variational inference fails to
converge on a fold; failures are logged and the fold uses the Gaussian fit.

Outputs:
  - dataset/processed/gp_oof_regression.csv       OOF mu, sigma, variance
  - dataset/processed/gp_oof_classification.csv   OOF P(deteriorate) - same schema as XGBoost
  - dataset/processed/gp_loyo_metrics.csv         per-fold ARD length scales and MLL
  - dataset/processed/gp_pooled_metrics.json      pooled metrics with bootstrap CIs
  - dataset/processed/gp_ard_lengthscales.csv     ARD length scales matrix (folds x features)
  - logs/gp_model_<timestamp>.log

Run:
    python gp_model.py [--seed 42] [--n-boot 2000] [--likelihood {studentt,gaussian}]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

import evaluation as ev

# ------------------------------- Config -------------------------------------

PROJECT_ROOT = Path(__file__).parent
RATIOS_PATH = PROJECT_ROOT / "dataset" / "processed" / "financial_ratios.csv"
XGB_OOF_PATH = PROJECT_ROOT / "dataset" / "processed" / "xgb_oof_predictions.csv"
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
REG_TARGET = "Delta_NPL_next_year"
CLF_TARGET = "Deteriorate_next_year"
DETERIORATION_THRESHOLD = 0.5  # ΔNPL > 0.5pp = deterioration (matches Stage 3 label rule)

# Optimisation
N_ADAM_STEPS = 200
ADAM_LR = 0.1
N_RANDOM_RESTARTS = 3  # per plan §7 — try 3 inits, pick lowest final loss


# ------------------------------- Setup --------------------------------------


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"gp_model_{ts}.log"
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


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RATIOS_PATH)
    required = ["Company", "FiscalYear", CLF_TARGET, REG_TARGET] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {RATIOS_PATH}: {missing}")
    mask = df[FEATURES + [CLF_TARGET, REG_TARGET]].notna().all(axis=1)
    sub = df.loc[mask].reset_index(drop=True)
    sub[CLF_TARGET] = sub[CLF_TARGET].astype(int)
    sub[REG_TARGET] = sub[REG_TARGET].astype(float)
    return sub


def assert_row_alignment_with_xgb(df: pd.DataFrame, log: logging.Logger) -> None:
    """Plan §6 sanity check #2: GP must operate on the same rows as XGBoost."""
    if not XGB_OOF_PATH.exists():
        log.warning("Cannot verify row alignment: %s does not exist yet", XGB_OOF_PATH)
        return
    xgb = pd.read_csv(XGB_OOF_PATH)
    gp_keys = set(zip(df["Company"].tolist(), df["FiscalYear"].tolist()))
    xgb_keys = set(zip(xgb["Company"].tolist(), xgb["FiscalYear"].tolist()))
    only_gp = gp_keys - xgb_keys
    only_xgb = xgb_keys - gp_keys
    if only_gp or only_xgb:
        raise RuntimeError(
            f"Row mis-alignment with XGBoost OOF! Only-in-GP: {len(only_gp)}, only-in-XGB: {len(only_xgb)}"
        )
    log.info("Row alignment with Stage 3 OOF: OK (%d shared (bank, year) keys)", len(gp_keys))


# ------------------------------- GP models ----------------------------------


class ExactGaussianGP(gpytorch.models.ExactGP):
    """Closed-form GP regression with Gaussian likelihood, ARD Matern 5/2."""

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.Likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ApproxStudentTGP(gpytorch.models.ApproximateGP):
    """Variational GP with Student-t likelihood, ARD Matern 5/2.

    Inducing points are set to all training points (so at N=80 the approximation
    is effectively the full GP). The Student-t likelihood needs variational
    inference since the closed-form Gaussian posterior no longer applies.
    """

    def __init__(self, inducing_points: torch.Tensor):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=False
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing_points.shape[-1])
        )

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ------------------------------- Per-fold fits ------------------------------


def fit_gaussian_gp(
    X_train: np.ndarray, y_train: np.ndarray, seed: int, log: logging.Logger,
) -> tuple[ExactGaussianGP, gpytorch.likelihoods.GaussianLikelihood, float]:
    """Fit ExactGP with Gaussian likelihood. Returns (model, likelihood, final_loss).

    Tries N_RANDOM_RESTARTS random inits, returns the best by final marginal NLL.
    """
    Xt = torch.as_tensor(X_train, dtype=torch.float64)
    yt = torch.as_tensor(y_train, dtype=torch.float64)

    best = None
    best_loss = float("inf")
    initial_losses = []

    for restart in range(N_RANDOM_RESTARTS):
        torch.manual_seed(seed + restart)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        model = ExactGaussianGP(Xt, yt, likelihood).double()
        model.train()
        likelihood.train()

        opt = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=ADAM_LR
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        with gpytorch.settings.cholesky_jitter(float_value=1e-4, double_value=1e-6):
            # initial loss
            opt.zero_grad()
            out = model(Xt)
            loss0 = -mll(out, yt)
            initial_losses.append(float(loss0.item()))

            for step in range(N_ADAM_STEPS):
                opt.zero_grad()
                out = model(Xt)
                loss = -mll(out, yt)
                loss.backward()
                opt.step()

        final_loss = float(loss.item())
        if final_loss < best_loss:
            best_loss = final_loss
            best = (model, likelihood)

    # Plan §6 sanity check #4: loss must improve
    if best_loss >= min(initial_losses):
        log.warning("Gaussian-GP fit did not improve any restart (initial losses %s -> best %.4f)",
                    [f"{l:.4f}" for l in initial_losses], best_loss)

    return best[0], best[1], best_loss


def predict_gaussian_gp(
    model: ExactGaussianGP, likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Xt = torch.as_tensor(X_test, dtype=torch.float64)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(Xt))
        mu = pred.mean.numpy()
        var = pred.variance.numpy()
    return mu, var


def fit_studentt_gp(
    X_train: np.ndarray, y_train: np.ndarray, seed: int, log: logging.Logger,
) -> tuple[ApproxStudentTGP, gpytorch.likelihoods.Likelihood, float] | None:
    """Fit ApproximateGP with Student-t likelihood (nu=4). Variational ELBO.

    Returns None if every restart fails to converge (caller should fall back to Gaussian).
    """
    Xt = torch.as_tensor(X_train, dtype=torch.float64)
    yt = torch.as_tensor(y_train, dtype=torch.float64)
    n = Xt.shape[0]

    best = None
    best_loss = float("inf")
    initial_losses = []

    for restart in range(N_RANDOM_RESTARTS):
        try:
            torch.manual_seed(seed + restart)
            model = ApproxStudentTGP(Xt).double()
            # StudentTLikelihood: nu is learnable; we initialise low for heavy tails
            likelihood = gpytorch.likelihoods.StudentTLikelihood().double()
            # initialise nu around 4 (heavy tails) -- access deg_free parameter
            try:
                likelihood.deg_free = torch.tensor(4.0, dtype=torch.float64)
            except Exception:
                pass  # if API differs, leave default and continue
            model.train()
            likelihood.train()

            opt = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()), lr=ADAM_LR
            )
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

            with gpytorch.settings.cholesky_jitter(float_value=1e-3, double_value=1e-5):
                opt.zero_grad()
                out = model(Xt)
                loss0 = -mll(out, yt)
                initial_losses.append(float(loss0.item()))

                for step in range(N_ADAM_STEPS):
                    opt.zero_grad()
                    out = model(Xt)
                    loss = -mll(out, yt)
                    loss.backward()
                    opt.step()

            final_loss = float(loss.item())
            if np.isfinite(final_loss) and final_loss < best_loss:
                best_loss = final_loss
                best = (model, likelihood)
        except Exception as e:
            log.warning("Student-t restart %d failed: %s", restart, e)
            continue

    if best is None:
        return None
    if initial_losses and best_loss >= min(initial_losses):
        log.warning("Student-t GP fit did not improve any restart (initial losses %s -> best %.4f)",
                    [f"{l:.4f}" for l in initial_losses], best_loss)
    return best[0], best[1], best_loss


def predict_studentt_gp(
    model: ApproxStudentTGP, likelihood: gpytorch.likelihoods.Likelihood, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Xt = torch.as_tensor(X_test, dtype=torch.float64)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # latent f distribution; for predictive uncertainty we want the marginal
        # of the (latent) function — Student-t adds heavy-tailed noise on top.
        # For the classification probability formula we want the function posterior,
        # so we use model(Xt).mean / .variance (not likelihood(...).variance which
        # would inflate by the Student-t scale).
        f_dist = model(Xt)
        mu = f_dist.mean.numpy()
        var = f_dist.variance.numpy()
    return mu, var


# --------------------- ARD length scale extraction --------------------------


def extract_lengthscales(model) -> np.ndarray:
    """Return the ARD length-scale vector from a fitted GP model."""
    # ScaleKernel(MaternKernel(...)) — the lengthscale lives in .base_kernel
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    return ls.astype(float)


def extract_outputscale(model) -> float:
    return float(model.covar_module.outputscale.detach().numpy().ravel()[0])


# ------------------------------- Main loop ----------------------------------


def run_loyo(
    df: pd.DataFrame, seed: int, likelihood_choice: str, log: logging.Logger,
) -> dict:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[REG_TARGET].to_numpy(dtype=float)

    # OOF predictions in original ΔNPL units
    oof_mu = np.full(len(df), np.nan)
    oof_var = np.full(len(df), np.nan)
    oof_proba = np.full(len(df), np.nan)
    fold_records = []
    lengthscale_rows = []

    for train_idx, test_idx, held_year in ev.loyo_splits(df):
        t0 = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Plan §6 sanity check #3 — fold-aware scaling, train-only fit
        x_scaler = ev.FoldScaler().fit(X_train)
        X_train_s = x_scaler.transform(X_train)
        X_test_s = x_scaler.transform(X_test)
        # Sanity: training scaled features should have mean ~0, std ~1
        assert abs(X_train_s.mean(axis=0)).max() < 1e-9
        assert abs(X_train_s.std(axis=0) - 1).max() < 1e-9

        # Target standardisation per fold (plan §2.3)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std())
        if y_std < 1e-9:
            log.warning("Fold year=%s has near-zero target std (%.2e) — skipping fold", held_year, y_std)
            continue
        y_train_s = (y_train - y_mean) / y_std

        # Fit
        used_likelihood = likelihood_choice
        if likelihood_choice == "studentt":
            res = fit_studentt_gp(X_train_s, y_train_s, seed=seed, log=log)
            if res is None:
                log.warning("Fold year=%s: Student-t failed entirely, falling back to Gaussian", held_year)
                model, lik, final_loss = fit_gaussian_gp(X_train_s, y_train_s, seed=seed, log=log)
                used_likelihood = "gaussian_fallback"
                mu_s, var_s = predict_gaussian_gp(model, lik, X_test_s)
            else:
                model, lik, final_loss = res
                mu_s, var_s = predict_studentt_gp(model, lik, X_test_s)
        else:
            model, lik, final_loss = fit_gaussian_gp(X_train_s, y_train_s, seed=seed, log=log)
            mu_s, var_s = predict_gaussian_gp(model, lik, X_test_s)

        # De-standardise to original ΔNPL units
        mu_orig = mu_s * y_std + y_mean
        var_orig = var_s * (y_std ** 2)
        sigma_orig = np.sqrt(np.clip(var_orig, 1e-12, None))

        # Sanity: σ must be finite and positive
        assert np.all(np.isfinite(sigma_orig)) and np.all(sigma_orig > 0), \
            f"Fold {held_year}: invalid sigma values"

        # Classification probability per plan §2.3 / proposal §6.4
        # P(ΔNPL > 0.5 | x*) = 1 - Φ((0.5 - μ) / σ)
        proba = 1.0 - norm.cdf((DETERIORATION_THRESHOLD - mu_orig) / sigma_orig)

        # Sanity: probabilities in [0, 1]
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

        oof_mu[test_idx] = mu_orig
        oof_var[test_idx] = var_orig
        oof_proba[test_idx] = proba

        ls = extract_lengthscales(model)
        outputscale = extract_outputscale(model)
        record = {
            "held_out_year": held_year,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "y_train_mean": round(y_mean, 4),
            "y_train_std": round(y_std, 4),
            "used_likelihood": used_likelihood,
            "final_loss": round(float(final_loss), 4),
            "outputscale": round(outputscale, 4),
            "fit_seconds": round(time.time() - t0, 2),
        }
        for f, l in zip(FEATURES, ls):
            record[f"ls_{f}"] = round(float(l), 4)
        fold_records.append(record)
        lengthscale_rows.append({"held_out_year": held_year, **{f: float(l) for f, l in zip(FEATURES, ls)}})

        log.info(
            "Fold year=%s  likelihood=%s  final_loss=%.4f  outputscale=%.3f  time=%.1fs",
            held_year, used_likelihood, float(final_loss), outputscale, time.time() - t0,
        )

    return {
        "df": df,
        "oof_mu": oof_mu,
        "oof_var": oof_var,
        "oof_proba": oof_proba,
        "fold_records": fold_records,
        "lengthscale_rows": lengthscale_rows,
    }


# ------------------------------- Persistence --------------------------------


def persist_results(res: dict, seed: int, n_boot: int, log: logging.Logger, log_path: Path) -> None:
    df = res["df"]
    have = ~np.isnan(res["oof_proba"])
    if not have.all():
        log.warning("%d rows have no GP prediction. Excluding from pooled metrics.", int((~have).sum()))

    # Regression OOF
    reg = pd.DataFrame({
        "Company": df["Company"].values,
        "FiscalYear": df["FiscalYear"].values,
        "FiscalYearAD": df["FiscalYearAD"].values if "FiscalYearAD" in df.columns else "",
        "y_true_delta": df[REG_TARGET].values,
        "y_pred_mean": res["oof_mu"],
        "y_pred_variance": res["oof_var"],
        "y_pred_std": np.sqrt(np.clip(res["oof_var"], 1e-12, None)),
    })
    reg_path = OUT_DIR / "gp_oof_regression.csv"
    reg.to_csv(reg_path, index=False)
    log.info("Wrote %s  (%d rows)", reg_path, len(reg))

    # Classification OOF (same schema as Stage 3's xgb_oof_predictions.csv)
    clf = pd.DataFrame({
        "Company": df["Company"].values,
        "FiscalYear": df["FiscalYear"].values,
        "FiscalYearAD": df["FiscalYearAD"].values if "FiscalYearAD" in df.columns else "",
        "y_true": df[CLF_TARGET].values,
        "y_pred_proba": res["oof_proba"],
    })
    clf_path = OUT_DIR / "gp_oof_classification.csv"
    clf.to_csv(clf_path, index=False)
    log.info("Wrote %s  (%d rows)", clf_path, len(clf))

    # Per-fold metrics
    fold_df = pd.DataFrame(res["fold_records"])
    fold_path = OUT_DIR / "gp_loyo_metrics.csv"
    fold_df.to_csv(fold_path, index=False)
    log.info("Wrote %s  (%d folds)", fold_path, len(fold_df))

    # ARD length scales matrix
    ls_df = pd.DataFrame(res["lengthscale_rows"])
    if not ls_df.empty:
        ls_df.loc["mean"] = ls_df.mean(numeric_only=True)
        ls_df.at["mean", "held_out_year"] = "MEAN"
    ls_path = OUT_DIR / "gp_ard_lengthscales.csv"
    ls_df.to_csv(ls_path, index=False)
    log.info("Wrote %s", ls_path)

    # Pooled classification metrics + CIs
    y_true = df[CLF_TARGET].values[have]
    y_proba = res["oof_proba"][have]
    clf_metrics = ev.compute_metrics(y_true, y_proba)
    clf_cis = ev.all_metrics_with_ci(y_true, y_proba, n_boot=n_boot, seed=seed)

    # Pooled regression metrics + CIs (RMSE, MAE, R²)
    y_true_d = df[REG_TARGET].values[have]
    y_pred_d = res["oof_mu"][have]

    def rmse(yt, yp): return float(np.sqrt(np.mean((yt - yp) ** 2)))
    def mae(yt, yp): return float(np.mean(np.abs(yt - yp)))
    def r2(yt, yp):
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    # Regression bootstrap CIs (non-stratified, no class label here)
    def boot_reg(metric_fn, n_boot=n_boot, seed=seed):
        rng = np.random.default_rng(seed)
        point = metric_fn(y_true_d, y_pred_d)
        samples = np.empty(n_boot)
        n = len(y_true_d)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            samples[b] = metric_fn(y_true_d[idx], y_pred_d[idx])
        return {
            "point": float(point),
            "lower": float(np.quantile(samples, 0.025)),
            "upper": float(np.quantile(samples, 0.975)),
        }

    reg_metrics = {
        "rmse": boot_reg(rmse),
        "mae": boot_reg(mae),
        "r2": boot_reg(r2),
    }

    pooled = {
        "n_rows_pooled": int(have.sum()),
        "class_balance": {
            "n_positive": int(y_true.sum()),
            "n_negative": int((1 - y_true).sum()),
            "positive_rate": float(y_true.mean()),
        },
        "classification": {
            "point": clf_metrics,
            "ci_95": {k: {"point": v.point, "lower": v.lower, "upper": v.upper} for k, v in clf_cis.items()},
        },
        "regression": reg_metrics,
    }
    pooled_path = OUT_DIR / "gp_pooled_metrics.json"
    with open(pooled_path, "w", encoding="utf-8") as f:
        json.dump(pooled, f, indent=2)
    log.info("Wrote %s", pooled_path)

    # Headline log summary
    log.info("=" * 72)
    log.info("POOLED OOF METRICS  (GP classification, N=%d)", int(have.sum()))
    for m, ci in clf_cis.items():
        log.info("  %-10s  point=%.4f  95%% CI [%.4f, %.4f]", m, ci.point, ci.lower, ci.upper)
    log.info("POOLED OOF METRICS  (GP regression on ΔNPL)")
    for m, v in reg_metrics.items():
        log.info("  %-6s  point=%.4f  95%% CI [%.4f, %.4f]", m, v["point"], v["lower"], v["upper"])
    log.info("Run log: %s", log_path)


# ------------------------------- Entry point --------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--likelihood", choices=["studentt", "gaussian"], default="studentt",
                        help="Default Student-t per plan §2.4; use 'gaussian' for fallback comparison")
    args = parser.parse_args()

    log_path = setup_logging()
    log = logging.getLogger(__name__)
    log.info("Stage 4+5 — GP regression and classification. seed=%d  n_boot=%d  likelihood=%s",
             args.seed, args.n_boot, args.likelihood)
    log.info("Per PLAN_GP.md. Modifications must be in the plan first.")

    set_all_seeds(args.seed)
    df = load_data()
    log.info("Loaded %d labeled rows. Class balance: %d pos / %d neg",
             len(df), int(df[CLF_TARGET].sum()), int((1 - df[CLF_TARGET]).sum()))
    log.info("ΔNPL target: mean=%.3f std=%.3f min=%.3f max=%.3f",
             df[REG_TARGET].mean(), df[REG_TARGET].std(), df[REG_TARGET].min(), df[REG_TARGET].max())

    assert_row_alignment_with_xgb(df, log)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = run_loyo(df, seed=args.seed, likelihood_choice=args.likelihood, log=log)

    # OOF coverage check (plan §6 #8)
    have = ~np.isnan(res["oof_proba"])
    assert have.all(), f"OOF coverage incomplete: {(~have).sum()} rows missing"

    persist_results(res, seed=args.seed, n_boot=args.n_boot, log=log, log_path=log_path)
    log.info("Stage 4+5 complete.")


if __name__ == "__main__":
    main()
