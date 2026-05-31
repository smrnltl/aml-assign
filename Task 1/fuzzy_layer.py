"""
Stage 6 - Gaussian fuzzy semantic layer.

Implementation strictly follows PLAN_Fuzzy_Linguistic.md. The system takes
OOF SHAP attributions from Stage 3 and OOF GP outputs from Stage 4+5,
fuzzifies them via Gaussian membership functions, runs an 8-rule Mamdani
inference, applies the confidence-override post-processing modulation (R4),
and emits a defuzzified Fundamental_Outlook + Risk_Flag per (bank, year).

We implement Mamdani manually rather than via scikit-fuzzy's ControlSystem:
  - 6 signals * 3 states = 18 MFs, 8 rules, 2 outputs is small enough that
    direct computation is clearer than threading through a control system.
  - The R4 override semantics (PLAN_Fuzzy_Linguistic.md section 2.6) need
    post-processing modulation that does not fit cleanly into standard
    max-aggregation. Doing the aggregation by hand keeps R4 transparent.
We still rely on scikit-fuzzy for the Gaussian MF primitive (gaussmf).

Run:
    python fuzzy_layer.py [--seed 42] [--no-tune]
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz

from sklearn.metrics import f1_score, accuracy_score

# ------------------------------- Config -------------------------------------

PROJECT_ROOT = Path(__file__).parent
SHAP_PATH = PROJECT_ROOT / "dataset" / "processed" / "shap_values_oof.csv"
GP_REG_PATH = PROJECT_ROOT / "dataset" / "processed" / "gp_oof_regression.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# 6 input signals (plan section 2.1)
INPUT_SIGNALS = [
    "SHAP_CD_Ratio",
    "SHAP_Base_Rate",
    "SHAP_Interest_Spread",
    "SHAP_CAR",
    "GP_mu",
    "GP_confidence",
]

# Human-readable state labels per signal (plan section 2.2). Order: (negative, neutral, positive).
STATE_LABELS = {
    "SHAP_CD_Ratio":       ("Liquidity-pressure reducing",   "Stable",               "Liquidity-pressure building"),
    "SHAP_Base_Rate":      ("Funding pressure easing",       "Stable",               "Funding pressure rising"),
    "SHAP_Interest_Spread":("Margin compression",            "Stable",               "Margin expansion"),
    "SHAP_CAR":            ("Capital pressure building",     "Stable",               "Capital cushion strengthening"),
    "GP_mu":               ("Improving outlook",             "Flat outlook",         "Deteriorating outlook"),
    "GP_confidence":       ("Low confidence",                "Moderate confidence",  "High confidence"),
}

# Output universes. Both outputs live on [0, 1]; centres at 0.2 / 0.5 / 0.8 per plan section 2.5.
OUTPUT_UNIVERSE = np.linspace(0.0, 1.0, 201)
OUTPUT_CENTRES = (0.2, 0.5, 0.8)
OUTPUT_WIDTH = (0.8 - 0.2) / 2.355  # overlap-0.5 heuristic from plan section 2.3

# R4 post-processing modulation parameters (plan section 2.6 implementation note).
R4_TRIGGER_THRESHOLD = 0.6        # low-confidence membership at/above this triggers override
R4_OUTLOOK_PULL = 0.7             # alpha in section 2.6
R4_RISK_FLOOR = 0.5               # Risk_Flag is clipped to at least this value

# Width-tuning grid per plan section 2.4
WIDTH_MULTIPLIERS = (0.75, 1.0, 1.25)


# ------------------------------- Setup --------------------------------------


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"fuzzy_layer_{ts}.log"
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


# ----------------------------- Data loading ---------------------------------


def load_signals(log: logging.Logger) -> pd.DataFrame:
    """Merge OOF SHAP and OOF GP signals on (Company, FiscalYear).

    Returns a DataFrame with: Company, FiscalYear, y_true, and the 6 input signals,
    plus diagnostic columns (gp_sigma) needed downstream.
    """
    shap_df = pd.read_csv(SHAP_PATH)
    gp_reg = pd.read_csv(GP_REG_PATH)

    # Sanity check #2 - row alignment
    shap_keys = set(zip(shap_df["Company"], shap_df["FiscalYear"]))
    gp_keys = set(zip(gp_reg["Company"], gp_reg["FiscalYear"]))
    if shap_keys != gp_keys:
        only_shap = shap_keys - gp_keys
        only_gp = gp_keys - shap_keys
        raise RuntimeError(
            f"Row mis-alignment between SHAP and GP OOF. only_shap={len(only_shap)} only_gp={len(only_gp)}"
        )
    log.info("Row alignment OK: %d shared (Company, FiscalYear) keys", len(shap_keys))

    merged = shap_df.merge(
        gp_reg[["Company", "FiscalYear", "y_pred_mean", "y_pred_variance", "y_pred_std", "y_true_delta"]],
        on=["Company", "FiscalYear"], how="inner", validate="one_to_one",
    )

    # Compute the 6 signals
    merged = merged.rename(columns={
        "shap_CD_Ratio": "SHAP_CD_Ratio",
        "shap_Base_Rate": "SHAP_Base_Rate",
        "shap_Interest_Spread": "SHAP_Interest_Spread",
        "shap_CAR": "SHAP_CAR",
        "y_pred_mean": "GP_mu",
    })
    # GP_confidence = 1 / variance, clipped to the 1st - 99th percentile of the raw 1/var values
    raw_conf = 1.0 / np.clip(merged["y_pred_variance"].to_numpy(), 1e-9, None)
    lo, hi = np.percentile(raw_conf, [1.0, 99.0])
    merged["GP_confidence"] = np.clip(raw_conf, lo, hi)

    # Keep a record of gp_sigma for the linguistic engine and figures
    merged["GP_sigma"] = merged["y_pred_std"]
    merged["y_true"] = merged["true_label"].astype(int)

    keep = ["Company", "FiscalYear"] + INPUT_SIGNALS + ["GP_sigma", "y_true", "y_true_delta"]
    out = merged[keep].copy()
    log.info("Loaded %d rows for fuzzy inference", len(out))
    return out


# ----------------------------- MF construction ------------------------------


def build_mf_params(df: pd.DataFrame, width_multipliers: dict[str, float]) -> dict[str, dict]:
    """For each of the 6 signals, set 3 Gaussian MFs at the 15th/50th/85th percentiles
    of the empirical distribution. Width per plan section 2.3 (overlap-0.5 heuristic),
    scaled by the per-signal multiplier from width_multipliers.

    Returns a dict {signal: {"centres": (c_low, c_mid, c_high), "width": sigma}}.
    """
    params: dict[str, dict] = {}
    for sig in INPUT_SIGNALS:
        values = df[sig].to_numpy()
        c_lo, c_mid, c_hi = np.percentile(values, [15, 50, 85])
        # Use the full low-to-high span for the width heuristic; if degenerate fall back to std
        span = c_hi - c_lo
        if span < 1e-9:
            span = max(float(values.std()), 1e-6)
        base_width = span / 2.355  # overlap-0.5 between adjacent Gaussians
        width = base_width * width_multipliers.get(sig, 1.0)
        params[sig] = {
            "centres": (float(c_lo), float(c_mid), float(c_hi)),
            "width": float(width),
            "obs_range": (float(values.min()), float(values.max())),
        }
    return params


def fuzzify(value: float, centres: tuple[float, float, float], width: float) -> tuple[float, float, float]:
    """Return the (negative, neutral, positive) membership values for a scalar input."""
    c_lo, c_mid, c_hi = centres
    # gaussmf signature is (x, mean, sigma); pass scalar via array wrapping
    arr = np.array([value], dtype=float)
    m_lo = float(fuzz.gaussmf(arr, c_lo, width)[0])
    m_mid = float(fuzz.gaussmf(arr, c_mid, width)[0])
    m_hi = float(fuzz.gaussmf(arr, c_hi, width)[0])
    return m_lo, m_mid, m_hi


# ----------------------------- Rule base ------------------------------------


def _output_gaussian(centre: float) -> np.ndarray:
    """Return a Gaussian membership function over OUTPUT_UNIVERSE for a given output centre."""
    return fuzz.gaussmf(OUTPUT_UNIVERSE, centre, OUTPUT_WIDTH)


# Pre-compute output MFs once at module load (cheap)
_OUT_WEAK = _output_gaussian(OUTPUT_CENTRES[0])
_OUT_NEUTRAL = _output_gaussian(OUTPUT_CENTRES[1])
_OUT_STRONG = _output_gaussian(OUTPUT_CENTRES[2])
_OUT_NONE = _output_gaussian(OUTPUT_CENTRES[0])     # Risk: None at 0.2
_OUT_WATCH = _output_gaussian(OUTPUT_CENTRES[1])    # Risk: Watch at 0.5
_OUT_ELEVATED = _output_gaussian(OUTPUT_CENTRES[2]) # Risk: Elevated at 0.8


def apply_rules(mems: dict[str, tuple[float, float, float]]) -> tuple[dict, dict, list[str]]:
    """Run the 8-rule Mamdani inference.

    mems: {signal: (neg, neutral, pos)} for each of 6 signals.
    Returns (outlook_aggregated, risk_aggregated, fired_rule_ids):
        outlook_aggregated: dict {"weak", "neutral", "strong"} -> firing strength
        risk_aggregated:    dict {"none", "watch", "elevated"} -> firing strength
        fired_rule_ids:     list of rule IDs (e.g. ["R1", "R3"]) where firing strength > 0.1
    """
    # Unpack mems for readability. _n = negative, _m = neutral, _p = positive.
    cd_n, cd_m, cd_p = mems["SHAP_CD_Ratio"]
    br_n, br_m, br_p = mems["SHAP_Base_Rate"]
    is_n, is_m, is_p = mems["SHAP_Interest_Spread"]
    car_n, car_m, car_p = mems["SHAP_CAR"]
    mu_n, mu_m, mu_p = mems["GP_mu"]
    conf_n, conf_m, conf_p = mems["GP_confidence"]

    # Aggregated firing strengths per output state (max over rules)
    outlook = {"weak": 0.0, "neutral": 0.0, "strong": 0.0}
    risk = {"none": 0.0, "watch": 0.0, "elevated": 0.0}
    fired: list[str] = []

    def fire(rule_id: str, target_dict, state_key, strength):
        if strength > target_dict[state_key]:
            target_dict[state_key] = strength
        if strength > 0.1 and rule_id not in fired:
            fired.append(rule_id)

    # R1: liquidity-pressure-building AND margin-compression -> Risk Elevated
    s = cd_p * is_n
    fire("R1", risk, "elevated", s)

    # R2: capital cushion strengthening AND funding pressure easing -> Outlook Strong
    s = car_p * br_n
    fire("R2", outlook, "strong", s)

    # R3: GP improving outlook AND high confidence -> Outlook Strong
    s = mu_n * conf_p
    fire("R3", outlook, "strong", s)

    # R4 is implemented as post-processing modulation in apply_r4_override() below.
    # We still record its firing strength here for the "fired" list and rule-attribution downstream.
    if conf_n > R4_TRIGGER_THRESHOLD:
        fired.append("R4")

    # R5: liquidity-pressure reducing AND capital cushion strengthening -> Outlook Strong
    s = cd_n * car_p
    fire("R5", outlook, "strong", s)

    # R6: GP deteriorating outlook AND high confidence -> Risk Elevated AND Outlook Weak
    s = mu_p * conf_p
    fire("R6", risk, "elevated", s)
    fire("R6", outlook, "weak", s)

    # R7: margin expansion AND funding pressure easing -> Outlook Strong
    s = is_p * br_n
    fire("R7", outlook, "strong", s)

    # R8: liquidity-pressure building AND capital pressure building -> Risk Elevated AND Outlook Weak
    s = cd_p * car_n
    fire("R8", risk, "elevated", s)
    fire("R8", outlook, "weak", s)

    # Backstop: if no outlook rule fired at all, give 'neutral' a base firing of 0.3 so the
    # centroid is well-defined. Same for risk -> 'none'. This prevents division-by-zero in
    # the defuzzification step.
    if max(outlook.values()) < 0.05:
        outlook["neutral"] = max(outlook["neutral"], 0.3)
    if max(risk.values()) < 0.05:
        risk["none"] = max(risk["none"], 0.3)

    return outlook, risk, fired


def defuzz_centroid(state_strengths: dict[str, float], state_mf_map: dict[str, np.ndarray]) -> float:
    """Standard Mamdani max-aggregation + centroid defuzzification."""
    # Aggregate: per-x point take max of (strength * mf) across states
    aggregated = np.zeros_like(OUTPUT_UNIVERSE)
    for state, strength in state_strengths.items():
        clipped = np.minimum(state_mf_map[state], strength)
        aggregated = np.maximum(aggregated, clipped)
    if aggregated.sum() < 1e-9:
        return 0.5  # truly empty - centre of universe
    return float(fuzz.defuzz(OUTPUT_UNIVERSE, aggregated, "centroid"))


def apply_r4_override(
    outlook_raw: float, risk_raw: float, low_conf_membership: float,
) -> tuple[float, float, bool]:
    """Plan section 2.6 implementation note: R4 as post-processing modulation.

    When GP_confidence Low membership >= R4_TRIGGER_THRESHOLD, pull the
    Fundamental_Outlook toward 0.5 (neutralising any otherwise-positive signal)
    and raise the Risk_Flag floor to R4_RISK_FLOOR.

    Returns (outlook_after, risk_after, fired). "fired" is True iff the override
    actually changed the outputs in a non-trivial way.
    """
    if low_conf_membership < R4_TRIGGER_THRESHOLD:
        return outlook_raw, risk_raw, False
    # Pull toward 0.5 in proportion to low-confidence membership above threshold
    pull = R4_OUTLOOK_PULL * low_conf_membership
    outlook_after = outlook_raw * (1.0 - pull) + 0.5 * pull
    risk_after = max(risk_raw, R4_RISK_FLOOR)
    fired = (abs(outlook_after - outlook_raw) > 1e-6) or (risk_after > risk_raw + 1e-6)
    return float(outlook_after), float(risk_after), bool(fired)


# ----------------------------- Inference loop -------------------------------


def run_inference(df: pd.DataFrame, mf_params: dict[str, dict], log: logging.Logger | None = None) -> pd.DataFrame:
    """Run fuzzy inference over all rows. Returns a DataFrame with raw and post-R4 outputs."""
    rows = []
    state_mf_outlook = {"weak": _OUT_WEAK, "neutral": _OUT_NEUTRAL, "strong": _OUT_STRONG}
    state_mf_risk = {"none": _OUT_NONE, "watch": _OUT_WATCH, "elevated": _OUT_ELEVATED}

    for _, r in df.iterrows():
        mems = {}
        for sig in INPUT_SIGNALS:
            value = float(r[sig])
            mems[sig] = fuzzify(value, mf_params[sig]["centres"], mf_params[sig]["width"])

        outlook_strengths, risk_strengths, fired = apply_rules(mems)
        outlook_raw = defuzz_centroid(outlook_strengths, state_mf_outlook)
        risk_raw = defuzz_centroid(risk_strengths, state_mf_risk)

        low_conf_mem = mems["GP_confidence"][0]  # negative state = Low confidence
        outlook_after, risk_after, r4_fired = apply_r4_override(outlook_raw, risk_raw, low_conf_mem)

        # Dominant fuzzy state per signal (for the linguistic engine downstream)
        dominant = {}
        for sig in INPUT_SIGNALS:
            neg, mid, pos = mems[sig]
            idx = int(np.argmax([neg, mid, pos]))
            label = STATE_LABELS[sig][idx]
            dominant[sig] = {"state": ["negative", "neutral", "positive"][idx],
                             "label": label,
                             "membership": float([neg, mid, pos][idx])}

        # Top positive / negative SHAP signal for the linguistic engine
        shap_signals = [s for s in INPUT_SIGNALS if s.startswith("SHAP_")]
        top_pos = max(shap_signals, key=lambda s: mems[s][2])
        top_neg = max(shap_signals, key=lambda s: mems[s][0])

        rows.append({
            "Company": r["Company"],
            "FiscalYear": r["FiscalYear"],
            "y_true": int(r["y_true"]),
            "y_true_delta": float(r["y_true_delta"]),
            "Fundamental_Outlook_raw": outlook_raw,
            "Fundamental_Outlook": outlook_after,
            "Risk_Flag_raw": risk_raw,
            "Risk_Flag": risk_after,
            "R4_fired": r4_fired,
            "low_conf_membership": low_conf_mem,
            "fired_rules": ",".join(fired),
            "dominant_states": json.dumps(dominant),
            "top_pos_signal": top_pos,
            "top_pos_signal_label": STATE_LABELS[top_pos][2],
            "top_pos_membership": float(mems[top_pos][2]),
            "top_neg_signal": top_neg,
            "top_neg_signal_label": STATE_LABELS[top_neg][0],
            "top_neg_membership": float(mems[top_neg][0]),
            "GP_sigma": float(r["GP_sigma"]),
        })
    return pd.DataFrame(rows)


# ----------------------------- Width tuning ---------------------------------


def evaluate_widths(df: pd.DataFrame, width_multipliers: dict[str, float]) -> dict:
    """Run inference at given width multipliers, return F1 + accuracy + degeneracy flag."""
    mf_params = build_mf_params(df, width_multipliers)
    out = run_inference(df, mf_params)
    y_true = out["y_true"].to_numpy()
    y_pred = (out["Fundamental_Outlook"].to_numpy() < 0.5).astype(int)
    # NOTE on direction: Fundamental_Outlook < 0.5 means "weak fundamentals" -> deteriorate.
    if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        # Degenerate constant predictor - per plan section 7 risk row, treat as zero F1
        return {"f1": 0.0, "accuracy": float(accuracy_score(y_true, y_pred)), "degenerate": True}
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "degenerate": False,
    }


def grid_search_widths(df: pd.DataFrame, log: logging.Logger) -> tuple[dict[str, float], pd.DataFrame]:
    """Search 3^6 = 729 combinations of per-signal width multipliers, return best + log."""
    combos = list(itertools.product(WIDTH_MULTIPLIERS, repeat=len(INPUT_SIGNALS)))
    log.info("Width-tuning grid search: %d combinations", len(combos))
    records = []
    best = None
    best_f1 = -1.0
    t0 = time.time()
    for i, combo in enumerate(combos):
        multipliers = dict(zip(INPUT_SIGNALS, combo))
        res = evaluate_widths(df, multipliers)
        record = {**{f"mult_{s}": m for s, m in multipliers.items()},
                  "f1": res["f1"], "accuracy": res["accuracy"], "degenerate": res["degenerate"]}
        records.append(record)
        if not res["degenerate"] and res["f1"] > best_f1:
            best_f1 = res["f1"]
            best = multipliers
    log.info("Grid search done in %.1fs. Best F1 = %.4f", time.time() - t0, best_f1)
    if best is None:
        log.warning("All combinations were degenerate; falling back to multiplier=1.0 for all signals.")
        best = {sig: 1.0 for sig in INPUT_SIGNALS}
    return best, pd.DataFrame(records)


# ----------------------------- Persistence ----------------------------------


def persist(df_out: pd.DataFrame, mf_params: dict[str, dict],
            tune_log: pd.DataFrame | None, log: logging.Logger) -> None:
    out_path = OUT_DIR / "fuzzy_oof_outputs.csv"
    df_out.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(df_out))

    mfp_path = OUT_DIR / "fuzzy_mf_params.json"
    with open(mfp_path, "w", encoding="utf-8") as f:
        json.dump(mf_params, f, indent=2)
    log.info("Wrote %s", mfp_path)

    if tune_log is not None:
        log_path = OUT_DIR / "fuzzy_tuning_log.csv"
        tune_log.to_csv(log_path, index=False)
        log.info("Wrote %s (%d rows)", log_path, len(tune_log))


# ----------------------------- Sanity checks --------------------------------


def assert_sanity(df_out: pd.DataFrame, n_combos: int | None, log: logging.Logger) -> None:
    """Plan section 6 checks."""
    # #3 no NaN in outputs
    for col in ["Fundamental_Outlook", "Risk_Flag", "Fundamental_Outlook_raw", "Risk_Flag_raw"]:
        assert df_out[col].notna().all(), f"NaN found in {col}"
    # #4 outputs in [0, 1]
    for col in ["Fundamental_Outlook", "Risk_Flag", "Fundamental_Outlook_raw", "Risk_Flag_raw"]:
        v = df_out[col].to_numpy()
        assert (v >= -1e-9).all() and (v <= 1.0 + 1e-9).all(), f"{col} out of [0,1]"
    # #5 R4 fires on at least one row
    n_r4 = int(df_out["R4_fired"].sum())
    assert n_r4 >= 1, "R4 never fired - cannot demonstrate headline confidence-override novelty"
    log.info("Sanity #5 OK: R4 fired on %d / %d rows", n_r4, len(df_out))
    # #6 grid was searched
    if n_combos is not None:
        assert n_combos == 729, f"Expected 729 width-multiplier combinations, got {n_combos}"
        log.info("Sanity #6 OK: %d combinations searched", n_combos)
    # #9 no NaN in any persisted column
    assert not df_out.isna().any().any(), "NaN found somewhere in persisted DataFrame"
    log.info("Stage-6 sanity checks passed.")


# ----------------------------- Entry point ----------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip grid search; use width multiplier 1.0 for all signals.")
    args = parser.parse_args()

    log_path = setup_logging()
    log = logging.getLogger(__name__)
    log.info("Stage 6 - Gaussian fuzzy semantic layer. seed=%d  no_tune=%s", args.seed, args.no_tune)
    log.info("Per PLAN_Fuzzy_Linguistic.md. Modifications must be in the plan first.")

    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_signals(log)

    if args.no_tune:
        best_multipliers = {sig: 1.0 for sig in INPUT_SIGNALS}
        tune_log = None
        n_combos = None
    else:
        best_multipliers, tune_log = grid_search_widths(df, log)
        n_combos = len(tune_log)
    log.info("Selected width multipliers: %s", best_multipliers)

    mf_params = build_mf_params(df, best_multipliers)
    # Stamp the tuned multipliers into mf_params for the audit trail
    for sig, mult in best_multipliers.items():
        mf_params[sig]["width_multiplier"] = float(mult)

    df_out = run_inference(df, mf_params, log)
    persist(df_out, mf_params, tune_log, log)
    assert_sanity(df_out, n_combos, log)

    # Quick headline summary
    y_true = df_out["y_true"].to_numpy()
    y_pred = (df_out["Fundamental_Outlook"].to_numpy() < 0.5).astype(int)
    log.info("=" * 72)
    log.info("Fuzzy classification (Fundamental_Outlook < 0.5 -> deteriorate, post-R4):")
    if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        log.info("  predictor is constant; F1 undefined.")
    else:
        log.info("  accuracy = %.4f", accuracy_score(y_true, y_pred))
        log.info("  F1       = %.4f", f1_score(y_true, y_pred))
    n_r4 = int(df_out["R4_fired"].sum())
    log.info("R4 confidence-override fired on %d / %d rows", n_r4, len(df_out))
    log.info("Run log: %s", log_path)
    log.info("Stage 6 complete.")


if __name__ == "__main__":
    main()
