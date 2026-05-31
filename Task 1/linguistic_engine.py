"""
Stage 7 - Deterministic template linguistic engine.

Implementation strictly follows PLAN_Fuzzy_Linguistic.md section 2.7. The engine
reads Stage 6's fuzzy_oof_outputs.csv and produces a one-paragraph supervisory
sentence per row. The system is fully deterministic - no LLM, no neural
language model - so every word can be traced to an upstream fuzzy/SHAP/GP signal.

Outputs:
  - dataset/processed/linguistic_outputs.csv    one sentence per OOF row (all 80)
  - dataset/processed/linguistic_examples.csv   the 8-10 hand-selected worked examples
                                                for the paper, with per-case annotations
  - logs/linguistic_engine_<timestamp>.log

Run:
    python linguistic_engine.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ------------------------------- Config -------------------------------------

PROJECT_ROOT = Path(__file__).parent
FUZZY_OOF_PATH = PROJECT_ROOT / "dataset" / "processed" / "fuzzy_oof_outputs.csv"
GP_REG_PATH = PROJECT_ROOT / "dataset" / "processed" / "gp_oof_regression.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# Membership threshold for "the GP is meaningfully low-confidence" used in
# the override-text decision (plan section 2.7.5). Must match the trigger
# threshold in fuzzy_layer.py so the textual override aligns with R4 firing.
R4_TEXT_TRIGGER = 0.6

# Threshold for "no clear signal" - if neither side has a SHAP membership
# above this, substitute placeholder text per plan section 2.7.2.
SIGNAL_PRESENCE_THRESHOLD = 0.3


# ------------------------------- Setup --------------------------------------


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"linguistic_engine_{ts}.log"
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


# ----------------------------- Template engine ------------------------------


def outlook_clause(
    outlook: float,
    outlook_raw: float,
    r4_fired: bool,
    low_conf_membership: float,
    top_pos_label: str,
    top_pos_membership: float,
    top_neg_label: str,
    top_neg_membership: float,
) -> str:
    """Build the OUTLOOK_CLAUSE per plan section 2.7.2 and the section 2.7.5 override.

    The 2.7.5 override substitutes a special sentence when R4 fires AND the
    pre-override outlook was >= 0.65 (i.e. the model would have said 'strong'
    without the override). This is the textual expression of the headline novelty.
    """
    # Override case (plan section 2.7.5)
    if r4_fired and low_conf_membership >= R4_TEXT_TRIGGER and outlook_raw >= 0.65:
        return (
            "Fundamentals appear strong on raw indicators, but the model's "
            "confidence is low; this signal should not be acted on without "
            "additional review."
        )
    # Standard mapping (plan section 2.7.2)
    pos_label = top_pos_label if top_pos_membership > SIGNAL_PRESENCE_THRESHOLD \
        else "no clear positive fundamental signal"
    neg_label = top_neg_label if top_neg_membership > SIGNAL_PRESENCE_THRESHOLD \
        else "no material downside attribution"
    if outlook <= 0.35:
        return f"Fundamentals indicate weakening, driven by {neg_label}."
    if outlook >= 0.65:
        return f"Fundamentals are strong, supported by {pos_label}."
    return f"Fundamentals are mixed: {pos_label} is offset by {neg_label}."


def risk_clause(risk: float) -> str:
    """Build the RISK_CLAUSE per plan section 2.7.3."""
    if risk <= 0.35:
        return "No material risk flag."
    if risk >= 0.65:
        return ("Elevated risk: NPL deterioration is plausible within the "
                "next reporting cycle.")
    return "Risk monitoring is warranted on liquidity and capital positions."


def confidence_clause(dominant_states: dict) -> str:
    """Build the CONFIDENCE_CLAUSE per plan section 2.7.4.

    dominant_states is the JSON-decoded dict from fuzzy_oof_outputs.csv that
    records {signal: {state, label, membership}} per signal. We look at
    GP_confidence and use its dominant fuzzy state.
    """
    gp_conf = dominant_states.get("GP_confidence", {})
    state = gp_conf.get("state", "neutral")
    membership = float(gp_conf.get("membership", 0.0))
    # "High-confidence (membership >= 0.5)" -> definite high
    if state == "positive" and membership >= 0.5:
        return "Model confidence in this assessment is high."
    # "Low-confidence (membership >= 0.5)" -> definite low
    if state == "negative" and membership >= 0.5:
        return ("Model confidence is low - this assessment should be treated "
                "as a preliminary signal pending supervisory review.")
    return ("Model confidence is moderate; supplementary review is recommended.")


def build_sentence(row: pd.Series) -> str:
    """Compose the three clauses into one paragraph for a single OOF row."""
    dominant = json.loads(row["dominant_states"])
    o = outlook_clause(
        outlook=float(row["Fundamental_Outlook"]),
        outlook_raw=float(row["Fundamental_Outlook_raw"]),
        r4_fired=bool(row["R4_fired"]),
        low_conf_membership=float(row["low_conf_membership"]),
        top_pos_label=row["top_pos_signal_label"],
        top_pos_membership=float(row["top_pos_membership"]),
        top_neg_label=row["top_neg_signal_label"],
        top_neg_membership=float(row["top_neg_membership"]),
    )
    r = risk_clause(float(row["Risk_Flag"]))
    c = confidence_clause(dominant)
    return f"{o} {r} {c}"


def run_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["generated_sentence"] = df.apply(build_sentence, axis=1)
    return df


# --------------------- Worked-example selection -----------------------------


def select_worked_examples(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Hand-pick 8-10 cases per plan section 2.8. Selection is deterministic
    so the table is reproducible; each case carries an explicit reason and a
    one-line annotation suitable for the paper.

    Selection criteria (in order applied):
      1. One clear strong-fundamentals case (high Fundamental_Outlook, high GP confidence, y_true=0)
      2. One clear weak-fundamentals case  (low  Fundamental_Outlook, high GP confidence, y_true=1)
      3. AT LEAST ONE confidence-override case: R4 fired AND raw outlook >= 0.65
      4. One correct elevated-risk case (Risk_Flag >= 0.65, y_true=1)
      5. One false-positive risk case (Risk_Flag >= 0.6, y_true=0)
      6. One false-negative case (predicted not-deteriorate, y_true=1)
      7. One 2078/2079 case (high-stress COVID year)
      8. 1-2 distinct top-signal combinations not yet covered
    """
    chosen = []
    chosen_keys = set()

    def add(row, reason, annotation):
        key = (row["Company"], row["FiscalYear"])
        if key in chosen_keys:
            return False
        chosen.append({**row.to_dict(), "selection_reason": reason, "annotation": annotation})
        chosen_keys.add(key)
        return True

    # 1. clear strong-fundamentals
    cand = df[(df.y_true == 0) & (~df.R4_fired)].sort_values(
        ["Fundamental_Outlook"], ascending=False)
    if len(cand) > 0:
        add(cand.iloc[0], "clear_strong_fundamentals",
            "Model correctly assesses stable bank-year as strong with high confidence.")

    # 2. clear weak-fundamentals
    cand = df[(df.y_true == 1) & (~df.R4_fired)].sort_values(
        ["Fundamental_Outlook"], ascending=True)
    if len(cand) > 0:
        add(cand.iloc[0], "clear_weak_fundamentals",
            "Model correctly flags deterioration with confident weak-fundamentals signal.")

    # 3. confidence-override (proposal headline novelty) - the most representative case
    cand = df[(df.R4_fired) & (df.Fundamental_Outlook_raw >= 0.65)].sort_values(
        "Fundamental_Outlook_raw", ascending=False)
    if len(cand) > 0:
        add(cand.iloc[0], "confidence_override_textual",
            "Raw fundamentals were strong, but high GP variance triggered R4. "
            "The linguistic engine substitutes the override sentence "
            "(see plan section 2.7.5) instead of asserting strength.")
        # second override case if available, with a different bank
        for _, r in cand.iloc[1:].iterrows():
            if r["Company"] != chosen[-1]["Company"]:
                add(r, "confidence_override_textual_alt",
                    "Second override demonstration on a different bank, "
                    "showing the override semantics generalise.")
                break

    # 4. correct elevated-risk
    cand = df[(df.Risk_Flag >= 0.65) & (df.y_true == 1)].sort_values(
        "Risk_Flag", ascending=False)
    if len(cand) > 0:
        add(cand.iloc[0], "correct_elevated_risk",
            "Model raises elevated-risk flag and the bank does deteriorate.")

    # 5. false-positive risk
    cand = df[(df.Risk_Flag >= 0.60) & (df.y_true == 0)].sort_values(
        "Risk_Flag", ascending=False)
    if len(cand) > 0:
        add(cand.iloc[0], "false_positive_risk",
            "Honest failure mode: model flagged risk but the bank remained stable.")

    # 6. false-negative
    pred_label = (df.Fundamental_Outlook < 0.5).astype(int)
    fn_mask = (pred_label == 0) & (df.y_true == 1)
    cand = df[fn_mask].sort_values("Fundamental_Outlook", ascending=False)
    if len(cand) > 0:
        add(cand.iloc[0], "false_negative",
            "Honest failure mode: model did not flag a true deterioration.")

    # 7. 2078/2079 COVID-stress year - the highest pre-R4 risk case from that year
    cand = df[df.FiscalYear == "2078/2079"].sort_values("Risk_Flag_raw", ascending=False)
    if len(cand) > 0:
        add(cand.iloc[0], "covid_stress_2078_79",
            "From the high-deterioration 2078/2079 fiscal year (COVID-era stress).")

    # 8. up to 2 additional distinct top-positive signal cases not yet covered
    covered_top_pos = {r["top_pos_signal"] for r in chosen}
    added_in_step_8 = 0
    for sig in ["SHAP_CAR", "SHAP_Interest_Spread", "SHAP_Base_Rate", "SHAP_CD_Ratio"]:
        if added_in_step_8 >= 2:
            break
        if sig in covered_top_pos:
            continue
        cand = df[df.top_pos_signal == sig].sort_values("top_pos_membership", ascending=False)
        for _, r in cand.iterrows():
            if add(r, f"distinct_top_signal_{sig}",
                   f"Distinct top-positive signal example for {sig}, not covered above."):
                added_in_step_8 += 1
                covered_top_pos.add(sig)
                break

    log.info("Selected %d worked examples", len(chosen))
    if len(chosen) < 6:
        log.warning("Fewer than 6 worked examples selected - paper table may be sparse.")
    return pd.DataFrame(chosen)


# ----------------------------- Persistence ----------------------------------


def persist(df_full: pd.DataFrame, df_examples: pd.DataFrame, log: logging.Logger) -> None:
    out_path = OUT_DIR / "linguistic_outputs.csv"
    df_full[["Company", "FiscalYear", "y_true",
             "Fundamental_Outlook", "Risk_Flag", "R4_fired",
             "generated_sentence"]].to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(df_full))

    ex_path = OUT_DIR / "linguistic_examples.csv"
    cols = ["Company", "FiscalYear", "y_true", "y_true_delta",
            "top_pos_signal", "top_pos_signal_label", "top_pos_membership",
            "top_neg_signal", "top_neg_signal_label", "top_neg_membership",
            "Fundamental_Outlook_raw", "Fundamental_Outlook",
            "Risk_Flag_raw", "Risk_Flag",
            "R4_fired", "low_conf_membership", "GP_sigma",
            "generated_sentence",
            "selection_reason", "annotation"]
    cols = [c for c in cols if c in df_examples.columns]
    df_examples[cols].to_csv(ex_path, index=False)
    log.info("Wrote %s (%d rows)", ex_path, len(df_examples))


# ----------------------------- Sanity checks --------------------------------


def assert_sanity(df_full: pd.DataFrame, df_examples: pd.DataFrame, log: logging.Logger) -> None:
    # Sanity #7 (plan): at least one worked example demonstrates the override
    override_phrase = "should not be acted on without additional review"
    has_override = df_examples["generated_sentence"].str.contains(override_phrase).any()
    assert has_override, (
        "No worked example contains the override sentence - "
        "headline novelty integrity check failed.")
    log.info("Sanity #7 OK: override sentence present in worked examples")
    # Sanity #3: no empty sentences
    assert df_full["generated_sentence"].str.len().min() > 20, "Empty/short sentences found"
    log.info("Stage-7 sanity checks passed.")


# ----------------------------- Entry point ----------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_path = setup_logging()
    log = logging.getLogger(__name__)
    log.info("Stage 7 - linguistic template engine. seed=%d", args.seed)
    log.info("Per PLAN_Fuzzy_Linguistic.md.")

    df_fuzzy = pd.read_csv(FUZZY_OOF_PATH)
    log.info("Loaded %d rows from %s", len(df_fuzzy), FUZZY_OOF_PATH)

    df_full = run_engine(df_fuzzy)
    df_examples = select_worked_examples(df_full, log)

    persist(df_full, df_examples, log)
    assert_sanity(df_full, df_examples, log)

    # Headline log: show one override case
    overrides = df_full[df_full.R4_fired & (df_full.Fundamental_Outlook_raw >= 0.65)]
    if len(overrides) > 0:
        ex = overrides.iloc[0]
        log.info("=" * 72)
        log.info("Headline override example: %s %s",
                 ex["Company"], ex["FiscalYear"])
        log.info("  raw outlook = %.3f -> post-R4 outlook = %.3f",
                 ex["Fundamental_Outlook_raw"], ex["Fundamental_Outlook"])
        log.info("  low-conf membership = %.3f, GP sigma = %.3f",
                 ex["low_conf_membership"], ex["GP_sigma"])
        log.info("  generated sentence: %s", ex["generated_sentence"])
    log.info("Run log: %s", log_path)
    log.info("Stage 7 complete.")


if __name__ == "__main__":
    main()
