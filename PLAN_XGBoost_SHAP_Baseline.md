# Implementation Plan — XGBoost + SHAP Baseline with Small-N Evaluation Machinery

**Status:** Design fixed, not yet implemented.
**Owner:** Smaran Luitel
**Date:** 2026-05-21
**Module:** STW7085CEM — Advanced Machine Learning

This document is the single source of truth for the XGBoost baseline stage of the project. Every modelling choice below has been deliberated and locked. When implementation begins, follow this document literally; if any choice needs to change, update this document *before* changing the code.

---

## 1. Purpose of this stage

Train a discriminative baseline classifier on the 80-row banking-fundamentals dataset, extract per-prediction SHAP attributions, and put in place the small-N evaluation machinery that *every* later component (GP regression, GP classification, fuzzy layer) will reuse. This stage delivers three things:

1. A reference classification benchmark that the GP must match or beat on accuracy and explicitly beat on calibration.
2. A matrix of SHAP values that becomes the input to the fuzzy semantic layer (the proposal's headline novelty).
3. A reusable, audited evaluation harness — cross-validation splitter, metrics, bootstrap CI generator, calibration plots — that the GP and fuzzy stages will plug into without modification.

This stage does **not** train the GP, design fuzzy membership functions, or generate linguistic explanations. Those are separate stages.

---

## 2. Frozen design choices

### 2.1 Dataset

- **Source file:** [`dataset/processed/financial_ratios.csv`](dataset/processed/financial_ratios.csv) — 110 rows × 20 columns produced by [`preprocess.py`](preprocess.py).
- **Modelling subset:** rows where all 8 base features and the binary label are present → **80 rows** with class balance 25 deteriorate / 55 stable (31% / 69%).
- **Lagged and bank-relative features (`*_lag1`, `*_dev`)**: *available in the CSV but not used in this stage.* Reserved for a later robustness analysis. The 8-base-feature configuration is the one reported in the paper.

### 2.2 Features (locked, 8)

| # | Feature | Source column | Notes |
|---|---|---|---|
| 1 | `CAR` | `Capital_Fund_to_RWA` | Capital adequacy ratio (%) |
| 2 | `NPL_Ratio` | `Non-Performing_Loan_(NPL)_to_Total_Loan` | Current-period NPL ratio (%) |
| 3 | `CD_Ratio` | `Credit_to_Deposit_Ratio` | Credit-to-deposit ratio (%) |
| 4 | `Cost_of_Funds` | `Cost_of_Funds` | Funding cost (%) |
| 5 | `Base_Rate` | `Base_Rate` | Bank's base rate (%) |
| 6 | `Interest_Spread` | `Interest_Rate_Spread` | Net interest margin proxy (%) |
| 7 | `ROE_derived` | `Net_Profit / SHAREHOLDERS_EQUITY × 100` | Profitability (%) |
| 8 | `Loan_Growth_YoY` | YoY %Δ in `Loans_and_Advances_to_Customers` | Growth (%) |

### 2.3 Targets (locked)

- **Classification target:** `Deteriorate_next_year` ∈ {0, 1}. `1` iff next-year NPL ratio exceeds current by more than 0.5 percentage points.
- **Regression target (also computed, for GP later):** `Delta_NPL_next_year` (continuous Δ in NPL ratio).
- **XGBoost stage uses the binary target only.**

### 2.4 Cross-validation scheme

- **Primary: Leave-One-Year-Out (LOYO).**
  - For each fiscal year *y* in the dataset, train on all rows with `FiscalYear ≠ y`, test on rows with `FiscalYear = y`.
  - Reflects the realistic supervisory question: "Given history up to last year, predict deterioration in the coming year."
  - Produces ~8 folds (years with ≥1 labeled row).
- **Secondary (robustness): Leave-One-Bank-Out (LOBO).**
  - For each bank *b*, train on other 9 banks, test on bank *b*.
  - Produces 10 folds, each with ~8 test rows.
  - Reports cross-sectional generalisation. Reported in the paper as a robustness table; not used to select hyperparameters or claim primary results.
- **No random K-fold anywhere.** Random K-fold would mix the same bank's adjacent years across train and test and inflate every metric. This is a hard rule.

### 2.5 Class imbalance handling

- **No SMOTE, no oversampling, no synthetic data.** Decided in prior discussion — the 35:65 imbalance is mild and synthesizing on a small panel of real-named regulated banks creates more academic-integrity and calibration risk than it solves.
- **XGBoost:** use `scale_pos_weight = n_neg / n_pos`, recomputed *per fold* on the training rows only. With ~55 neg / 25 pos overall, this is ≈ 2.2; per-fold it will vary slightly. Computing per-fold prevents test-set information leaking into the loss weighting.
- **Decision threshold for binary metrics:** 0.5 by default; also report results with the threshold tuned per fold on the *training* fold via maximising F1 (no peeking at test). Both numbers go in the paper.

### 2.6 Model — XGBoost configuration

- **Library:** `xgboost.XGBClassifier`. Pinned version in `requirements.txt` (see §6).
- **Objective:** `binary:logistic`. Output: probability of class 1.
- **Hyperparameters:** small ranges only — N=80 cannot support a large search.
  - `n_estimators`: {50, 100, 200}
  - `max_depth`: {2, 3, 4}  *(deliberately shallow — deeper trees overfit at this N)*
  - `learning_rate`: {0.05, 0.1}
  - `min_child_weight`: {1, 3}
  - `reg_lambda`: {1.0, 5.0}
  - `subsample`: 0.8 (fixed)
  - `colsample_bytree`: 0.8 (fixed)
  - `scale_pos_weight`: computed per fold (see §2.5)
- **Hyperparameter selection:** **nested cross-validation** inside each LOYO outer fold. Inner CV = 3-fold stratified, restricted to the outer-train rows only. Inner CV uses `GridSearchCV` over the grid above, optimising mean F1. The selected hyperparameters are then refit on the full outer-train set and evaluated on the held-out outer year.
  - This is computationally cheap (3 × 3 × 2 × 2 × 2 = 72 grid points × 3 inner folds × 8 outer folds = ~1700 fits, each on ~50 rows). Will complete in minutes.
  - Nested CV is the *only* honest way to report metrics when hyperparameters are tuned on the same data. With N=80, skipping nested CV would silently overstate every metric.
- **Random seed:** fixed at 42 across all calls. Re-running must produce bitwise-identical metrics.

### 2.7 Calibration

- **Raw XGBoost probabilities are typically miscalibrated** — that's the whole reason the proposal frames GP as the calibration-superior alternative. Report **uncalibrated** XGBoost probabilities so the GP has a real bar to clear.
- **Also compute** isotonic-regression-calibrated XGBoost probabilities (`CalibratedClassifierCV(method='isotonic', cv='prefit')`) using a held-out slice of the training fold (last 20% of train fold by year), and report both.
  - This shows the marker that calibration was considered, not ignored.
  - Story in the paper: "XGBoost raw is miscalibrated (ECE = X); XGBoost+isotonic improves to (ECE = Y); GP without any post-hoc calibration achieves (ECE = Z ≤ Y)." If GP doesn't clear isotonic-calibrated XGBoost, that's a legitimate finding to report honestly, not a failure to hide.

### 2.8 SHAP

- **Library:** `shap.TreeExplainer` (exact Shapley values for tree models — no approximation needed).
- **Computation point:** SHAP values computed on the **test rows of each outer LOYO fold**, using the model trained on that fold's training rows. This produces an out-of-fold SHAP matrix of shape `(80, 8)` that is the honest per-prediction explanation (no test rows seen by the model that explains them).
- **Storage:** persist as `dataset/processed/shap_values_oof.csv` with columns `Company, FiscalYear, <8 feature SHAPs>, base_value, predicted_proba, true_label`. This file becomes the input to the fuzzy semantic layer in the next stage.
- **Sanity check (run once, document in the paper):** sum of SHAP values + base value should equal the log-odds of `predicted_proba`. Assert `|sum + base - logit(p)| < 1e-6` per row.
- **Global summary plots in the paper:** SHAP beeswarm and mean(|SHAP|) bar chart, both computed from the OOF SHAP matrix.

---

## 3. Small-N evaluation machinery (reusable)

This is the part most students get wrong with small datasets. Designed once here, reused by GP and fuzzy stages without modification.

### 3.1 Metrics — every result reports

For each fold (and aggregated across folds):

| Metric | Why |
|---|---|
| Accuracy | Sanity. Useful only with class balance reported alongside. |
| F1 (class 1) | Primary classification metric — captures both precision and recall on the minority class. |
| Precision (class 1) | False-alarm cost matters for supervisory triage. |
| Recall (class 1) | Missed-deterioration cost matters more. |
| ROC-AUC | Threshold-independent ranking quality. |
| PR-AUC | More informative than ROC-AUC under class imbalance. |
| Brier score | Probabilistic accuracy — pairs with ECE for calibration story. |
| Expected Calibration Error (ECE) | Headline calibration metric. Use 5 bins on the full N=80 pooled OOF predictions, *not* per-fold (per-fold has too few rows). State bin count in the paper. |

### 3.2 Bootstrap confidence intervals

- **Compute on the pooled OOF prediction matrix**, not per-fold (per-fold CIs are too noisy at this N).
- **Procedure:** B = 2000 bootstrap resamples of the 80 OOF (predicted_proba, true_label) pairs with replacement; for each resample compute every metric in §3.1; report the 2.5th and 97.5th percentile as the 95% CI.
- **Why this works:** the bootstrap CI captures sampling uncertainty in the *metric*, which is the right uncertainty to report when N is small. It does *not* claim to add training data.
- **Implementation:** one reusable function `bootstrap_ci(y_true, y_proba, metric_fn, B=2000, seed=42)` that returns `(point_estimate, lower, upper)`.
- **Important nuance — stratified bootstrap:** the resampling must preserve roughly the original class proportions, otherwise some resamples will contain zero positives and metrics like F1 are undefined. Use stratified bootstrap by class label.

### 3.3 Calibration reporting

- **Reliability diagram** computed on pooled OOF predictions, with 5 quantile-spaced bins (not equal-width — equal-width leaves bins empty at this N).
- **Plot annotations:** show bin sample sizes on the diagram so the reader sees the limited support behind each point.
- **ECE formula:** `Σ (|bin| / N) × |mean_predicted_proba_in_bin − fraction_positive_in_bin|`. Document the formula in the paper.
- **Brier score** reported alongside as a single scalar.

### 3.4 Per-bank and per-year error tables

In addition to pooled metrics, generate two diagnostic tables:

- **Per-bank table:** for each of the 10 banks, the bank's average predicted probability vs its true deterioration rate across its ~8 years. Surfaces banks the model systematically misjudges.
- **Per-year table:** for each year, the year's average predicted probability vs its true deterioration rate. Surfaces years (e.g. COVID era) the model fails on.

These two tables are required outputs even if they don't go in the main paper — they go in the appendix and they catch silent failure modes.

### 3.5 What is *not* in scope at this stage

- No SMOTE or other data synthesis.
- No deep learning baselines.
- No multi-class extension.
- No time-series-aware features beyond what's already in the CSV.
- No external data (no stock prices, no macroeconomic indicators).
- No deployment, FastAPI, or serving code.

---

## 4. Deliverables from this stage

| Artefact | Path | Purpose |
|---|---|---|
| Baseline training script | `xgb_baseline.py` | Reproducible end-to-end run |
| Evaluation harness | `evaluation.py` | Reusable CV splitters, metrics, bootstrap CIs |
| OOF predictions | `dataset/processed/xgb_oof_predictions.csv` | 80 rows: Company, FiscalYear, y_true, y_pred_proba, y_pred_proba_isotonic |
| OOF SHAP matrix | `dataset/processed/shap_values_oof.csv` | 80 rows × 8 SHAPs + base + proba + label — input to fuzzy stage |
| Per-fold metrics | `dataset/processed/xgb_loyo_metrics.csv` | One row per outer LOYO fold |
| Pooled metrics with CIs | `dataset/processed/xgb_pooled_metrics.json` | Headline numbers for the paper |
| Reliability diagram | `figures/xgb_reliability.png` | Calibration plot, with bin sizes annotated |
| SHAP global plots | `figures/shap_beeswarm.png`, `figures/shap_bar.png` | Global feature attribution figures |
| Per-bank, per-year tables | `dataset/processed/xgb_error_per_bank.csv`, `xgb_error_per_year.csv` | Diagnostic tables for appendix |
| Run log | `logs/xgb_baseline_<timestamp>.log` | Hyperparameters chosen per fold, fit time, sanity assertions |

---

## 5. Execution order (when implementation begins)

1. **Stub the evaluation harness first** (`evaluation.py`): LOYO splitter, LOBO splitter, all 8 metrics, stratified bootstrap CI. Unit-test against a tiny toy dataset before any modelling.
2. **Write `xgb_baseline.py`:** nested CV loop, per-fold isotonic calibration, OOF prediction collection.
3. **Add SHAP computation** to the per-fold loop. Persist OOF SHAP matrix.
4. **Run end-to-end once** with seed 42. Verify SHAP additivity assertion passes.
5. **Generate all paper figures** from the persisted CSVs (figures must be regenerable from the CSVs alone, without rerunning the model).
6. **Write a 1-page results summary** to `RESULTS_XGBoost.md` for later reference — what the headline numbers were, what surprised, what the diagnostic tables showed.

This sequencing is deliberate: the harness in step 1 is what the GP stage will reuse, so it must be solid before any modelling code is written on top of it.

---

## 6. Dependencies and environment

Pandas is not currently installed in the project Python (verified earlier). Either:

- **Option A (recommended):** create a `requirements.txt` with pinned versions and install into a virtualenv before implementation begins:
  ```
  numpy
  pandas
  scikit-learn
  xgboost
  shap
  matplotlib
  ```
  Versions to be pinned at install time. Document the pinned versions in this file under §6.1 after install.
- **Option B:** stay stdlib-only as `preprocess.py` does. Not practical for this stage — SHAP, XGBoost, and bootstrap CIs all genuinely need NumPy.

**Decision:** Option A. Install before coding starts. Pin versions in this document at that time.

### 6.1 Pinned versions

*(To be filled in once virtualenv is created. Do not start implementation until this section is populated.)*

---

## 7. Sanity checks the implementation must pass

These are non-negotiable acceptance criteria. If any fail, the stage is incomplete.

1. **Reproducibility:** running `python xgb_baseline.py` twice produces bitwise-identical OOF prediction CSVs.
2. **No leakage:** for every outer fold, assert that `set(train_years) ∩ {test_year} == ∅` and that `scale_pos_weight` was computed only from `train_years` rows.
3. **SHAP additivity:** for each test row, `|base_value + sum(shap_values) − logit(predicted_proba)| < 1e-6`.
4. **OOF coverage:** the OOF prediction CSV has exactly 80 rows, one per labeled (bank, year) pair, with no duplicates.
5. **Class-weight correctness:** the `scale_pos_weight` logged per fold matches `(train_negative_count / train_positive_count)` recomputed from the persisted CSVs.
6. **Hyperparameter selection variance:** log the chosen hyperparameters per outer fold; if the *same* hyperparameters are selected on >75% of folds, the search space is too narrow — flag for review. If a wildly different set is chosen on each fold, the search is overfitting the inner CV — also flag.
7. **Bootstrap CI containment:** the point estimate of every metric must fall inside its 95% CI. This is a trivial check but catches off-by-one errors in the bootstrap code.
8. **No SMOTE, no random K-fold, no peeking:** assert these in code comments and CI naming. Reviewer-checkable.

---

## 8. Risks and pre-decided mitigations

| Risk | Likelihood | Pre-decided response |
|---|---|---|
| XGBoost overfits at N=80, OOF metrics collapse | Medium | Already mitigated via shallow trees (max_depth ≤ 4), small `n_estimators`, regularisation. If F1 < 0.4 OOF, fall back to plain logistic regression as the baseline and document this in the paper as a finding. |
| Some LOYO folds have zero positive test rows | Possible (early years) | Skip those folds for F1/AUC reporting; include in accuracy reporting. Document which folds were skipped and why. |
| ECE on 5 bins is dominated by 1–2 bins with most of the data | Likely at N=80 | Acknowledge in paper; show bin-size-annotated reliability diagram. This is exactly *why* we report Brier score alongside. |
| SHAP additivity assertion fails | Unlikely with TreeExplainer | Indicates a bug in feature ordering between model and explainer. Halt run, debug before persisting outputs. |
| Hyperparameter grid is wrong size — too coarse or too fine | Medium | After first run, inspect per-fold chosen hyperparameters (sanity check #6). Adjust grid once and rerun. Do not iterate beyond two grid revisions — that becomes hyperparameter overfitting on the small CV. |
| Bootstrap CIs are very wide (e.g. F1 CI = [0.2, 0.8]) | Likely at N=80 | This is the *honest* answer at this sample size. Report it. Wide CIs are the academic finding — "this is a small-N regime where confidence in any point estimate must be modest." |

---

## 9. How this plan plugs into the proposal and downstream stages

- **Proposal alignment:** this stage delivers the XGBoost + SHAP comparator promised in [Proposal_v2_OnePage.md](Proposal_v2_OnePage.md) §5(2). The OOF SHAP matrix is the input to §5(3) (the Gaussian fuzzy semantic layer).
- **What the GP stage will inherit:** the entire `evaluation.py` harness, the LOYO/LOBO splitters, the bootstrap CI machinery, the reliability-diagram plotter, the per-bank/per-year diagnostic tables. The GP stage should produce the same set of OOF CSVs in the same format so the two models are comparable line-by-line.
- **What the fuzzy stage will inherit:** the OOF SHAP matrix `shap_values_oof.csv`. Fuzzy membership functions will be defined over these signed SHAP signals, *not* over raw feature values. The GP posterior mean and variance from the GP stage will join this matrix later as additional signals to fuzzify.

---

## 10. Changelog

- **2026-05-21** — Initial plan written. 8-base-feature configuration, LOYO primary / LOBO robustness, nested CV for hyperparameters, no synthesis, class-weighted XGBoost, isotonic calibration as a second reporting line, bootstrap CIs on pooled OOF predictions.
