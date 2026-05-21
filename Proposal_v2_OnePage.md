# RESEARCH PROPOSAL

**Title:** Gaussian Process-Driven Fuzzy Linguistic Assessment of Forward Non-Performing Loan Risk in Nepali Commercial Banks

**Module:** STW7085CEM — Advanced Machine Learning (15 credits)
**Author:** Smaran Luitel
**Date:** 2026-05-21

---

## 1. Problem

Nepal's commercial banking sector operates under Nepal Rastra Bank (NRB) prudential thresholds — Capital Adequacy Ratio (CAR) ≥ 11%, Credit-to-Deposit (CD) ratio ≤ 90%, and supervisory targets on Non-Performing Loans (NPL). Existing supervisory and analyst models for forecasting bank health typically return point estimates without uncertainty, and numerical outputs without linguistic interpretation. In a small, autocorrelated, frontier-market panel — 10 commercial banks observed quarterly over a decade — an overconfident prediction is more dangerous than no prediction, and a bare NPL forecast number is less useful to a supervisor than a calibrated, explained one.

This research asks: **can a Gaussian Process model, combined with a Gaussian fuzzy semantic layer, produce calibrated and linguistically interpretable forecasts of forward NPL risk for Nepali commercial banks?**

## 2. Data

A normalized panel of 10 NEPSE-listed commercial banks (ADBL, EBL, GBIME, HBL, NABIL, NICA, NMB, PCBL, SANIMA, SCB), quarterly from FY 2072/73 to 2082/83 BS (≈ 2015/16 – 2025/26 AD). Source: bank disclosures aggregated into `financial_data_all_normalized.csv`. After long-to-wide pivoting (joining Balance Sheet, Income Statement, Distributable Profit, and NRB Ratios rows on `(Company, FiscalYear, Quarter)`) and aligning a 4-quarter-ahead label, approximately **360 usable observations** remain.

## 3. Inputs (8 features, satisfying GP ≥ 4-input requirement)

| # | Feature | Source | Role |
|---|---|---|---|
| 1 | Capital Adequacy Ratio | `Capital_Fund_to_RWA` | Stability |
| 2 | NPL Ratio (current) | `NPL_to_Total_Loan` | Asset quality |
| 3 | Credit-to-Deposit Ratio | `Credit_to_Deposit_Ratio` | Liquidity |
| 4 | Cost of Funds | `Cost_of_Funds` | Profitability driver |
| 5 | Base Rate | `Base_Rate` | Pricing |
| 6 | Interest Rate Spread | `Interest_Rate_Spread` | Margin |
| 7 | ROE (derived) | `Net_Profit / Shareholders_Equity` | Profitability |
| 8 | Loan Growth (derived) | YoY change in `Loans_and_Advances_to_Customers` | Growth |

## 4. Target (dual mode, satisfies GP regression + classification rubric)

- **Regression target:** `ΔNPL = NPL(t+4) − NPL(t)` — continuous forward 4-quarter change in NPL ratio.
- **Classification target:** binary deterioration class — `1` if `ΔNPL > 0.5` percentage points, else `0`. Threshold motivated by NRB supervisory practice.

## 5. Methods

1. **Gaussian Process (primary, LO1).** GP regression for `ΔNPL`; GP classification via thresholding the posterior mean. Matérn 5/2 kernel; hyperparameters by maximum marginal likelihood. Implemented in `GPyTorch` (`ExactGP` suitable for N < 500).
2. **Baseline + explainability comparator.** XGBoost classifier on the same task, with SHAP `TreeExplainer` producing per-feature attributions for each prediction.
3. **Gaussian fuzzy semantic layer (LO4).** SHAP values, GP posterior mean, and GP posterior variance are fuzzified using Gaussian membership functions into supervisory-vocabulary linguistic states (e.g. *Asset-quality risk building / Stable / Reducing*; *Confidence Low / Moderate / High*). A Mamdani rule base aggregates these into composite signals.
4. **Confidence-override rule (headline novelty).** A dedicated rule encodes GP uncertainty into the final linguistic output: when GP posterior variance is high, the system produces a cautious-language explanation regardless of the strength of fundamental signals.
5. **Template linguistic engine.** Defuzzified outputs and dominant fuzzy state labels drive a deterministic sentence template producing a human-readable supervisory commentary.

## 6. Evaluation

| Component | Metric | Method |
|---|---|---|
| GP regression | RMSE, MAE, R² on ΔNPL | Rolling-origin time-series CV by year |
| GP classification | Accuracy, F1, AUC-ROC | Same split; no look-ahead |
| XGBoost baseline | Same classification metrics | Direct comparison vs GP |
| Calibration | Expected Calibration Error, Brier score | Reliability diagram |
| Fuzzy linguistic output | Expert walkthrough on 10 sampled cases | Qualitative annotation in paper |

## 7. Academic Contribution

The contribution is **methodological**, not empirical claims about frontier markets generally. Specifically: (i) the use of GP posterior variance as a *live input* to a downstream fuzzy rule engine that modulates the linguistic confidence of supervisory explanations — a form of uncertainty-aware natural-language assessment; (ii) the use of SHAP attributions, rather than raw ratios, as fuzzification inputs, decoupling linguistic interpretation from hard supervisory thresholds; (iii) demonstration on a small, autocorrelated, frontier-market banking panel where calibration matters more than headline accuracy.

## 8. Plan (Task 1, ~14 weeks)

| Phase | Weeks | Deliverable |
|---|---|---|
| Data pivot, derived-ratio computation, forward-label alignment | 1–2 | Wide-format analytical CSV (~360 rows × 8 features + 2 labels) |
| XGBoost baseline + SHAP | 3–4 | Baseline metrics; SHAP value matrix |
| GP regression + classification with calibration analysis | 5–7 | GP model; reliability diagram; comparison table |
| Gaussian fuzzy layer design + rule base + confidence-override | 8–9 | Membership functions; rule base; sample outputs |
| Linguistic engine + expert walkthrough | 10–11 | Annotated example outputs |
| Paper write-up | 12–14 | Final 6-page research paper |

## 9. Scope Limitations (declared up front)

- **Banking sector only.** Generalisation to hydropower, insurance, or manufacturing is not claimed.
- **No price-based features or returns target.** PE, PB, and forward stock-return prediction are explicitly out of scope.
- **Sample size ~360 observations.** ExactGP is appropriate; no claim of state-of-the-art predictive performance is made — the contribution is in calibration and linguistic interpretability, not raw accuracy.
- **No ANFIS / FastAPI / multi-sector benchmarking.** Cut from earlier draft to keep scope realistic for a 15-credit module.