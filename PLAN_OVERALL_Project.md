# Overall Project Plan — GP + Fuzzy Linguistic Assessment of Forward NPL Risk

**Status:** Active. Emergency-pace execution (overdue).
**Owner:** Smaran Luitel
**Date:** 2026-05-21
**Module:** STW7085CEM — Advanced Machine Learning, Task 1 only
**Sister documents:**
- [Proposal_v2_OnePage.md](Proposal_v2_OnePage.md) — the rewritten 1-page proposal this plan implements
- [PLAN_XGBoost_SHAP_Baseline.md](PLAN_XGBoost_SHAP_Baseline.md) — detailed plan for Stage 3 (the most-designed stage)

This document is the master schedule and stage map. It defers detail to per-stage plans where they exist; for stages that have not been planned in detail yet, it captures the design intent so nothing gets lost.

---

## 0. TL;DR

Deliver a 6-page research paper showing that a **Gaussian Process** model gives **better-calibrated** forward NPL deterioration forecasts than an **XGBoost+SHAP** baseline on 80 (bank-year) observations from 10 NEPSE commercial banks, with the GP's posterior variance feeding a **Gaussian fuzzy semantic layer** that produces **human-readable supervisory commentary**. Submit it ASAP — every stage is scoped to the minimum that defends the headline claim.

The contribution is **methodological**, not empirical. The novelty is the **GP-confidence-override rule** that propagates posterior variance into the linguistic explanation.

---

## 1. The whole project, in one table

| Stage | What it produces | Key inputs | Status |
|---|---|---|---|
| **0. Document & align** | This plan + [Proposal_v2_OnePage.md](Proposal_v2_OnePage.md) | Assignment brief, raw CSV | ✅ Done |
| **1. Raw data audit** | Long→wide pivot decision, schema reconciliation | `financial_data_all_normalized.csv` | ✅ Done |
| **2. Preprocessing** | 80 labeled (bank, year) rows × 8 base features + labels | Raw CSV | ✅ Done — [preprocess.py](preprocess.py), [financial_ratios.csv](dataset/processed/financial_ratios.csv) |
| **3. XGBoost + SHAP baseline** | OOF predictions, OOF SHAP matrix, evaluation harness | Stage 2 output | 📋 Planned — [PLAN_XGBoost_SHAP_Baseline.md](PLAN_XGBoost_SHAP_Baseline.md) |
| **4. GP regression** | Posterior mean + variance over ΔNPL, calibration curves | Stage 2 features | ⏳ Not started |
| **5. GP classification** | Posterior-mean-threshold classifier, ECE/Brier vs XGBoost | Stage 4 model | ⏳ Not started |
| **6. Fuzzy semantic layer** | Membership functions over SHAP + GP signals; rule base; confidence-override | Stage 3 SHAP matrix + Stage 5 GP outputs | ⏳ Not started |
| **7. Linguistic engine** | Template-based supervisory commentary on test cases | Stage 6 defuzzified outputs | ⏳ Not started |
| **8. Paper write-up** | 6-page research paper, code appendix, all figures | All preceding stages | ⏳ Interleaved with stages — see §4 |
| **9. Submission packaging** | Final PDF, code zip / GitHub link, file rename per assignment spec | Stage 8 | ⏳ Not started |

Everything from Stage 4 onward depends on Stage 3's evaluation harness — that's why Stage 3 has its own detailed plan and is the next thing to build.

---

## 2. Architecture — how the stages fit together

```
        Raw CSV (1445×138)
              │
              ▼
   ┌──────────────────────┐
   │ Stage 2: Preprocess  │  preprocess.py
   └──────────────────────┘
              │
              ▼
   financial_ratios.csv  (80 rows × 8 features + binary label + Δ regression label)
              │
       ┌──────┴──────┐
       ▼             ▼
 ┌──────────┐  ┌──────────┐
 │ Stage 3: │  │ Stage 4: │
 │ XGBoost  │  │ GP regr. │  (Matern 5/2, ARD, ExactGP)
 │ + SHAP   │  └──────────┘
 └──────────┘       │
       │            ▼
       │      Stage 5: GP classification via posterior-mean threshold
       │            │
       ▼            ▼
   shap_oof.csv   gp_oof.csv    ← both share schema: Company, FY, y_true,
                                  y_pred, [per-feature attributions / GP variance]
       │            │
       └────┬───────┘
            ▼
   ┌──────────────────────┐
   │ Stage 6: Fuzzy layer │  Gaussian MFs over SHAP signals + GP posterior μ, σ²
   │   - Membership fns   │  Mamdani rule base
   │   - Rule engine      │  Confidence-override rule (HEADLINE NOVELTY)
   └──────────────────────┘
            │
            ▼
   ┌──────────────────────┐
   │ Stage 7: Linguistic  │  Template-based sentence generation
   │   - Decision table   │  Deterministic, auditable
   │   - Sample outputs   │
   └──────────────────────┘
            │
            ▼
   Paper §6 examples — annotated commentary on selected test cases
```

**Why this architecture, in one sentence:** the GP gives uncertainty, SHAP gives feature-level attribution, the fuzzy layer fuses them into linguistic states, and the confidence-override rule is what makes the integration academically defensible rather than three loosely-stapled techniques.

---

## 3. Stages 4–7 — design intent (until each gets its own plan)

These stages don't have detailed planning documents yet. The text below is the design intent locked in by the proposal and prior conversation — enough to start implementation when each stage's turn comes, with the understanding that each stage will get its own plan document at kickoff (mirroring how Stage 3 was treated).

### Stage 4 — GP regression

- **Target:** `Delta_NPL_next_year` (continuous).
- **Library:** `GPyTorch`, `ExactGP` (N=80 supports exact inference comfortably).
- **Kernel:** Matérn 5/2 with **ARD** (one length scale per feature). Hyperparameters by maximum marginal likelihood with `Adam`, ~200 steps.
- **Inputs:** the same 8 features as Stage 3, standardised (zero mean, unit variance) using *training-fold* statistics inside each LOYO outer fold.
- **Outputs per test point:** posterior mean μ(x*), posterior variance σ²(x*).
- **Reuses from Stage 3:** the LOYO splitter, the standardisation must be fold-aware (Stage 3 doesn't standardise — XGBoost doesn't need it — so a new fold-aware scaler helper goes into `evaluation.py`).
- **Metrics reported:** RMSE, MAE, R² on pooled OOF predictions, plus bootstrap CIs.
- **Persistence:** `dataset/processed/gp_oof_regression.csv` with columns `Company, FiscalYear, y_true_delta, y_pred_mean, y_pred_variance`.

### Stage 5 — GP classification via threshold

- **Approach:** train the GP as in Stage 4, then classify by `μ(x*) > 0.5` (deterioration > 0.5 percentage points). This *satisfies the assignment's "thresholded GP classification" requirement* without training a second model.
- **Probability of deterioration:** computed from the GP posterior — `P(ΔNPL > 0.5 | x*) = 1 − Φ((0.5 − μ) / σ)`, where Φ is the standard normal CDF. This is a *proper* probabilistic forecast — the calibration story rides on this.
- **Metrics:** the same 8 metrics as Stage 3 (accuracy, F1, precision, recall, ROC-AUC, PR-AUC, Brier, ECE) on the same 80 pooled OOF predictions, using the same evaluation harness. Direct comparability with XGBoost.
- **Headline expectation in the paper:** GP should match XGBoost on ranking metrics (AUC) and *beat* XGBoost on calibration metrics (Brier, ECE). If it doesn't, that's still a legitimate finding — report it honestly.
- **Persistence:** `dataset/processed/gp_oof_classification.csv` mirroring `xgb_oof_predictions.csv` schema.

### Stage 6 — Gaussian fuzzy semantic layer

- **Library:** `scikit-fuzzy` (Mamdani inference, Gaussian MFs, defuzzification).
- **Signals fuzzified (locked):**
  - 4 SHAP-based: `SHAP(NPL_Ratio)`, `SHAP(CAR)`, `SHAP(CD_Ratio)`, `SHAP(ROE_derived)` — these are the four highest-impact features that the marker would expect supervisory analysts to talk about. Other SHAP values computed but not fuzzified at this stage to keep the rule base tractable.
  - 2 GP-based: GP posterior mean μ, GP confidence proxy `1 / σ²` (clipped).
- **Fuzzy states per signal:** 3 states — `{Negative, Neutral, Positive}` for SHAP signals, `{Negative-return, Flat, Positive-return}` for GP mean, `{Low, Moderate, High}` for GP confidence. Total: 6 signals × 3 states = 18 MFs.
- **MF parameter initialisation:** centres at 15th / 50th / 85th percentiles of each signal's *training-fold* empirical distribution; widths σ set so adjacent MFs overlap at 0.5 membership (sklearn-style heuristic).
- **MF parameter tuning:** **none — grid search over MF widths only** (per the prior scope cut; no ANFIS). Three width multipliers tested: {0.75, 1.0, 1.25}. The combination that maximises F1 on a held-out validation fold becomes the reported configuration.
- **Rule base:** ~6–8 hand-designed Mamdani rules in supervisory vocabulary. Includes the **confidence-override rule** (Rule 4 in the proposal: high GP variance → cautious linguistic output regardless of fundamental signals).
- **Defuzzification:** centroid method, producing a scalar "outlook score" in [0, 1].
- **Persistence:** `dataset/processed/fuzzy_oof_outputs.csv` with `Company, FiscalYear, outlook_score, dominant_states (JSON), fired_rules (JSON)`.

### Stage 7 — Linguistic engine

- **Approach:** deterministic template-based sentence generator. **No LLM, no neural language model.** Auditable, reproducible, appropriate to 15-credit scope.
- **Input:** Stage 6's defuzzified outlook score + dominant fuzzy state labels per signal + which rules fired.
- **Template engine:** a decision table mapping `(dominant_state, confidence_level, risk_flag)` → sentence fragment. Fragments concatenated with simple connectives. Implementation: a dict of (key tuple → string template).
- **Paper deliverable:** a table of 8–10 hand-selected test cases showing input ratios → GP prediction with uncertainty → SHAP values → fuzzy states → final generated sentence, with a 1-sentence annotation per row explaining why the output is sensible. This *is* the qualitative evaluation; the human Likert protocol from the original proposal is cut.

---

## 4. Schedule — emergency pace, interleaved writing

Total elapsed effort assumed: **2 weeks of focused work**, distributed roughly as follows. "D" = day of work, not calendar day.

| Block | Days | Stage(s) | Paper sections drafted |
|---|---|---|---|
| **A** | D1 | Stage 3 implementation: env setup, `evaluation.py`, `xgb_baseline.py` skeleton | §3 (Data) — draft from preprocessing facts |
| **B** | D2 | Stage 3 finish: nested CV run, SHAP OOF, all sanity checks | §4 (Methods – XGBoost + SHAP subsection) |
| **C** | D3 | Stage 4: GP regression with fold-aware scaling, calibration curves | §4 (Methods – GP subsection) |
| **D** | D4 | Stage 5: GP classification via threshold, paired metrics table vs XGBoost | §5 (Results – classification & calibration table); §1 (Introduction) — first draft |
| **E** | D5 | Stage 6: fuzzy MFs from OOF SHAP + GP outputs, hand-designed rule base, confidence-override rule wired | §2 (Related Work / Literature Review) — first draft |
| **F** | D6 | Stage 7: linguistic engine + 8–10 worked examples | §5 (Results – linguistic examples); §6 (Discussion) — first draft |
| **G** | D7 | Robustness: LOBO secondary, per-bank / per-year diagnostic tables, bootstrap CIs on every metric | §5 (Results – robustness subsection) |
| **H** | D8 | Paper revision pass 1 — tighten Methods + Results, finalise figures | All sections — coherence pass |
| **I** | D9 | Paper revision pass 2 — Introduction, Discussion, Conclusion polishing; ethics/social/legal section | §7 (Social/Ethical/Legal); §8 (Conclusion); §0 (Abstract) |
| **J** | D10 | Code appendix, references final-format, file rename (`NAME_studentID`), final PDF export, submission | — |

**Working rules during this block:**
- One stage starts only when the previous one's persistence file exists on disk. No verbal handoffs.
- Each stage gets a `RESULTS_<stage>.md` written immediately after the stage runs — captures headline numbers, what surprised, what changed from the plan. This is the source material for the paper sections.
- Paper sections drafted in the column above are **first drafts**, not final. Revision happens in blocks H–I.
- If a day blows budget, the response is to **cut scope inside the running stage**, not to push the schedule. Pre-cut targets identified in §6.

---

## 5. What "done" looks like per stage

Concrete acceptance criteria. No optimism, no qualitative language.

- **Stage 3 done:** [PLAN_XGBoost_SHAP_Baseline.md](PLAN_XGBoost_SHAP_Baseline.md) §7 sanity checks all pass. `xgb_oof_predictions.csv` and `shap_values_oof.csv` exist with 80 rows each. `RESULTS_XGBoost.md` written.
- **Stage 4 done:** `gp_oof_regression.csv` exists with 80 rows. RMSE / MAE / R² with bootstrap CIs computed. Per-feature ARD length scales reported (this shows the GP discovered which features matter — a free interpretability win to mention in the paper).
- **Stage 5 done:** `gp_oof_classification.csv` exists with 80 rows. Side-by-side metrics table (XGBoost raw / XGBoost isotonic / GP) computed for all 8 metrics. ECE and Brier comparison is the headline result.
- **Stage 6 done:** `fuzzy_oof_outputs.csv` exists. The confidence-override rule has been *demonstrated* on at least one test case where a positive fundamental signal is correctly overridden by high GP variance — verify by inspection, document the case in the paper.
- **Stage 7 done:** 8–10 worked examples in a paper-ready table format. Each example shows: bank, year, input ratios, GP μ/σ, top SHAP values, fuzzy states, generated sentence, 1-line annotation.
- **Stage 8 done:** PDF of paper exists, ≤ 6 pages excluding references and code appendix, all figures regenerable from persisted CSVs, all references in consistent format.
- **Stage 9 done:** file renamed `Smaran_Luitel_<studentID>.pdf` (insert real student ID), code zipped or pushed to GitHub with README, submission link receipt saved.

---

## 6. Pre-decided scope cuts (in priority order if time pressed)

When a day blows budget, cut from the **bottom up**. Do not negotiate.

1. **First to cut:** LOBO robustness analysis (Stage 3 + Stage 5). LOYO alone is defensible. Mention as "future work."
2. **Second to cut:** Bootstrap CIs on all metrics — keep on Brier and F1 only. Other metrics get point estimates.
3. **Third to cut:** Per-bank and per-year diagnostic tables. Move to appendix only; not referenced in main text.
4. **Fourth to cut:** Isotonic-calibrated XGBoost. Report only raw XGBoost as comparator. Weakens the calibration story slightly; mention as a limitation.
5. **Fifth to cut:** Grid search over fuzzy MF widths. Use heuristic widths only. Acknowledge as a limitation.
6. **Last resort:** Drop Stage 6 width tuning *and* drop the LOBO robustness *and* the bootstrap CIs. The paper still has: GP vs XGBoost on calibration, SHAP-based fuzzification, confidence-override rule. That is the minimum viable paper.

**What we never cut, no matter what:**
- GP regression + GP classification (assignment requirement).
- SHAP attributions feeding fuzzy MFs (the proposal's novelty hinge).
- The confidence-override rule (the proposal's headline novelty).
- LOYO cross-validation with no leakage (basic methodological integrity).
- Honest reporting of N=80 limitation in the paper.

---

## 7. Paper outline — locked target structure

Six pages, single column, written to a generic ML conference template (per assignment §"You are encouraged to target a certain conference or journal").

| Section | ~Length | Drafted in block |
|---|---|---|
| Abstract | 200 words | I |
| 1. Introduction | 0.7 page | D |
| 2. Related Work | 0.6 page | E |
| 3. Data and Preprocessing | 0.6 page | A |
| 4. Methods (XGBoost+SHAP / GP / Fuzzy / Linguistic) | 1.5 pages | B–F |
| 5. Results | 1.5 pages | D–G |
| 6. Discussion (and Confidence-Override case study) | 0.5 page | F |
| 7. Social / Ethical / Legal / Professional considerations | 0.3 page | I |
| 8. Conclusion | 0.2 page | I |
| References + Appendix (code, hyperparams) | beyond page limit | J |

**Ethics/social/legal section** is required by the marking rubric and easy to forget. Three concrete points to cover: (a) supervisory decisions made on small samples carry policy weight — model overconfidence is a real harm; (b) data is from public NRB disclosures, no personal data, no consent issue; (c) the linguistic output is *advisory* and explicitly framed as such — no replacement of professional judgment.

---

## 8. Code & artifact organisation (target end state)

```
Assignment/
├── Proposal_v2_OnePage.md              # rewritten 1-page proposal
├── PLAN_OVERALL_Project.md             # this document
├── PLAN_XGBoost_SHAP_Baseline.md       # Stage 3 detailed plan
├── PLAN_GP.md                          # Stage 4–5 detailed plan (TBD at Stage 4 kickoff)
├── PLAN_Fuzzy_Linguistic.md            # Stage 6–7 detailed plan (TBD at Stage 6 kickoff)
├── RESULTS_<stage>.md                  # one per stage, written immediately after stage runs
├── requirements.txt                    # pinned dependencies
├── preprocess.py                       # Stage 2 (done)
├── evaluation.py                       # shared harness (Stage 3 creates, Stages 4–6 reuse)
├── xgb_baseline.py                     # Stage 3
├── gp_model.py                         # Stages 4–5
├── fuzzy_layer.py                      # Stage 6
├── linguistic_engine.py                # Stage 7
├── make_figures.py                     # regenerates all paper figures from persisted CSVs
├── dataset/
│   ├── financial_data_all_normalized.csv      # raw
│   └── processed/
│       ├── financial_data_wide.csv            # Stage 2 output
│       ├── financial_ratios.csv               # Stage 2 output — modelling input
│       ├── xgb_oof_predictions.csv            # Stage 3 output
│       ├── shap_values_oof.csv                # Stage 3 output — input to Stage 6
│       ├── gp_oof_regression.csv              # Stage 4 output
│       ├── gp_oof_classification.csv          # Stage 5 output — head-to-head with XGBoost
│       └── fuzzy_oof_outputs.csv              # Stage 6 output — input to Stage 7
├── figures/                                   # all paper figures, regenerable from CSVs alone
├── logs/                                      # one log per stage run
└── paper/
    ├── paper.tex (or .docx)
    └── paper.pdf                              # final submission artefact
```

Every output CSV is the contract between stages. If a downstream stage needs to inspect upstream output, it reads the CSV — never re-runs the upstream stage. This makes every stage independently re-runnable and the paper figures independently regenerable.

---

## 9. Risks specific to the overall plan

| Risk | Likelihood | Pre-decided response |
|---|---|---|
| GP doesn't beat XGBoost on calibration | Possible | Report honestly. Paper pivots to: "GP and isotonic-calibrated XGBoost both achieve good calibration; GP's advantage is *unsupervised* uncertainty (no calibration set needed) and *direct integration* with the fuzzy layer via posterior variance." Story still holds. |
| Fuzzy outputs feel arbitrary in worked examples | Medium | Worked examples are hand-selected anyway. Pick cases where the confidence-override rule clearly fires. Document selection criterion in the paper as "illustrative, not random." |
| 6 pages too short for everything | Likely | First cut: shrink Related Work to 0.4 page (cite tightly, don't summarise). Second cut: shrink Methods by referencing equations to literature instead of re-deriving GP posterior formulas. |
| Page-limit confusion (assignment says "6 pages A4, up to 4000 words" for paper but "12 pages A4, up to 6000" overall) | Assumption needs confirmation | The 6-page limit applies to the paper proper; appendix (code) is beyond. If the marker treats the limit as inclusive of appendix, move code to a GitHub link — the rubric explicitly allows this. |
| Reproducibility breaks between stages because of seed/state mismatch | Medium | Every stage script accepts `--seed` argument, defaults to 42. `evaluation.py` exposes the LOYO splitter as a deterministic generator. Sanity check at the start of each stage: verify the loaded input CSV's row count and column hash match the upstream stage's manifest. |

---

## 10. Open decisions (to be made at the latest possible moment)

These are deliberately *not* decided yet. Each has a default that will be used if no decision is made by the listed deadline.

- **Conference template:** ICML, NeurIPS, or a generic single-column format. Default: `acmart` single-column at submission time. Decide by D8.
- **GP library:** `GPyTorch` (proposal-stated) vs `scikit-learn` GP (simpler API but no ARD by default). Default: GPyTorch. Decide at Stage 4 start.
- **Final paper format:** LaTeX or Word. Default: Word (.docx → PDF) for speed given the emergency pace. Decide at start of block H.
- **Code submission method:** GitHub link vs zipped appendix. Default: GitHub link with a tagged release commit. Decide at Stage 9.

---

## 11. What this plan is *not* doing

To prevent scope creep, these are explicitly excluded:

- No Bayesian network. (Considered in the proposal discussion; the user opted to keep XGBoost+SHAP as the only comparator.)
- No ANFIS. (Cut from the proposal.)
- No FastAPI or any serving layer. (Cut from the proposal.)
- No multi-sector expansion. (Banking-only; declared as a scope limitation in the proposal.)
- No external data — no stock prices, no macroeconomic series.
- No quarterly modelling. (Decided against on label-overlap grounds.)
- No data synthesis or SMOTE. (Decided against on academic-integrity and calibration grounds.)
- No deep learning baselines.
- No human Likert evaluation. (Replaced by hand-selected worked examples.)
- No Task 2 work. (User confirmed Task 1 only.)

If at any point during execution one of these is reconsidered, the change must first be reflected in this document, then in the affected stage plan.

---

## 12. Changelog

- **2026-05-21** — Initial overall plan written. Emergency-pace 10-day execution schedule. XGBoost-runs-anyway gating (no early-exit on baseline failure). Interleaved paper drafting. Confidence-override rule confirmed as headline novelty. Scope cuts pre-ordered in §6.
