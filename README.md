# Forward NPL Risk: GP Prediction with Fuzzy Linguistic Explanation

Code and data for **STW7085CEM Advanced Machine Learning, Task 1**.

This repository predicts forward Non-Performing Loan (NPL) change for ten Nepali commercial banks and explains each prediction in plain language. It combines four stages:

1. **XGBoost + SHAP** — a gradient-boosted classifier with per-feature attributions.
2. **Gaussian Process (GP)** — a Matern 5/2 ARD regression with a Student-t likelihood that also gives a deterioration probability and a per-row posterior variance.
3. **Gaussian fuzzy layer** — joins SHAP attributions and GP outputs into fuzzy linguistic states, with a confidence override (Rule 4) that softens the output when the GP is unsure.
4. **Linguistic engine** — a fixed template that writes one short supervisory comment per bank year. No language model is used; every word traces back to a model signal.

The full method, results, and discussion are in [`paper/paper.md`](paper/paper.md).

---

## Requirements

- Python 3.12
- The pinned dependencies in [`requirements.txt`](requirements.txt) (numpy, pandas, scikit-learn, xgboost, shap, matplotlib, scipy, torch, gpytorch, scikit-fuzzy)

### Set up the environment

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running the pipeline

Run the scripts in order from the repository root. Each script reads from and writes to `dataset/processed/`, so later stages depend on earlier ones.

```bash
python preprocess.py
python xgb_baseline.py --seed 42 --n-boot 2000
python gp_model.py --likelihood studentt --seed 42 --n-boot 2000
python fuzzy_layer.py --seed 42
python linguistic_engine.py --seed 42
python make_figures.py
```

Everything is seeded at 42. Two consecutive runs of any script produce bitwise-identical output files. The GP step is the slowest (variational inference across eight folds with three restarts each); the rest run in seconds.

### What each script does

| Script | Reads | Writes |
|---|---|---|
| `preprocess.py` | `dataset/financial_data_all_normalized.csv` (raw long table) | `financial_data_wide.csv`, `financial_ratios.csv` |
| `evaluation.py` | — (shared helper module) | cross-validation, metrics, and bootstrap-interval functions used by the others |
| `xgb_baseline.py` | `financial_ratios.csv` | `xgb_oof_predictions.csv`, `shap_values_oof.csv`, `xgb_*_metrics.*`, per-year/per-bank error |
| `gp_model.py` | `financial_ratios.csv` | `gp_oof_regression.csv`, `gp_oof_classification.csv`, `gp_ard_lengthscales.csv`, `gp_*_metrics.*` |
| `fuzzy_layer.py` | SHAP + GP out-of-fold outputs | `fuzzy_oof_outputs.csv`, `fuzzy_mf_params.json`, `fuzzy_tuning_log.csv` |
| `linguistic_engine.py` | fuzzy + SHAP + GP outputs | `linguistic_outputs.csv`, `linguistic_examples.csv` |
| `make_figures.py` | all of the above | eleven PNG figures in `figures/` |

---

## Repository layout

```
.
├── preprocess.py              # long-to-wide pivot, derives 7 ratios + forward label
├── evaluation.py              # shared CV / metrics / bootstrap harness
├── xgb_baseline.py            # XGBoost classifier + SHAP attributions
├── gp_model.py                # GP regression + GP classification (Student-t, ARD Matern 5/2)
├── fuzzy_layer.py             # Gaussian membership functions, Mamdani rules, Rule 4 override
├── linguistic_engine.py       # template engine, worked-example selection
├── make_figures.py            # generates all paper figures
├── requirements.txt           # pinned dependencies
├── dataset/
│   ├── financial_data_all_normalized.csv   # raw input (long table)
│   └── processed/             # all generated CSV / JSON outputs
├── figures/                   # eleven generated PNG figures
└── paper/
    └── paper.md               # the report
```

---

## Data

The input is public quarterly disclosures for ten Nepali commercial banks (ADBL, EBL, GBIME, HBL, NABIL, NICA, NMB, PCBL, SANIMA, SCB) summarised by Nepal Rastra Bank, covering BS 2072/2073–2082/2083 (≈ AD 2015/2016–2025/2026). After pivoting to one row per bank year and aligning a forward label, the modelling subset has **80 bank-year rows** with seven base ratios as inputs: CAR, NPL Ratio, CD Ratio, Cost of Funds, Base Rate, Interest Spread, and a derived Return on Equity.

- **Regression target:** `Delta_NPL_next_year` — next-year change in NPL ratio (percentage points).
- **Classification target:** `Deteriorate_next_year` — 1 when `Delta_NPL_next_year` > 0.5 pp (25 deteriorate / 55 stable).

There is no personal data; all values are public bank disclosures.

---

## Key results

Pooled out-of-fold metrics on 80 rows (95% stratified bootstrap intervals; see [`paper/paper.md`](paper/paper.md) for the full tables):

- The GP and XGBoost **agree on the top two features** (CD Ratio and Base Rate) despite using completely different mechanisms.
- The GP **does not improve calibration** over XGBoost at this sample size — intervals are wide and the classifiers are hard to tell apart.
- The fuzzy system **matches XGBoost on F1 (0.44)** and **beats every upstream model on accuracy (0.75)**.
- The Rule 4 confidence override fires on **37 of 80 rows**, split across both classes, turning confident-sounding wrong readings into explicit "low confidence" warnings.

The main worked example (ADBL 2074/2075) shows the override in action.

---

## Reproducibility notes

- All seeds are fixed at 42 (`--seed 42`).
- Bootstrap intervals use 2,000 resamples (`--n-boot 2000`).
- Cross-validation is Leave-One-Year-Out for both XGBoost and the GP, so out-of-fold row indices align line by line across files.
- Fuzzy width multipliers are tuned **in sample** on the same 80 rows the fuzzy system reports on; the upstream models are honest out of fold. This is stated in the report's limitations.
