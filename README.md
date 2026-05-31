# STW7085CEM Advanced Machine Learning — Assignment

This repository contains both tasks for the module:

- **[Task 1](#task-1--forward-npl-risk-gp-prediction-with-fuzzy-linguistic-explanation)** — Forward NPL risk prediction for Nepali commercial banks, combining a Gaussian Process, XGBoost + SHAP, a fuzzy layer, and a linguistic engine (Python, in [`Task 1/`](Task%201/)).
- **[Task 2](#task-2--fuzzy-logic-control-and-evolutionary-optimisation)** — A Smart Apartment fuzzy logic controller with GA optimisation and a GA-vs-PSO benchmark study (MATLAB, in [`Task 2 Assignment/`](Task%202%20Assignment/)).

---

# Task 1 — Forward NPL Risk: GP Prediction with Fuzzy Linguistic Explanation

Code and data for **Task 1**. All paths in this section are relative to [`Task 1/`](Task%201/).

This part predicts forward Non-Performing Loan (NPL) change for ten Nepali commercial banks and explains each prediction in plain language. It combines four stages:

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

---

# Task 2 — Fuzzy Logic Control and Evolutionary Optimisation

A separate MATLAB project in [`Task 2 Assignment/`](Task%202%20Assignment/). All paths in this section are relative to that folder. It has two parts:

1. **A Smart Apartment Fuzzy Logic Controller (FLC)** — a Mamdani fuzzy system that decides HVAC, lighting, and blinds settings from six environmental and behavioural inputs.
2. **Evolutionary optimisation** — a Genetic Algorithm (GA) that tunes the FLC's membership functions, plus a GA-vs-PSO comparison on two CEC'2005 benchmark functions.

## Requirements

- MATLAB (R2019b or later recommended — uses the `mamfis` / `addInput` / `addMF` object API)
- **Fuzzy Logic Toolbox** (for `mamfis`, `evalfis`, `writeFIS`, the rule viewer, and control-surface plots)

No other toolboxes are required; the GA and PSO are implemented from scratch in the scripts.

## Files

| File | What it is |
|---|---|
| `SmartApartmentFLC.m` | Builds the Mamdani FLC, plots membership functions and control surfaces, runs a 24-hour scenario simulation, and writes the FIS to file |
| `SmartApartmentFLC.fis` | The base FLC, saved (6 inputs, 3 outputs, 25 rules) |
| `SmartApartmentFLC_optimised.fis` | The FLC after GA tuning of the input membership functions |
| `GA_FLC_Optimizer.m` | Genetic Algorithm that optimises the FLC membership-function breakpoints against a reference dataset |
| `CEC2005_Benchmark_Comparison.m` | Compares GA vs PSO on CEC'2005 F1 (Shifted Sphere) and F9 (Shifted Rastrigin) at D=2 and D=10 |
| `matlab.mat` | Saved workspace / results |
| `AML TASK 2.tex`, `Advance_ML TASK 2.pdf`, `Task_2_Fuzzy_Logic.pdf` | The written report (source and PDFs) |

## Part 1 — Smart Apartment FLC

A Mamdani fuzzy inference system with **6 inputs**, **3 outputs**, and **25 rules**.

**Inputs:**

| Input | Range | Membership functions |
|---|---|---|
| `temperature` (°C) | 0–40 | cold, comfortable, hot |
| `humidity` (%) | 0–100 | dry, comfortable, humid |
| `external_light` (lux) | 0–1000 | dark, moderate, bright |
| `time_of_day` (hour) | 0–24 | night, morning, afternoon, evening (4 MFs) |
| `occupancy` | — | low, medium, high |
| `user_activity` | — | resting, active, away |

**Outputs:** `hvac_output` (−100…100), `lighting_output` (0…100), `blinds_output` (0…100).

Inference uses `min` (AND), `max` (OR / aggregation), `min` implication, and **centroid** defuzzification.

Run it:

```matlab
SmartApartmentFLC
```

This builds the FIS, writes `SmartApartmentFLC.fis`, plots the membership functions and control surfaces, runs a 24-hour operational scenario, and opens the rule viewer for a sample input. To open the system interactively instead: `fuzzyLogicDesigner('SmartApartmentFLC.fis')`.

## Part 2 — GA optimisation of the FLC

`GA_FLC_Optimizer.m` tunes the **input** membership-function breakpoints with a real-coded Genetic Algorithm.

- **Chromosome:** 76 real-valued genes across the six inputs (trapmf = 4 params, trimf = 3 padded to 4).
- **Fitness:** `1 / (1 + RMSE)`, where RMSE is measured against a synthetic reference dataset of 200 input–output examples (higher fitness = lower RMSE).
- **GA settings:** population 60, 80 generations, crossover probability 0.80, per-gene mutation 0.05, elitism of the top 2.

Run it:

```matlab
GA_FLC_Optimizer
```

It prints per-generation best/mean fitness and best RMSE, reports the base-vs-optimised RMSE improvement, and writes `SmartApartmentFLC_optimised.fis`. The script also discusses how the chromosome length would change for a full-system encoding and for a Sugeno (TSK) model instead of Mamdani.

## Part 3 — GA vs PSO on CEC'2005 benchmarks

`CEC2005_Benchmark_Comparison.m` compares a real-coded GA (SBX crossover) against an inertia-weight PSO on:

- **F1 — Shifted Sphere** (unimodal), global optimum −450
- **F9 — Shifted Rastrigin** (multimodal), global optimum −330

at dimensions **D = 2** and **D = 10**. Protocol: 15 independent runs per algorithm per function per dimension, with a budget of 10,000 function evaluations per run. The script reports mean / std / best / worst final values and produces convergence curves, box plots, and a summary table.

Run it:

```matlab
CEC2005_Benchmark_Comparison
```

## Reproducing Task 2

From the `Task 2 Assignment` folder in MATLAB:

```matlab
SmartApartmentFLC              % Part 1: build + visualise the FLC
GA_FLC_Optimizer               % Part 2: GA-tune the membership functions
CEC2005_Benchmark_Comparison   % Part 3: GA vs PSO benchmark study
```

The GA and PSO are stochastic; to reproduce exact numbers, set a fixed seed with `rng(42)` at the top of each script before running. The full method and results write-up is in `Task_2_Fuzzy_Logic.pdf` (and `AML TASK 2.tex`).
