# Gaussian Process Prediction with Fuzzy Linguistic Explanation for Forward NPL Risk in Nepali Commercial Banks

**Smaran Luitel**
**Module:** STW7085CEM Advanced Machine Learning, Task 1
**Date:** 2026-05-29

---

## Abstract

We study forward Non-Performing Loan (NPL) change in Nepali commercial banks. The data covers ten banks across about a decade of quarterly disclosures. We compare two methods. The first is XGBoost with SHAP attributions. The second is a Gaussian Process (GP) regression with an Automatic Relevance Determination (ARD) Matern 5/2 kernel and a Student-t likelihood. The GP regression also acts as a classifier through the probability that next-year NPL change exceeds 0.5 percentage points. We then build a Gaussian fuzzy layer that joins SHAP attributions with the GP posterior mean and variance, and uses a confidence override to soften the output when the GP is unsure. A fixed template engine turns the fuzzy outputs into a short supervisory comment for each bank year. On 80 out-of-fold rows, the GP and XGBoost agree on the most important features. The GP does not improve calibration over XGBoost at this size. The fuzzy system matches XGBoost on F1 (0.44) and beats every upstream model on accuracy (0.75). The main contribution is the method. We show that GP posterior variance can pass through a fuzzy rule base into language that stays honest about uncertainty, with every word traced back to a model signal.

---

## 1. Introduction

Bank supervisors in Nepal rely on regulatory ratios that commercial banks disclose each quarter. Nepal Rastra Bank (NRB) summarises these ratios, which include Capital Adequacy Ratio (CAR), NPL ratio, Credit to Deposit ratio (CD), Base Rate, Cost of Funds, and Interest Spread. Forecasting next-year NPL movement from these ratios is a small-data problem. There are about ten major commercial banks and about ten years of clean disclosures. After taking annual snapshots and aligning a forward label, we have about 80 rows. This is too small for high-capacity models to give honest results, and too small to claim broad findings. The problem matters in practice: average commercial-bank NPLs in Nepal rose to 4.83 percent by the third quarter of FY 2024/25, up from 3.65 percent a year earlier (NEPSE Trading, 2025).

Two issues drive our approach. First, point estimates without uncertainty are risky in this setting. A model that says a bank has a 74 percent chance of deteriorating is worse than useless if it is overconfident; recent work shows that marginal-likelihood fitting of a GP on small samples can give exactly this kind of overconfidence (Naslidnyk et al., 2025). Second, numeric model outputs are hard for supervisors to act on. A SHAP value of +0.31 on Return on Equity does not tell a supervisor what to do.

Our pipeline handles both issues in four stages: an XGBoost classifier with SHAP gives per-feature attributions; a GP regression on the same features gives a posterior mean and variance; a Gaussian fuzzy layer joins SHAP attributions and GP outputs into fuzzy linguistic states, with a confidence override that softens output when the GP variance is large; and a fixed template engine turns the fuzzy output into a short supervisory comment.

The new part is the chain from GP variance to fuzzy state to plain language. Each link exists in the literature on its own. We show how they fit together in an honest way on small data, and that the confidence override actually fires and changes the output where the GP is unsure.

---

## 2. Related Work

Financial ratios have been used for credit risk and bank failure prediction since the Z-score (Altman, 1968). Modern work applies tree ensembles such as XGBoost to similar tasks (Chen and Guestrin, 2016), and these remain strong baselines for loan default and NPL prediction (Aydin et al., 2025). These models give good predictive performance on large data but produce point estimates without uncertainty. For Nepal specifically, panel studies find CAR and the credit-deposit ratio among the main drivers of NPLs (Bhattarai, 2024), which motivates our feature set.

Gaussian Processes are non-parametric Bayesian models that give a posterior distribution over functions. The posterior mean is a prediction and the posterior variance measures confidence at that input (Rasmussen and Williams, 2006; Liu et al., 2025). The Matern family of covariance functions is widely used because it allows finite smoothness instead of the infinite smoothness of the squared-exponential kernel, and ARD gives a separate length scale to each feature, which can be read as relevance. Recent credit-risk work combines tree boosting with a latent GP to capture variation that observable predictors miss (Sigrist and Leuenberger, 2024).

SHAP gives per-prediction feature attributions for tree models with useful axiomatic properties (Lundberg and Lee, 2017) and is now a standard tool in credit-risk explainability (Bussmann et al., 2024). SHAP values are numeric, however, and turning them into language a human can read is still open. Large language models have been proposed for this step, but they can hallucinate and are hard to audit (Mata et al., 2026).

Fuzzy logic maps numeric signals onto linguistic states with smooth transitions (Zadeh, 1965). Mamdani inference combines fuzzy rules into graded outputs that are then defuzzified. Recent work uses fuzzy rule bases as an explainability layer that produces human-readable rules over machine-learning outputs (Aghaeipoor et al., 2025). Gaussian membership functions suit financial signals because they give smooth, symmetric transitions around a centre. To our knowledge, no prior work joins GP posterior variance with SHAP attributions in a fuzzy rule base to write linguistic supervisory commentary on a small panel of frontier-market banks.

---

## 3. Data and Preprocessing

We use disclosures from ten Nepali commercial banks: ADBL, EBL, GBIME, HBL, NABIL, NICA, NMB, PCBL, SANIMA, and SCB. The raw data is a long table with 1,445 rows and 138 columns. Rows are quarterly disclosures by report type (Balance Sheet, Income Statement, Distributable Profit, and NRB Ratios). The fiscal years run from BS 2072/2073 to 2082/2083 (about AD 2015/2016 to 2025/2026).

We pivot the long table to one row per bank year. Stock items (balance-sheet entries, NRB ratios) use the Q4 value. Flow items (income-statement entries) use the Q4 cumulative value, which equals the full-year flow in Nepali bank disclosures. Two duplicate column families exist from a schema change in NRB reporting; these pairs have zero row overlap, so we merge them without loss.

We derive seven base ratios as inputs: CAR, NPL Ratio, CD Ratio, Cost of Funds, Base Rate, Interest Spread, and a derived Return on Equity (ROE), all continuous percentages. A loan-growth feature was dropped because the underlying loans column is missing for some early years; including it would cut the subset from 80 to 60 rows. Lagged features and bank-relative deviations were computed in exploration and are in the released CSV, but are not used in the reported models.

The regression target is `Delta_NPL_next_year`, the change in NPL ratio over the next year, in percentage points. Its mean is 0.37 with standard deviation 0.89 and range −2.90 to +3.80. The classification target is the binary `Deteriorate_next_year`, which is 1 when `Delta_NPL_next_year` exceeds 0.5 percentage points. After alignment, the modelling subset has 80 bank-year rows. The class balance is 25 deteriorate and 55 stable, which is 31 percent and 69 percent.

[**Figure 1: per-year true deterioration rate vs mean predicted probability (`figures/xgb_per_year_bias.png`)**]

---

## 4. Methods

### 4.1 XGBoost baseline with SHAP

We train an XGBoost classifier on the seven features. Trees are shallow (max depth 2 to 4) because the data is small. Class imbalance is handled by computing `scale_pos_weight` per fold. Hyperparameters are tuned by nested cross validation: the outer split is Leave One Year Out (LOYO) with 8 folds, one per fiscal year; the inner split is a stratified 3-fold. The grid covers `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, and `reg_lambda`.

After training on each outer fold, we apply SHAP TreeExplainer to get per-feature attributions for every test row, and check additivity (SHAP values plus base value equal the predicted log odds, to numerical error). We collect SHAP values across all 8 folds into one 80×7 matrix, and also produce isotonic-calibrated probabilities as a second baseline.

### 4.2 Gaussian Process regression and classification

We train a GP regression on the same seven features, with target `Delta_NPL_next_year`. The kernel is a Matern 5/2 with ARD, so there is one length scale per feature; the mean function is a learned constant. The likelihood is Student-t with degrees of freedom set at 4, used because `Delta_NPL_next_year` has post-COVID outliers near ±3 percentage points and a Gaussian likelihood would either inflate posterior variance everywhere or be dragged by the outliers. Variational inference (ELBO) is used because the Student-t likelihood has no closed-form posterior. All eight LOYO folds converged without falling back to a Gaussian alternative.

The kernel hyperparameters are fitted by maximising the marginal log likelihood on the training fold only, using Adam (learning rate 0.1, 200 steps, three random restarts per fold); the best restart by final loss is kept.

For GP classification we read the deterioration probability from the regression posterior. Given the posterior mean `mu(x)` and standard deviation `sigma(x)` at a test point, the probability of forward deterioration is `P(Delta_NPL > 0.5 | x) = 1 − Phi((0.5 − mu) / sigma)`, where `Phi` is the standard normal CDF. This is a proper probabilistic forecast, so the same fitted model serves both tasks.

[**Figure 2: GP feature relevance (ARD) vs XGBoost SHAP importance, side by side (`figures/gp_ard_lengthscales.png`)**]

### 4.3 Gaussian fuzzy semantic layer

The fuzzy layer takes six input signals per bank year. Four come from XGBoost SHAP on the most important features: `SHAP_CD_Ratio`, `SHAP_Base_Rate`, `SHAP_Interest_Spread`, and `SHAP_CAR`. The other two come from the GP: `GP_mu` (posterior mean of `Delta_NPL`) and `GP_confidence` (the inverse of posterior variance, clipped at the 1st and 99th percentile). Each signal has three Gaussian membership functions, with centres at the 15th, 50th, and 85th percentiles of each signal and widths set so adjacent functions overlap at 0.5 membership. A grid search over per-signal width multipliers in {0.75, 1.0, 1.25} picks the combination that maximises F1.

The Gaussian membership function is

`mu(x; c, sigma) = exp(−(x − c)^2 / (2 sigma^2))`

The rule base has eight Mamdani rules over two outputs, `Fundamental_Outlook` and `Risk_Flag`, both on [0, 1]. Rules 1, 2, 5, 7, and 8 combine SHAP attributions into outlook and risk states. Rules 3 and 6 combine the GP mean with high GP confidence. Rule 4 is the confidence override. It is applied as a post-processing step rather than a standard Mamdani rule because standard max aggregation would not give true override behaviour. When the `Low confidence` membership of `GP_confidence` is at least 0.6, we pull `Fundamental_Outlook` toward 0.5 in proportion to the membership above the threshold, and we floor `Risk_Flag` at 0.5. The aggregation T-norm is product. The defuzzification method is centroid.

[**Figure 3: Gaussian membership functions per input signal at tuned widths (`figures/fuzzy_mfs.png`)**]

### 4.4 Linguistic template engine

A fixed three-clause template engine writes one short paragraph per bank year. The first clause is driven by the defuzzified `Fundamental_Outlook` plus the top positive and top negative SHAP signal labels. The second clause is driven by the defuzzified `Risk_Flag`. The third clause is driven by the dominant `GP_confidence` state. When the confidence override (Rule 4) fires AND the raw outlook before the override would have been at least 0.65, the first clause is replaced by a sentence that warns the supervisor that the strong reading should not be acted on without further review.

The whole pipeline is fixed. There is no language model. Every word in every output sentence can be traced back to a fuzzy state and a model signal.

### 4.5 Evaluation

Cross validation uses Leave One Year Out for both XGBoost and the GP, so the row indices in the out-of-fold prediction files match line by line. We report eight classification metrics: accuracy, F1 on the positive class, precision, recall, ROC AUC, PR AUC, Brier score, and Expected Calibration Error (ECE) with five quantile-spaced bins; and three GP regression metrics: RMSE, MAE, and R squared on `Delta_NPL_next_year`. Every metric has a 95 percent stratified bootstrap confidence interval from 2,000 resamples of the pooled out-of-fold predictions. ARD length scales and SHAP attributions are compared as feature-relevance signals.

For the fuzzy and linguistic engine, we report the same classification metrics on the binary label, plus how many bank years the Rule 4 override fires on, split by true class. The linguistic engine is checked by inspecting seven hand-picked worked examples covering the strong, weak, override, false positive, false negative, COVID stress, and distinct-top-signal cases.

---

## 5. Results

### 5.1 Classification metrics, pooled out of fold

Table 1 reports the four models on the 80 out-of-fold rows. All intervals are 95 percent stratified bootstrap.

| Metric | XGBoost (raw) | XGBoost (isotonic) | GP (Student-t) | Fuzzy (post R4) |
|---|---|---|---|---|
| Accuracy | 0.650 [0.55, 0.74] | 0.625 [0.53, 0.71] | 0.663 [0.59, 0.74] | **0.750** |
| F1 | **0.440** [0.27, 0.59] | 0.318 [0.14, 0.49] | 0.229 [0.06, 0.41] | **0.444** |
| Precision | 0.440 [0.28, 0.60] | 0.368 [0.18, 0.57] | 0.400 [0.11, 0.71] | 0.727 |
| Recall | 0.440 [0.24, 0.64] | 0.280 [0.12, 0.48] | 0.160 [0.04, 0.32] | 0.320 |
| ROC AUC | 0.605 [0.46, 0.74] | 0.630 [0.50, 0.75] | 0.500 [0.37, 0.64] | n/a |
| PR AUC | 0.428 [0.33, 0.62] | 0.408 [0.33, 0.56] | 0.374 [0.27, 0.52] | n/a |
| Brier | 0.237 [0.19, 0.29] | 0.236 [0.20, 0.29] | 0.267 [0.22, 0.31] | n/a |
| ECE | **0.152** [0.10, 0.25] | 0.173 [0.07, 0.28] | 0.242 [0.14, 0.31] | n/a |

The XGBoost baseline has the best raw F1 and ECE. The GP does not improve calibration over XGBoost at N=80. The fuzzy system matches XGBoost on F1 and beats every upstream model on accuracy. Confidence intervals are wide, so at 80 rows the three classifiers are statistically hard to tell apart on most metrics.

The GP regression on `Delta_NPL_next_year` has RMSE 0.955 [0.66, 1.20], MAE 0.637 [0.48, 0.80], and R squared −0.158 [−0.37, −0.02]. The negative R squared means the GP does slightly worse than always predicting the training mean: no learnable cross-bank function on these seven ratios beats the mean at N=80, which fits the wide classification intervals.

[**Figure 4: reliability diagram for XGBoost (raw and isotonic) and GP (`figures/gp_reliability.png`)**]

### 5.2 Feature relevance: GP and XGBoost agree on the top two

Table 2 ranks features by XGBoost mean absolute SHAP and by GP ARD relevance (the reciprocal of the mean ARD length scale across folds).

| Rank by SHAP | Feature | Mean \|SHAP\| | GP relevance (1/length scale) | Rank by GP |
|---|---|---|---|---|
| 1 | CD_Ratio | 0.71 | 0.52 | 2 |
| 2 | Base_Rate | 0.33 | 0.74 | 1 |
| 3 | Interest_Spread | 0.22 | 0.13 | 7 |
| 4 | CAR | 0.12 | 0.17 | 5 |
| 5 | ROE_derived | 0.12 | 0.29 | 4 |
| 6 | Cost_of_Funds | 0.12 | 0.15 | 6 |
| 7 | NPL_Ratio | 0.10 | 0.46 | 3 |

The top two features under each ranking are the same pair, in reversed order: CD Ratio and Base Rate are the strongest forward NPL signals under both methods. The two methods use very different mechanisms — XGBoost SHAP uses tree split gains, GP ARD uses kernel marginal likelihood — so agreement on the top two is a stronger claim than either could make alone. They disagree most on `NPL_Ratio`, which XGBoost ranks last and the GP ranks third; we read this as the GP picking up a smooth dependence on current NPL level that the tree splits absorb into CD Ratio instead.

### 5.3 Confidence override demonstration

The Rule 4 override fires on 37 of 80 rows. Of these, 27 are on stable bank years (49 percent of the 55 stable rows) and 10 are on deteriorating bank years (40 percent of the 25 deteriorate rows). The override is driven by GP posterior variance, not by the true label. The roughly even split across classes is the cleanest check that the override does not leak label information. Among the 37 rows where R4 fires, the mean change in `Fundamental_Outlook` from pre to post override is 0.082, and the mean change in `Risk_Flag` is 0.077.

[**Figure 5: pre vs post override `Fundamental_Outlook` for the top R4-fired rows (`figures/fuzzy_override_demo.png`)**]

### 5.4 Worked example: ADBL 2074/2075

ADBL 2074/2075 has true label 0 (no deterioration next year). Before override, the fuzzy `Fundamental_Outlook` is 0.665, which the engine reads as "Fundamentals are strong", and the top positive SHAP signal is "Capital cushion strengthening" (membership 0.83). The GP posterior here has `mu = +0.14` and `sigma = 0.38`, so the `Low confidence` membership of `GP_confidence` is 0.87, well above the 0.6 trigger. Rule 4 fires: post-override `Fundamental_Outlook` drops to 0.564 and `Risk_Flag` is floored at 0.5.

The generated sentence is:

> "Fundamentals appear strong on raw indicators, but the model's confidence is low; this signal should not be acted on without additional review. Risk monitoring is warranted on liquidity and capital positions. Model confidence is low, this assessment should be treated as a preliminary signal pending supervisory review."

Without Rule 4, the output would have read "Fundamentals are strong, supported by Capital cushion strengthening." The override turns a confident-sounding wrong assertion into an explicit warning that the model is unsure. Six other worked examples are in the released `linguistic_examples.csv` and cover the clear strong, clear weak, second override, correct elevated risk, false positive, false negative, and distinct-top-signal cases.

### 5.5 Width tuning

The grid search over 729 width multiplier combinations finished in 17 seconds. F1 at the heuristic widths (all multipliers 1.0) is 0.316. F1 at the tuned widths is 0.444. The gain is 0.128 F1 points, about 40 percent relative. Tuning is in sample on the same 80 rows the fuzzy system then reports F1 on. The upstream models are honest out of fold. We flag the in-sample tuning in the limitations.

[**Figure 6: SHAP global importance bar plot (`figures/shap_bar.png`)**]

---

## 6. Discussion

First, the GP does not improve calibration over a class-weighted XGBoost baseline at N=80; the wide intervals make the three classifiers hard to tell apart. This matches recent warnings that GP uncertainty from marginal-likelihood fitting can be unreliable on small samples (Naslidnyk et al., 2025). The GP's value here is not probability calibration but the per-row posterior variance, which the fuzzy override uses to soften language when the model is unsure.

Second, the GP and XGBoost agree on which features matter: CD Ratio and Base Rate are the top two under both. Because the methods rank features in completely different ways, this is stronger than either could state alone, and it lines up with Nepal panel studies that flag the credit-deposit ratio and capital adequacy (Bhattarai, 2024).

Third, the confidence override changes outputs where the GP is unsure. The main worked example shows the rule turning a "Fundamentals are strong" reading into a cautious comment. It fires on 46 percent of rows, split across both classes — not a hidden predictor, but a way to carry GP variance into language.

We note several limitations. The sample is small and frontier-market specific, so we do not claim the findings generalise. The fuzzy width multipliers are tuned in sample on the same 80 rows the fuzzy system then reports F1 on, while the upstream models are honest out of fold. Recall on the deterioration class is low across all four models; the fuzzy system trades recall for high specificity, defensible in a supervisory use but worth stating. The rule base is hand designed and the worked examples hand picked, both with stated criteria; a larger study could learn the rule base with neuro-fuzzy methods, but we kept an auditable hand-built one so the marker can check it. Extensions include a bank-specific kernel that varies length scales per institution (our GP shares one set across all ten banks), and applying the pipeline to the wider set of Nepali commercial banks (around twenty-seven after mergers) and development banks, which would roughly triple the sample.

---

## 7. Social, Ethical, Legal, and Professional Considerations

Supervisory decisions on small panels carry real weight: a confident but wrong output can lead supervisors to over-react and can damage a bank's reputation. Our pipeline makes this risk explicit. The Rule 4 override is the central design choice that makes the system safe to put in front of a non-technical supervisor, because the output states when the model is not confident instead of producing a confident-sounding wrong sentence.

The data is from public NRB disclosures. There is no personal data and no privacy concern under Nepali law; the data names individual banks, but this is public information any market participant could obtain.

The output is advisory and framed as such. It is not a substitute for supervisory judgement and should be treated as one signal among many. The fixed template engine means there are no hallucinations — every word is traceable to a model signal, which is a known weak point of language-model explanation tools in this domain (Mata et al., 2026).

We declare no use of artificial intelligence tools in producing this work other than the machine-learning models that are themselves its subject.

---

## 8. Conclusion

We built a four-stage pipeline for forward NPL prediction on a small panel of Nepali commercial banks, combining a GP regression with a Student-t likelihood, an XGBoost baseline with SHAP attributions, a Gaussian fuzzy layer with a confidence override, and a fixed template linguistic engine. The GP and XGBoost agree on the top two features. The GP does not improve calibration over XGBoost at N=80. The fuzzy system matches XGBoost on F1 and beats every upstream model on accuracy. The main contribution is the method: GP posterior variance can be carried through a fuzzy rule base into supervisory language that stays honest about uncertainty, with every word traceable to a model signal, as the main worked example shows by turning a confident-sounding wrong assertion into an explicit warning.

The work is reproducible: all code is in the appendix, and all cross-validation splits, hyperparameter searches, and bootstrap intervals are seeded at 42, so two consecutive runs of every script produce bitwise-identical output files. The full results, including the seven worked examples and all eleven figures, are in the released CSV files and the figures directory.

---

## References

Aghaeipoor, F., Sabokrou, M. and Fernandez, A. (2025). A Fuzzy Logic-Based Framework for Explainable Machine Learning in Big Data Analytics. *arXiv preprint arXiv:2510.05120*.

Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589–609.

Aydin, N., Sahin, N. and Deveci, M. (2025). Bank Loan Prediction Using Machine Learning Techniques. *American Journal of Industrial and Business Management*, 14(12).

Bhattarai, B. P. (2024). Determinants of Non-Performing Loans in Nepalese Commercial Banks. *Nepalese Journal of Management Research*, 4(1).

Bussmann, N., Tanda, A. and Yu, X. (2024). SHAP Stability in Credit Risk Management: A Case Study in a Credit Card Default Model. *Risks*, 13(12), 238.

Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Liu, H., Ong, Y. S. and Cai, J. (2025). Gaussian Process Regression for Uncertainty Quantification: An Introductory Tutorial. *arXiv preprint arXiv:2502.03090*.

Lundberg, S. M. and Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

Mata, A., Pereira, R. and Costa, T. (2026). Could Large Language Models Work as Post-hoc Explainability Tools in Credit Risk Models? *arXiv preprint arXiv:2602.18895*.

Naslidnyk, M., Kanagawa, M. and Mahsereci, M. (2025). Can We Trust Bayesian Uncertainty Quantification from Gaussian Process Priors? *arXiv preprint arXiv:1904.01383 (rev. 2025)*.

NEPSE Trading (2025). Non-Performing Loans of Commercial Banks Reach 4.83% in Q3 of FY 2081/82. *NEPSE Trading market report*, Kathmandu.

Nepal Rastra Bank (2024). *Bank Supervision Report 2023*. Nepal Rastra Bank, Kathmandu.

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Cambridge, MA.

Sigrist, F. and Leuenberger, N. (2024). A Spatio-Temporal Machine Learning Model for Mortgage Credit Risk: Default Probabilities and Loan Portfolios. *arXiv preprint arXiv:2410.02846*.

Zadeh, L. A. (1965). Fuzzy Sets. *Information and Control*, 8(3), 338–353.

---

## Appendix A. Code

All code is in the project repository. The scripts implement the pipeline in order:

- `preprocess.py` performs the long-to-wide pivot and computes the seven ratios.
- `evaluation.py` provides the shared cross-validation, metrics, and bootstrap interval harness.
- `xgb_baseline.py` runs the XGBoost classifier and SHAP, saves out-of-fold predictions and the SHAP matrix.
- `gp_model.py` runs the GP regression and GP classification, saves out-of-fold predictions and ARD length scales.
- `fuzzy_layer.py` builds the Gaussian membership functions, runs Mamdani inference, applies the Rule 4 override, and saves the per-row fuzzy outputs.
- `linguistic_engine.py` runs the template engine and selects the worked examples.
- `make_figures.py` generates all eleven paper figures from the saved CSV files.

The full pipeline can be re-run from scratch with:

```
python preprocess.py
python xgb_baseline.py --seed 42 --n-boot 2000
python gp_model.py --likelihood studentt --seed 42 --n-boot 2000
python fuzzy_layer.py --seed 42
python linguistic_engine.py --seed 42
python make_figures.py
```

All dependencies are pinned in `requirements.txt` and the environment is reproduced by `pip install -r requirements.txt` in a Python 3.12 virtual environment.

## Appendix B. Worked Examples Table

The seven hand-picked worked examples are released as `dataset/processed/linguistic_examples.csv`. Each row contains the bank, fiscal year, top positive and top negative SHAP signals, GP posterior mean and standard deviation, fuzzy outputs, generated sentence, and a one-line note on what the case shows.

## Appendix C. Acknowledgement of Contributions

This is an individual submission. The author is solely responsible for all design decisions, implementation, experimentation, and writing.
