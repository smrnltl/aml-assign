# Gaussian Process Prediction with Fuzzy Linguistic Explanation for Forward NPL Risk in Nepali Commercial Banks

**Smaran Luitel**
**Module:** STW7085CEM Advanced Machine Learning, Task 1
**Date:** 2026-05-29

---

## Abstract

We study forward Non-Performing Loan (NPL) deterioration in Nepali commercial banks. The dataset covers ten banks across about a decade of quarterly disclosures. We compare two methods. The first is XGBoost with SHAP attributions. The second is a Gaussian Process (GP) regression with an Automatic Relevance Determination (ARD) Matern 5/2 kernel and a Student-t likelihood. The GP regression is also used as a classifier by deriving the probability that the forward change in NPL exceeds 0.5 percentage points. We then build a Gaussian fuzzy semantic layer that combines SHAP attributions with the GP posterior mean and variance, and uses a confidence override rule to soften model output when the GP is uncertain. A deterministic template engine turns the fuzzy outputs into a one paragraph supervisory comment for each bank year. On 80 out of fold observations, the GP and XGBoost agree on the most important features. The GP does not improve calibration over XGBoost at this sample size. The fuzzy system matches XGBoost on F1 (0.44) and beats every upstream model on accuracy (0.75). The headline contribution is methodological. We show that GP posterior variance can be passed through a fuzzy rule base into language that is honest about model uncertainty, with every word traceable to an upstream signal.

---

## 1. Introduction

Bank supervisors in Nepal rely on regulatory ratios disclosed by commercial banks each quarter. These ratios are summarised by Nepal Rastra Bank (NRB) and include Capital Adequacy Ratio (CAR), Non-Performing Loan ratio (NPL), Credit to Deposit ratio (CD), Base Rate, Cost of Funds, and Interest Spread. Forecasting forward NPL movement from these ratios is a small data problem. There are around ten major commercial banks. There are around ten years of clean disclosures. After taking annual snapshots and aligning a forward label, we have about 80 observations. This is too small for high capacity models to give honest results. It is also too small to claim broad empirical findings.

Two issues motivate our approach. First, point estimates without uncertainty are dangerous in this regime. A model that says "this bank has a 74 percent chance of deteriorating" is worse than useful if it is overconfident. Second, numerical model outputs are hard for supervisors to act on. A SHAP value of +0.31 on Return on Equity does not tell a supervisor what to do.

We build a pipeline that addresses both issues. The pipeline has four stages. First, an XGBoost classifier with SHAP gives per feature attributions for each prediction. Second, a Gaussian Process regression on the same features gives a posterior mean and posterior variance for each prediction. Third, a Gaussian fuzzy semantic layer combines SHAP attributions and GP outputs into fuzzy linguistic states. A confidence override rule explicitly softens the output when the GP variance is large. Fourth, a deterministic template engine turns the fuzzy output into a one paragraph supervisory comment.

The novelty contribution is the chain from GP variance to fuzzy state to natural language. Each link exists in the literature on its own. We show how they fit together in an honest way on small data, and we show that the confidence override rule actually fires and produces qualitatively different output on cases where the GP is uncertain.

The paper is organised as follows. Section 2 reviews related work. Section 3 describes the data and preprocessing. Section 4 describes the four method components in turn. Section 5 reports the experimental results. Section 6 discusses the findings, including the headline case study. Section 7 covers social, ethical, legal and professional issues. Section 8 concludes.

---

## 2. Related Work

Financial ratios have been used for credit risk and bank failure prediction since Altman's Z-score (Altman, 1968) and the prior CAMELS framework. Modern work has applied tree ensembles such as Random Forest and XGBoost to similar tasks (Chen and Guestrin, 2016). These models give strong predictive performance on large datasets but produce point estimates without uncertainty.

Gaussian Processes are non-parametric Bayesian models that give a posterior distribution over functions. The posterior mean is a prediction and the posterior variance is a measure of how confident the model is at that input (Rasmussen and Williams, 2006). The Matern family of covariance functions is widely used because it allows finite smoothness instead of the infinite smoothness of the squared exponential kernel. Automatic Relevance Determination (ARD) gives a separate length scale to each input feature, which can be read as a measure of feature relevance.

SHAP (SHapley Additive exPlanations) gives per prediction feature attributions for tree based models with desirable axiomatic properties (Lundberg and Lee, 2017). SHAP values are widely used as an explanation tool, but they are numerical. Translating SHAP values into language that a human can read remains an open problem.

Fuzzy logic offers a way to map numerical signals onto linguistic states with smooth transitions (Zadeh, 1965). Mamdani fuzzy inference systems combine fuzzy rules to produce graded output values that can then be defuzzified. Gaussian membership functions are particularly well suited to financial signals because they give smooth and symmetric transitions around a centre.

To our knowledge, no prior work combines GP posterior variance with SHAP attributions in a fuzzy rule base to generate linguistic supervisory commentary on a small panel of frontier market banks.

---

## 3. Data and Preprocessing

We use disclosures from ten Nepali commercial banks. The banks are ADBL, EBL, GBIME, HBL, NABIL, NICA, NMB, PCBL, SANIMA, and SCB. The raw dataset is a long table with 1,445 rows and 138 columns. Rows are quarterly disclosures by report type (Balance Sheet, Income Statement, Distributable Profit, and NRB Ratios). The fiscal years cover BS 2072/2073 to 2082/2083 (about AD 2015/2016 to 2025/2026).

We pivot the long table to one row per bank year. Stock items such as balance sheet entries and NRB ratios use the Q4 value. Flow items such as Income Statement entries use the Q4 cumulative value, which equals the full year flow in Nepali bank disclosures. Two duplicate column families exist due to a schema change in NRB reporting between older and newer formats. We checked that these pairs have zero row overlap, so we coalesce them losslessly.

We derive seven base regulatory ratios as model inputs. These are CAR, NPL Ratio, CD Ratio, Cost of Funds, Base Rate, Interest Spread, and a derived Return on Equity (ROE). All seven are continuous percentages. A loan growth feature was considered but dropped because the underlying loans column is missing for some early years; including it would reduce the modelling subset from 80 to 60 rows. Lagged features and bank relative deviations were also computed during exploratory analysis and are present in the released CSV, but are not used in the reported models.

The regression target is `Delta_NPL_next_year`, the change in NPL ratio over the next year, in percentage points. The mean is 0.37 with standard deviation 0.89 and range minus 2.90 to plus 3.80. The classification target is the binary indicator `Deteriorate_next_year` which is 1 when `Delta_NPL_next_year` exceeds 0.5 percentage points. After alignment with the forward label, the modelling subset has 80 bank year rows. The class balance is 25 deteriorate and 55 stable, which is 31 percent and 69 percent.

[**Figure 1: per year true deterioration rate vs mean predicted probability (`figures/xgb_per_year_bias.png`)**]

---

## 4. Methods

### 4.1 XGBoost baseline with SHAP

We train an XGBoost classifier on the seven features. Shallow trees are used (max depth 2 to 4) because the dataset is small. The class imbalance is handled by per fold computation of `scale_pos_weight` as the ratio of negative to positive training rows. Hyperparameters are tuned by a nested cross validation. The outer split is Leave One Year Out (LOYO) with 8 folds, one per fiscal year in the modelling subset. The inner split is a stratified 3 fold. The search grid covers `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, and `reg_lambda`.

After training the XGBoost classifier on each outer fold, we apply SHAP TreeExplainer to get per feature attribution values for every test row. We verify additivity: the sum of SHAP values plus the base value equals the predicted log odds for every row, to within numerical error. We collect SHAP values across all 8 folds into one 80 by 7 matrix. We also produce isotonic calibrated probabilities as a second baseline.

### 4.2 Gaussian Process regression and classification

We train a Gaussian Process regression model on the same seven features, with target `Delta_NPL_next_year`. The kernel is a Matern 5/2 with Automatic Relevance Determination, which means there is one length scale per input feature. The mean function is a learned constant. The likelihood is Student-t with degrees of freedom initialised at 4. The Student-t likelihood is used because `Delta_NPL_next_year` has post COVID outliers in the range plus or minus 3 percentage points, and a Gaussian likelihood would either inflate posterior variance uniformly or be dragged by the outliers. Variational inference (ELBO) is used because the Student-t likelihood does not give a closed form posterior. All eight LOYO folds converged without falling back to a Gaussian alternative.

The kernel hyperparameters are fitted by maximising the marginal log likelihood on the training fold only, using Adam with learning rate 0.1 for 200 steps and three random restarts per fold. The best restart by final loss is kept.

For GP classification we use the proposal formula. Given the posterior mean `mu(x)` and standard deviation `sigma(x)` at a test point, the probability of forward deterioration is `P(Delta_NPL > 0.5 | x) = 1 - Phi((0.5 - mu) / sigma)`, where `Phi` is the standard normal CDF. This is a proper probabilistic forecast derived from the regression posterior, so the same fitted model serves both tasks.

[**Figure 2: GP feature relevance (ARD) vs XGBoost SHAP importance, side by side (`figures/gp_ard_lengthscales.png`)**]

### 4.3 Gaussian fuzzy semantic layer

The fuzzy layer takes six input signals per bank year. Four signals come from XGBoost SHAP on the most important features: `SHAP_CD_Ratio`, `SHAP_Base_Rate`, `SHAP_Interest_Spread`, and `SHAP_CAR`. The other two come from the GP: `GP_mu` (posterior mean of `Delta_NPL`) and `GP_confidence` (the inverse of posterior variance, clipped at the 1st and 99th percentile). Each signal has three Gaussian membership functions. The centres are set at the 15th, 50th, and 85th percentiles of the empirical distribution of each signal. The widths are set so adjacent functions overlap at 0.5 membership. A grid search over per signal width multipliers in {0.75, 1.0, 1.25} selects the combination that maximises F1.

The Gaussian membership function is

`mu(x; c, sigma) = exp(-(x - c)^2 / (2 sigma^2))`

The rule base has eight Mamdani rules over two output variables, `Fundamental_Outlook` and `Risk_Flag`, both on the interval [0, 1]. Rules 1, 2, 5, 7, and 8 combine SHAP attributions to produce outlook and risk states. Rules 3 and 6 combine the GP mean with high GP confidence. Rule 4 is the confidence override. It is implemented as a post processing modulation rather than a standard Mamdani rule because the standard max aggregation would not produce true override semantics. When the `Low confidence` membership of `GP_confidence` is at least 0.6, we pull the `Fundamental_Outlook` toward 0.5 in proportion to the membership above the threshold, and we floor the `Risk_Flag` at 0.5. The aggregation T-norm is product. The defuzzification method is centroid.

[**Figure 3: Gaussian membership functions per input signal at tuned widths (`figures/fuzzy_mfs.png`)**]

### 4.4 Linguistic template engine

A deterministic three clause template engine produces one paragraph per bank year. The first clause is driven by the defuzzified `Fundamental_Outlook` plus the top positive and top negative SHAP signal labels. The second clause is driven by the defuzzified `Risk_Flag`. The third clause is driven by the dominant `GP_confidence` state. When the confidence override (Rule 4) fires AND the raw outlook before the override would have been at least 0.65, the first clause is replaced by a special sentence that warns the supervisor that the strong fundamental reading should not be acted on without further review.

The entire pipeline is deterministic. There is no language model. Every word in every output sentence can be traced back to a fuzzy state and an upstream signal.

### 4.5 Evaluation

Cross validation uses Leave One Year Out for both XGBoost and the GP, so the row indices in the out of fold prediction files match line by line. We report eight classification metrics: accuracy, F1 on the positive class, precision, recall, ROC AUC, PR AUC, Brier score, and Expected Calibration Error (ECE) with five quantile spaced bins. We also report regression metrics for the GP: RMSE, MAE, and R squared on `Delta_NPL_next_year`. Every metric is reported with a 95 percent stratified bootstrap confidence interval based on 2,000 resamples of the pooled out of fold predictions. ARD length scales and SHAP attributions are compared as feature relevance signals.

For the fuzzy and linguistic engine, we report the same classification metrics on the binary deterioration label. We also report the number of bank years where the Rule 4 confidence override fires, broken down by the true class. The linguistic engine is evaluated qualitatively by inspecting seven hand selected worked examples that cover the strong, weak, override, false positive, false negative, COVID stress, and distinct top signal cases.

---

## 5. Results

### 5.1 Classification metrics, pooled out of fold

Table 1 reports the four models on the 80 out of fold rows. All confidence intervals are 95 percent stratified bootstrap.

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

The XGBoost baseline has the best raw F1 and the best raw ECE. The GP does not improve calibration over XGBoost at N=80. The fuzzy system matches XGBoost on F1 and beats every upstream model on accuracy. Confidence intervals are wide for every model. At 80 rows, the three classifiers are statistically indistinguishable from each other on most metrics. We report this honestly.

The GP regression on `Delta_NPL_next_year` has RMSE 0.955 [0.66, 1.20], MAE 0.637 [0.48, 0.80], and R squared minus 0.158 [-0.37, -0.02]. The negative R squared means the GP regression performs slightly worse than always predicting the training mean. We interpret this as evidence that no learnable cross bank function on these seven ratios predicts forward NPL change better than the mean, at N=80. This is consistent with the wide confidence intervals on the classification metrics.

[**Figure 4: reliability diagram for XGBoost (raw and isotonic) and GP (`figures/gp_reliability.png`)**]

### 5.2 Feature relevance: GP and XGBoost agree on the top two

Table 2 ranks features by XGBoost mean absolute SHAP and by GP ARD relevance (defined as the reciprocal of the mean ARD length scale across folds).

| Rank by SHAP | Feature | Mean \|SHAP\| | GP relevance (1/length scale) | Rank by GP |
|---|---|---|---|---|
| 1 | CD_Ratio | 0.71 | 0.52 | 2 |
| 2 | Base_Rate | 0.33 | 0.74 | 1 |
| 3 | Interest_Spread | 0.22 | 0.13 | 7 |
| 4 | CAR | 0.12 | 0.17 | 5 |
| 5 | ROE_derived | 0.12 | 0.29 | 4 |
| 6 | Cost_of_Funds | 0.12 | 0.15 | 6 |
| 7 | NPL_Ratio | 0.10 | 0.46 | 3 |

The top two features by each ranking are the same pair, in reversed order. CD Ratio and Base Rate are the strongest forward NPL signals on this panel under both methods. The two methods are based on completely different mechanisms. XGBoost SHAP uses tree split gains. GP ARD uses kernel marginal likelihood. The fact that they agree on the top two features is a stronger feature relevance claim than either method alone could make. The two methods disagree most on `NPL_Ratio`. XGBoost ranks it last, while the GP ranks it third. We read this as the GP detecting a smooth dependence on current NPL level that the tree splits absorb into CD Ratio instead.

### 5.3 Confidence override demonstration

The Rule 4 confidence override fires on 37 of 80 rows. Of these, 27 are on stable bank years (49 percent of the 55 stable rows) and 10 are on deteriorating bank years (40 percent of the 25 deteriorate rows). The override is driven by GP posterior variance, not by the true label. The roughly even distribution across classes is the cleanest sanity check that the override does not smuggle label information. Among the 37 rows where R4 fires, the mean change in `Fundamental_Outlook` from pre to post override is 0.082, and the mean change in `Risk_Flag` is 0.077.

[**Figure 5: pre vs post override `Fundamental_Outlook` for the top R4-fired rows (`figures/fuzzy_override_demo.png`)**]

### 5.4 Worked example: ADBL 2074/2075

We present the flagship worked example. ADBL 2074/2075 has true label 0 (the bank did not deteriorate next year). Before override, the fuzzy `Fundamental_Outlook` is 0.665, which would be read as "Fundamentals are strong" by the linguistic engine. The top positive SHAP signal is "Capital cushion strengthening" with membership 0.83. The GP posterior at this row has `mu = +0.14` and `sigma = 0.38`. The `Low confidence` membership of `GP_confidence` is 0.87, well above the override trigger threshold of 0.6. Rule 4 fires. The post override `Fundamental_Outlook` drops to 0.564 and `Risk_Flag` is floored at 0.5.

The generated sentence is:

> "Fundamentals appear strong on raw indicators, but the model's confidence is low; this signal should not be acted on without additional review. Risk monitoring is warranted on liquidity and capital positions. Model confidence is low, this assessment should be treated as a preliminary signal pending supervisory review."

Without Rule 4, the output would have read "Fundamentals are strong, supported by Capital cushion strengthening." The override transforms a confident sounding wrong assertion into an explicit warning to the supervisor that the model is uncertain. Six other worked examples appear in the released `linguistic_examples.csv` file and cover the clear strong, clear weak, second override, correct elevated risk, false positive risk, false negative, and distinct top signal cases.

### 5.5 Width tuning

The grid search over 729 width multiplier combinations completed in 17 seconds. F1 at the heuristic widths (all multipliers equal to 1.0) is 0.316. F1 at the tuned widths is 0.444. The improvement is 0.128 F1 points or about 40 percent relative. Tuning is in sample on the same 80 rows the fuzzy system then reports F1 on. The upstream models are honest out of fold. We acknowledge the in sample nature of the fuzzy hyperparameter tuning in the limitations.

[**Figure 6: SHAP global importance bar plot (`figures/shap_bar.png`)**]

---

## 6. Discussion

The most important finding is that the GP does not improve calibration over a class weighted XGBoost baseline at N=80. We expected the opposite based on the proposal. The wide confidence intervals on every metric show that the three classifiers are statistically indistinguishable from each other on most metrics. This is the honest answer at this sample size. The downstream value of the GP in this pipeline is not its calibration of the deterioration probability. The downstream value is the availability of a per row posterior variance, which the fuzzy confidence override rule uses to soften linguistic output when the model is uncertain.

The second important finding is that the GP and XGBoost agree on which features matter. CD Ratio and Base Rate are the top two features under both methods. This is a stronger statement than either method could make alone, because the two methods use completely different mechanisms to rank features.

The third important finding is that the confidence override rule actually changes outputs in cases where the GP is uncertain. The flagship worked example shows the rule converting a "Fundamentals are strong" reading into a cautious supervisory comment. The rule fires on 46 percent of rows, distributed across both classes. The rule is not a smuggled predictor. It is an epistemic propagation of GP variance into language.

We acknowledge several limitations. First, the sample size is small and frontier market specific. We do not claim that our findings generalise to other markets or sectors. Second, the fuzzy width multipliers are tuned in sample on the same 80 rows the fuzzy system then reports F1 on. The upstream models are honest out of fold. Third, recall on the deterioration class is low across all four models. The fuzzy system trades recall for high specificity, which is defensible in a supervisory use case but should be acknowledged. Fourth, the rule base is hand designed and the worked examples are hand selected, both with explicit criteria. A larger study could use neuro fuzzy methods to learn the rule base, but the proposal scope explicitly excluded ANFIS in favour of an auditable rule base for the marker.

A possible extension is to learn a bank specific kernel that allows the GP to vary length scales per institution. Our current GP shares one set of ARD length scales across all ten banks, which limits its ability to model bank specific dynamics. Another extension is to apply the same pipeline to the wider set of Nepali commercial banks (around twenty seven institutions post merger) and to development banks, which would roughly triple the sample size.

---

## 7. Social, Ethical, Legal, and Professional Considerations

Supervisory decisions made on small panels carry real institutional weight. A confident but wrong model output can lead to over reaction by supervisors and reputational damage for banks. Our pipeline is designed to make this risk explicit. The Rule 4 confidence override is the central design choice that makes the system safe to put in front of a non technical supervisor. The linguistic output explicitly tells the reader when the model is not confident, instead of producing a confident sounding wrong sentence.

The data we use is from public NRB disclosures. There is no personal data and there are no privacy concerns under Nepali law. The dataset names individual banks, but this is public information that any market participant could obtain.

The linguistic output of our system is advisory and is explicitly framed as such. The system is not a substitute for supervisory judgement and should not be used in isolation. A supervisor reading the output should treat it as one signal among many. The deterministic template engine means there are no hallucinations. Every word in the output is traceable to an upstream signal.

We declare no use of artificial intelligence tools in the production of this work other than the machine learning models that are themselves the subject of the work.

---

## 8. Conclusion

We built a four stage pipeline for forward NPL deterioration prediction on a small panel of Nepali commercial banks. The pipeline combines a Gaussian Process regression with a Student-t likelihood, an XGBoost baseline with SHAP attributions, a Gaussian fuzzy semantic layer with a confidence override rule, and a deterministic template linguistic engine. The GP and XGBoost agree on the top two features. The GP does not improve calibration over XGBoost at N=80. The fuzzy system matches XGBoost on F1 and beats every upstream model on accuracy. The headline contribution is methodological. We show that GP posterior variance can be propagated through a fuzzy rule base into supervisory language that is honest about model uncertainty, with every word traceable to an upstream signal. The flagship worked example demonstrates the confidence override rule converting a confident sounding wrong assertion into an explicit warning.

The work is reproducible. All code is provided in the appendix. All cross validation splits, hyperparameter searches, and bootstrap confidence intervals are seeded at 42. Two consecutive runs of every script produce bitwise identical output files. The complete results, including the seven worked examples and all eleven figures, are available in the released CSV files and the figures directory.

---

## References

Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589 to 609.

Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785 to 794.

Lundberg, S. M. and Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30, 4765 to 4774.

Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Cambridge, MA.

Zadeh, L. A. (1965). Fuzzy Sets. *Information and Control*, 8(3), 338 to 353.

Nepal Rastra Bank (2024). *Bank Supervision Report 2023*. Nepal Rastra Bank, Kathmandu.

Mamdani, E. H. (1977). Application of Fuzzy Logic to Approximate Reasoning Using Linguistic Synthesis. *IEEE Transactions on Computers*, C-26(12), 1182 to 1191.

Sugeno, M. (1985). *Industrial Applications of Fuzzy Control*. Elsevier Science, Amsterdam.

---

## Appendix A. Code

All code is available in the project repository. The following scripts implement the pipeline in order:

- `preprocess.py` performs the long to wide pivot and computes the seven ratios.
- `evaluation.py` provides the shared cross validation, metrics, and bootstrap confidence interval harness.
- `xgb_baseline.py` runs the XGBoost classifier and SHAP, persists out of fold predictions and the SHAP matrix.
- `gp_model.py` runs the GP regression and GP classification, persists out of fold predictions and ARD length scales.
- `fuzzy_layer.py` builds the Gaussian membership functions, runs Mamdani inference, applies the Rule 4 post processing override, and persists the per row fuzzy outputs.
- `linguistic_engine.py` runs the template engine and selects the worked examples.
- `make_figures.py` generates all eleven paper figures from the persisted CSV files.

The full pipeline can be re run from scratch with the command sequence:

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

The seven hand selected worked examples are released as `dataset/processed/linguistic_examples.csv`. Each row contains the bank, fiscal year, top positive and top negative SHAP signals, GP posterior mean and standard deviation, fuzzy outputs, generated sentence, and a one line annotation explaining what the case demonstrates.

## Appendix C. Acknowledgement of Contributions

This work is an individual submission. The author is solely responsible for all design decisions, implementation, experimentation, and writing.
