# Final Binary Elevated-Risk Experiment

## Outcome

The binary target is derived from the same self-reported question:

- **Not Elevated Risk (0):** response 1, 2, or 3
- **Elevated Risk (1):** response 4 or 5

This remains a measure of self-reported dropout consideration due to academic stress. It is not confirmed future dropout.

After applying the existing cleaning rules, the dataset contains 274 Not Elevated responses and 77 Elevated responses. Elevated prevalence is 21.94%, which is also the no-skill Average Precision baseline.

## Validation Design

The stratified holdout was separated before model comparison. The 280-record training partition was evaluated with 5-fold, 5-repeat `RepeatedStratifiedKFold`. Every imputer, encoder, scaler, and estimator was fitted inside its fold. Timestamp, the original target and normalized target score, and academic-overwhelm frequency were excluded.

All 20 leakage-safe candidate predictors were used. No feature reduction was performed because the full model did not pass the decision gate.

## Repeated-CV Comparison

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Logistic Regression | 0.623 ± 0.062 | 0.343 ± 0.066 |
| Random Forest | **0.653 ± 0.065** | **0.383 ± 0.079** |
| XGBoost | 0.635 ± 0.075 | 0.374 ± 0.080 |

Random Forest was selected using repeated-CV Average Precision, not accuracy. Its averaged out-of-fold probabilities had ROC-AUC 0.657 and Average Precision 0.344.

## Training-Only Threshold Analysis

| Threshold | Precision | Recall | F1 | TP | FP | FN | TN | Predicted positive |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.219 | 1.000 | 0.360 | 61 | 217 | 0 | 2 | 99.3% |
| 0.20 | 0.227 | 0.984 | 0.369 | 60 | 204 | 1 | 15 | 94.3% |
| 0.25 | 0.249 | 0.984 | 0.397 | 60 | 181 | 1 | 38 | 86.1% |
| 0.30 | 0.254 | 0.803 | 0.386 | 49 | 144 | 12 | 75 | 68.9% |
| 0.35 | 0.263 | 0.689 | 0.380 | 42 | 118 | 19 | 101 | 57.1% |
| **0.40** | **0.302** | **0.574** | **0.395** | **35** | **81** | **26** | **138** | **41.4%** |
| 0.45 | 0.343 | 0.410 | 0.373 | 25 | 48 | 36 | 171 | 26.1% |
| 0.50 | 0.436 | 0.279 | 0.340 | 17 | 22 | 44 | 197 | 13.9% |
| 0.55 | 0.353 | 0.098 | 0.154 | 6 | 11 | 55 | 208 | 6.1% |
| 0.60 | 0.400 | 0.033 | 0.061 | 2 | 3 | 59 | 216 | 1.8% |

Threshold 0.40 was selected from training predictions because it offered the best F1 among thresholds that flagged no more than half the cohort. Lower thresholds produced an unusable false-positive burden.

## Selected CV Operating Point

- Precision: 0.302
- Recall: 0.574
- F1: 0.395
- Balanced accuracy: 0.602
- Accuracy: 0.618
- Confusion matrix: TN 138, FP 81, FN 26, TP 35
- Predicted-positive rate: 41.4%
- ROC-AUC: 0.657
- Average Precision: 0.344

## Untouched Holdout Sanity Check

At the locked 0.40 threshold:

- Precision: 0.179
- Recall: 0.313
- F1: 0.227
- Balanced accuracy: 0.447
- Accuracy: 0.521
- Confusion matrix: TN 32, FP 23, FN 11, TP 5
- Predicted-positive rate: 39.4%
- ROC-AUC: 0.427
- Average Precision: 0.224

Holdout Average Precision is effectively at the 0.219 no-skill baseline, and ROC-AUC is below chance. The apparent repeated-CV signal did not reproduce.

## Full Versus Reduced Features

The full 20-feature candidate model failed the required stability and holdout-agreement gate. Under the stated protocol, feature reduction was permitted only after meaningful full-model signal was established. Therefore no reduced 8–12 feature model was fit, no final website feature set was approved, and no binary model artifact was saved.

## Final Decision

**DO NOT BUILD PREDICTIVE CLASSIFIER**

Recommended project title:

**Machine Learning-Based Analysis and Profiling of Student Dropout Risk Factors among Bangladeshi University Students**

The project can still present rigorous cleaning, descriptive profiling, leakage analysis, and transparent failed-model evaluation. It should not claim reliable individual prediction.
