# Modeling Setup

- Dataset: [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- Prediction point: enrollment, before first-semester teaching
- Target: `Dropout = 1`, `Graduate = 0`
- Excluded outcome: 794 `Enrolled` rows because outcomes are unresolved
- Resolved cohort: 3,630 students (1,421 dropout; 2,209 graduate)
- Input features: 24 enrollment-time/contextual variables
- Held-out split: 80/20 with `random_state=42` and target stratification
- Training set: 2,904 (1,137 dropout; 1,767 graduate)
- Test set: 726 (284 dropout; 442 graduate)
- Cross-validation: 5-fold `StratifiedKFold`, shuffled with `random_state=42`, training set only
- Leakage control: splitting occurs before fitting; encoding/scaling and the estimator remain inside each scikit-learn pipeline

# Logistic Regression

Logistic Regression uses balanced class weights, one-hot encoding for nominal code fields, and standardized numeric features.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.772 | 0.017 |
| Precision (Dropout) | 0.702 | 0.023 |
| Recall (Dropout) | 0.727 | 0.044 |
| F1 (Dropout) | 0.713 | 0.025 |
| ROC-AUC | 0.848 | 0.026 |

# Random Forest

Random Forest uses 400 trees, maximum depth 12, minimum leaf size 5, balanced-subsample weights, one-hot categorical inputs, and unscaled numeric inputs.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.772 | 0.019 |
| Precision (Dropout) | 0.707 | 0.026 |
| Recall (Dropout) | 0.713 | 0.030 |
| F1 (Dropout) | 0.710 | 0.024 |
| ROC-AUC | 0.845 | 0.017 |

# Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.772 | 0.702 | 0.727 | 0.713 | 0.848 |
| Random Forest | 0.772 | 0.707 | 0.713 | 0.710 | 0.845 |

The selected model had the highest mean cross-validated dropout F1. ROC-AUC and recall were used as tie-breakers, so accuracy alone did not determine the choice. Logistic Regression offers simpler coefficient interpretation; Random Forest can capture nonlinear relationships.

# Threshold Selection

Thresholds were evaluated using out-of-fold probabilities generated only from the training set. The held-out test set was not used for model or threshold selection.

| Threshold | Precision | Recall | F1 | False positives | False negatives |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.577 | 0.870 | 0.694 | 724 | 148 |
| 0.35 | 0.607 | 0.834 | 0.702 | 614 | 189 |
| 0.40 **(selected)** | 0.651 | 0.807 | 0.721 | 493 | 219 |
| 0.45 | 0.674 | 0.768 | 0.718 | 423 | 264 |
| 0.50 | 0.701 | 0.726 | 0.714 | 352 | 311 |
| 0.55 | 0.737 | 0.682 | 0.709 | 277 | 361 |
| 0.60 | 0.771 | 0.654 | 0.708 | 221 | 393 |
| 0.65 | 0.796 | 0.608 | 0.689 | 177 | 446 |
| 0.70 | 0.831 | 0.563 | 0.671 | 130 | 497 |

- Default threshold: 0.50
- Selected threshold: **0.40**
- Reason: Selected from training out-of-fold predictions. It improved dropout recall by at least 0.02 versus 0.50, retained precision of at least 0.55, and had the best F1 among thresholds meeting those safeguards.

# Final Model

**Logistic Regression** was selected. It is saved as a complete fitted pipeline, including all preprocessing, at `models/dropout_model.joblib`.

# Held-Out Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 0.736 |
| Precision (Dropout) | 0.620 |
| Recall (Dropout) | 0.835 |
| F1 (Dropout) | 0.712 |
| ROC-AUC | 0.855 |

The selected threshold was fixed before this one-time test evaluation. There is no clear evidence of serious overfitting: held-out F1 and ROC-AUC are within 0.10 of the selected model's cross-validation means.

# Confusion Matrix Interpretation

- Correctly detected dropouts (true positives): **237**
- Missed dropouts (false negatives): **47**
- False dropout warnings for graduates (false positives): **145**
- Correctly identified graduates (true negatives): **297**

# Feature Interpretation

The plot and table aggregate one-hot category contributions back to their original source fields where practical. Larger values mean the feature was more influential in the fitted model's predictions; they do not mean the feature caused dropout. High-cardinality variables can receive more total Random Forest importance.

| Rank | Original feature | Association measure |
|---:|---|---:|
| 1 | `Tuition fees up to date` | 1.5030 |
| 2 | `Scholarship holder` | 0.6790 |
| 3 | `Course` | 0.5827 |
| 4 | `Debtor` | 0.5036 |
| 5 | `Application mode` | 0.4352 |
| 6 | `Marital status` | 0.3856 |
| 7 | `Nacionality` | 0.3801 |
| 8 | `Mother's occupation` | 0.3582 |
| 9 | `Mother's qualification` | 0.3503 |
| 10 | `Gender` | 0.3497 |

For Logistic Regression, positive coefficients are associated with a higher predicted dropout log-odds and negative coefficients with lower predicted dropout log-odds, holding other encoded inputs constant. These are associations, not causal effects.

| Strong positive encoded term | Coefficient | Strong negative encoded term | Coefficient |
|---|---:|---|---:|
| `Course=9119` | 1.639 | `Tuition fees up to date=1` | -1.441 |
| `Tuition fees up to date=0` | 1.565 | `Mother's occupation=191` | -1.375 |
| `Mother's occupation=0` | 1.458 | `Application mode=15` | -1.069 |
| `Application mode=7` | 1.211 | `Nacionality=26` | -1.035 |
| `Course=9853` | 1.162 | `Course=9500` | -0.989 |
| `Nacionality=109` | 1.046 | `Course=9003` | -0.926 |
| `Application mode=39` | 1.034 | `Application mode=16` | -0.918 |
| `Father's occupation=90` | 0.947 | `Mother's qualification=4` | -0.901 |

# Limitations

- Data comes from one Portuguese higher-education institution and may not generalize to Bangladesh or other institutions.
- The dataset combines multiple study programs rather than one Computer Science program.
- Dropout dates are unavailable, which prevents a validated post-semester future-outcome cutoff.
- Socioeconomic and economic-context variables can change across countries and cohorts.
- Sensitive and proxy variables require subgroup monitoring and careful support-oriented governance.
- A random split cannot test future-cohort drift because enrollment dates are unavailable.
- The model estimates statistical risk, not certainty or causation.
