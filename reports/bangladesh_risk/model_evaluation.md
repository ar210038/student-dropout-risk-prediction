# Bangladesh Dropout-Risk Model Evaluation

## Model Comparison

Results are mean ± standard deviation across 25 training-only repeated-stratified validation folds.

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Multinomial Logistic Regression | 0.382 ± 0.042 | 0.382 ± 0.048 | 0.374 ± 0.048 | 0.382 ± 0.048 | 0.369 ± 0.046 | 0.383 ± 0.044 |
| Random Forest | 0.390 ± 0.053 | 0.373 ± 0.055 | 0.371 ± 0.063 | 0.373 ± 0.055 | 0.366 ± 0.057 | 0.388 ± 0.055 |
| XGBoost | 0.381 ± 0.053 | 0.384 ± 0.059 | 0.376 ± 0.055 | 0.384 ± 0.059 | **0.372 ± 0.056** | 0.381 ± 0.055 |

## Per-Class Repeated-CV Results

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Logistic Regression | Low Risk | 0.497 ± 0.077 | 0.383 ± 0.089 | 0.429 ± 0.079 |
| Logistic Regression | Moderate Risk | 0.334 ± 0.059 | 0.380 ± 0.119 | 0.351 ± 0.081 |
| Logistic Regression | High Risk | 0.291 ± 0.094 | 0.383 ± 0.129 | 0.327 ± 0.101 |
| Random Forest | Low Risk | 0.490 ± 0.061 | 0.450 ± 0.098 | 0.465 ± 0.072 |
| Random Forest | Moderate Risk | 0.320 ± 0.081 | 0.350 ± 0.118 | 0.329 ± 0.084 |
| Random Forest | High Risk | 0.301 ± 0.102 | 0.317 ± 0.122 | 0.303 ± 0.100 |
| XGBoost | Low Risk | 0.483 ± 0.072 | 0.371 ± 0.089 | 0.416 ± 0.078 |
| XGBoost | Moderate Risk | 0.332 ± 0.072 | 0.380 ± 0.093 | 0.351 ± 0.073 |
| XGBoost | High Risk | **0.313 ± 0.082** | **0.402 ± 0.128** | **0.348 ± 0.093** |

## Selection and Sanity Check

XGBoost ranked first by the required selection metric, Macro F1, and also produced the strongest High Risk precision, recall, and F1 in repeated CV. It is therefore the selected experimental model.

The 71-response holdout was evaluated once as a sanity check:

- Accuracy: 0.423
- Balanced accuracy: 0.412
- Macro precision: 0.407
- Macro recall: 0.412
- Macro F1: 0.400
- Weighted F1: 0.424
- High Risk precision: 0.222
- High Risk recall: 0.250
- Confusion matrix (rows actual, columns predicted Low/Moderate/High): `[[13, 10, 10], [5, 13, 4], [5, 7, 4]]`

The holdout contains only 16 High Risk responses, so its class metrics are highly uncertain. It does not override the repeated-CV result.

## Associated Features

Training-only validation-fold permutation importance identified the following features as most positively associated with dropout-risk classification performance:

1. Employment / tuition work
2. Scholarship / stipend
3. Attendance
4. Study-material satisfaction
5. Academic level

The remaining selected features had near-zero or negative mean permutation importance with standard deviations overlapping zero: sleep duration, commute time, academic-resource access, GPA, study-space quality, study routine, and internet quality. This instability is important evidence that the sample is too small for dependable individual-level prediction. None of these associations is causal.

## Performance and Artifact Decision

The selected model's repeated-CV Macro F1 is only 0.372 ± 0.056, close to a weak three-class baseline, and High Risk recall varies substantially. The holdout High Risk recall is 0.25. This is not sufficiently dependable for a live risk-assessment website.

Accordingly, the trained estimator was **not saved** as `models/bangladesh_risk_model.joblib`, and no final model metadata artifact was created. The existing active Portuguese model remains unchanged. The code, cleaned dataset, results, and confusion matrix are retained for reproducible academic analysis.

The project remains valid as a dataset-analysis and modeling experiment, but the model should not be presented as a reliable student-screening instrument.
