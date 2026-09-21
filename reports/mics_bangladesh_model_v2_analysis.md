# Bangladesh MICS Dropout Model v2: Grouped Validation Report

## Decision: DO NOT MIGRATE

The controlled v2 experiment improved discrimination and dropout recall, but the practical alert burden remains unacceptable for a public-facing tool. At the training-selected threshold, the PSU-grouped test recalled 74.7% of dropout transitions but flagged 39.3% of all children, with 5.3% precision and 2,314 false positives for 130 true positives. Probabilities are also poorly calibrated.

The target and cohort are unchanged from v1. `app.py` and the active Portuguese UCI model were not modified.

## Controlled changes from v1

- Added one defensible feature: father's education from `hl.sav`.
- Replaced random row splitting with PSU-disjoint held-out and cross-validation splits.
- Added Average Precision/PR-AUC, Brier score, PR curve, and calibration diagnostics.
- Compared only the predeclared Logistic Regression, regularized Random Forest, and one conservative XGBoost configuration.
- Selected the operating threshold using grouped out-of-fold training probabilities only.

## Educational-progress review

| Concept | Official variable | Timing | Decision |
|---|---|---|---|
| Previous education level | `fs.sav: CB10A` | Level attended in previous school year | Keep; explicitly predates current-year outcome. |
| Previous/last grade | `fs.sav: CB10B` | Grade attended in previous school year | Keep; explicitly predates current-year outcome. |
| Highest level/grade ever attended | `fs.sav: CB5A`, `CB5B` | Measured at survey time | Exclude; may incorporate current-year progress for continuing students. |
| Grade completion | `fs.sav: CB6` | Whether highest grade/year had ever been completed, measured at survey time | Exclude; not tied to the previous-year cutoff and may contain post-outcome information. |
| Father's education | `hl.sav: felevel` | Derived parental background | Keep after audited one-to-one child-roster merge. |

## Father's education merge

Father's education is not directly present in `fs.sav`. It was obtained from the child's corresponding household-member row in `hl.sav`:

```text
fs.sav (HH1, HH2, FS3/LN) -> hl.sav (HH1, HH2, HL1)
```

- `LN` equalled `FS3` for all 40,617 `fs.sav` records.
- Both source keys were unique.
- The merge used `validate="one_to_one"`.
- All 31,101 cohort children matched exactly once.
- Child rows after merge: 31,101; unmatched: 0; duplicate expansion: 0.
- `felevel` labels: pre-primary/none, primary, secondary, higher secondary+, no information, and missing/DK.
- 5,140 records have the explicit `No information` category; 14 `Missing/DK` values are imputed within training folds.

The merge adds no current-attendance or post-dropout variable.

## Final feature set

1. Age (`CB3`)
2. Sex (`HL4`)
3. Division (`HH7`)
4. Urban/Rural Area (`HH6`)
5. Previous School Level (`CB10A`)
6. Previous Grade (`CB10B`)
7. Mother's Education (`melevel`)
8. Father's Education (`hl.sav: felevel`)
9. Household Wealth Quintile (`windex5`)
10. Functional Difficulty (`fsdisability`)
11. Children Aged 5–17 in Household (`HH52`)

No current-year attendance, current grade/level, child labour, current literacy, current homework, survey identifier, PSU, stratum, or weight is a predictor.

## PSU-grouped split

| Quantity | Result |
|---|---:|
| Total cohort | 31,101 |
| Total PSUs | 3,220 |
| Training rows | 24,877 |
| Test rows | 6,224 |
| Training PSUs | 2,576 |
| Test PSUs | 644 |
| PSU overlap | 0 |
| Training dropout cases | 792 |
| Test dropout cases | 174 |
| Training prevalence | 3.184% |
| Test prevalence/no-skill AP | 2.796% |

`StratifiedGroupKFold` created five training folds with no PSU overlap. Fold validation prevalence ranged from 3.155% to 3.213%.

## Grouped cross-validation

Metrics are mean ± standard deviation at threshold 0.50. Average Precision is compared with the approximately 3.18% training prevalence baseline.

| Model | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.070 ± 0.002 | 0.662 ± 0.026 | 0.127 ± 0.004 | 0.739 ± 0.016 | 0.103 ± 0.020 |
| Random Forest | 0.115 ± 0.009 | 0.506 ± 0.047 | 0.188 ± 0.015 | 0.762 ± 0.017 | 0.153 ± 0.036 |
| XGBoost | 0.088 ± 0.005 | 0.544 ± 0.038 | 0.152 ± 0.009 | 0.748 ± 0.015 | 0.143 ± 0.026 |

Random Forest was selected because it had the strongest grouped-CV Average Precision, ROC-AUC, F1, and precision. Logistic Regression had higher recall but materially worse precision and PR-AUC. XGBoost did not outperform the regularized Random Forest, so no further model search was performed.

## Threshold analysis

The following table uses Random Forest out-of-fold probabilities from the grouped training folds only.

| Threshold | Precision | Recall | F1 | FP | FN | Predicted positive |
|---:|---:|---:|---:|---:|---:|---:|
| 0.100 | 0.032 | 0.999 | 0.063 | 23,665 | 1 | 98.3% |
| 0.125 | 0.033 | 0.992 | 0.064 | 22,813 | 6 | 94.9% |
| 0.150 | 0.035 | 0.986 | 0.067 | 21,619 | 11 | 90.0% |
| 0.175 | 0.037 | 0.975 | 0.071 | 20,284 | 20 | 84.6% |
| 0.200 | 0.038 | 0.952 | 0.074 | 18,941 | 38 | 79.2% |
| 0.225 | 0.040 | 0.924 | 0.077 | 17,501 | 60 | 73.3% |
| 0.250 | 0.042 | 0.886 | 0.081 | 15,871 | 90 | 66.6% |
| 0.275 | 0.046 | 0.854 | 0.087 | 14,066 | 116 | 59.3% |
| 0.300 | 0.051 | 0.822 | 0.096 | 12,136 | 141 | 51.4% |
| 0.325 | 0.056 | 0.779 | 0.105 | 10,328 | 175 | 44.0% |
| **0.350** | **0.063** | **0.745** | **0.116** | **8,754** | **202** | **37.6%** |
| 0.375 | 0.070 | 0.697 | 0.127 | 7,351 | 240 | 31.8% |
| 0.400 | 0.078 | 0.657 | 0.140 | 6,119 | 272 | 26.7% |
| 0.425 | 0.087 | 0.616 | 0.153 | 5,114 | 304 | 22.5% |
| 0.450 | 0.095 | 0.569 | 0.163 | 4,288 | 341 | 19.0% |
| 0.475 | 0.105 | 0.535 | 0.175 | 3,625 | 368 | 16.3% |
| 0.500 | 0.115 | 0.506 | 0.188 | 3,075 | 391 | 14.0% |
| 0.525 | 0.123 | 0.458 | 0.194 | 2,594 | 429 | 11.9% |
| 0.550 | 0.131 | 0.426 | 0.200 | 2,245 | 455 | 10.4% |
| 0.575 | 0.138 | 0.388 | 0.204 | 1,914 | 485 | 8.9% |
| 0.600 | 0.148 | 0.360 | 0.209 | 1,645 | 507 | 7.8% |
| 0.625 | 0.159 | 0.332 | 0.215 | 1,393 | 529 | 6.7% |
| 0.650 | 0.170 | 0.298 | 0.216 | 1,156 | 556 | 5.6% |
| 0.675 | 0.183 | 0.259 | 0.214 | 917 | 587 | 4.5% |
| 0.700 | 0.203 | 0.236 | 0.218 | 734 | 605 | 3.7% |

Threshold 0.350 was selected because it was the highest-precision point in the specified grid that still achieved training OOF recall of at least 0.70. It was not chosen by maximizing F1. The table shows that no threshold produces both high recall and practical precision: achieving approximately 70% recall requires flagging more than one-third of children.

## PSU-grouped held-out test

| Metric | Result |
|---|---:|
| Accuracy | 0.621 |
| Precision | 0.053 |
| Recall | 0.747 |
| F1 | 0.099 |
| ROC-AUC | 0.761 |
| Average Precision / PR-AUC | 0.125 |
| No-skill Average Precision | 0.028 |
| Predicted-positive rate | 39.27% |
| Brier score | 0.1403 |

Confusion matrix:

| | Predicted continue | Predicted dropout |
|---|---:|---:|
| Actual continue | TN = 3,736 | FP = 2,314 |
| Actual dropout | FN = 44 | TP = 130 |

Average Precision is approximately 4.5 times the test prevalence baseline, so the ranking contains useful signal. Operational precision is nevertheless only 5.3%, meaning roughly 19 children must be flagged to identify one observed dropout transition.

## Calibration

The Brier score is 0.1403, and the calibration curve shows substantial overprediction: predicted probabilities are much higher than observed dropout frequencies. This is expected in part from balanced class weighting. No calibration was applied because that would require a separately validated training-only calibration design and would add complexity without solving the weak precision/alert-burden problem.

## Associated features

Random Forest one-hot importances aggregated to original features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Previous Grade | 0.226 |
| 2 | Age | 0.189 |
| 3 | Father's Education | 0.126 |
| 4 | Division | 0.093 |
| 5 | Previous School Level | 0.081 |
| 6 | Household Wealth Quintile | 0.073 |
| 7 | Mother's Education | 0.069 |
| 8 | Sex | 0.066 |
| 9 | Children Aged 5–17 in Household | 0.034 |
| 10 | Urban/Rural Area | 0.023 |
| 11 | Functional Difficulty | 0.020 |

These are associations with predictions, not causal effects.

## Subgroup performance

### Sex and area

| Group | N | Dropouts | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| Female | 3,170 | 88 | 0.071 | 0.739 | 0.130 | 28.9% |
| Male | 3,054 | 86 | 0.043 | 0.756 | 0.081 | 49.8% |
| Rural | 4,969 | 145 | 0.054 | 0.772 | 0.100 | 41.6% |
| Urban | 1,255 | 29 | 0.051 | 0.621 | 0.095 | 28.2% |

Male children are flagged far more often than female children despite similar prevalence, while urban recall is substantially lower than rural recall.

### Wealth quintile

| Group | N | Dropouts | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| Poorest | 1,337 | 48 | 0.064 | 0.917 | 0.120 | 51.4% |
| Second | 1,350 | 36 | 0.045 | 0.750 | 0.085 | 44.5% |
| Middle | 1,311 | 34 | 0.048 | 0.765 | 0.090 | 41.7% |
| Fourth | 1,175 | 27 | 0.038 | 0.593 | 0.072 | 35.4% |
| Richest | 1,051 | 29 | 0.088 | 0.586 | 0.153 | 18.4% |

Flagging rates range from 18.4% for the richest group to 51.4% for the poorest group. This creates a serious socioeconomic fairness and intervention-capacity concern.

### Division

Division recall ranges from 0.615 to 0.813 and precision from 0.025 to 0.076. Rangpur has only 9 dropout cases and Mymensingh only 13, so these estimates are unstable and should not be interpreted as proof of fairness or regional causation.

## Comparison with v1

| Metric | v1 random split | v2 PSU-grouped split |
|---|---:|---:|
| Precision | 0.113 | 0.053 |
| Recall | 0.301 | 0.747 |
| F1 | 0.164 | 0.099 |
| ROC-AUC | 0.714 | 0.761 |
| False positives | 456 | 2,314 |
| Predicted-positive rate | 8.3% | 39.3% |

The experiments use different held-out groups and thresholds, so the comparison is not perfectly paired. V2 materially improves recall and ranking discrimination, but not practical overall performance: precision and F1 fall sharply, and false-positive burden increases approximately fivefold.

## Saved artifacts

- `models/mics_bangladesh_dropout_model_v2.joblib`
- `models/mics_bangladesh_model_metadata_v2.json`
- `reports/mics_training_results_v2.json`
- `reports/mics_figures/precision_recall_curve.png`
- `reports/mics_figures/calibration_curve.png`
- `reports/mics_figures/confusion_matrix_v2.png`
- `reports/mics_figures/roc_curve_v2.png`
- `reports/mics_figures/feature_importance_v2.png`

The model is labelled candidate-only and is not loaded by the Streamlit application.

## Migration gate

**DO NOT MIGRATE.** The v2 model finds substantially more dropout cases, but doing so requires flagging nearly 40% of children, produces 2,314 false positives in 6,224 test records, has only 5.3% precision, and is poorly calibrated. It may support continued research, but it is not ready for a student-facing or operational risk system.
