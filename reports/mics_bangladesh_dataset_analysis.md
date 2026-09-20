# Bangladesh MICS 2019 Dropout Dataset and Model Analysis

> A controlled PSU-grouped v2 experiment with father education and XGBoost is documented in `reports/mics_bangladesh_model_v2_analysis.md`. Its decision is **DO NOT MIGRATE**.

## Decision

The target is a defensible school-attendance transition outcome, and the dataset is suitable for a university research prototype. However, the candidate model is **not strong enough to justify migrating the deployed website yet**. On the untouched test set, the selected Random Forest identified 58 of 193 dropout cases (30.1% recall), while 456 continuing students were false positives (11.3% precision).

The current Portuguese UCI application and model remain unchanged. The Bangladesh model is saved under a separate candidate filename for review only.

# Dataset Source

- Survey: Bangladesh Multiple Indicator Cluster Survey 2019, Round 6
- Reference ID: `BGD_2019_MICS_v01_M`
- Producers: Bangladesh Bureau of Statistics and UNICEF
- Official catalog: <https://microdata.worldbank.org/catalog/4147>
- UNICEF report: <https://www.unicef.org/bangladesh/en/reports/progotir-pathey-bangladesh>
- Original package: `BGD_2019_MICS6_v01_M.zip`
- `fs.sav` SHA-256: `029281044f533c87279e1da19b4475177e35ea6760a2fcffa8093d181ffcbbea`
- Raw files remain unchanged under `data/mics_bangladesh_2019/raw/`.

The package README states that the datasets are distributed for legitimate research after disclosure of the research purpose and asks users to send resulting reports/publications to the named Bangladesh Bureau of Statistics and UNICEF contacts.

# Files Used

| File | Use |
|---|---|
| `Read me_Bangladesh_MICS6.txt` | Access conditions, file inventory, and data-unit descriptions |
| `fs.sav` | Primary child-level dataset and sole modeling source |

No merge with `hh.sav` or `hl.sav` was required. This avoids one-to-many merge errors and duplicated child records.

`fs.sav` contains **40,617 rows and 268 columns**. It represents selected children aged 5–17.

# Attendance Variables

SPSS metadata was read with `pyreadstat` without overwriting or converting the source file.

## CB7

- Variable label: **Attended school or early childhood programme during current school year**
- Value labels: `1 = YES`, `2 = NO`, `9 = NO RESPONSE`
- Observed counts: Yes 33,569; No 4,356; system-missing 2,692; observed code 9: 0

## CB9

- Variable label: **Attended school or early childhood programme during previous school year**
- Value labels: `1 = YES`, `2 = NO`, `9 = NO RESPONSE`
- Observed counts: Yes 31,196; No 6,729; system-missing 2,692; observed code 9: 0

The 2,692 records missing both attendance variables follow questionnaire/interview eligibility patterns rather than representing observed No responses. They were not classified as dropouts.

# Exact Dropout Definition

For completed interviews only:

```text
Dropout (1):  CB9 = 1 (previous-year Yes) AND CB7 = 2 (current-year No)
Continued (0): CB9 = 1 (previous-year Yes) AND CB7 = 1 (current-year Yes)
```

Children who did not attend in the previous year, had unknown/system-missing attendance, or did not have a completed child interview were excluded. Never-enrolled/non-previous-year attendees were not mislabeled as dropouts.

This is a genuine observed transition from attendance in the 2018 school year to non-attendance in the 2019 school year. Current attendance is recorded separately from the background predictors. It remains a survey-reported transition rather than an institutional withdrawal record: there is no exact dropout date, reason, or confirmation that non-attendance is permanent.

# Cohort Construction

Sequential exclusions were:

| Step | Records excluded | Records remaining |
|---|---:|---:|
| Initial `fs.sav` records | — | 40,617 |
| Incomplete/non-completed child interview (`FS17 != 1`) | 1,231 | 39,386 |
| Previous attendance unknown/system-missing among completed interviews | 1,574 | 37,812 |
| Did not attend in previous school year (`CB9 = 2`) | 6,711 | 31,101 |
| Current attendance unknown among previous-year attendees | 0 | 31,101 |

Final modeling cohort: **31,101 children**.

# Class Distribution

| Class | Unweighted count | Unweighted % | Survey-weighted % |
|---|---:|---:|---:|
| Continued attendance | 30,135 | 96.894% | 96.848% |
| Dropout transition | 966 | 3.106% | 3.152% |

The target is strongly imbalanced. Models used class weights; SMOTE was not used.

# Survey Design

The following official design fields are present:

- `fsweight`: child age 5–17 survey weight
- `stratum`: sampling stratum, 128 represented in the cohort
- `PSU`: primary sampling unit, 3,220 represented in the cohort
- `HH1`: cluster number

Survey weights were used for descriptive class prevalence. Classifier training and predictive metrics were unweighted because the goal is a conventional ML prototype and ordinary scikit-learn cross-validation does not constitute full complex-survey inference. Therefore, the reported accuracy and ROC-AUC must not be described as design-corrected nationally representative model performance.

The split is random and stratified rather than PSU-grouped. Children from similar sampled communities may occur in both training and test data, so future work should add PSU-grouped sensitivity validation.

# Candidate Features

| Website label | MICS variable | Meaning | Type | Keep? | Reason |
|---|---|---|---|---|---|
| Age | `CB3` | Child age in completed years | Numeric | Keep | Stable background characteristic; 5–17 range verified. |
| Sex | `HL4` | Male/Female label | Categorical, sensitive | Keep with audit | Basic demographic association; subgroup performance is reported. |
| Division | `HH7` | Bangladesh administrative division | Categorical/geographic | Keep with caution | Understandable regional context; may encode unequal services and location. |
| Area | `HH6` | Urban/Rural residence | Categorical/geographic | Keep with caution | Stable context; geographic portability must be monitored. |
| Previous School Level | `CB10A` | Education level attended in the previous school year | Categorical | Keep | Predates the attendance-transition outcome. |
| Previous Grade | `CB10B` | Grade attended in the previous school year | Categorical | Keep | Predates the target; ECE has a documented not-applicable category. |
| Mother's Education | `melevel` | Derived maternal education level | Ordinal categorical | Keep | Stable household-background feature already present in `fs.sav`. |
| Household Economic Group | `windex5` | Wealth-index quintile | Ordinal categorical, sensitive | Keep with audit | Relevant economic context; association only, never causation. |
| Functional Difficulty / Disability | `fsdisability` | Child functional-difficulty indicator | Categorical, highly sensitive | Keep only for supportive research | Potential support need; must never be used punitively. |
| Children Aged 5–17 in Household | `HH52` | Number of school-age children in household | Numeric | Keep | Stable, understandable household characteristic. |
| District | `HH7A` | District | High-cardinality geographic | Exclude | High memorization/sparsity risk; division is simpler. |
| Father's Education | Not available directly | Proposed parental education feature | — | Exclude | No verified direct `fs.sav` field; unnecessary merge avoided. |
| Child Labour / Work | `CL*` | Work during week near survey | Current/post-outcome | Exclude | Could be a consequence of leaving school. |
| Current literacy/homework | `FL*`, `PR*` | Current assessments and school participation | Current/post-outcome | Exclude | Timing and skip patterns may reveal or follow dropout. |

# Final Recommended Feature Set

Ten proposed website inputs:

1. Age
2. Sex
3. Division
4. Urban/Rural Area
5. Previous School Level
6. Previous Grade
7. Mother's Education
8. Household Economic Group
9. Functional Difficulty / Disability
10. Number of Children Aged 5–17 in the Household

All current attendance variables and target-derived fields are absent from the model pipeline.

# Leakage Assessment

Mandatory exclusions include:

- `CB7`, because it directly defines the target.
- `CB8A` and `CB8B`, because current level/grade occur only for current attendees and their missingness reveals the target.
- `CB9` from predictors, because it defines cohort eligibility.
- All variables derived from current attendance.
- Current labour, work, homework, and assessment variables, because they could be consequences of dropout or have target-dependent skip patterns.
- Survey weights, PSU, stratum, household/child IDs, and interviewer fields.

The retained education variables (`CB10A`, `CB10B`) explicitly refer to the previous school year and therefore precede the current-year attendance outcome.

# Data Quality

- Exact duplicate final-cohort rows: 0
- Duplicate `HH1` + `HH2` + `LN` child keys: 0
- One `CB10A = 9` No Response becomes missing and is imputed inside each training fold.
- `CB10B` is structurally missing for 3,409 ECE records; these receive an explicit `Not applicable (ECE)` category rather than an invented grade.
- All other final input values are complete after excluding incomplete interviews.
- Rare observed categories include previous level Higher (9 records), Grade 13 (8), and Grade 14 (1). `OneHotEncoder(handle_unknown="ignore")` prevents failures, but effects for rare categories are unreliable.
- `HH52` ranges from 1 to 10; only 9 records exceed 6 children aged 5–17.
- `fsweight` is complete and positive in the cohort.
- Preprocessing is entirely inside scikit-learn pipelines. Numeric fields use training-fold median imputation and scaling; categorical fields use training-fold most-frequent imputation and one-hot encoding.

# Modeling Design

- Random state: 42
- Split: stratified 80% training / 20% untouched test
- Training rows: 24,880
- Test rows: 6,221
- Cross-validation: 5-fold shuffled `StratifiedKFold`, training set only
- Positive class: Dropout = 1
- Both models: balanced class weights
- Random Forest controls: 400 trees, `max_depth=10`, `min_samples_leaf=15`

## Cross-validation at threshold 0.50

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.699 ± 0.004 | 0.064 ± 0.002 | 0.634 ± 0.013 | 0.116 ± 0.003 | 0.728 ± 0.014 |
| Random Forest | 0.829 ± 0.004 | 0.091 ± 0.004 | 0.501 ± 0.022 | 0.154 ± 0.007 | 0.747 ± 0.017 |

Random Forest was selected because it had better precision, F1, ROC-AUC, and overall false-positive burden, with stable fold variation. Logistic Regression had higher recall but generated substantially more false positives and lower F1. Selection was not based on accuracy alone.

## Threshold selection

Thresholds 0.20–0.60 were evaluated exclusively with Random Forest out-of-fold training probabilities. Threshold **0.60** produced the highest training OOF F1 in the specified grid:

- Accuracy: 0.905
- Precision: 0.120
- Recall: 0.323
- F1: 0.175
- ROC-AUC: 0.746

This threshold reduces false-positive burden but sacrifices recall. It is acceptable for comparing a research candidate, not for operational intervention without stakeholder review. Lower thresholds can recover more dropout cases but produce very low precision.

## Held-out test results

The test set was evaluated after model and threshold selection:

| Metric | Result |
|---|---:|
| Accuracy | 0.905 |
| Precision | 0.113 |
| Recall | 0.301 |
| F1 | 0.164 |
| ROC-AUC | 0.714 |

Confusion matrix:

| | Predicted continue | Predicted dropout |
|---|---:|---:|
| Actual continue | TN = 5,572 | FP = 456 |
| Actual dropout | FN = 135 | TP = 58 |

Accuracy is misleadingly high because 96.9% of the cohort continues attending. The model misses roughly 70% of held-out dropout transitions and only about 1 in 9 flagged children is a true dropout case.

## Overfitting assessment

At threshold 0.60, training OOF F1 was 0.175 versus held-out F1 0.164; OOF accuracy was 0.905 versus held-out 0.905. ROC-AUC decreased from 0.746 to 0.714. This is not serious overfitting, but it indicates modest generalization loss and overall limited predictive signal.

# Feature Interpretation

Random Forest one-hot importances were aggregated back to their original fields. These are variables **associated with model predictions**, not causes of dropout.

| Rank | Original feature | Aggregated importance |
|---:|---|---:|
| 1 | Previous Grade | 0.2421 |
| 2 | Age | 0.2285 |
| 3 | Division | 0.1128 |
| 4 | Previous School Level | 0.1015 |
| 5 | Mother's Education | 0.0876 |
| 6 | Household Economic Group | 0.0847 |
| 7 | Sex | 0.0629 |
| 8 | Children Aged 5–17 in Household | 0.0371 |
| 9 | Urban/Rural Area | 0.0262 |
| 10 | Functional Difficulty / Disability | 0.0167 |

# Fairness and Subgroup Analysis

Held-out metrics at threshold 0.60:

## Sex and area

| Group | N | Dropout prevalence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Female | 3,222 | 2.79% | 0.122 | 0.333 | 0.179 |
| Male | 2,999 | 3.43% | 0.104 | 0.272 | 0.151 |
| Rural | 5,041 | 3.15% | 0.112 | 0.321 | 0.166 |
| Urban | 1,180 | 2.88% | 0.121 | 0.206 | 0.152 |

Urban recall is materially lower than rural recall. Male recall is also lower than female recall.

## Division

| Division | N | Dropout cases | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Barishal | 600 | 15 | 0.140 | 0.533 | 0.222 |
| Chattogram | 1,091 | 41 | 0.141 | 0.317 | 0.195 |
| Dhaka | 1,197 | 38 | 0.125 | 0.237 | 0.164 |
| Khulna | 956 | 38 | 0.116 | 0.289 | 0.165 |
| Mymensingh | 360 | 10 | 0.075 | 0.300 | 0.120 |
| Rajshahi | 770 | 18 | 0.069 | 0.222 | 0.105 |
| Rangpur | 808 | 18 | 0.105 | 0.333 | 0.160 |
| Sylhet | 439 | 15 | 0.093 | 0.267 | 0.138 |

Division results are unstable because each subgroup has only 10–41 dropout cases. They do not establish fairness or causal regional effects.

## Household economic group

| Group | N | Dropout prevalence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Poorest | 1,345 | 4.01% | 0.163 | 0.426 | 0.236 |
| Second | 1,382 | 3.18% | 0.069 | 0.182 | 0.100 |
| Middle | 1,280 | 3.36% | 0.131 | 0.302 | 0.183 |
| Fourth | 1,201 | 2.83% | 0.089 | 0.265 | 0.133 |
| Richest | 1,013 | 1.78% | 0.088 | 0.278 | 0.133 |

Performance varies meaningfully across wealth groups, especially recall. The model must not be described as fair or used for punitive decisions.

# Limitations

- MICS is a cross-sectional household survey, not longitudinal school-administration data.
- Dropout is inferred from attendance in two school years; temporary absence and re-entry are not observed.
- Previous-year attendance can be affected by recall or proxy-reporting error.
- Background household characteristics are measured at survey time, not necessarily before dropout.
- The ordinary ML split does not provide complex-survey design-corrected national performance.
- Geographic variables may encode unequal access and local socioeconomic conditions.
- Very low dropout prevalence produces low precision even with moderate ROC-AUC.
- The model estimates associations and risk; it does not identify causes.
- The survey is from 2019 and may not represent current conditions.

# Saved Candidate Artifacts

- `models/mics_bangladesh_dropout_model.joblib`
- `models/mics_bangladesh_model_metadata.json`
- `reports/mics_training_results.json`
- `reports/mics_figures/confusion_matrix.png`
- `reports/mics_figures/roc_curve.png`
- `reports/mics_figures/feature_importance.png`

These do not replace `models/dropout_model.joblib` and are not loaded by `app.py`.

# Migration Recommendation

The MICS dataset is methodologically stronger for a **Bangladesh school-attendance dropout-transition project** than applying the Portuguese university dataset to Bangladesh: it is locally collected, uses a real attendance transition, and avoids a fabricated target. It is not directly comparable to the Portuguese university population because the goal has changed from university dropout to school-age attendance transition.

Do **not** migrate the website yet. Before migration, improve or reassess the operating point, perform PSU-grouped validation, investigate additional clearly pre-outcome household predictors, test temporal/external validation if possible, and agree on an acceptable recall/false-positive tradeoff. The current held-out recall and precision are too weak for a public-facing risk tool.
