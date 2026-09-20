# OULAD Withdrawal-Risk Dataset and Model Analysis

## Dataset Source

OULAD is published by The Open University and described in Kuzilek, Hlosta, and Zdrahal (2017), *Open University Learning Analytics dataset*, Scientific Data 4, 170171. The Open University dataset page and DOI-linked Figshare record were treated as the primary sources. Their download endpoints were unavailable during this run, so the identical official UCI repository distribution was used rather than a third-party copy.

- Open University: https://research.stem.open.ac.uk/ouanalyse/dataset/
- Paper: https://doi.org/10.1038/sdata.2017.171
- DOI-linked data record: https://doi.org/10.6084/m9.figshare.5081998.v1
- Download fallback: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Local archive: `data/oulad/oulad_official.zip` (46,748,244 bytes; MD5 `EA9F1293B549C3A93B70B8D349B442E2`)

The source CSV files are preserved unchanged under `data/oulad/raw/`. This project predicts withdrawal from a particular module presentation, not permanent departure from all education.

## Tables Used

| Table | Rows | Columns | Verified key / join role | Duplicate risk |
|---|---:|---:|---|---|
| `studentInfo.csv` | 32,593 | 12 | Unique on module, presentation, student; outcome and background | 0 duplicate rows/key records |
| `studentRegistration.csv` | 32,593 | 5 | Unique on module, presentation, student | 0 duplicate rows/key records |
| `assessments.csv` | 206 | 6 | `id_assessment` is unique; maps an assessment to module/presentation | 0 duplicates |
| `studentAssessment.csv` | 173,912 | 5 | Unique on assessment/student; joins through `id_assessment` | 0 duplicate rows/assessment-student keys |
| `studentVle.csv` | 10,655,280 | 6 | Aggregated by module, presentation, student after cutoff filtering | 787,170 exact repeated rows; the official rows were retained and their `sum_click` values summed |
| `vle.csv` | 6,364 | 6 | `id_site` is unique; resource metadata | Inspected but not needed for the final compact feature set |
| `courses.csv` | 22 | 3 | Unique on module/presentation | Inspected but course length is not an early-risk input |

The core student-course identifier is `(code_module, code_presentation, id_student)`. A person can appear in more than one student-course record, so `id_student` is the grouping variable for splitting.

## Prediction Point

Prediction is made at the end of day 30 after the module presentation starts. Dates in OULAD are measured relative to course start, so this gives one consistent temporal boundary. Only registrations known by day 30, assessment deadlines on or before day 30, submissions made on or before day 30, and VLE events on or before day 30 are used.

Day 30 is usable but imperfect: 17 assessment definitions across 16 of the 22 presentations are due by then. Some presentations therefore have no early assessment deadline, making VLE engagement and registration timing especially important.

## Target Definition

The observed labels are Pass (12,361), Withdrawn (10,156), Fail (7,052), and Distinction (3,024). The binary target is:

- `1`: `final_result == "Withdrawn"`
- `0`: `final_result` is Fail, Pass, or Distinction

Fail is deliberately non-withdrawal because the student remained through the course outcome. The model is therefore a course-withdrawal classifier, not a general academic-success classifier.

## Cohort Construction

1. Join student information to registration one-to-one on the verified composite key.
2. Exclude 5,119 withdrawn records with `date_unregistration <= 30`; their withdrawal was already known at the prediction point.
3. Exclude 23 additional records whose registration was missing or occurred after day 30 and which were not already excluded as early withdrawals. These students were not demonstrably active and registered at the prediction point.
4. Retain 27,451 student-course records representing 24,781 unique students.
5. Split by `id_student`: 21,985 training rows / 19,824 students and 5,466 test rows / 4,957 students. Student overlap is zero.

There are 93 withdrawn records with no unregistration date in the source. They remain eligible because the outcome is Withdrawn but no evidence shows it occurred on or before day 30. This uncertainty is documented as a limitation rather than repaired by guessing a date.

## Early Withdrawal Exclusions

The policy is to predict later withdrawal only among students still observable as active at day 30. `date_unregistration` is used solely to construct this eligible cohort. It is never a feature. This avoids predicting an event after it has already happened.

## Class Distribution

Within the final cohort:

| Class | Count | Percentage |
|---|---:|---:|
| Withdrawn | 5,033 | 18.33% |
| Non-withdrawn | 22,418 | 81.67% |
| Total | 27,451 | 100.00% |

The positive class is moderately imbalanced. Class weighting was evaluated for Logistic Regression and Random Forest. Threshold selection used grouped out-of-fold training probabilities only; no SMOTE was used.

## Feature Definitions

| Website label | Model feature | Type | Definition at day 30 | Example |
|---|---|---|---|---|
| Age group | `age_group` | Select | Source age band | 35–55 |
| Previous education level | `previous_education` | Select | Highest education reported on entry | A-level equivalent |
| Previous attempts | `previous_attempts` | Integer | Previous attempts at the module | 1 |
| Study load | `studied_credits` | Number | Credits currently studied | 60 |
| Registration lead time | `registration_lead_days` | Integer | `-date_registration`; positive means registered before course start | 25 |
| Assessments submitted | `early_assessments_submitted` | Integer | Distinct assessments due by day 30 and submitted by day 30 | 1 |
| Missed assessments | `early_assessments_missed` | Integer | Assessments due by day 30 minus qualifying submissions | 0 |
| Early assessment average | `early_average_score` | Number / optional | Mean score of qualifying early submissions | 72 |
| Learning activity | `early_total_clicks` | Integer | Sum of VLE clicks through day 30 | 350 |
| Active learning days | `early_active_days` | Integer | Distinct VLE dates through day 30 | 18 |
| Learning resources accessed | `early_unique_resources` | Integer | Distinct VLE resources used through day 30 | 42 |

The final recommended website has 11 inputs. Raw module codes, presentation codes, student identifiers, UK region, IMD band, gender, and disability are excluded. Disability was removed because it is sensitive, creates fairness concerns, and did not resolve the model's practical performance problem.

## Leakage Assessment

| Variable / risk | What it represents | Known at day 30? | Decision |
|---|---|---|---|
| `final_result` | End-of-course outcome | No | Target only; never a predictor |
| `date_unregistration` | Withdrawal date | It can directly reveal the outcome | Used only to remove withdrawals already observed by day 30; excluded from features |
| VLE rows after day 30 | Later engagement | No | Filtered out before aggregation |
| Assessments due after day 30 | Future opportunities/results | No | Excluded; they are not counted as missed |
| Submissions after day 30 | Later academic behavior | No | Excluded before score/count aggregation |
| Full-course totals | Outcome-adjacent cumulative behavior | No | Not constructed |
| Student identifier | Identity and repeat-enrolment link | Yes, but not a causal input | Used only for joins/group splits; excluded from predictors |
| Module/presentation codes | Institution-specific course identity | Yes | Used for correct joins only; excluded from predictors to keep the form country-neutral |
| Registration date | Enrollment timing | Yes if registration occurred by cutoff | Converted to lead time; post-cutoff/missing registrations excluded |

Residual leakage is low under the stated day-30 prediction design. The main remaining concern is not target leakage but context encoding: counts of due/missed assessments reflect presentation-specific assessment schedules even though the raw module code is hidden.

## Missing Data

- `assessments.date`: 11 missing values, primarily exams; such assessments cannot be shown to be due by day 30 and are excluded from early assessment construction.
- `studentAssessment.score`: 173 missing values; numeric imputation is performed inside each model pipeline.
- `studentInfo.imd_band`: 1,111 missing, but IMD is excluded.
- `studentRegistration.date_registration`: 45 missing; eligible rows with missing registration timing are excluded.
- `studentRegistration.date_unregistration`: 22,521 missing, expected mainly for completers; 93 Withdrawn records also lack it.
- `vle.week_from` and `week_to`: 5,243 missing each, but these fields are not modeled.
- Students with no qualifying early score retain a missing `early_average_score`; median imputation occurs inside the pipeline while submitted/missed counts preserve the reason context.

## Model Evaluation

Five-fold `StratifiedGroupKFold` was used inside the training partition. Thresholds 0.20 to 0.60 were assessed from grouped out-of-fold probabilities. XGBoost was tried only because Logistic Regression and Random Forest produced low precision and average precision.

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.40 | 0.488 | 0.238 | 0.803 | 0.367 | 0.675 | 0.311 | 0.226 |
| Random Forest | 0.30 | 0.554 | 0.255 | 0.738 | 0.380 | 0.683 | 0.336 | 0.168 |
| XGBoost | 0.20 | 0.700 | 0.314 | 0.526 | 0.393 | 0.685 | 0.352 | 0.139 |

XGBoost had the highest grouped-CV average precision, so it was evaluated once on the untouched grouped holdout. Test results were accuracy 0.697, precision 0.299, recall 0.526, F1 0.381, ROC-AUC 0.680, average precision 0.330, and Brier score 0.136. The confusion matrix was TN 3,303, FP 1,196, FN 458, TP 509. Calibration is materially better than the class-weighted Logistic Regression and acceptable in aggregate by Brier score, but the calibration curve remains a single-dataset check rather than evidence of transportability.

The most influential model inputs were early assessments missed, early average score, early assessments submitted, study load, previous education, previous attempts, active learning days, total clicks, registration lead time, and unique resources. These are associated with predicted withdrawal risk; they are not causal findings.

## Migration Gate

The candidate gate required test precision at least 0.35, recall at least 0.70, average precision at least 0.40, and Brier score at most 0.18. XGBoost met only the Brier criterion. No OULAD model or metadata artifact was saved under `models/oulad/`.

Decision: **DO NOT MIGRATE**. OULAD provides a more defensible temporal withdrawal target than the Portuguese UCI dataset, but the current compact country-neutral feature set produces too many false positives and misses nearly half of later withdrawals at the selected threshold.

## Limitations

- OULAD covers distance-learning modules at one UK institution in 2013–2014. External validation is required before use in Bangladesh, a campus university, or a modern learning platform.
- Withdrawal is module-level discontinuation, not necessarily university dropout.
- Course/presentation schedule differences remain even after identifiers are removed; six presentations have no assessment due by day 30.
- Exact repeated rows in `studentVle.csv` may inflate total-click counts. They were retained because the official log does not supply an event identifier that proves they are erroneous duplicates.
- Threshold performance is operationally weak: at threshold 0.20 the model flags 31.2% of the held-out cohort, but only 29.9% of flags are true withdrawals.
- The 93 withdrawn records with missing unregistration dates cannot be checked for early-withdrawal exclusion.
- Feature importance describes model association, not causes or intervention effects.
