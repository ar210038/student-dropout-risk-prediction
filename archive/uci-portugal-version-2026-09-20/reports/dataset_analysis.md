# Dataset Overview

## Source and scope

- **Dataset:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Publisher:** UCI Machine Learning Repository
- **Citation:** Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). DOI: [`10.24432/C5MC89`](https://doi.org/10.24432/C5MC89)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Institutional context:** Students from a Portuguese higher-education institution across several undergraduate programs. This is not a Computer Science-only dataset.
- **Rows:** 4,424 students
- **Columns:** 37 total: 36 predictors and 1 target
- **Saved file:** `data/student_dropout.csv`, copied from the official UCI archive without changing its data values. The official file uses a semicolon delimiter.
- **Missing values declared by UCI:** None

UCI states that the dataset combines information available at enrollment with academic performance at the end of the first and second semesters. The target records status at the end of the normal duration of the course.

## Original target distribution

| Target class | Students | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |
| **Total** | **4,424** | **100.00%** |

# Target Definition

## Strategies considered

### A. Dropout versus Graduate only

- `Dropout = 1`
- `Graduate = 0`
- Exclude `Enrolled`

This uses only resolved outcomes. It avoids asserting that a student who is merely enrolled will never drop out. The cost is removing 794 students and learning mainly from two outcome endpoints.

### B. Dropout versus Non-Dropout

- `Dropout = 1`
- `Graduate` or `Enrolled = 0`

This retains all 4,424 students, but treats the 794 enrolled students as confirmed non-dropouts. Their courses were not yet resolved at the recorded endpoint, so some may later drop out. This is right-censoring and introduces label uncertainty into the negative class.

## Recommendation

Use **Strategy A: Dropout versus Graduate only**, excluding currently enrolled students. Methodological validity is more important than retaining every row. The resulting modeling dataset has 3,630 students:

| Binary target | Meaning | Students | Percentage |
|---:|---|---:|---:|
| 0 | Graduate | 2,209 | 60.85% |
| 1 | Dropout | 1,421 | 39.15% |

The target is moderately imbalanced, so accuracy must not be the main evaluation measure.

# Recommended Prediction Point

**Predict dropout risk at enrollment, before first-semester teaching begins.**

The proposed end-of-second-semester point was considered but rejected for the primary model. UCI supplies the final status but does not supply a dropout date. A student labelled `Dropout` may have left during the first or second semester. In that case, second-semester zeros or low activity would describe an event that already occurred rather than predict a future event.

An enrollment-time cutoff has a clear temporal order: the input exists before academic study and before the eventual resolved outcome. It also produces a genuinely early intervention model. The first- and second-semester variables remain valuable for a later model only if a future dataset provides dropout dates or confirms that all labeled outcomes occur after the chosen cutoff.

# Feature Review

Descriptions are based on the official UCI variable documentation. Integer codes are treated as categories where UCI defines category mappings; their numeric values must not be interpreted as measured quantities.

| Feature | Type | Stage Available | Meaning | Keep? | Reason |
|---|---|---|---|---|---|
| `Marital status` | Categorical code | Admission-time feature | Marital-status category: single, married, widower, divorced, facto union, or legally separated. | Yes | Known at enrollment. Sensitive personal information; include only for support-oriented use and audit group performance. |
| `Application mode` | Categorical code | Admission-time feature | Admission route or legal/application regime. | Yes | Known at application and potentially informative; one-hot encode rather than treating codes as ordered. |
| `Application order` | Ordinal integer | Admission-time feature | Choice order from 0 (first choice) to 9 (last choice). | Yes | Known at application and has a documented order. |
| `Course` | Categorical code | Admission-time feature | Undergraduate program code, covering 17 programs. | Yes | Known at enrollment and may capture genuine program-level differences. Monitor program fairness and avoid treating codes as continuous. |
| `Daytime/evening attendance` | Binary categorical | Admission-time feature | 1 daytime, 0 evening. | Yes | Known at enrollment; may reflect work and access constraints, so audit subgroup performance. |
| `Previous qualification` | Categorical code | Admission-time feature | Type/level of qualification before entering the course. | Yes | Known before enrollment. Category codes are nominal, not numeric magnitude. |
| `Previous qualification (grade)` | Continuous, 0–200 | Admission-time feature | Grade of the previous qualification. | Yes | Genuine pre-enrollment academic evidence. |
| `Nacionality` | Categorical code | Admission-time feature | Nationality code. The misspelling is in the official schema. | Yes, with audit | Known at enrollment but sensitive and sparse. Use one-hot encoding and assess nationality/international-group disparities. |
| `Mother's qualification` | Categorical code | Admission-time feature | Mother's education level; code 34 represents unknown. | Yes, with audit | Available at enrollment and may proxy socioeconomic advantage. Treat unknown as its own category. |
| `Father's qualification` | Categorical code | Admission-time feature | Father's education level; code 34 represents unknown. | Yes, with audit | Same proxy and missing-code concerns as mother's qualification. |
| `Mother's occupation` | Categorical code | Admission-time feature | Mother's occupation group; code 99 is documented as blank. | Yes, with audit | Available at enrollment but high-cardinality and socioeconomic. Treat blank as a category and regularize/limit model complexity. |
| `Father's occupation` | Categorical code | Admission-time feature | Father's occupation group; code 99 is documented as blank. | Yes, with audit | Same concerns as mother's occupation. |
| `Admission grade` | Continuous, 0–200 | Admission-time feature | Grade at admission. | Yes | Strong, legitimate admission-time academic predictor. |
| `Displaced` | Binary categorical | Admission-time feature | Whether the student is displaced: 1 yes, 0 no. | Yes, with audit | Known at enrollment; may represent relocation and access barriers. |
| `Educational special needs` | Binary categorical | Admission-time feature | Whether educational special needs are recorded. | Yes, restricted use | May help direct support but is sensitive disability-related information. It must not be used to restrict opportunity, and access should be limited. |
| `Debtor` | Binary categorical | Admission-time feature* | Whether the student is a debtor. | Yes, verify timestamp | UCI groups it with enrollment information, but deployment must confirm it is the enrollment-time value rather than a later snapshot. Financially sensitive. |
| `Tuition fees up to date` | Binary categorical | Admission-time feature* | Whether tuition fees are up to date. | Yes, verify timestamp | Potentially strong and legitimate if captured at enrollment. Exact operational timestamp must be confirmed. Financially sensitive. |
| `Gender` | Binary categorical | Admission-time feature | 1 male, 0 female. | Yes, with audit | Known at enrollment but protected/sensitive. Compare performance with and without it and report subgroup metrics. |
| `Scholarship holder` | Binary categorical | Admission-time feature | Whether the student holds a scholarship. | Yes, with audit | Known at enrollment; financial-support and socioeconomic proxy. |
| `Age at enrollment` | Integer years | Admission-time feature | Student age at enrollment. | Yes, with audit | Known at enrollment; range is 17–70, reflecting nontraditional students as well as typical entrants. |
| `International` | Binary categorical | Admission-time feature | Whether the student is international. | Yes, with audit | Known at enrollment but sensitive and potentially sparse. |
| `Curricular units 1st sem (credited)` | Integer count | First-semester feature | Number of curricular units credited in semester 1. | No | Not available at the enrollment cutoff. Legitimate for a later cutoff, not intrinsically leakage. |
| `Curricular units 1st sem (enrolled)` | Integer count | First-semester feature | Number of curricular units enrolled in semester 1. | No | Post-cutoff for the recommended model. |
| `Curricular units 1st sem (evaluations)` | Integer count | First-semester feature | Number of evaluations in semester-1 curricular units. | No | Post-cutoff for the recommended model. |
| `Curricular units 1st sem (approved)` | Integer count | First-semester feature | Number of semester-1 curricular units approved. | No | A legitimate strong predictor after semester 1, but unavailable at enrollment. |
| `Curricular units 1st sem (grade)` | Continuous, 0–20 | First-semester feature | Average semester-1 grade. | No | A legitimate strong predictor after semester 1, but unavailable at enrollment. |
| `Curricular units 1st sem (without evaluations)` | Integer count | First-semester feature | Number of semester-1 curricular units without evaluations. | No | Post-cutoff for the recommended model. |
| `Curricular units 2nd sem (credited)` | Integer count | Second-semester feature | Number of curricular units credited in semester 2. | No | Post-cutoff and potentially records progress after an early dropout. |
| `Curricular units 2nd sem (enrolled)` | Integer count | Second-semester feature | Number of curricular units enrolled in semester 2. | No | Post-cutoff; a zero may encode that dropout has already occurred. |
| `Curricular units 2nd sem (evaluations)` | Integer count | Second-semester feature | Number of evaluations in semester-2 curricular units. | No | Post-cutoff; missing activity may describe an existing outcome. |
| `Curricular units 2nd sem (approved)` | Integer count | Second-semester feature | Number of semester-2 curricular units approved. | No | Strong but unavailable at enrollment; not target leakage at a verified later cutoff. |
| `Curricular units 2nd sem (grade)` | Continuous, 0–20 | Second-semester feature | Average semester-2 grade. | No | Strong but unavailable at enrollment; zeros need contextual interpretation. |
| `Curricular units 2nd sem (without evaluations)` | Integer count | Second-semester feature | Number of semester-2 curricular units without evaluations. UCI's public description mistakenly says first semester. | No | Post-cutoff; documentation contains a wording error. |
| `Unemployment rate` | Continuous percentage | Contextual/economic feature | Unemployment rate associated with the student's entry context. | Yes | Available context at enrollment. May encode cohort/time and may generalize poorly to unseen economic conditions. |
| `Inflation rate` | Continuous percentage | Contextual/economic feature | Inflation rate associated with the student's entry context. | Yes | Same cohort/generalization concern as unemployment. |
| `GDP` | Continuous | Contextual/economic feature | GDP indicator associated with the student's entry context. UCI does not specify units in the variable table. | Yes | Available context according to UCI, but units and temporal reference should be clarified. |
| `Target` | Categorical outcome | Target | Final status: Dropout, Enrolled, or Graduate at the end of the normal course duration. | Target only | Direct outcome. Never include in predictors or preprocessing fitted to predictors. |

# Data Quality

- **Missing values:** 0 pandas null cells, consistent with UCI's declaration. This does not mean every value is observed: parental qualification code 34 means unknown, and parental occupation code 99 is documented as blank.
- **Duplicate rows:** 0 exact duplicates.
- **Identifiers:** No student ID, name, row index, or identifier-like column is present.
- **Physical types:** 29 integer columns, 7 floating-point columns, and the string target. Seventeen predictor columns are semantic categorical/code variables despite being stored as integers.
- **Header issue:** The official file contains trailing whitespace in the `Daytime/evening attendance` header. The inspection script strips header whitespace in memory but leaves the downloaded file unchanged.
- **Schema wording:** The official column `Nacionality` is misspelled. The official description of `Curricular units 2nd sem (without evaluations)` refers to the first semester, which appears to be a documentation typo.
- **Ranges:** Age at enrollment is 17–70. Admission and previous-qualification grades remain within their documented 0–200 scale. Semester averages remain within the documented 0–20 scale.
- **Zeros in academic fields:** Zero enrolled units, evaluations, approvals, or grades can be a valid observed state. At a post-enrollment cutoff, it may also indicate nonparticipation or an already-completed dropout event. Do not replace these zeros with missing values without further evidence.
- **Rare categories:** Nationality, parental occupation, educational special needs, international status, and some application/course codes contain small groups. One-hot encoding and subgroup evaluation are required; merge categories only with a documented rule fitted on training data.
- **Generalizability:** The dataset comes from one Portuguese institution and several programs. It is much stronger than the retired 113-row dataset for a university project, but results should not be claimed as universal or Computer Science-specific.

# Data Leakage Assessment

## Direct leakage

`Target` is the only direct outcome field and must never enter the feature matrix.

## Timing leakage at the recommended enrollment cutoff

All 12 first- and second-semester curricular-unit variables are unavailable at enrollment and must be excluded. They are not inherently “bad” variables: approved units and grades are legitimate strong predictors when a model is explicitly run after those results exist. They become leakage only when used before their availability or when they describe a dropout that already occurred.

## Why the second-semester cutoff is not selected

The dataset has no dropout date. Consequently, it is impossible to verify that every `Dropout` outcome happens after second-semester results. Semester-2 non-enrollment, no evaluations, no approvals, and zero grades can effectively encode that a student has already left. A later cutoff would therefore mix future prediction with retrospective detection.

## Administrative timing ambiguity

`Debtor` and `Tuition fees up to date` are retained because UCI describes the non-academic block as information known at enrollment. Before deployment, confirm that local versions are frozen at enrollment. If they are updated later, using their current value would violate the cutoff.

## Cohort/context risk

`Unemployment rate`, `Inflation rate`, and `GDP` are known contextual predictors, not direct leakage. Their combinations can identify enrollment cohorts. A random split may therefore look better than performance on a genuinely future cohort. The later modeling report should disclose this and, if cohort dates can be obtained, prefer a temporal validation split.

# Recommended Input Features

Use exactly these 24 features for the next modeling round:

1. `Marital status`
2. `Application mode`
3. `Application order`
4. `Course`
5. `Daytime/evening attendance`
6. `Previous qualification`
7. `Previous qualification (grade)`
8. `Nacionality`
9. `Mother's qualification`
10. `Father's qualification`
11. `Mother's occupation`
12. `Father's occupation`
13. `Admission grade`
14. `Displaced`
15. `Educational special needs`
16. `Debtor`
17. `Tuition fees up to date`
18. `Gender`
19. `Scholarship holder`
20. `Age at enrollment`
21. `International`
22. `Unemployment rate`
23. `Inflation rate`
24. `GDP`

This feature set is available at the chosen cutoff, understandable to a supervisor, and compatible with both Logistic Regression and Random Forest. The modeling pipeline should one-hot encode nominal code fields. It should leave ordered/count/continuous fields numeric and scale numeric fields for Logistic Regression inside the training pipeline.

# Fairness and Privacy Assessment

The dataset includes gender, age, nationality, international status, marital status, special-needs status, displacement, parental education/occupation, debt, tuition, scholarship, course, and schedule. These may be legally protected, sensitive, or strong proxies for socioeconomic disadvantage.

They are not removed solely for being sensitive because the proposed use is to offer support, and excluding them does not guarantee fairness. The later evaluation must:

- report recall, false-negative rate, and precision across sufficiently sized groups;
- suppress or clearly qualify metrics for very small groups;
- compare overall performance with and without the most sensitive attributes;
- avoid using predictions to deny admission, financial aid, or educational opportunity;
- minimize access to row-level personal and financial data;
- document that group differences reflect this institution and historical processes, not student capability.

# Evaluation Metrics for the Modeling Round

Report:

- precision for dropout alerts;
- recall for dropouts;
- F1-score;
- ROC-AUC using predicted probabilities;
- the confusion matrix with false positives and false negatives clearly labelled.

**Dropout recall should receive particular attention** because a false negative is a student at risk who receives no alert. Recall should not be maximized without regard to precision: too many false alerts can overwhelm support services and stigmatize students. Accuracy should be secondary because the recommended target is 60.85% graduate versus 39.15% dropout.

# Suitability and Pre-Training Decision

The UCI dataset is strong enough for this university project. It has 4,424 documented student records, a clear license, no direct identifiers, no null cells, and a useful mixture of admission and academic variables. Its main limitations are single-institution scope, multiple courses rather than a CSE-only population, sensitive attributes, rare codes, and missing dropout-event dates.

The exact next step, after review, is to create a leakage-safe training pipeline on the 3,630 resolved-outcome rows using the 24 enrollment/context features, a stratified held-out test set, one-hot encoding fitted only on training data, numeric scaling for Logistic Regression, and cross-validated comparison of Logistic Regression and Random Forest. No training has been performed in this round.
