# Bangladesh Student Dropout-Risk Survey: Dataset Analysis

## Scope and Interpretation

The supplied workbook is a Bangladesh university-student survey collected in 2025. It contains self-reported answers about circumstances, study behavior, and likelihood of considering dropout because of academic stress. It does **not** contain a longitudinally observed withdrawal outcome.

The project must therefore be presented as an academic **risk-assessment prototype**. Its classes summarize self-reported dropout consideration and must not be described as confirmed future dropout predictions. The supplied dataset is identified as CC BY 4.0, but the sample is not claimed to represent all university students in Bangladesh.

## Raw Workbook

- File: `data/bangladesh_risk/raw/Student Dropout Risk Survey Dataset.xlsx`
- Worksheet: `Sheet1`
- Rows: 368 submitted responses
- Columns: 23
- Structure: one Timestamp, 21 predictor questions, and one target question
- Missing cells: none
- Timestamp range confirms 2025 collection

The raw workbook was not modified. The cleaned output is saved separately as `data/bangladesh_risk/processed/student_dropout_risk_cleaned.csv`.

## Target Definition

Target question:

> How likely are you to consider dropping out due to academic stress? (Just let your feelings out)

Mixed numeric and annotated answers such as `1` and `1 (Very unlikely)` are normalized to the leading ordinal value from 1 through 5. The model/presentation classes are:

| Original score | Derived class |
|---|---|
| 1–2 | Low Risk |
| 3 | Moderate Risk |
| 4–5 | High Risk |

These classes are derived from an intention scale. They are not observed dropout outcomes.

## Duplicate Investigation

Duplicates were detected across all 22 survey answers while ignoring Timestamp. Two exact duplicate profile groups were found:

| Group | Original copies | Timestamp behavior | Copies removed |
|---|---:|---|---:|
| 1 | 16 | 16 submissions between 2025-06-16 23:49:34.795 and 23:50:01.978; consecutive gaps 1.493–3.482 seconds | 15 |
| 2 | 3 | 2025-06-01 21:32:49.649, 21:53:59.423, and 21:55:36.249; gaps 1,269.774 and 96.826 seconds | 2 |

One representative from each group was retained. Seventeen duplicate copies were removed, reducing the dataset from 368 to 351 responses.

## Cleaned Target Distribution

| Risk class | Count | Percentage |
|---|---:|---:|
| Low Risk | 165 | 47.01% |
| Moderate Risk | 109 | 31.05% |
| High Risk | 77 | 21.94% |
| Total | 351 | 100.00% |

## Normalization Mappings

The following deterministic mappings were applied before splitting:

- Both Public University wordings were mapped to `Public University`.
- The private-university example wording was mapped to `Private University`.
- `National University (Affiliated Colleges)` was mapped to `National University / Affiliated College`.
- Mixed 1–5 representations for internet quality, material satisfaction, and the target were converted to their explicit integer rating.
- Corrupted range separators displayed as `�` were normalized to an en dash, for example `5–6 hours` and `20,000–50,000`.
- No categories were combined merely because they appeared similar; only genuine wording/representation equivalences were normalized.

## Leakage and Construct-Overlap Review

The following fields are forbidden in the primary model:

| Field | Decision | Reason |
|---|---|---|
| Timestamp | Exclude | Submission timing is not a meaningful student-risk input and can identify duplicate batches |
| Target question / normalized score | Exclude from predictors | Direct target leakage |
| “How often do you feel overwhelmed by academic workload?” | Exclude | Strong construct overlap with considering dropout due to academic stress |

A sensitivity run adding academic overwhelm did not improve repeated-CV XGBoost Macro F1 (0.366 ± 0.061 versus 0.372 ± 0.056 without it). It remains excluded on methodological grounds.

## Leakage-Safe Candidate Pool

The initial predictor pool contained 20 fields: age group, gender, academic level, university type, prior residence, living arrangement, sleep duration, employment type, meal skipping, commute time, internet quality, study-space quality, GPA, study routine, attendance, family income, scholarship, personal income, academic-resource access, and study-material satisfaction.

Feature selection was performed only within the 280-record training partition. Five-fold training-only permutation importance was considered alongside stability, privacy/fairness, interpretability, and ease of answering. Gender, detailed family/personal income, meal skipping, residence variables, age, and university type were not retained merely for small or unstable performance differences.

## Final 12-Feature Set

| Original survey question | Website label | Inclusion reason |
|---|---|---|
| Current academic level | Academic Level | Simple study-stage context |
| Average sleep duration per night | Sleep Duration | Understandable wellbeing/routine measure |
| Part-time job or freelance work | Employment / Tuition Work | Stable training importance and workload context |
| One-way commute time | Commute Time | Practical time burden |
| Internet quality for studying | Internet Quality | Academic access constraint |
| Dedicated study space and noise | Study Space | Practical learning environment |
| Current GPA or equivalent | Current GPA | Core academic-performance measure |
| Typical study timing | Study Routine | Direct, actionable study behavior |
| Attendance during the past month | Attendance | Stronger and comparatively stable training importance |
| Scholarship or stipend | Scholarship / Stipend | Financial-support context and training importance |
| Access to academic resources | Academic Resource Access | Measures access to learning support |
| Satisfaction with university study materials | Study Material Satisfaction | Institutional learning-support perception |

## Evaluation Design

A reproducible stratified 80/20 split created 280 training and 71 untouched sanity-check records. Model comparison used only the training partition with 5-fold, 5-repeat `RepeatedStratifiedKFold` (25 validation folds; random state 42). All learned imputation, one-hot encoding, and scaling remained inside scikit-learn pipelines.

## Limitations

- Only 351 deduplicated responses are available.
- The data are a convenience/survey sample and are not nationally representative.
- Every variable is self-reported and may contain recall, social-desirability, or response bias.
- The target measures intention/risk, not confirmed later dropout.
- There is no longitudinal outcome or external validation.
- Feature rankings are unstable across folds and do not establish causal effects.
- The derived three-class boundaries are presentation/model choices imposed on a five-point response scale.
