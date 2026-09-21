# Bangladesh Dataset Target-Validation Report

## Decision

**STOP: do not train or migrate the application with this dataset in its published form.**

The official Mendeley record advertises a target named `student status`, but the only downloadable file contains **31 predictor columns and no target/outcome column**. The target's values, class definitions, construction rule, and distribution therefore cannot be verified. The workbook cannot currently support supervised dropout or academic-risk modelling.

The repository's existing Portuguese UCI application, model, metadata, and reports remain active and unchanged. Git tag `uci-portugal-version` records the working version before this investigation.

## Source and provenance

- Dataset: *Students_Academic_ Performance_Evaluation_Dataset*
- Official record: <https://data.mendeley.com/datasets/dc3797vf3t/1>
- DOI: `10.17632/dc3797vf3t.1`
- Version: 1; published 16 September 2024
- Licence: CC BY 4.0
- Context stated by the publisher: CSE students at a private institution in Bangladesh
- Official file: `Students_Performance_data_set.xlsx`
- Local verified copy: `data/bangladesh_students_performance.xlsx`
- Official/local SHA-256: `04ebaf9f3a047afe96d9bff22dbe65b2a320e6395f230032735495e7e58277cf`

The official record describes 1,195 individuals and 31 features. The actual workbook has 1,195 spreadsheet rows **including the header**, so it contains **1,194 student records** and 31 columns. Only one file is published; there is no accompanying codebook, README, notebook, paper, or separate label file in the Mendeley record.

## Target validation gate

### What the official record claims

The record says the targeted column is `student status` and that it was chosen based on present CGPA, SGPA, attendance, daily study hours, study frequency, social-media time, and English proficiency.

### What the official workbook contains

- One visible sheet: `Students_Performance_data_set`
- 1,194 data rows and 31 columns
- No column named `student status`
- No other outcome-like column such as `target`, `class`, `dropout`, `status`, `result`, or `label`
- No hidden sheet or hidden column
- No formulas that calculate a target
- No class definitions or target values

### Classification of the target

The exact meaning of `student status` **cannot be verified from the official dataset**. It cannot responsibly be classified as an actual dropout/continuation outcome. The wording in the official description indicates a constructed academic status or performance category based on concurrent academic and behavioural measurements, rather than an independently observed dropout event.

Accordingly:

- **A. Actual dropout/continuation:** not supported.
- **B. Academic risk/status:** possible, but not verifiable from the published file.
- **C. Academic performance category:** most consistent with the available description and later secondary uses, but not present in the official workbook.
- **D. Other:** the precise construction remains undocumented.

There is no target distribution to report because the official file contains no target values.

## Dataset overview

| Item | Finding |
|---|---|
| Students/records | 1,194 |
| Predictor columns | 31 |
| Target column | Missing from the published workbook |
| Workbook sheets | 1 visible sheet |
| Duplicate rows | 0 |
| Missing cells | 1 |
| Formula cells | 0 |
| Hidden rows/columns | 0 |
| Numerical columns | 12 |
| Categorical/mixed columns | 19 |

## Data quality

### Missing values and duplicates

- `What are the skills do you have ?` has one missing value.
- All other columns have no pandas-null values.
- There are no exact duplicate rows.
- Several apparent zero/sentinel values require a codebook before they can be interpreted as genuine measurements.

### Notable anomalies

- `H.S.C passing year` includes 2028, and one row has an HSC year later than its university admission year.
- `Current Semester` ranges from 1 to 24; 36 records exceed semester 12. This may reflect a trimester system or extended enrolment, but it is undocumented.
- `Average attendance on class` mixes numeric values with one range string, `94-98`.
- `Program` is constant (`BCSE`) and has no predictive variation.
- Previous SGPA and current CGPA each include 39 zero values; credits completed includes 37 zeros. In 35 rows, all three are zero, which may mean first-semester/not-yet-recorded data rather than true academic values.
- Daily study hours range from 0 to 13, social-media hours from 0 to 20, and skill-development hours from 0 to 12. These are self-reported and should be validated.
- `Do you have any health issues?` uses inconsistent labels (`No`, `no`, `N`, `Yes`).
- Relationship, skill, and interest fields contain spelling/capitalisation variants and semantically duplicated categories.
- Skills (55 observed labels) and interests (25 observed labels) are noisy, high-cardinality free text.
- Monthly family income ranges from 4,000 to 2,000,000; currency, period, and outlier handling are undocumented.

### Class imbalance

Unknown. No target values are present, so class counts and percentages cannot be calculated.

### Sensitive variables

Gender, age, relationship status, health issues, physical disability, living arrangement, scholarship, device ownership, and family income may encode protected, health, or socioeconomic information. They require a documented necessity, fairness analysis, and careful user communication. Health/disability fields should not be used for an operational score without an especially strong ethical and institutional justification.

## Feature analysis

`Keep?` below is only a provisional judgement for a future, correctly labelled dataset. **No feature is approved for training until an independent target and prediction time are defined.**

| Feature | Type | Meaning | Known Before Target? | Keep? | Reason |
|---|---|---|---|---|---|
| University Admission year | Numerical/year | Calendar year of university admission | Usually | Review | Cohort and policy proxy; useful only with temporal validation. |
| Gender | Categorical, sensitive | Self-reported gender category | Yes | Fairness audit only | Potential discrimination; not necessary for intervention targeting. |
| Age | Numerical, sensitive | Student age | Yes | Review | May add context but requires fairness assessment. |
| H.S.C passing year | Numerical/year | Year of Higher Secondary Certificate completion | Yes | Review | One impossible/future value must be resolved; may duplicate age/admission timing. |
| Program | Categorical | Degree programme | Yes | Exclude | Constant `BCSE`; no predictive information. |
| Current Semester | Numerical/ordinal | Current semester number | Yes, if snapshot date is defined | Candidate | Important academic-timeline context, but values above 12 need documentation. |
| Do you have meritorious scholarship ? | Binary categorical, sensitive proxy | Merit-scholarship status | Usually | Review | Potential academic/socioeconomic proxy and policy-dependent. |
| Do you use University transportation? | Binary categorical | Use of university transport | Usually | Candidate | Practical context; relevance must be validated. |
| How many hour do you study daily? | Numerical/self-report | Daily study hours | Concurrent | Exclude if published target is used | Official description says this helps define `student status`; circular leakage. |
| How many times do you seat for study in a day? | Numerical/self-report | Daily study-session frequency | Concurrent | Exclude if published target is used | Official description says this helps define the target; circular leakage. |
| What is your preferable learning mode? | Categorical | Preferred online/offline learning mode | Yes | Candidate | Understandable contextual feature; confirm collection timing. |
| Do you use smart phone? | Binary categorical, socioeconomic proxy | Smartphone access/use | Yes | Review | Nearly ubiquitous/possibly low value; socioeconomic fairness concern. |
| Do you have personal Computer? | Binary categorical, socioeconomic proxy | Personal-computer access | Yes | Review | May reflect study resources but also income. |
| How many hour do you spent daily in social media? | Numerical/self-report | Daily social-media time | Concurrent | Exclude if published target is used | Official description says this helps define the target; circular leakage. |
| Status of your English language proficiency | Ordinal categorical | Self-assessed English proficiency | Concurrent | Exclude if published target is used | Official description says this helps define the target; circular leakage. |
| Average attendance on class | Numerical/mixed | Average class attendance, apparently percent | Concurrent | Exclude if published target is used | Target-defining per official description; also contains a range string. |
| Did you ever fall in probation? | Binary categorical | History of academic probation | Potentially | Review/exclude | Strong proximal outcome; leaks future information unless prediction occurs after the recorded event. |
| Did you ever got suspension? | Binary categorical | History of suspension | Potentially | Review/exclude | Strong proximal outcome and disciplinary sensitivity; timing is unspecified. |
| Do you attend in teacher consultancy for any kind of academical problems? | Binary categorical | Whether student sought faculty consultation | Potentially | Review | May be a response to existing difficulty; snapshot timing is essential. |
| What are the skills do you have ? | High-cardinality text | Self-reported skills | Yes | Exclude initially | One missing value, 55 noisy variants, difficult to encode and explain. |
| How many hour do you spent daily on your skill development? | Numerical/self-report | Daily skill-development time | Concurrent | Review | Self-reported; clarify whether it predates the outcome window. |
| What is you interested area? | High-cardinality text | Preferred technical/academic interest | Yes | Exclude initially | 25 noisy categories and unclear predictive rationale. |
| What is your relationship status? | Categorical, sensitive | Relationship/marital status | Yes | Exclude | Sensitive, inconsistent categories, and weak intervention rationale. |
| Are you engaged with any co-curriculum activities? | Binary categorical | Participation in co-curricular activities | Concurrent | Candidate | Potentially actionable context if collected before outcome. |
| With whom you are living with? | Categorical, socioeconomic proxy | Family versus bachelor/shared living | Yes | Review | Contextual but sensitive and possibly confounded by location/income. |
| Do you have any health issues? | Categorical, health-sensitive | Self-reported health issue | Yes | Fairness audit only | Sensitive health information and inconsistent labels. |
| What was your previous SGPA? | Numerical | Previous-semester grade-point average | Potentially | Exclude if published target is used | Official description says SGPA defines target; zeros also need interpretation. |
| Do you have any physical disabilities? | Binary categorical, highly sensitive | Self-reported physical disability | Yes | Fairness audit only | High discrimination risk; should not drive adverse predictions. |
| What is your current CGPA? | Numerical | Current cumulative grade-point average | Concurrent | Exclude if published target is used | Official description says CGPA defines target; direct circular leakage. |
| How many Credit did you have completed? | Numerical | Completed academic credits | Potentially | Review | Strong progress/semester proxy; timing and zero values need clarification. |
| What is your monthly family income? | Numerical, sensitive proxy | Reported monthly household income | Yes | Review/exclude | Currency/unit and outliers undocumented; major socioeconomic fairness concern. |

## Data leakage assessment

### Direct target-construction leakage

The publisher states that `student status` was chosen based on:

1. Current CGPA
2. Previous SGPA
3. Average class attendance
4. Daily study hours
5. Study frequency per day
6. Daily social-media hours
7. English-language proficiency

If the missing label was deterministically or manually derived from these fields, training on those same fields would teach the model to reproduce the label-creation rule. High test accuracy would not demonstrate prediction of an independent future outcome. These seven variables must therefore be excluded from any model using the published `student status` label unless independent documentation proves that they did not define it.

Because the label column and construction rule are absent, it is not possible to quantify the leakage, reconstruct the threshold rule, or test a leakage-free alternative.

### Temporal/outcome leakage requiring investigation

- Probation and suspension histories may occur after the intended prediction point.
- Faculty consultation can be a response to already-observed academic difficulty.
- Current CGPA, completed credits, and current semester capture the present academic state and must be aligned to a defined snapshot date.
- Cohort/admission year can encode time-dependent institutional policies if random splitting is used.

## Recommended prediction point

No final prediction point can be approved for this unlabelled dataset. For a replacement longitudinal dataset, use one defensible point: **the end of a student's first completed semester, predicting independently recorded dropout/withdrawal before the start of the third semester**. Only information recorded on or before that first-semester cutoff should be used. This supports early intervention while enforcing a clear time boundary.

## Proposed ML target

For the stated university-project goal, the required target should be **binary classification: actual dropout/withdrawal before the defined follow-up date versus continued enrolment**. It must come from institutional enrolment records, not a score created from the input variables.

The missing, undocumented `student status` label should not be recreated or guessed. A performance-category classifier would be a different project and would require a different title and objective.

## Recommended input features

**Final usable training list: none approved.** The absence of a verified target makes feature selection and supervised evaluation invalid.

If the dataset owner supplies an independently observed, time-stamped outcome, provisional fields worth reviewing for a compact early-semester form are:

- `Current Semester`
- `Do you have meritorious scholarship ?`
- `Do you use University transportation?`
- `What is your preferable learning mode?`
- `Do you have personal Computer?`
- `Do you attend in teacher consultancy for any kind of academical problems?`
- `How many hour do you spent daily on your skill development?`
- `Are you engaged with any co-curriculum activities?`
- `With whom you are living with?`
- `How many Credit did you have completed?`

This is a review list, not a model schema. It must be reconsidered against the eventual outcome definition and observation date. Sensitive fields should remain outside the operational model unless a documented benefit outweighs fairness risks.

## Required remediation before modelling

Obtain a corrected, documented release containing:

1. The actual target column and all unique class values.
2. A codebook defining every target class.
3. The target's creation method and any thresholds/rules.
4. An observation date for every predictor and a later outcome date.
5. Confirmation of whether labels are real institutional outcomes or constructed categories.
6. Preferably, an independently recorded dropout/withdrawal/continuation outcome.

After receipt, repeat the target, leakage, class-balance, and temporal-split review before training Logistic Regression or Random Forest.

## Migration and naming decision

- Do not replace the active UCI model.
- Do not change the Streamlit form, metrics, or website title.
- Do not call this candidate dataset a dropout dataset.
- If a future dataset supports only an independent academic-risk outcome, a suitable title would be **“Bangladeshi University Student Academic Risk Prediction Using Machine Learning.”** That title is not activated now because the target has not been verified.

## Methodological suitability

The data is more locally relevant to Bangladeshi CSE students than the Portuguese UCI data, but it is **not methodologically stronger for dropout prediction**. In its official published form it has no outcome column, no verifiable class definitions, no longitudinal timing, and a stated risk of circular label construction. A corrected target or a different longitudinal Bangladesh dataset is required.
