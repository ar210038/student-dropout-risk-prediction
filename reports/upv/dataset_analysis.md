# UPV Dataset Analysis and Target-Validation Gate

## Decision Summary

The dataset is authentic, longitudinal, and relevant, but modeling was stopped. The official paper defines university dropout and identifies `abandono_hash` as the outcome field, yet it documents only the anonymized values `A` and `B`. It does not state which value means dropout and which means non-dropout. The Zenodo archives contain only CSV files and provide no supplementary mapping.

Inferring the positive class from the field name, prevalence, correlations, or a later third-party analysis would violate the required target-validation rule. No UPV target was constructed, no prediction cohort was finalized, no model was trained, and no candidate artifact was saved.

Final dataset decision: **USE OULAD**.

## Official Sources

- Zenodo record: https://zenodo.org/records/17239943
- Dataset DOI: https://doi.org/10.5281/zenodo.17239943
- Official paper: https://doi.org/10.3390/data10100162
- Publisher page: https://www.mdpi.com/2306-5729/10/10/162

The three official archives were downloaded without modification and verified against the checksums published by Zenodo.

| Archive | Size on disk | Published and verified MD5 | Extracted file |
|---|---:|---|---|
| `dataset_2018_hash.zip` | 11.9 MB | `43fc067c88f09797dacbcd83ee83cc28` | `dataset_2018_hash.csv` |
| `dataset_2021_hash.zip` | 14.6 MB | `50cfb2d0f0a07bf3460bc259ffa84064` | `dataset_2021_hash.csv` |
| `dataset_2022_hash.zip` | 15.1 MB | `b5f724fde2c33961f07f26e8fa25b4b7` | `dataset_2022_hash.csv` |

The archives are under `data/upv/`; extracted source CSVs are under `data/upv/raw/`.

## Dataset Structure

| Academic-year file | Raw course-enrolment rows | Physical columns | Unique students | Unique degrees | Unique courses | Exact duplicate rows |
|---|---:|---:|---:|---:|---:|---:|
| 2018 | 152,446 | 178 | 20,233 | 124 | 3,747 | 0 |
| 2021 | 153,120 | 178 | 20,264 | 140 | 3,859 | 0 |
| 2022 | 159,173 | 169 | 20,427 | 142 | 3,787 | 0 |
| Combined | 464,739 | — | 39,364 | 163 | 4,989 | 0 |

There are 60,924 distinct student-year observations. A total of 17,729 students appear in more than one released year, so any valid future model would require student-grouped validation even when using a temporal test year.

The paper describes 77 conceptual variables. The CSV files have 169–178 physical columns because monthly LMS, resource, and Wi-Fi measures are expanded into separate year-month columns. Seventy base columns are shared by all three files; the remaining differences are primarily monthly time windows.

Each raw row represents a student-course-degree enrolment in one academic year:

- Student identifier: `dni_hash`
- Degree identifier: `tit_hash`
- Course identifier: `asi_hash`
- Academic year: `caca`
- Course/group identifier: `grupos_por_tipocredito_hash`

`dni_hash`, `tit_hash`, `asi_hash`, campus hashes, and group hashes are join/grouping fields, not suitable website predictors.

## Official Dropout Definition

The paper defines dropout at the university level as a student who:

1. still has credits pending to complete the degree; and
2. does not enroll at UPV for two consecutive academic years.

Internal transfers between UPV degrees are explicitly not treated as dropout. Graduation is conceptually distinguishable because a graduate has no credits pending. The definition treats two consecutive years of non-enrolment as dropout; it does not publish a separate temporary-interruption class, and a student may later return after satisfying that retrospective definition.

The outcome becomes knowable only after observing two complete consecutive academic years without enrolment. The public rows contain `abandono_hash`, but they do not contain an explicit outcome-observation date.

## Target Gate Failure

The official dictionary lists:

| Variable | Type | Published values | Published semantic mapping |
|---|---|---|---|
| `abandono_hash` | Categorical | `A`, `B` | **Not provided** |

Observed raw course-row counts are:

| File | A | B |
|---|---:|---:|
| 2018 | 9,879 | 142,567 |
| 2021 | 8,943 | 144,177 |
| 2022 | 10,789 | 148,384 |

At the student-year level, the counts are 4,294 `A` and 56,630 `B`. These are deliberately not labeled “dropout” and “non-dropout” because the official source does not map them.

The field is internally consistent within each student-year: every course row for a student in a given released year has the same `abandono_hash`. Across released years, 1,853 students change between A and B. That observation does not establish the meaning of either value and is compatible with several retrospective administrative constructions.

Because the class mapping is absent, the following mandatory facts cannot be established safely:

- which exact value is the positive dropout class;
- the dropout and non-dropout counts or prevalence;
- which rows represent graduation/completion versus continued enrolment at prediction time;
- whether a derived student-year target faithfully matches the paper's two-year definition.

## Longitudinal and Timing Assessment

A defensible candidate prediction point would have been the end of the first semester, using only information available through December of the academic year. September–December LMS fields are explicitly month-indexed and could support this cutoff. Admission information, registered study load, study stage, and prior-year performance could also be known by then.

Current-year final grades, current-year credits passed, full-year performance, later monthly LMS/Wi-Fi activity, `baja_fecha`, `matricula_activa`, pending-degree credits measured after the year, and final administrative outcomes would require detailed timing review or exclusion.

This cutoff was not adopted as a modeling cutoff because the target gate failed first.

## Missing Data and Categorical Structure

The release contains substantial, structured missingness:

- Admission year, route, and admission grades are missing for many course rows, particularly when the admission pathway is not applicable.
- Prior-year performance fields become increasingly missing for newer students: `rend_total_ultimo`, `rend_total_penultimo`, and `rend_total_antepenultimo` reflect available history.
- Some flags use missing values to mean “No,” according to the official dictionary, while other missing cells mean unavailable or no activity. These meanings cannot be collapsed blindly.
- Monthly LMS, assignment, test, resource, and Wi-Fi fields are sparse and vary by cohort and calendar month.
- `rendimiento_total` is entirely missing in the 2018 file and present with missing values in later files, demonstrating cross-cohort schema/availability differences.

Documented categorical fields include admission route, parents' education, full-time/part-time dedication, displaced-student status, selection preference, active enrollment, exemption, adapted/returning-student flags, and anonymized identifiers. Several are Spain- or UPV-specific and unsuitable for a country-neutral website.

## Candidate Feature Review (Not Approved for Modeling)

| Website label | Dataset variable(s) | Meaning | Available by proposed cutoff? | Keep? | Reason |
|---|---|---|---|---|---|
| Prior academic performance | `rend_total_ultimo` | Previous academic year's completion/performance rate | Yes when history exists | Deferred | Transferable concept; missing for new students |
| Admission performance | `nota10_hash` / `nota14_hash` | Modified admission grade | Usually | Deferred | Transferable concept but two scales and substantial missingness |
| Study load | `cred_mat_total` | Credits registered in the academic year | Yes | Deferred | Transferable after explaining UPV/ECTS scale |
| Study stage | `curso_mas_alto` | Highest course/year level currently enrolled | Yes | Deferred | Transferable concept |
| Full-time or part-time | `dedicacion` | Study dedication | Yes | Deferred | Transferable and simple |
| Years since entry | `caca`, `anyo_inicio_estudios` | Time since studies began | Yes | Deferred | Derived from documented dates/years |
| Early LMS events | `pft_events_{year}_{month}` | Platform events from September–December | Yes | Deferred | Transferable concept; exact LMS measurement is institution-specific |
| Early LMS visits | `pft_visits_{year}_{month}` | Platform visits from September–December | Yes | Deferred | Potentially understandable but platform-specific |
| Early active days | `pft_days_logged_{year}_{month}` | Days logged in during early months | Yes | Deferred | Aggregation across course rows needs care |
| Assignments submitted | `pft_assignment_submissions_{year}_{month}` | Early LMS assignment submissions | Yes | Deferred | Transferable concept |
| Online tests completed | `pft_test_submissions_{year}_{month}` | Early LMS test submissions | Yes | Deferred | Transferable concept |
| Early platform time | `pft_total_minutes_{year}_{month}` | Minutes recorded in the LMS | Yes | Deferred | Measurement depends on UPV's platform instrumentation |

No final feature list or website input count is approved because no valid binary target was established.

## Leakage Assessment

High-risk fields include:

- `abandono_hash`: intended outcome; never a predictor.
- `baja_fecha`: withdrawal/cancellation-related administrative date and potentially outcome-revealing.
- `matricula_activa`: active-enrollment status that may be recorded after the intended cutoff.
- `fecha_datos`: extraction timestamp, not a student predictor.
- `nota_asig_hash`: final subject grade, unavailable at an early-semester cutoff.
- Current-year credits passed and performance: `cred_sup*`, `rendimiento_cuat_*`, and `rendimiento_total` can contain end-of-semester/year information.
- `cred_pend_sup_tit`: pending degree credits can incorporate later outcomes and is part of the official dropout definition.
- January–August digital activity for a December cutoff: post-cutoff information.
- Identifiers and institutional hashes: can memorize student, degree, course, campus, or group effects and are unsuitable website inputs.

No model was trained, so no suspiciously high performance or model-level leakage was observed.

## Validation Feasibility

If the target mapping were officially resolved, the most defensible design would train on 2018 and 2021 student-year rows and test once on 2022. Any student appearing in 2022 would need removal from historical training to guarantee zero identity overlap. Group-aware cross-validation by `dni_hash` would be required within historical training.

That design was not executed. Creating a model with an assumed A/B mapping would make every metric, threshold, confusion matrix, calibration result, and comparison with OULAD potentially inverted and therefore invalid.

## Comparison with OULAD

UPV is conceptually stronger than OULAD because its published definition concerns institution-level university dropout rather than module withdrawal, and it covers in-person higher education across multiple years. However, OULAD has an explicit, directly interpretable target value: `final_result == "Withdrawn"`. UPV's public release does not officially disclose which anonymized class value means dropout.

The target ambiguity outweighs UPV's conceptual advantages for a defensible university project. OULAD remains reproducible, auditable, and label-valid even though its predictive performance is modest and its outcome is course withdrawal.

## Final Decision

**USE OULAD**
