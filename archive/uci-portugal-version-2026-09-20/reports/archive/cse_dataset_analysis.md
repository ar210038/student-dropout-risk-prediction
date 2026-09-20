# Dataset Overview

- **Dataset source:** [`dataset/anonymized_dataset.csv`](https://github.com/mvoassis/dropout_prediction/blob/main/dataset/anonymized_dataset.csv) from the [`mvoassis/dropout_prediction`](https://github.com/mvoassis/dropout_prediction) research repository. The repository describes data from the Computer Science Education program at the Federal University of Paraná, Brazil. The local copy is `data/anonymized_dataset.csv` (34,695 bytes; SHA-256 `02f624726fb06d1f8e280ba8df2e702a143a9734a943e35b7fc96a0ceb872114`).
- **Why this file:** It is the repository's only dataset file and its README explicitly links it as the anonymized dataset. The original non-anonymized file referenced by the notebook is not published.
- **Number of students:** 113 rows, apparently one row per student.
- **Number of features:** The CSV has 69 columns: 1 target and 68 non-target columns. One non-target column (`Unnamed: 0`) is an exported row index, leaving 67 substantive predictor fields before timing/leakage exclusions.
- **Target variable:** `situacao` (student situation/status).
- **Target classes:** The source notebook defines `0 = enrolled` and `1 = dropout`.
- **Scope limitation:** The source provides a cross-sectional/latest-record snapshot, not semester-stamped longitudinal snapshots. The extraction date and observation window are not documented in the public repository.
- **License/permission:** No license file is present in the source repository. Before publishing or redistributing this dataset with the final project, obtain permission or a clear license from the owner.

## Target distribution

| Class | Meaning | Students | Percentage |
|---:|---|---:|---:|
| 0 | Enrolled | 62 | 54.87% |
| 1 | Dropout | 51 | 45.13% |

The target itself is reasonably balanced. However, “enrolled” is not the same as “eventually will not drop out”; currently enrolled students may be right-censored and later become dropouts.

# Feature Analysis

Interpretations below use the Portuguese column names, the source notebook, and labels in the source application. Where the repository does not define a field or acronym, the uncertainty is stated rather than guessed.

| Feature | Type | Description/Interpretation | Keep? | Reason |
|---|---|---|---|---|
| `Unnamed: 0` | Integer/index | Pandas-exported original row index (unique, values 0–120 with gaps). | No | Useless identifier/index; also reveals preprocessing gaps. |
| `situacao` | Binary categorical | Student status: 0 enrolled, 1 dropout. | Target only | Correct outcome, never a predictor. |
| `anoConclusaoEnsinoMedio` | Numeric year | Year of high-school completion; 0 means not reported in the source app. | Yes, with cleaning | Known by entry/cutoff, but 0 must be treated as unknown rather than a real year. |
| `chIntegralizadaMatriculada` | Numeric | Credit-hour/workload field labelled “integrated/enrolled”; exact calculation is undocumented. | No as supplied | Cumulative/latest snapshot and definition unclear; likely contains post-cutoff progress. |
| `chintegralizadaACE` | Numeric | Integrated credit hours in curriculum category `ACE`; acronym not defined publicly. | No as supplied | Cumulative/latest snapshot; could be recomputed at cutoff only after definition is confirmed. |
| `chintegralizadaAF` | Numeric | Integrated credit hours in category `AF`; acronym not defined publicly. | No as supplied | Same timing and documentation problem. |
| `chintegralizadaCF` | Numeric | Integrated credit hours in category `CF`; acronym not defined publicly. | No as supplied | Same timing and documentation problem. |
| `chintegralizadaObrigatoria` | Numeric | Integrated compulsory-course credit hours. | No as supplied | Cumulative progress measured after the proposed cutoff for many students. |
| `chintegralizadaOptativa` | Numeric | Integrated elective-course credit hours. | No as supplied | Cumulative progress measured after the proposed cutoff for many students. |
| `chintegralizadaTotal` | Numeric | Total integrated credit hours. | No as supplied | Directly reflects how far the student ultimately progressed. |
| `fracaoChIntegralizadaTotal` | Numeric percentage | Fraction/percentage of total curriculum credit hours completed. | No as supplied | Strong post-cutoff progress signal and likely leakage for early warning. |
| `cidade` | Categorical (22 values) | City associated with the student; whether this is home, residence, or registration city is undocumented. | Provisional yes | Available early, but high cardinality, sparse cities, fairness/proxy, and geographic generalization need review. |
| `formaIngInstituicao` | Categorical (5 values) | Admission route: Vestibular, SISU, PSS ENEM, prior-higher-education credit, or program re-selection. | Yes | Known at admission; rare categories require careful encoding/grouping. |
| `ira` | Numeric (0–0.8973) | Academic Performance Index (`Índice de Rendimento Acadêmico`). | Cutoff version only | Useful performance summary, but the supplied value appears cumulative through extraction. Recompute through semester 2. |
| `sexo` | Binary categorical code | Encoded sex with values 0 and 1; code-to-label mapping is not documented. | Provisional yes | Known early, but mapping, ethical purpose, and fairness impact must be documented. Do not present it as gender without evidence. |
| `tempoUniversidade` | Numeric count | Time at university in semesters (source app); range 1–17. | No | At a fixed end-of-semester-2 prediction point this should be constant; supplied values reveal variable follow-up duration. |
| `Cancelado` | Numeric count | Number of course cancellations in the observed record. | Cutoff version only | Later cancellations leak future history; recompute using only semesters 1–2. |
| `Reprovado por frequência` | Numeric count | Number of failures due to insufficient attendance. | Cutoff version only | Potentially useful early warning, but supplied count is cumulative through extraction. |
| `Reprovado por nota` | Numeric count | Number of failures due to grade. | Cutoff version only | Potentially useful early warning, but supplied count is cumulative through extraction. |
| `DEE374_nota`, `DEE374_freq` | Numeric grade/attendance | Pre-Calculus, first semester, according to the source app. | Yes, cutoff-controlled | Direct first-year evidence; distinguish not taken/missing from a genuine zero. |
| `DEE341_nota`, `DEE341_freq` | Numeric grade/attendance | Programming Laboratory I, first semester, according to the source app. | Yes, cutoff-controlled | Direct first-year evidence; distinguish not taken/missing from a genuine zero. |
| `DEE345_nota`, `DEE345_freq` | Numeric grade/attendance | Computer Systems Security, second semester, according to the source app. | Yes, cutoff-controlled | Direct first-year evidence; distinguish not taken/missing from a genuine zero. |
| `DEC013_nota`, `DEE346_nota`, `DEE375_nota`, `DEC008_nota`, `DEC012_nota`, `DEC219_nota`, `DEE238_nota`, `DEE338_nota`, `DEE342_nota`, `DEE344_nota`, `DEC009_nota`, `DEC014_nota`, `DEC220_nota`, `DEE239_nota`, `DEE347_nota`, `DEE349_nota`, `DEE351_nota`, `DEE358_nota`, `DEE245_nota`, `DEE352_nota`, `DEE359_nota`, `DEE360_nota` | Numeric grades | Mean grade/result fields for coded courses. Public documentation does not identify course names or scheduled semesters. | No for proposed model | Cannot prove they are available by the end of semester 2; many zeros appear to mean no recorded attempt. |
| `DEC013_freq`, `DEE346_freq`, `DEE375_freq`, `DEC008_freq`, `DEC012_freq`, `DEC219_freq`, `DEE238_freq`, `DEE338_freq`, `DEE342_freq`, `DEE344_freq`, `DEC009_freq`, `DEC014_freq`, `DEC220_freq`, `DEE239_freq`, `DEE347_freq`, `DEE349_freq`, `DEE351_freq`, `DEE358_freq`, `DEE245_freq`, `DEE352_freq`, `DEE359_freq`, `DEE360_freq` | Numeric attendance | Mean attendance fields paired with the coded courses above. | No for proposed model | Course timing is undocumented, so availability by semester 2 cannot be established. |

The original repository reports an eight-feature selection: `tempoUniversidade`, `cidade`, `anoConclusaoEnsinoMedio`, `ira`, `DEE345_nota`, `DEE341_freq`, `DEE374_freq`, and `Reprovado por frequência`. That result is useful background, not a feature list to copy: several are cumulative or measured at inconsistent times, and the selection was made on the same very small dataset.

# Data Quality

- **Missing values:** Pandas reports 0 null cells. This is misleadingly clean: `anoConclusaoEnsinoMedio` is 0 for 28 students (24.78%), and zeros occupy an average of 66.83% of each course grade/attendance column. The source app explicitly treats high-school year 0 as “not informed”; course zeros likely mix genuine zero performance with not taken/no record. These states must be separated before modeling.
- **Duplicates:** 0 exact duplicates, both with and without the exported index.
- **Class imbalance:** Mild/no major imbalance: 62 enrolled (54.87%) versus 51 dropout (45.13%). Accuracy alone is still inappropriate because missing a student at risk has a different cost from a false alert.
- **Categorical variables:** `cidade` (22 values), `formaIngInstituicao` (5), and `sexo` (2). `situacao` is the binary target. Several city/admission categories occur only once, creating unstable encodings in a 113-row dataset.
- **Numerical variables:** 64 substantive numerical predictor fields after excluding the target, row index, and three categorical predictors. This includes 50 sparse course-grade/attendance fields.
- **Sample size:** 113 students is very small relative to 67 substantive predictors. Feature selection, evaluation variance, and overfitting are major concerns; preprocessing and feature selection must occur inside cross-validation in the later modeling round.
- **Documentation:** Most course codes, curriculum-category acronyms, collection dates, cohort years, and the `sexo` code mapping are undocumented publicly.
- **Generalizability:** This is a single-program, single-institution sample. It may support a local classroom prototype but cannot establish performance at other universities.

# Data Leakage Assessment

The intended task is prospective early warning: predict a future dropout using only facts known at a stated historical cutoff. The published CSV instead contains each student's cumulative record at a variable extraction point.

| Suspicious variable(s) | What it represents | Why it may leak | Recommendation |
|---|---|---|---|
| `chIntegralizadaMatriculada`, `chintegralizadaACE`, `chintegralizadaAF`, `chintegralizadaCF`, `chintegralizadaObrigatoria`, `chintegralizadaOptativa`, `chintegralizadaTotal`, `fracaoChIntegralizadaTotal` | Cumulative curriculum workload/progress. | Values recorded after semester 2 reveal how far the student ultimately progressed; low completion is close to a consequence of dropout. | Exclude supplied values. Only use semester-2 reconstructions if definitions and timestamps become available. |
| `tempoUniversidade` | Elapsed semesters in the record. | It measures follow-up/record duration and varies from 1 to 17. For an end-of-semester-2 model it should be 2 for everyone. | Exclude. |
| `ira` | Cumulative academic performance index. | The final/current index can incorporate grades earned after the prediction date. | Retain only an explicitly recomputed `ira_as_of_semester_2`. |
| `Cancelado`, `Reprovado por frequência`, `Reprovado por nota` | Cumulative event counts. | Counts can include events occurring after the proposed prediction date. | Retain only counts through semester 2. |
| All 25 `*_nota` fields | Mean course grades/results. | A course may occur after semester 2; even early-course fields may average later retakes. | Keep only verified first-year attempts as observed by cutoff: currently `DEE374_nota`, `DEE341_nota`, `DEE345_nota`. Exclude other course fields until the curriculum is documented. |
| All 25 `*_freq` fields | Mean course attendance. | Same timing/retake problem as grades; later attendance cannot be known early. | Keep only cutoff-safe first-year attendance: currently `DEE374_freq`, `DEE341_freq`, `DEE345_freq`. |
| `situacao` | Final/current status. | It is the outcome itself. | Target only; never pass it into preprocessing as a feature. |

## Selection bias in the published CSV

The exported index is unique but skips exactly `[34, 35, 59, 62, 90, 93, 94, 103]`. The [source notebook](https://github.com/mvoassis/dropout_prediction/blob/main/notebooks/Student_dropout_predictor.ipynb) identifies these same eight rows as enrolled students with at least 15 attendance failures, calls them likely dropout cases, removes them from the main data, and keeps them for a separate test. The notebook's saved `dados_full.info()` output then reports 113 rows, and the published CSV contains those same index gaps.

This is not ordinary random test-set selection. Rows were selected using the observed label and a strong predictor, removing difficult/ambiguous negatives from training and conventional evaluation. It can inflate apparent performance and distort the class boundary. The original 121-row anonymized dataset—including these eight rows—should be recovered for an unbiased split. Their official label should not be silently changed; instead, establish a later outcome date or mark them as censored/unknown.

# Recommended Prediction Point

**Make one prediction immediately after second-semester results are finalized: predict whether a student will subsequently drop out, using only admission information and academic activity recorded through the end of semester 2.**

This point is defensible because the source application explicitly identifies `DEE374` (Pre-Calculus) and `DEE341` (Programming Laboratory I) as first-semester courses and `DEE345` (Computer Systems Security) as a second-semester course. It provides a full academic year of behavioral/performance evidence while still leaving time for intervention. It also gives every feature a common information cutoff.

The current CSV does **not** directly implement that design. Semester-stamped attempts are needed to reconstruct the cutoff features and to ensure that dropout occurred after the cutoff. Students who dropped out before completing semester 2 need a pre-specified policy (normally excluded from this particular prediction cohort, or handled by a separate admission/semester-1 model).

# Proposed ML Target

Use **binary classification: dropout versus non-dropout after the semester-2 cutoff**.

The repository defines only two labels, and the university-project goal is dropout risk rather than distinguishing graduation, transfer, and active enrollment. For a valid future-outcome target, however, “non-dropout” should mean a sufficiently observed student who remained enrolled or graduated through a fixed follow-up horizon—not simply “enrolled on the extraction date.” A practical label definition would be: dropout within the next defined number of semesters versus no dropout during that same follow-up window. The horizon must be agreed with the university/data owner before training.

# Recommended Input Features

The exact provisional feature list for the next modeling round is:

1. `anoConclusaoEnsinoMedio`
2. `cidade`
3. `formaIngInstituicao`
4. `sexo`
5. `ira` — recomputed through semester 2 only
6. `Cancelado` — count through semester 2 only
7. `Reprovado por frequência` — count through semester 2 only
8. `Reprovado por nota` — count through semester 2 only
9. `DEE374_nota` — first-semester result available by cutoff
10. `DEE374_freq` — first-semester attendance available by cutoff
11. `DEE341_nota` — first-semester result available by cutoff
12. `DEE341_freq` — first-semester attendance available by cutoff
13. `DEE345_nota` — second-semester result available by cutoff
14. `DEE345_freq` — second-semester attendance available by cutoff

This list is **conditional on reconstructing every academic field as of semester 2**. Add explicit missing/not-taken indicators rather than treating all zero values as actual zero performance. `cidade` and `sexo` should be subjected to fairness/sensitivity analysis, and a second feature set excluding them should be compared later. `tempoUniversidade`, all cumulative completion fields, all undocumented later-course fields, the exported index, and of course `situacao` must not be predictors.

# Suitability Decision and Pre-Training Gate

The dataset is suitable for schema exploration and a carefully labelled university-project prototype, but **not suitable as-is for a credible leakage-free early-warning model**. Before model training, the following must be resolved:

1. Recover the eight excluded enrolled rows or obtain the pre-removal anonymized 121-row dataset.
2. Obtain semester/timestamp information and reconstruct all predictor values at the end of semester 2.
3. Define a fixed future follow-up horizon and resolve right-censoring among currently enrolled students.
4. Clarify zero sentinels, course mappings/semesters, curriculum acronyms, and the `sexo` encoding.
5. Confirm permission/licensing for use and redistribution.

No model should be trained on the published snapshot until at least items 1–3 are addressed or the project explicitly narrows its claim to retrospective status classification rather than future dropout prediction.
