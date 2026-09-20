# Machine Learning-Based Student Profiling and Dropout Risk Factor Analysis among Bangladeshi University Students

This university project combines unsupervised student profiling with exploratory analysis of self-reported dropout consideration. It does **not** predict whether an individual student will drop out.

## Dataset and cleaning

The active **Student Dropout Risk Survey Dataset** covers Bangladesh university students. It contains academic, socioeconomic, and lifestyle responses plus a 1–5 question about considering dropout because of academic stress.

- Raw responses: 368
- Duplicate copies removed: 17
- Cleaned sample: 351 unique responses
- Duplicate groups: 16 near-immediate repeated submissions and 3 repeated submissions
- No respondent information is displayed

## K-Means methodology

The saved scikit-learn pipeline applies one-hot encoding to nominal variables, defensible ordinal encoding to ordered variables, scaling where appropriate, and K-Means clustering. The target response, Timestamp, derived risk classes, and academic-overwhelm frequency were excluded from clustering.

The 12 inputs are age group, academic level, personal income, employment type, internet quality, living arrangement, study routine, family income, scholarship/stipend, study-space quality, academic-resource access, and current GPA.

Values of k from 2 through 5 were evaluated using silhouette score, Davies–Bouldin index, cluster sizes, interpretability, and stability across random seeds.

| Result | Value |
|---|---:|
| Profile 1 size | 213 (60.7%) |
| Profile 2 size | 138 (39.3%) |
| Silhouette score | 0.1335 |
| Davies–Bouldin score | 2.2859 |
| Mean pairwise ARI | 0.9894 |
| Minimum ARI | 0.9772 |

The profiles are highly reproducible but weakly separated geometrically:

- **Profile 1 — Early-Stage, Mostly Non-Working Students**
- **Profile 2 — Advanced-Stage, Working Students**

## Exploratory factor analysis

Elevated dropout consideration is descriptively defined as responses 4–5; responses 1–3 are Not Elevated. The cleaned sample contains 77 Elevated responses (21.9%) and 274 Not Elevated responses (78.1%). This grouping is not an ML training label.

Among the stronger associations observed were study-material satisfaction (Cramér’s V 0.185), scholarship/stipend (0.171), study-space quality (0.170), employment type (0.169), prior residence (0.167), and sleep duration (0.153). These are small exploratory associations and do not establish causation.

Profile 1 contained 20.7% Elevated responses and Profile 2 contained 23.9%. The difference was not statistically significant: χ²(1)=0.346, p=0.557, Cramér’s V=0.038. Profile membership is not a risk category.

## Website functionality

The Streamlit website provides four pages:

1. Home
2. Student Profile Explorer
3. Dropout Risk Factor Analysis
4. Methodology & Dataset

The explorer loads the saved profiler with Streamlit resource caching, accepts the exact 12 inputs, and assigns the nearest descriptive profile. It never retrains in the application and never reports an individual forecast.

## Key files

- `app.py` — active four-page interface
- `models/bangladesh_student_profiler.joblib` — preprocessing and K-Means pipeline
- `models/bangladesh_student_profiler_metadata.json` — schema and analysis metadata
- `reports/bangladesh_risk/profile_definitions.json` — profile descriptions
- `reports/bangladesh_risk/figures/cluster_pca.png` — verified PCA view
- `reports/bangladesh_risk/clustering_analysis.md` — clustering report
- `reports/bangladesh_risk/factor_analysis.md` — association report
- `archive/uci-portugal-version-2026-09-20/` — recoverable historical implementation

## Requirements and local use

Python 3.12 is recommended. The active application requires Streamlit, pandas, NumPy, scikit-learn, joblib, and Matplotlib. Other retained dependencies support archived research scripts.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

The application uses repository-relative paths and needs no secret or external service.

## Limitations

- Small, self-reported, cross-sectional sample
- Survey-selection bias may exist
- Not nationally representative
- No confirmed future dropout outcome
- Profiles depend on selected variables and preprocessing
- Weak silhouette separation
- Profile membership was not significantly associated with dropout consideration
- Factor associations are exploratory; association does not imply causation

Approved claim: “This system uses unsupervised machine learning to identify common student profiles from academic, socioeconomic and lifestyle characteristics and examines how those profiles relate to self-reported dropout consideration.”
