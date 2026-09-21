# Early Student Dropout Risk Prediction Using Machine Learning

This university project estimates the probability that a newly enrolled student may later drop out. It is intended as an early-support prototype, not an automated decision system.

The Streamlit website accepts the 24 demographic, application, financial, and contextual fields available at enrollment. It loads a previously fitted scikit-learn pipeline, reports the dropout probability, applies the saved 0.40 decision threshold, and presents a careful per-student model explanation.

## Dataset

The active dataset is [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), DOI [`10.24432/C5MC89`](https://doi.org/10.24432/C5MC89), licensed under CC BY 4.0.

- Original dataset: 4,424 students, 36 predictors, and one target
- Original outcomes: 2,209 Graduate, 1,421 Dropout, and 794 Enrolled
- Modeling cohort: 3,630 students with resolved outcomes
- Binary target: `Dropout = 1`, `Graduate = 0`
- Prediction point: enrollment, before first-semester teaching

Students whose outcome is still `Enrolled` are excluded from modeling. First- and second-semester academic variables are also excluded so that every predictor is available at the stated prediction point.

## Final model

The selected model is a class-weighted Logistic Regression pipeline with one-hot encoding for categorical fields and standardization for numerical fields. Model selection and threshold selection use only training-set cross-validation and out-of-fold predictions. The saved pipeline contains preprocessing and the fitted classifier together.

Held-out test performance at the saved 0.40 threshold:

| Metric | Result |
|---|---:|
| Accuracy | 73.55% |
| Precision | 62.04% |
| Recall | 83.45% |
| F1-score | 71.17% |
| ROC-AUC | 85.48% |

The confusion matrix contains 237 correctly detected dropouts, 47 missed dropouts, 145 false dropout warnings, and 297 correctly identified graduates. Recall is prioritized because the intended use is to identify students who may benefit from supportive review.

## Technology stack

- Python
- Pandas and NumPy
- scikit-learn
- Matplotlib
- Streamlit
- pytest

## Project structure

```text
cse-dropout-prediction/
├── .streamlit/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Important files:

- `data/student_dropout.csv` — active UCI dataset
- `src/inspect_dataset.py` — dataset inspection and leakage-oriented checks
- `src/train_model.py` — reproducible offline training and evaluation
- `models/dropout_model.joblib` — fitted preprocessing-and-classifier pipeline
- `models/model_metadata.json` — input schema, threshold, and metrics
- `reports/dataset_analysis.md` — dataset and leakage analysis
- `reports/model_evaluation.md` — model evaluation details
- `reports/fairness_analysis.md` — subgroup diagnostics and limitations
- `reports/local_validation_plan.md` — plan for a future Bangladeshi external-validation study
- `.streamlit/config.toml` — minimal light-theme configuration; contains no secrets
- `app.py` — four-page Streamlit interface; inference only

## Installation

**Recommended deployment version: Python 3.12.** Streamlit Community Cloud currently defaults to Python 3.12, and every direct dependency in this project declares Python 3.12 support. The current local machine has only Python 3.14.5 installed, so the completed test run described below used 3.14.5 rather than pretending that 3.12 was tested locally.

The scikit-learn version is fixed at 1.9.1 because the saved model artifact was created with that version. From the project directory, create and activate a virtual environment if desired, then install the requirements:

```bash
python -m pip install -r requirements.txt
```

## Inspect the dataset

```bash
python src/inspect_dataset.py
```

## Train and save the model

The repository includes the validated artifacts. To reproduce them locally:

```bash
python src/train_model.py
```

Training is an offline step. The Streamlit application never fits or retrains a model.

## Run the website

```bash
python -m streamlit run app.py
```

Use the sidebar to open Home, Predict Dropout Risk, Model Performance, and About Dataset.

## Deployment

The project is prepared for [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). It uses repository-relative `pathlib` paths and needs no database, external service, environment variable, or secret.

1. Create a GitHub repository and push the complete project, including the saved model, metadata, figures, requirements, and Streamlit configuration.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app**.
4. Choose the GitHub repository and branch.
5. Set the entrypoint file to `app.py`.
6. Open **Advanced settings** and select **Python 3.12**.
7. Leave the secrets field empty; this project requires no secrets.
8. Deploy the app.
9. After deployment, verify all four pages, confirm that all three evaluation figures appear, and submit one sample prediction.

Before pushing, confirm that these deployment-critical files are committed:

```text
app.py
requirements.txt
.streamlit/config.toml
models/dropout_model.joblib
models/model_metadata.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/feature_importance.png
```

`.streamlit/secrets.toml` is intentionally excluded by `.gitignore` and must never be committed if secrets are added in the future.

## Run tests

```bash
python -m pytest -q
```

## Responsible-use limitations

The data comes from one Portuguese higher-education institution, covers multiple degree programs, and does not include exact dropout dates. Results may not transfer to Bangladesh or another university without local validation. Demographic and socioeconomic fields can introduce fairness concerns. Predictions are statistical associations—not causal findings—and must not be the sole basis for academic, financial, disciplinary, or admission decisions.

See [`reports/local_validation_plan.md`](reports/local_validation_plan.md) for a realistic external-validation protocol covering local feature mapping, frozen-model evaluation, calibration, fairness, privacy, and governance. The plan does not fabricate local data or results.
