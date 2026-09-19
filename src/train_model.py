"""Train and evaluate the enrollment-time student dropout classifier.

Running this module from the project root performs the complete reproducible
workflow: cohort preparation, train/test split, cross-validation, training-only
threshold selection, one final held-out evaluation, reports, figures, and model
serialization. It intentionally does not implement the Streamlit application.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cse-dropout-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "student_dropout.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "dropout_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "model_evaluation.md"
FAIRNESS_REPORT_PATH = REPORTS_DIR / "fairness_analysis.md"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Target"
TARGET_MAPPING = {"Graduate": 0, "Dropout": 1}
DATASET_SOURCE = (
    "https://archive.ics.uci.edu/dataset/697/"
    "predict+students+dropout+and+academic+success"
)

# Exact list approved in reports/dataset_analysis.md. Do not add predictors here
# without first revising and reviewing that analysis.
FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Integer category codes are nominal and must be one-hot encoded. Application
# order is intentionally numeric because UCI defines an order from first to last
# choice rather than unrelated category labels.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERIC_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def load_raw_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the untouched UCI CSV and normalize header whitespace in memory."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, sep=detect_delimiter(path), encoding="utf-8-sig")
    df.columns = [str(column).strip() for column in df.columns]
    return df


def prepare_modeling_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return resolved outcomes and the exact approved feature matrix."""
    required = set(FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    unexpected_targets = sorted(set(df[TARGET_COLUMN].dropna()) - {"Graduate", "Dropout", "Enrolled"})
    if unexpected_targets:
        raise ValueError(f"Unexpected target labels: {unexpected_targets}")

    cohort = df[df[TARGET_COLUMN].isin(TARGET_MAPPING)].copy()
    X = cohort.loc[:, FEATURES].copy()
    y = cohort[TARGET_COLUMN].map(TARGET_MAPPING).astype("int8")

    missing = X.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"Unexpected missing predictor values:\n{missing.to_string()}")
    if y.isna().any():
        raise ValueError("Target mapping produced missing values.")

    numeric_values = X[NUMERIC_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric predictors contain infinite or non-numeric values.")
    return X, y


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer: StandardScaler | str = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipelines() -> dict[str, Pipeline]:
    """Create conservative pipelines with all preprocessing inside each model."""
    logistic = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"Logistic Regression": logistic, "Random Forest": forest}


def summarize_cross_validation(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, dict[str, dict[str, float]]]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        summaries[model_name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"], ddof=1)),
            }
            for metric in SCORING
        }
    return summaries


def select_model(cv_summaries: dict[str, dict[str, dict[str, float]]]) -> str:
    """Select by F1 first, then ROC-AUC and recall; never by accuracy alone."""
    return max(
        cv_summaries,
        key=lambda name: (
            cv_summaries[name]["f1"]["mean"],
            cv_summaries[name]["roc_auc"]["mean"],
            cv_summaries[name]["recall"]["mean"],
        ),
    )


def threshold_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in np.round(np.arange(0.30, 0.701, 0.05), 2):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(results: pd.DataFrame) -> tuple[float, str]:
    """Prefer useful recall gain while retaining acceptable precision."""
    default = results.loc[np.isclose(results["threshold"], 0.50)].iloc[0]
    candidates = results[
        (results["threshold"] <= 0.50)
        & (results["recall"] >= default["recall"] + 0.02)
        & (results["precision"] >= 0.55)
    ]
    if candidates.empty:
        return 0.50, (
            "The default threshold was retained because no tested lower threshold improved "
            "training out-of-fold recall by at least 0.02 while keeping precision at or above 0.55."
        )
    selected = candidates.sort_values(
        ["f1", "recall", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return float(selected["threshold"]), (
        "Selected from training out-of-fold predictions. It improved dropout recall by at least "
        "0.02 versus 0.50, retained precision of at least 0.55, and had the best F1 among thresholds "
        "meeting those safeguards."
    )


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, predictions, matrix


def aggregate_feature_effects(
    fitted_pipeline: Pipeline, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Aggregate encoded effects to original fields and optionally return LR category effects."""
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]
    encoder: OneHotEncoder = preprocessor.named_transformers_["categorical"]
    model = fitted_pipeline.named_steps["model"]

    if model_name == "Random Forest":
        encoded_values = np.asarray(model.feature_importances_, dtype=float)
        aggregation = "sum"
        encoded_signed = None
    else:
        encoded_signed = np.asarray(model.coef_[0], dtype=float)
        encoded_values = np.abs(encoded_signed)
        aggregation = "mean"

    rows: list[dict[str, float | str]] = []
    encoded_rows: list[dict[str, float | str]] = []
    offset = 0
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True):
        width = len(categories)
        values = encoded_values[offset : offset + width]
        importance = float(values.sum() if aggregation == "sum" else values.mean())
        rows.append({"feature": feature, "importance": importance})
        if encoded_signed is not None:
            for category, coefficient in zip(
                categories, encoded_signed[offset : offset + width], strict=True
            ):
                encoded_rows.append(
                    {
                        "encoded_feature": f"{feature}={category}",
                        "coefficient": float(coefficient),
                    }
                )
        offset += width

    for feature in NUMERIC_FEATURES:
        rows.append({"feature": feature, "importance": float(encoded_values[offset])})
        if encoded_signed is not None:
            encoded_rows.append(
                {"encoded_feature": feature, "coefficient": float(encoded_signed[offset])}
            )
        offset += 1

    if offset != len(encoded_values):
        raise RuntimeError("Encoded feature-effect length did not match preprocessing output.")

    summary = pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)
    encoded = pd.DataFrame(encoded_rows) if encoded_rows else None
    return summary, encoded


def save_confusion_matrix(matrix: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Graduate", "Dropout"],
        yticklabels=["Graduate", "Dropout"],
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Held-out test confusion matrix",
    )
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:d}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true: pd.Series, probabilities: np.ndarray, auc_value: float) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities, pos_label=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive_rate, true_positive_rate, color="#245b8a", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"Held-out ROC curve (AUC = {auc_value:.3f})",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(feature_summary: pd.DataFrame, model_name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = feature_summary.head(10).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#4878a8")
    measure = (
        "Aggregated impurity importance"
        if model_name == "Random Forest"
        else "Mean absolute standardized coefficient"
    )
    ax.set(xlabel=measure, title=f"Top original features: {model_name}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_fairness_groups(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = X_test.copy()
    frame["actual"] = np.asarray(y_test)
    frame["predicted"] = predictions
    frame["Age group"] = pd.cut(
        frame["Age at enrollment"],
        bins=[-np.inf, 20, 25, np.inf],
        labels=["17-20", "21-25", "26+"],
    )

    group_specs = {
        "Gender": {0: "Female", 1: "Male"},
        "Scholarship holder": {0: "No", 1: "Yes"},
        "Debtor": {0: "No", 1: "Yes"},
        "Age group": None,
    }
    rows: list[dict[str, float | int | str]] = []
    skipped: list[str] = []
    for attribute, labels in group_specs.items():
        for value, subset in frame.groupby(attribute, observed=True):
            label = labels.get(value, str(value)) if labels else str(value)
            positives = int(subset["actual"].sum())
            negatives = int(len(subset) - positives)
            if len(subset) < 30 or positives < 10 or negatives < 10:
                skipped.append(
                    f"{attribute}={label} (n={len(subset)}, dropouts={positives}, graduates={negatives})"
                )
                continue
            rows.append(
                {
                    "attribute": attribute,
                    "group": label,
                    "sample_count": int(len(subset)),
                    "dropout_prevalence": float(subset["actual"].mean()),
                    "precision": float(
                        precision_score(subset["actual"], subset["predicted"], zero_division=0)
                    ),
                    "recall": float(
                        recall_score(subset["actual"], subset["predicted"], zero_division=0)
                    ),
                    "f1": float(f1_score(subset["actual"], subset["predicted"], zero_division=0)),
                }
            )

    results = pd.DataFrame(rows)
    findings: list[str] = []
    if not results.empty:
        for attribute, group in results.groupby("attribute"):
            if len(group) < 2:
                continue
            recall_gap = float(group["recall"].max() - group["recall"].min())
            precision_gap = float(group["precision"].max() - group["precision"].min())
            if recall_gap >= 0.15 or precision_gap >= 0.15:
                findings.append(
                    f"{attribute}: recall gap {recall_gap:.3f}, precision gap {precision_gap:.3f}."
                )
    return results, skipped, findings


def format_cv_detail(summary: dict[str, dict[str, float]]) -> str:
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision (Dropout)",
        "recall": "Recall (Dropout)",
        "f1": "F1 (Dropout)",
        "roc_auc": "ROC-AUC",
    }
    lines = ["| Metric | Mean | Standard deviation |", "|---|---:|---:|"]
    for metric in SCORING:
        lines.append(
            f"| {labels[metric]} | {summary[metric]['mean']:.3f} | {summary[metric]['std']:.3f} |"
        )
    return "\n".join(lines)


def write_model_evaluation_report(
    *,
    total_count: int,
    class_counts: dict[int, int],
    train_count: int,
    test_count: int,
    train_counts: dict[int, int],
    test_counts: dict[int, int],
    cv_summaries: dict[str, dict[str, dict[str, float]]],
    selected_model: str,
    thresholds: pd.DataFrame,
    selected_threshold: float,
    threshold_reason: str,
    test_metrics: dict[str, float | int],
    feature_summary: pd.DataFrame,
    encoded_effects: pd.DataFrame | None,
) -> None:
    comparison_rows = []
    for model_name, summary in cv_summaries.items():
        comparison_rows.append(
            f"| {model_name} | {summary['accuracy']['mean']:.3f} | "
            f"{summary['precision']['mean']:.3f} | {summary['recall']['mean']:.3f} | "
            f"{summary['f1']['mean']:.3f} | {summary['roc_auc']['mean']:.3f} |"
        )

    threshold_rows = [
        "| Threshold | Precision | Recall | F1 | False positives | False negatives |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in thresholds.itertuples(index=False):
        selected_mark = " **(selected)**" if np.isclose(row.threshold, selected_threshold) else ""
        threshold_rows.append(
            f"| {row.threshold:.2f}{selected_mark} | {row.precision:.3f} | {row.recall:.3f} | "
            f"{row.f1:.3f} | {row.false_positives} | {row.false_negatives} |"
        )

    top_rows = ["| Rank | Original feature | Association measure |", "|---:|---|---:|"]
    for rank, row in enumerate(feature_summary.head(10).itertuples(index=False), start=1):
        top_rows.append(f"| {rank} | `{row.feature}` | {row.importance:.4f} |")

    coefficient_section = ""
    if encoded_effects is not None:
        strongest_positive = encoded_effects.nlargest(8, "coefficient")
        strongest_negative = encoded_effects.nsmallest(8, "coefficient")
        coefficient_lines = [
            "\nFor Logistic Regression, positive coefficients are associated with a higher predicted "
            "dropout log-odds and negative coefficients with lower predicted dropout log-odds, holding "
            "other encoded inputs constant. These are associations, not causal effects.\n",
            "| Strong positive encoded term | Coefficient | Strong negative encoded term | Coefficient |",
            "|---|---:|---|---:|",
        ]
        for positive, negative in zip(
            strongest_positive.itertuples(index=False),
            strongest_negative.itertuples(index=False),
            strict=True,
        ):
            coefficient_lines.append(
                f"| `{positive.encoded_feature}` | {positive.coefficient:.3f} | "
                f"`{negative.encoded_feature}` | {negative.coefficient:.3f} |"
            )
        coefficient_section = "\n".join(coefficient_lines)

    selected_cv = cv_summaries[selected_model]
    f1_gap = float(test_metrics["f1"]) - selected_cv["f1"]["mean"]
    auc_gap = float(test_metrics["roc_auc"]) - selected_cv["roc_auc"]["mean"]
    overfit_text = (
        "There is no clear evidence of serious overfitting: held-out F1 and ROC-AUC are within "
        "0.10 of the selected model's cross-validation means."
        if f1_gap >= -0.10 and auc_gap >= -0.10
        else "Held-out performance falls more than 0.10 below cross-validation on F1 or ROC-AUC, which is concerning evidence of overfitting or split instability."
    )

    model_tradeoff = (
        "The selected model had the highest mean cross-validated dropout F1. ROC-AUC and recall were "
        "used as tie-breakers, so accuracy alone did not determine the choice. Logistic Regression "
        "offers simpler coefficient interpretation; Random Forest can capture nonlinear relationships."
    )
    content = f"""# Modeling Setup

- Dataset: [UCI Predict Students' Dropout and Academic Success]({DATASET_SOURCE})
- Prediction point: enrollment, before first-semester teaching
- Target: `Dropout = 1`, `Graduate = 0`
- Excluded outcome: 794 `Enrolled` rows because outcomes are unresolved
- Resolved cohort: {total_count:,} students ({class_counts[1]:,} dropout; {class_counts[0]:,} graduate)
- Input features: {len(FEATURES)} enrollment-time/contextual variables
- Held-out split: 80/20 with `random_state=42` and target stratification
- Training set: {train_count:,} ({train_counts[1]:,} dropout; {train_counts[0]:,} graduate)
- Test set: {test_count:,} ({test_counts[1]:,} dropout; {test_counts[0]:,} graduate)
- Cross-validation: 5-fold `StratifiedKFold`, shuffled with `random_state=42`, training set only
- Leakage control: splitting occurs before fitting; encoding/scaling and the estimator remain inside each scikit-learn pipeline

# Logistic Regression

Logistic Regression uses balanced class weights, one-hot encoding for nominal code fields, and standardized numeric features.

{format_cv_detail(cv_summaries['Logistic Regression'])}

# Random Forest

Random Forest uses 400 trees, maximum depth 12, minimum leaf size 5, balanced-subsample weights, one-hot categorical inputs, and unscaled numeric inputs.

{format_cv_detail(cv_summaries['Random Forest'])}

# Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

{model_tradeoff}

# Threshold Selection

Thresholds were evaluated using out-of-fold probabilities generated only from the training set. The held-out test set was not used for model or threshold selection.

{chr(10).join(threshold_rows)}

- Default threshold: 0.50
- Selected threshold: **{selected_threshold:.2f}**
- Reason: {threshold_reason}

# Final Model

**{selected_model}** was selected. It is saved as a complete fitted pipeline, including all preprocessing, at `models/dropout_model.joblib`.

# Held-Out Test Performance

| Metric | Value |
|---|---:|
| Accuracy | {float(test_metrics['accuracy']):.3f} |
| Precision (Dropout) | {float(test_metrics['precision']):.3f} |
| Recall (Dropout) | {float(test_metrics['recall']):.3f} |
| F1 (Dropout) | {float(test_metrics['f1']):.3f} |
| ROC-AUC | {float(test_metrics['roc_auc']):.3f} |

The selected threshold was fixed before this one-time test evaluation. {overfit_text}

# Confusion Matrix Interpretation

- Correctly detected dropouts (true positives): **{test_metrics['tp']}**
- Missed dropouts (false negatives): **{test_metrics['fn']}**
- False dropout warnings for graduates (false positives): **{test_metrics['fp']}**
- Correctly identified graduates (true negatives): **{test_metrics['tn']}**

# Feature Interpretation

The plot and table aggregate one-hot category contributions back to their original source fields where practical. Larger values mean the feature was more influential in the fitted model's predictions; they do not mean the feature caused dropout. High-cardinality variables can receive more total Random Forest importance.

{chr(10).join(top_rows)}
{coefficient_section}

# Limitations

- Data comes from one Portuguese higher-education institution and may not generalize to Bangladesh or other institutions.
- The dataset combines multiple study programs rather than one Computer Science program.
- Dropout dates are unavailable, which prevents a validated post-semester future-outcome cutoff.
- Socioeconomic and economic-context variables can change across countries and cohorts.
- Sensitive and proxy variables require subgroup monitoring and careful support-oriented governance.
- A random split cannot test future-cohort drift because enrollment dates are unavailable.
- The model estimates statistical risk, not certainty or causation.
"""
    EVALUATION_REPORT_PATH.write_text(content, encoding="utf-8")


def write_fairness_report(
    results: pd.DataFrame,
    skipped: list[str],
    findings: list[str],
    selected_model: str,
    threshold: float,
) -> None:
    table_lines = [
        "| Attribute | Group | Sample count | Dropout prevalence | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        table_lines.append(
            f"| {row.attribute} | {row.group} | {row.sample_count} | "
            f"{row.dropout_prevalence:.3f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} |"
        )
    finding_lines = (
        [f"- {finding}" for finding in findings]
        if findings
        else ["- No eligible attribute showed a precision or recall gap of 0.15 or more on this split."]
    )
    skipped_lines = [f"- {item}" for item in skipped] or ["- None"]
    content = f"""# Fairness and Sensitivity Analysis

This is a diagnostic review of the final held-out test predictions from **{selected_model}** at threshold **{threshold:.2f}**. It does not establish that the model is fair. The test set was examined only after model and threshold selection and was not used for tuning.

Groups are reported only when they contain at least 30 students, 10 dropouts, and 10 graduates.

{chr(10).join(table_lines)}

## Notable differences

{chr(10).join(finding_lines)}

Differences may reflect prevalence, sample composition, historical processes, or random test-split variation. They should not be interpreted as inherent differences in student ability.

## Groups not reported because of sample size

{chr(10).join(skipped_lines)}

## Use limitations

- Gender, age, debt, scholarship, nationality, disability-related information, and family background are sensitive or proxy variables.
- Predictions should be used to offer support, never to deny admission, aid, or educational opportunity.
- Subgroup metrics need external validation on the intended institution and future cohorts.
- Small-group metrics should remain suppressed or explicitly qualified.
"""
    FAIRNESS_REPORT_PATH.write_text(content, encoding="utf-8")


def save_metadata(
    *,
    selected_model: str,
    threshold: float,
    threshold_reason: str,
    train_count: int,
    test_count: int,
    cv_summary: dict[str, dict[str, float]],
    test_metrics: dict[str, float | int],
) -> None:
    metadata: dict[str, Any] = {
        "model_name": selected_model,
        "threshold": threshold,
        "threshold_selection": threshold_reason,
        "feature_list": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_mapping": TARGET_MAPPING,
        "excluded_target": "Enrolled",
        "prediction_point": "Enrollment, before first-semester teaching begins",
        "dataset_source": DATASET_SOURCE,
        "dataset_path": "data/student_dropout.csv",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cross_validation": "5-fold StratifiedKFold(shuffle=True, random_state=42)",
        "training_rows": train_count,
        "test_rows": test_count,
        "selected_model_cv": cv_summary,
        "held_out_test_metrics": test_metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    X, y = prepare_modeling_data(raw)
    class_counts = {int(label): int(count) for label, count in y.value_counts().items()}
    print(f"Resolved cohort: {len(y)}")
    print(
        f"Dropout: {class_counts[1]} ({class_counts[1] / len(y) * 100:.2f}%); "
        f"Graduate: {class_counts[0]} ({class_counts[0] / len(y) * 100:.2f}%)"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_counts = {int(label): int(count) for label, count in y_train.value_counts().items()}
    test_counts = {int(label): int(count) for label, count in y_test.value_counts().items()}
    print(f"Training set: {len(y_train)}; classes={train_counts}")
    print(f"Held-out test set: {len(y_test)}; classes={test_counts}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipelines = build_model_pipelines()
    cv_summaries = summarize_cross_validation(pipelines, X_train, y_train, cv)
    for name, summary in cv_summaries.items():
        print(f"\n{name} cross-validation")
        for metric, values in summary.items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    selected_model = select_model(cv_summaries)
    selected_pipeline = pipelines[selected_model]
    print(f"\nSelected model from training CV: {selected_model}")

    out_of_fold_probabilities = cross_val_predict(
        selected_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    thresholds = threshold_table(y_train, out_of_fold_probabilities)
    selected_threshold, threshold_reason = select_threshold(thresholds)
    print("\nTraining out-of-fold threshold analysis")
    print(thresholds.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Selected threshold: {selected_threshold:.2f}")

    selected_pipeline.fit(X_train, y_train)
    test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
    test_metrics, test_predictions, matrix = classification_metrics(
        y_test, test_probabilities, selected_threshold
    )
    print("\nFinal held-out test performance")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {metric}: {float(test_metrics[metric]):.4f}")
    print(
        f"  TP={test_metrics['tp']} TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} FN={test_metrics['fn']}"
    )

    feature_summary, encoded_effects = aggregate_feature_effects(
        selected_pipeline, selected_model
    )
    print("\nTop 10 original features associated with predictions")
    print(feature_summary.head(10).to_string(index=False))

    fairness_results, fairness_skipped, fairness_findings = evaluate_fairness_groups(
        X_test, y_test, test_predictions
    )

    save_confusion_matrix(matrix)
    save_roc_curve(y_test, test_probabilities, float(test_metrics["roc_auc"]))
    save_feature_importance_plot(feature_summary, selected_model)
    joblib.dump(selected_pipeline, MODEL_PATH)
    save_metadata(
        selected_model=selected_model,
        threshold=selected_threshold,
        threshold_reason=threshold_reason,
        train_count=len(y_train),
        test_count=len(y_test),
        cv_summary=cv_summaries[selected_model],
        test_metrics=test_metrics,
    )
    write_model_evaluation_report(
        total_count=len(y),
        class_counts=class_counts,
        train_count=len(y_train),
        test_count=len(y_test),
        train_counts=train_counts,
        test_counts=test_counts,
        cv_summaries=cv_summaries,
        selected_model=selected_model,
        thresholds=thresholds,
        selected_threshold=selected_threshold,
        threshold_reason=threshold_reason,
        test_metrics=test_metrics,
        feature_summary=feature_summary,
        encoded_effects=encoded_effects,
    )
    write_fairness_report(
        fairness_results,
        fairness_skipped,
        fairness_findings,
        selected_model,
        selected_threshold,
    )
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved evaluation report: {EVALUATION_REPORT_PATH}")
    print(f"Saved fairness report: {FAIRNESS_REPORT_PATH}")


if __name__ == "__main__":
    main()
