"""Train and evaluate candidate Bangladesh MICS dropout-risk models.

This script never modifies app.py or the active Portuguese UCI model. It saves
the selected Bangladesh candidate under distinct filenames for review.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mics-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from mics_pipeline import (
    CATEGORICAL_FEATURES,
    FINAL_FEATURES,
    FS_DATA_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    WEBSITE_LABELS,
    construct_dropout_cohort,
    load_fs_dataset,
    make_preprocessor,
    validate_attendance_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "mics_figures"
MODEL_PATH = MODELS_DIR / "mics_bangladesh_dropout_model.joblib"
METADATA_PATH = MODELS_DIR / "mics_bangladesh_model_metadata.json"
RESULTS_PATH = REPORTS_DIR / "mics_training_results.json"

THRESHOLDS = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
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
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=15,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def binary_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        row = {"threshold": float(threshold)}
        row.update(binary_metrics(y_true, probabilities, float(threshold)))
        rows.append(row)
    chosen = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return float(chosen["threshold"]), rows


def aggregate_importance(fitted: Pipeline) -> pd.DataFrame:
    preprocessor = fitted.named_steps["preprocessor"]
    model = fitted.named_steps["model"]
    transformed = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.abs(np.asarray(model.coef_[0], dtype=float))

    original_features: list[str] = []
    ordered = sorted(FINAL_FEATURES, key=len, reverse=True)
    for name in transformed:
        clean = name.split("__", 1)[-1]
        matched = next(
            (feature for feature in ordered if clean == feature or clean.startswith(feature + "_")),
            clean,
        )
        original_features.append(matched)

    result = pd.DataFrame({"feature": original_features, "importance": values})
    aggregation = "sum" if hasattr(model, "feature_importances_") else "max"
    result = result.groupby("feature", as_index=False)["importance"].agg(aggregation)
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def subgroup_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    groups: pd.Series,
) -> list[dict[str, Any]]:
    predictions = (probabilities >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    aligned_groups = groups.reset_index(drop=True)
    aligned_y = y_true.reset_index(drop=True)
    for group in sorted(aligned_groups.dropna().unique(), key=str):
        mask = aligned_groups.eq(group).to_numpy()
        y_group = aligned_y[mask]
        pred_group = predictions[mask]
        if len(y_group) < 30:
            continue
        rows.append(
            {
                "group": str(group),
                "n": int(len(y_group)),
                "dropout_count": int(y_group.sum()),
                "prevalence": float(y_group.mean()),
                "precision": float(precision_score(y_group, pred_group, zero_division=0)),
                "recall": float(recall_score(y_group, pred_group, zero_division=0)),
                "f1": float(f1_score(y_group, pred_group, zero_division=0)),
            }
        )
    return rows


def save_figures(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    importance: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Continue", "Dropout"])
    ax.set_yticks([0, 1], labels=["Continue", "Dropout"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"MICS Test Confusion Matrix (threshold={threshold:.2f})")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_value = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("MICS Held-out ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)

    plot_data = importance.sort_values("importance").tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([WEBSITE_LABELS.get(x, x) for x in plot_data.feature], plot_data.importance)
    ax.set_xlabel("Aggregated model importance")
    ax.set_title("Features Associated with Model Predictions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw, metadata = load_fs_dataset()
    validate_attendance_metadata(metadata)
    X, y, context, audit = construct_dropout_cohort(raw)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    context_test = context.iloc[test_idx].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_results: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in build_models().items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        oof_probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold, threshold_table = choose_threshold(y_train, oof_probabilities)
        cv_summary = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in scoring
        }
        model_results[name] = {
            "cv_at_0_5": cv_summary,
            "oof_selected_threshold": threshold,
            "oof_metrics_at_selected_threshold": binary_metrics(
                y_train, oof_probabilities, threshold
            ),
            "threshold_analysis": threshold_table,
        }

    ranked = sorted(
        model_results,
        key=lambda name: (
            model_results[name]["oof_metrics_at_selected_threshold"]["f1"],
            model_results[name]["oof_metrics_at_selected_threshold"]["roc_auc"],
        ),
        reverse=True,
    )
    selected_name = ranked[0]
    threshold = model_results[selected_name]["oof_selected_threshold"]
    selected = clone(build_models()[selected_name]).fit(X_train, y_train)
    fitted_models[selected_name] = selected

    test_probabilities = selected.predict_proba(X_test)[:, 1]
    test_metrics = binary_metrics(y_test, test_probabilities, threshold)
    test_predictions = (test_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()

    importance = aggregate_importance(selected)
    subgroup = {
        "sex": subgroup_metrics(
            y_test, test_probabilities, threshold, X_test["sex"]
        ),
        "area": subgroup_metrics(
            y_test, test_probabilities, threshold, X_test["area"]
        ),
        "division": subgroup_metrics(
            y_test, test_probabilities, threshold, X_test["division"]
        ),
        "wealth_quintile": subgroup_metrics(
            y_test, test_probabilities, threshold, X_test["wealth_quintile"]
        ),
    }

    weighted = pd.DataFrame(
        {"target": y.to_numpy(), "weight": context["fsweight"].to_numpy()}
    ).groupby("target")["weight"].sum()
    weighted_percent = (weighted / weighted.sum() * 100).to_dict()

    results = {
        "dataset": {
            "path": str(FS_DATA_PATH.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(FS_DATA_PATH),
            "rows": int(raw.shape[0]),
            "columns": int(raw.shape[1]),
        },
        "attendance_variables": {
            "CB7_label": metadata.column_names_to_labels["CB7"],
            "CB7_value_labels": metadata.variable_value_labels["CB7"],
            "CB7_counts": raw["CB7"].value_counts(dropna=False).to_dict(),
            "CB9_label": metadata.column_names_to_labels["CB9"],
            "CB9_value_labels": metadata.variable_value_labels["CB9"],
            "CB9_counts": raw["CB9"].value_counts(dropna=False).to_dict(),
        },
        "cohort_audit": audit,
        "class_distribution": {
            "continuing": int((y == 0).sum()),
            "dropout": int((y == 1).sum()),
            "continuing_percent": float((y == 0).mean() * 100),
            "dropout_percent": float((y == 1).mean() * 100),
            "weighted_continuing_percent": float(weighted_percent[0]),
            "weighted_dropout_percent": float(weighted_percent[1]),
        },
        "features": FINAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split": {
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "test_size": 0.20,
        },
        "models": model_results,
        "selected_model": selected_name,
        "selected_threshold": float(threshold),
        "test_metrics": test_metrics,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "feature_importance": importance.to_dict(orient="records"),
        "subgroup_metrics": subgroup,
        "survey_design": {
            "weight": "fsweight",
            "stratum": "stratum",
            "psu": "PSU",
            "training_weighted": False,
            "descriptive_prevalence_weighted": True,
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_PATH)
    metadata_payload = {
        "status": "candidate_for_review_not_active",
        "model_name": selected_name,
        "threshold": float(threshold),
        "target": "CB9=YES and CB7=NO versus CB9=YES and CB7=YES",
        "positive_class": 1,
        "features": FINAL_FEATURES,
        "website_labels": WEBSITE_LABELS,
        "dataset_sha256": results["dataset"]["sha256"],
        "test_metrics": test_metrics,
        "confusion_matrix": results["confusion_matrix"],
        "training_weighted": False,
        "active_application_model": False,
    }
    METADATA_PATH.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    save_figures(y_test, test_probabilities, threshold, importance)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
