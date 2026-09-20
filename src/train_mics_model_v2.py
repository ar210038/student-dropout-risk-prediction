"""Controlled v2 experiment for the Bangladesh MICS dropout model.

Changes from v1 are deliberately limited to an audited father-education join,
PSU-separated validation, PR-AUC/calibration reporting, and one additional
model (XGBoost).  The target and cohort definition are unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mics-v2-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mics_pipeline import (
    FS_DATA_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    V2_FINAL_FEATURES,
    WEBSITE_LABELS,
    construct_dropout_cohort_v2,
    load_fs_dataset,
    load_household_roster,
    make_preprocessor_v2,
    validate_attendance_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "mics_figures"
MODEL_PATH = MODELS_DIR / "mics_bangladesh_dropout_model_v2.joblib"
METADATA_PATH = MODELS_DIR / "mics_bangladesh_model_metadata_v2.json"
RESULTS_PATH = REPORTS_DIR / "mics_training_results_v2.json"
THRESHOLDS = np.round(np.arange(0.10, 0.7001, 0.025), 3)


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    """Return the three predeclared, conservative candidate pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", make_preprocessor_v2()),
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
            [
                ("preprocessor", make_preprocessor_v2()),
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
        "XGBoost": Pipeline(
            [
                ("preprocessor", make_preprocessor_v2()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        min_child_weight=5,
                        reg_lambda=2.0,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(prediction.mean()),
    }


def threshold_analysis(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        row = {"threshold": float(threshold)}
        row.update(metrics(y_true, probabilities, float(threshold)))
        rows.append(row)

    preferred = [row for row in rows if row["recall"] >= 0.70]
    if preferred:
        selected = max(preferred, key=lambda row: (row["precision"], row["f1"], row["threshold"]))
    else:
        acceptable = [row for row in rows if row["recall"] >= 0.60]
        if acceptable:
            selected = max(acceptable, key=lambda row: (row["precision"], row["f1"], row["threshold"]))
        else:
            selected = max(rows, key=lambda row: (row["f1"], row["recall"]))
    return float(selected["threshold"]), rows


def aggregate_importance(fitted: Pipeline) -> pd.DataFrame:
    names = fitted.named_steps["preprocessor"].get_feature_names_out()
    estimator = fitted.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        aggregation = "sum"
    else:
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        aggregation = "max"
    ordered = sorted(V2_FINAL_FEATURES, key=len, reverse=True)
    originals = []
    for transformed in names:
        clean = transformed.split("__", 1)[-1]
        originals.append(
            next(
                (feature for feature in ordered if clean == feature or clean.startswith(feature + "_")),
                clean,
            )
        )
    frame = pd.DataFrame({"feature": originals, "importance": values})
    return (
        frame.groupby("feature", as_index=False)["importance"]
        .agg(aggregation)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def subgroup_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    groups: pd.Series,
) -> list[dict[str, Any]]:
    prediction = (probabilities >= threshold).astype(int)
    y_aligned = y_true.reset_index(drop=True)
    group_aligned = groups.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for group in sorted(group_aligned.dropna().unique(), key=str):
        mask = group_aligned.eq(group).to_numpy()
        y_group = y_aligned[mask]
        pred_group = prediction[mask]
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
                "predicted_positive_rate": float(pred_group.mean()),
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
    prediction = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Continue", "Dropout"])
    ax.set_yticks([0, 1], labels=["Continue", "Dropout"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Grouped Test Confusion Matrix (threshold={threshold:.3f})")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_v2.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("MICS v2 Grouped Held-out ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve_v2.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)
    prevalence = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, label=f"Average precision = {average_precision:.3f}")
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"No-skill = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("MICS v2 Precision–Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(predicted, observed, marker="o", label="Candidate model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed dropout fraction")
    ax.set_title("MICS v2 Calibration Curve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=180)
    plt.close(fig)

    plot_data = importance.sort_values("importance").tail(11)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([WEBSITE_LABELS.get(x, x) for x in plot_data.feature], plot_data.importance)
    ax.set_xlabel("Aggregated model importance")
    ax.set_title("MICS v2 Features Associated with Predictions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_v2.png", dpi=180)
    plt.close(fig)


def main() -> None:
    raw, source_metadata = load_fs_dataset()
    roster, roster_metadata = load_household_roster()
    validate_attendance_metadata(source_metadata)
    X, y, context, audit = construct_dropout_cohort_v2(raw, roster)
    groups = context["PSU"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    context = context.reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)
    context_test = context.iloc[test_idx].reset_index(drop=True)

    train_psus = set(groups_train.unique())
    test_psus = set(groups_test.unique())
    if train_psus.intersection(test_psus):
        raise RuntimeError("PSU leakage detected between grouped train and test sets.")

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive
    models = build_models(scale_pos_weight)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["precision", "recall", "f1", "roc_auc", "average_precision"]

    cv_fold_audit: list[dict[str, Any]] = []
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X_train, y_train, groups_train), start=1):
        fit_groups = set(groups_train.iloc[fit_idx])
        valid_groups = set(groups_train.iloc[valid_idx])
        if fit_groups.intersection(valid_groups):
            raise RuntimeError(f"PSU leakage detected in CV fold {fold}.")
        cv_fold_audit.append(
            {
                "fold": fold,
                "train_rows": int(len(fit_idx)),
                "validation_rows": int(len(valid_idx)),
                "train_psus": int(len(fit_groups)),
                "validation_psus": int(len(valid_groups)),
                "train_dropout_prevalence": float(y_train.iloc[fit_idx].mean()),
                "validation_dropout_prevalence": float(y_train.iloc[valid_idx].mean()),
            }
        )

    comparison: dict[str, Any] = {}
    oof_probabilities: dict[str, np.ndarray] = {}
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        oof_probabilities[name] = probabilities
        threshold, table = threshold_analysis(y_train, probabilities)
        comparison[name] = {
            "grouped_cv_at_0_5": {
                metric: {
                    "mean": float(np.mean(scores[f"test_{metric}"])),
                    "std": float(np.std(scores[f"test_{metric}"])),
                }
                for metric in scoring
            },
            "oof_selected_threshold": threshold,
            "oof_metrics_at_selected_threshold": metrics(y_train, probabilities, threshold),
            "threshold_analysis": table,
        }

    selected_name = max(
        comparison,
        key=lambda name: (
            comparison[name]["grouped_cv_at_0_5"]["average_precision"]["mean"],
            comparison[name]["grouped_cv_at_0_5"]["roc_auc"]["mean"],
            comparison[name]["grouped_cv_at_0_5"]["f1"]["mean"],
        ),
    )
    threshold = comparison[selected_name]["oof_selected_threshold"]
    selected = clone(models[selected_name]).fit(X_train, y_train)
    test_probabilities = selected.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test, test_probabilities, threshold)
    importance = aggregate_importance(selected)

    subgroups = {
        "sex": subgroup_metrics(y_test, test_probabilities, threshold, X_test["sex"]),
        "area": subgroup_metrics(y_test, test_probabilities, threshold, X_test["area"]),
        "wealth_quintile": subgroup_metrics(
            y_test, test_probabilities, threshold, X_test["wealth_quintile"]
        ),
        "division": subgroup_metrics(
            y_test, test_probabilities, threshold, X_test["division"]
        ),
    }

    results = {
        "status": "controlled_candidate_v2",
        "target_unchanged": True,
        "dataset_rows": int(raw.shape[0]),
        "dataset_columns": int(raw.shape[1]),
        "cohort_audit": audit,
        "features": V2_FINAL_FEATURES,
        "father_education": {
            "source_file": "hl.sav",
            "source_variable": "felevel",
            "variable_label": roster_metadata.column_names_to_labels["felevel"],
            "join_keys": ["HH1", "HH2", "FS3/LN -> HL1"],
            "one_to_one": True,
        },
        "grade_completion": {
            "variable": "CB6",
            "included": False,
            "reason": "Highest-grade completion is measured at survey time and is not explicitly tied to the previous-year cutoff; it may contain post-outcome information.",
        },
        "grouped_split": {
            "total_psus": int(groups.nunique()),
            "train_psus": int(len(train_psus)),
            "test_psus": int(len(test_psus)),
            "psu_overlap": 0,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_dropout_count": int(y_train.sum()),
            "test_dropout_count": int(y_test.sum()),
            "train_dropout_prevalence": float(y_train.mean()),
            "test_dropout_prevalence": float(y_test.mean()),
        },
        "cv_fold_audit": cv_fold_audit,
        "training_scale_pos_weight": float(scale_pos_weight),
        "model_comparison": comparison,
        "selected_model": selected_name,
        "selected_threshold": float(threshold),
        "test_metrics": test_metrics,
        "no_skill_average_precision": float(y_test.mean()),
        "feature_importance": importance.to_dict(orient="records"),
        "subgroup_metrics": subgroups,
        "calibration_applied": False,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_figures(y_test, test_probabilities, threshold, importance)

    # Saving a candidate does not activate it.  It must at least exceed the
    # no-skill AP baseline substantially and meet the training-selected recall
    # objective on the grouped held-out data.
    better_candidate = (
        test_metrics["average_precision"] >= 2 * float(y_test.mean())
        and test_metrics["roc_auc"] > 0.714
        and test_metrics["recall"] >= 0.60
    )
    results["candidate_saved"] = bool(better_candidate)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if better_candidate:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(selected, MODEL_PATH)
        metadata = {
            "status": "candidate_v2_for_review_not_active",
            "model_name": selected_name,
            "threshold": float(threshold),
            "target": "CB9=YES and CB7=NO versus CB9=YES and CB7=YES",
            "positive_class": 1,
            "features": V2_FINAL_FEATURES,
            "website_labels": WEBSITE_LABELS,
            "grouped_by": "PSU",
            "test_metrics": test_metrics,
            "active_application_model": False,
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
