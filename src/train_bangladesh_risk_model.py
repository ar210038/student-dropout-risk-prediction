"""Train the final Bangladesh self-reported dropout-risk prototype."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from bangladesh_risk_pipeline import (
    NUMERIC_RATING_FEATURES,
    PRIMARY_CANDIDATE_FEATURES,
    QUESTION_TO_FEATURE,
    RISK_ORDER,
    TIMESTAMP_QUESTION,
    load_and_clean,
    primary_model_frame,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "bangladesh_risk" / "raw" / "Student Dropout Risk Survey Dataset.xlsx"
PROCESSED_DIR = ROOT / "data" / "bangladesh_risk" / "processed"
REPORT_DIR = ROOT / "reports" / "bangladesh_risk"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_PATH = ROOT / "models" / "bangladesh_risk_model.joblib"
METADATA_PATH = ROOT / "models" / "bangladesh_risk_model_metadata.json"
RESULTS_PATH = REPORT_DIR / "training_results.json"

LABEL_TO_ID = {label: index for index, label in enumerate(RISK_ORDER)}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

# Frozen after training-only permutation analysis, with usability/fairness review.
FINAL_FEATURES = [
    "academic_level",
    "sleep_duration",
    "employment_type",
    "commute_time",
    "internet_quality",
    "study_space_quality",
    "current_gpa",
    "study_routine",
    "attendance",
    "scholarship",
    "academic_resource_access",
    "study_material_satisfaction",
]


def make_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    numeric = [feature for feature in features if feature in NUMERIC_RATING_FEATURES]
    categorical = [feature for feature in features if feature not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            ("numeric", Pipeline(numeric_steps), numeric),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Multinomial Logistic Regression": Pipeline(
            [
                ("preprocess", make_preprocessor(FINAL_FEATURES, True)),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", make_preprocessor(FINAL_FEATURES, False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", make_preprocessor(FINAL_FEATURES, False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=160,
                        max_depth=2,
                        learning_rate=0.04,
                        min_child_weight=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=3.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": {
            ID_TO_LABEL[index]: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
            for index in range(3)
        },
    }


def repeated_cv(model_name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    fold_metrics = []
    for train_index, valid_index in splitter.split(X, y):
        fit_kwargs = {}
        if model_name == "XGBoost":
            fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
                "balanced", y.iloc[train_index]
            )
        model.fit(X.iloc[train_index], y.iloc[train_index], **fit_kwargs)
        predictions = model.predict(X.iloc[valid_index])
        fold_metrics.append(score_predictions(y.iloc[valid_index].to_numpy(), predictions))

    scalar_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]
    summary = {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_metrics])),
            "std": float(np.std([fold[name] for fold in fold_metrics], ddof=1)),
        }
        for name in scalar_names
    }
    summary["per_class"] = {
        label: {
            metric: {
                "mean": float(
                    np.mean([fold["per_class"][label][metric] for fold in fold_metrics])
                ),
                "std": float(
                    np.std(
                        [fold["per_class"][label][metric] for fold in fold_metrics],
                        ddof=1,
                    )
                ),
            }
            for metric in ["precision", "recall", "f1"]
        }
        for label in RISK_ORDER
    }
    summary["fold_count"] = len(fold_metrics)
    return summary


def training_only_importance(
    model_name: str, X: pd.DataFrame, y: pd.Series
) -> list[dict]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    values = {feature: [] for feature in FINAL_FEATURES}
    for train_index, valid_index in splitter.split(X, y):
        model = candidate_models()[model_name]
        fit_kwargs = {}
        if model_name == "XGBoost":
            fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
                "balanced", y.iloc[train_index]
            )
        model.fit(X.iloc[train_index], y.iloc[train_index], **fit_kwargs)
        importance = permutation_importance(
            model,
            X.iloc[valid_index],
            y.iloc[valid_index],
            scoring="f1_macro",
            n_repeats=20,
            random_state=42,
        )
        for feature, value in zip(FINAL_FEATURES, importance.importances_mean):
            values[feature].append(value)
    return sorted(
        [
            {
                "feature": feature,
                "mean_macro_f1_decrease": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
            }
            for feature, scores in values.items()
        ],
        key=lambda item: item["mean_macro_f1_decrease"],
        reverse=True,
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cleaned, audit = load_and_clean(RAW_PATH)
    cleaned.to_csv(PROCESSED_DIR / "student_dropout_risk_cleaned.csv", index=False)
    model_data = primary_model_frame(cleaned, FINAL_FEATURES)
    X = model_data[FINAL_FEATURES]
    y = model_data["risk_class"].map(LABEL_TO_ID)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv_results = {
        name: repeated_cv(name, model, X_train, y_train)
        for name, model in candidate_models().items()
    }
    selected_name = max(cv_results, key=lambda name: cv_results[name]["macro_f1"]["mean"])
    importance = training_only_importance(selected_name, X_train, y_train)
    selected_model = candidate_models()[selected_name]
    fit_kwargs = {}
    if selected_name == "XGBoost":
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight("balanced", y_train)
    selected_model.fit(X_train, y_train, **fit_kwargs)
    test_predictions = selected_model.predict(X_test)
    test_metrics = score_predictions(y_test.to_numpy(), test_predictions)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center")
    ax.set_xticks(range(3), RISK_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(3), RISK_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    results = {
        "duplicate_audit": audit,
        "class_distribution": cleaned["risk_class"].value_counts().to_dict(),
        "class_percentages": (
            cleaned["risk_class"].value_counts(normalize=True).mul(100).to_dict()
        ),
        "primary_candidate_features": PRIMARY_CANDIDATE_FEATURES,
        "final_features": FINAL_FEATURES,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_results": cv_results,
        "selected_model": selected_name,
        "holdout_metrics": test_metrics,
        "holdout_confusion_matrix": matrix.tolist(),
        "training_only_permutation_importance": importance,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # A modest but above-chance repeated-CV macro F1 is sufficient only for an
    # explicitly limited academic survey-risk prototype.
    meaningful_signal = cv_results[selected_name]["macro_f1"]["mean"] >= 0.38
    metadata = {
        "project_title": "Machine Learning-Based Student Dropout Risk Assessment for Bangladeshi University Students",
        "dataset_source": "Student Dropout Risk Survey Dataset (CC BY 4.0), supplied workbook",
        "dataset_year": 2025,
        "target_definition": "Self-reported likelihood of considering dropout due to academic stress",
        "target_mapping": {
            "1-2": "Low Risk",
            "3": "Moderate Risk",
            "4-5": "High Risk",
        },
        "label_to_id": LABEL_TO_ID,
        "final_features": FINAL_FEATURES,
        "question_to_feature": QUESTION_TO_FEATURE,
        "category_mappings": audit["category_mappings"],
        "selected_model": selected_name,
        "repeated_cv_metrics": cv_results[selected_name],
        "holdout_metrics": test_metrics,
        "limitations": [
            "Small convenience survey sample",
            "Self-reported intention rather than observed future dropout",
            "Possible response bias",
            "No longitudinal outcome or external validation",
            "Associations are not causal effects",
        ],
        "prototype_only": True,
    }
    if meaningful_signal:
        joblib.dump(selected_model, MODEL_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        MODEL_PATH.unlink(missing_ok=True)
        METADATA_PATH.unlink(missing_ok=True)
    print(json.dumps({**results, "meaningful_signal": meaningful_signal}, indent=2))


if __name__ == "__main__":
    main()
