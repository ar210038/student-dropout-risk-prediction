"""Final binary experiment for elevated self-reported dropout risk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from bangladesh_risk_pipeline import (
    NUMERIC_RATING_FEATURES,
    PRIMARY_CANDIDATE_FEATURES,
    elevated_risk,
    load_and_clean,
    primary_model_frame,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "bangladesh_risk" / "raw" / "Student Dropout Risk Survey Dataset.xlsx"
REPORT_DIR = ROOT / "reports" / "bangladesh_risk"
RESULTS_PATH = REPORT_DIR / "binary_training_results.json"
THRESHOLDS = np.arange(0.15, 0.61, 0.05)


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


def candidate_models(features: list[str]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", make_preprocessor(features, True)),
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
                ("preprocess", make_preprocessor(features, False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=350,
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
                ("preprocess", make_preprocessor(features, False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=2,
                        learning_rate=0.035,
                        min_child_weight=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=3.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
    }


def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(round(threshold, 2)),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(prediction.mean()),
        **probability_metrics(y, probability),
    }


def repeated_cv_probabilities(
    model_name: str,
    features: list[str],
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, dict]:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    probability_sum = np.zeros(len(y), dtype=float)
    probability_count = np.zeros(len(y), dtype=int)
    fold_probability_metrics = []
    for train_index, valid_index in splitter.split(X, y):
        model = candidate_models(features)[model_name]
        fit_kwargs = {}
        if model_name == "XGBoost":
            fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
                "balanced", y.iloc[train_index]
            )
        model.fit(X.iloc[train_index], y.iloc[train_index], **fit_kwargs)
        probability = model.predict_proba(X.iloc[valid_index])[:, 1]
        probability_sum[valid_index] += probability
        probability_count[valid_index] += 1
        fold_probability_metrics.append(
            probability_metrics(y.iloc[valid_index].to_numpy(), probability)
        )
    if not np.all(probability_count == 5):
        raise RuntimeError("Each training row must receive one prediction per repeat")
    oof_probability = probability_sum / probability_count
    summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in fold_probability_metrics])),
            "std": float(np.std([row[metric] for row in fold_probability_metrics], ddof=1)),
        }
        for metric in ["roc_auc", "average_precision"]
    }
    summary["fold_count"] = len(fold_probability_metrics)
    return oof_probability, summary


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, list[dict]]:
    rows = [threshold_metrics(y, probability, threshold) for threshold in THRESHOLDS]
    # An intervention list covering most students is not operationally useful.
    # Keep the selected alert rate at or below half the cohort, then maximize F1.
    practical = [row for row in rows if row["predicted_positive_rate"] <= 0.50]
    pool = practical or rows
    chosen = max(pool, key=lambda row: (row["f1"], row["precision"]))
    return chosen["threshold"], rows


def evaluate_feature_set(
    features: list[str], X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    results = {}
    for name in candidate_models(features):
        probability, fold_summary = repeated_cv_probabilities(
            name, features, X_train[features], y_train
        )
        threshold, threshold_table = choose_threshold(y_train.to_numpy(), probability)
        results[name] = {
            "fold_probability_metrics": fold_summary,
            "oof_metrics": threshold_metrics(
                y_train.to_numpy(), probability, threshold
            ),
            "threshold_table": threshold_table,
        }
    return results


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned, audit = load_and_clean(RAW_PATH)
    cleaned["elevated_risk"] = cleaned["dropout_intention_score"].map(elevated_risk)
    model_data = primary_model_frame(cleaned, PRIMARY_CANDIDATE_FEATURES)
    X = model_data[PRIMARY_CANDIDATE_FEATURES]
    y = cleaned.loc[model_data.index, "elevated_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    full_results = evaluate_feature_set(PRIMARY_CANDIDATE_FEATURES, X_train, y_train)
    selected_name = max(
        full_results,
        key=lambda name: (
            full_results[name]["fold_probability_metrics"]["average_precision"]["mean"],
            full_results[name]["fold_probability_metrics"]["roc_auc"]["mean"],
        ),
    )
    selected_threshold = full_results[selected_name]["oof_metrics"]["threshold"]
    selected_model = candidate_models(PRIMARY_CANDIDATE_FEATURES)[selected_name]
    fit_kwargs = {}
    if selected_name == "XGBoost":
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight(
            "balanced", y_train
        )
    selected_model.fit(X_train, y_train, **fit_kwargs)
    holdout_probability = selected_model.predict_proba(X_test)[:, 1]
    holdout_metrics = threshold_metrics(
        y_test.to_numpy(), holdout_probability, selected_threshold
    )

    prevalence = float(y.mean())
    selected_cv = full_results[selected_name]
    meaningful_signal = (
        selected_cv["fold_probability_metrics"]["average_precision"]["mean"]
        >= prevalence + 0.10
        and selected_cv["fold_probability_metrics"]["roc_auc"]["mean"] >= 0.62
        and selected_cv["oof_metrics"]["precision"] >= prevalence + 0.08
        and selected_cv["oof_metrics"]["recall"] >= 0.60
        and holdout_metrics["average_precision"] >= prevalence
        and holdout_metrics["roc_auc"] >= 0.55
    )
    output = {
        "duplicate_audit": audit,
        "class_counts": {
            "not_elevated": int((y == 0).sum()),
            "elevated": int((y == 1).sum()),
        },
        "positive_prevalence": prevalence,
        "no_skill_average_precision": prevalence,
        "train_size": len(X_train),
        "holdout_size": len(X_test),
        "full_features": PRIMARY_CANDIDATE_FEATURES,
        "full_model_results": full_results,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "holdout_metrics": holdout_metrics,
        "meaningful_signal": meaningful_signal,
        "reduced_feature_evaluation": None,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
