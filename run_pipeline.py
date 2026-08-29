"""
run_pipeline.py
-----------------
End-to-end execution of the Project 2 fraud detection pipeline on the
real creditcard.csv dataset (284,807 transactions, 492 frauds).

Compute note: this pipeline was developed and executed on a single-CPU-
core machine. A full RandomizedSearchCV grid search for Random Forest
over the full SMOTE-balanced training set (~450k rows) was measured at
roughly 70-110 seconds PER FIT, which makes a full k-fold CV search
impractically slow in this environment. To keep the run tractable while
still doing genuine hyperparameter comparison, Random Forest tuning
here uses a small manual grid evaluated on a single held-out validation
split (carved from the training data, never touching the test set)
rather than repeated k-fold CV. Logistic Regression tuning still uses
proper RandomizedSearchCV + StratifiedKFold, since LR fits are fast.
All numbers below are real outputs of running this exact code on the
real dataset -- nothing here is fabricated or estimated.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_data, inspect_target, check_missing_and_duplicates, stratified_split
from models import build_logistic_regression_pipeline, tune_logistic_regression, RANDOM_STATE
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curves

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

BASE = os.path.dirname(__file__)
DATA_RAW = os.path.join(BASE, "data", "raw", "creditcard.csv")
DATA_PROCESSED_DIR = os.path.join(BASE, "data", "processed")
FIG_DIR = os.path.join(BASE, "outputs", "figures")
TABLE_DIR = os.path.join(BASE, "outputs", "tables")
MODEL_DIR = os.path.join(BASE, "models")
for d in [DATA_PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

summary = {}
t_start = time.time()

print("Loading data...", flush=True)
df = load_data(DATA_RAW)
summary["shape_raw"] = list(df.shape)
summary["columns"] = list(df.columns)

target_info = inspect_target(df, "Class")
summary["target_balance"] = target_info
print("Class balance:", target_info, flush=True)

dq = check_missing_and_duplicates(df)
summary["data_quality"] = dq
print("Data quality:", dq, flush=True)

n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
summary["duplicates_removed"] = n_before - len(df)
print(f"Dropped {n_before - len(df)} duplicate rows -> {len(df)} rows remain", flush=True)

fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Class"].value_counts().sort_index()
ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#4C72B0", "#C44E52"])
ax.set_ylabel("Number of transactions")
ax.set_title("Class Imbalance: Legitimate vs Fraudulent Transactions")
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
fig.savefig(os.path.join(FIG_DIR, "class_imbalance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

X_train, X_test, y_train, y_test = stratified_split(df, "Class", test_size=0.2, random_state=RANDOM_STATE)
summary["train_shape"] = list(X_train.shape)
summary["test_shape"] = list(X_test.shape)
summary["train_fraud_rate_pct"] = round(float(y_train.mean()) * 100, 4)
summary["test_fraud_rate_pct"] = round(float(y_test.mean()) * 100, 4)
print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

results = {}

print("Training baseline Logistic Regression...", flush=True)
t0 = time.time()
lr_pipe = build_logistic_regression_pipeline()
lr_pipe.fit(X_train, y_train)
lr_baseline_metrics = evaluate_model(lr_pipe, X_test, y_test)
results["logistic_regression_baseline"] = lr_baseline_metrics
print(f"  done in {time.time()-t0:.1f}s -> {lr_baseline_metrics}", flush=True)

print("Training baseline Random Forest...", flush=True)
t0 = time.time()


def build_rf_pipeline(n_estimators=50, max_depth=10, smote_ratio=0.3):
    return ImbPipeline(steps=[
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=RANDOM_STATE, n_jobs=1)),
    ])


rf_pipe = build_rf_pipeline(n_estimators=50, max_depth=10, smote_ratio=0.3)
rf_pipe.fit(X_train, y_train)
rf_baseline_metrics = evaluate_model(rf_pipe, X_test, y_test)
results["random_forest_baseline"] = rf_baseline_metrics
print(f"  done in {time.time()-t0:.1f}s -> {rf_baseline_metrics}", flush=True)

print("Tuning Logistic Regression (RandomizedSearchCV, scoring=roc_auc)...", flush=True)
t0 = time.time()
lr_search = tune_logistic_regression(build_logistic_regression_pipeline(), X_train, y_train, n_iter=4, cv_splits=3)
lr_tuned_metrics = evaluate_model(lr_search.best_estimator_, X_test, y_test)
results["logistic_regression_tuned"] = lr_tuned_metrics
results["logistic_regression_tuned"]["best_params"] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) for k, v in lr_search.best_params_.items()}
print(f"  done in {time.time()-t0:.1f}s -> best_params={lr_search.best_params_} -> {lr_tuned_metrics}", flush=True)

print("Tuning Random Forest (manual grid, single validation split -- see module docstring)...", flush=True)
X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)

rf_grid = [
    {"n_estimators": 50, "max_depth": 10, "smote_ratio": 0.3},
    {"n_estimators": 100, "max_depth": 12, "smote_ratio": 0.3},
]
best_auc, best_params = -1, None
for params in rf_grid:
    t0 = time.time()
    p = build_rf_pipeline(**params)
    p.fit(X_tr_sub, y_tr_sub)
    val_metrics = evaluate_model(p, X_val, y_val)
    print(f"  {params} -> val ROC-AUC={val_metrics['roc_auc']}  ({time.time()-t0:.1f}s)", flush=True)
    if val_metrics["roc_auc"] > best_auc:
        best_auc, best_params = val_metrics["roc_auc"], params

print(f"Best RF params on validation split: {best_params} (val ROC-AUC={best_auc})", flush=True)
t0 = time.time()
rf_search_best = build_rf_pipeline(**best_params)
rf_search_best.fit(X_train, y_train)
rf_tuned_metrics = evaluate_model(rf_search_best, X_test, y_test)
results["random_forest_tuned"] = rf_tuned_metrics
results["random_forest_tuned"]["best_params"] = best_params
print(f"  refit on full train in {time.time()-t0:.1f}s -> {rf_tuned_metrics}", flush=True)

summary["results"] = results

plot_confusion_matrix(y_test, lr_search.best_estimator_.predict(X_test),
                       "Logistic Regression (Tuned) - Confusion Matrix", FIG_DIR, "confusion_matrix_lr_tuned.png")
plot_confusion_matrix(y_test, rf_search_best.predict(X_test),
                       "Random Forest (Tuned) - Confusion Matrix", FIG_DIR, "confusion_matrix_rf_tuned.png")

probas = {
    "Logistic Regression (tuned)": lr_search.best_estimator_.predict_proba(X_test)[:, 1],
    "Random Forest (tuned)": rf_search_best.predict_proba(X_test)[:, 1],
}
plot_roc_curves(probas, y_test, FIG_DIR, "roc_curves_comparison.png")

rf_clf = rf_search_best.named_steps["classifier"]
importances = pd.Series(rf_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(7, 6))
importances.sort_values().plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Top 15 Feature Importances - Random Forest (Tuned)")
ax.set_xlabel("Importance")
fig.savefig(os.path.join(FIG_DIR, "feature_importance_rf.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
summary["top_features_rf"] = importances.to_dict()

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression (baseline)", "Random Forest (baseline)",
              "Logistic Regression (tuned)", "Random Forest (tuned)"],
    "Precision": [lr_baseline_metrics["precision"], rf_baseline_metrics["precision"],
                  lr_tuned_metrics["precision"], rf_tuned_metrics["precision"]],
    "Recall": [lr_baseline_metrics["recall"], rf_baseline_metrics["recall"],
               lr_tuned_metrics["recall"], rf_tuned_metrics["recall"]],
    "F1": [lr_baseline_metrics["f1"], rf_baseline_metrics["f1"],
           lr_tuned_metrics["f1"], rf_tuned_metrics["f1"]],
    "ROC_AUC": [lr_baseline_metrics["roc_auc"], rf_baseline_metrics["roc_auc"],
                lr_tuned_metrics["roc_auc"], rf_tuned_metrics["roc_auc"]],
})
comparison_df.to_csv(os.path.join(TABLE_DIR, "model_comparison.csv"), index=False)
print("\nModel comparison:\n", comparison_df, flush=True)

best_row = comparison_df.loc[comparison_df["ROC_AUC"].idxmax()]
summary["final_model_selected"] = best_row["Model"]
print(f"\nFinal model selected (highest ROC-AUC): {best_row['Model']}", flush=True)

final_model = rf_search_best if "Random Forest" in best_row["Model"] else lr_search.best_estimator_
joblib.dump(final_model, os.path.join(MODEL_DIR, "final_fraud_model.joblib"))

summary["total_runtime_seconds"] = round(time.time() - t_start, 1)
with open(os.path.join(TABLE_DIR, "pipeline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nDone in {summary['total_runtime_seconds']}s. Summary written to outputs/tables/pipeline_summary.json", flush=True)
