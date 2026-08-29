"""
evaluation.py
--------------
Evaluation metrics and plots for the fraud detection models.
Accuracy is deliberately NOT the headline metric: on a dataset that is
99.8% legitimate transactions, a model that predicts "legitimate" for
every single row would score ~99.8% accuracy while catching zero fraud.
We evaluate with Precision, Recall, ROC-AUC, and the confusion matrix
instead.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, confusion_matrix,
    roc_curve, f1_score
)

sns.set_theme(style="whitegrid")


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute precision, recall, F1, and ROC-AUC on the held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def plot_confusion_matrix(y_test, y_pred, title, out_dir, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(models_probas: dict, y_test, out_dir, filename="roc_curves.png"):
    """
    models_probas: dict of {model_name: predicted_probabilities}
    Plots all ROC curves on one figure for direct model comparison.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_proba in models_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
