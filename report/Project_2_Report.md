# Project 2 Report: Fraud Detection Pipeline

## 1. Title
Supervised Learning for Imbalanced Fraud Detection

## 2. Introduction
Fraud detection is a canonical example of an imbalanced classification
problem: fraudulent events are rare by definition, but missing one is far
more costly than a false alarm. This project builds a leakage-free pipeline
that handles that imbalance correctly and evaluates models with metrics that
actually reflect business cost, rather than Accuracy.

## 3. Objective
Build and tune a classifier that identifies fraudulent transactions in a
highly imbalanced dataset, using SMOTE correctly (training data only) and
selecting the final model on Precision, Recall, and ROC-AUC rather than
Accuracy.

## 4. Dataset Description
**Credit Card Fraud Detection dataset** (public; downloaded from a GitHub
mirror of the original Kaggle release, since no dataset was supplied for
this project — see README for details): 284,807 transactions made by
European cardholders in September 2013, with 492 (0.173%) labeled fraud.
Features `V1`–`V28` are the result of a PCA transformation applied by the
dataset's original authors to protect confidential transaction details;
`Time` (seconds since the first transaction) and `Amount` are the only
untransformed features. `Class` is the binary target.

## 5. Methodology
- Loaded the data, confirmed 0 missing values, found and removed 1,081 exact
  duplicate rows (283,726 rows remained).
- Split 80/20 with `stratify=y` so train and test both preserve the 0.17%
  fraud rate (train: 0.1665%, test: 0.1674% — matching by design).
- Built pipelines with `imblearn.pipeline.Pipeline`, placing `SMOTE` and (for
  Logistic Regression) `StandardScaler` *inside* the pipeline so they are
  refit on each training fold only, never on the test set.
- Random Forest does **not** use a scaler — tree splits partition on raw
  feature values, so scaling has no effect on its structure.

## 6. Data Preprocessing
No missing-value imputation was needed (0 missing values in the raw data).
The only cleaning step was duplicate removal. `Time` and `Amount` were left
in their raw units for the tree model; for Logistic Regression they are
standardized inside the pipeline alongside the already-PCA'd `V1`–`V28`.

## 7. Modeling
Four models were trained and evaluated on the same held-out test set:
1. Logistic Regression (baseline: default `C=1.0`, `class_weight=None`, SMOTE
   sampling_strategy=0.3 in training pipeline)
2. Random Forest (baseline: 50 trees, max_depth=10, SMOTE sampling_strategy=0.3)
3. Logistic Regression (tuned via `RandomizedSearchCV`, scoring=`roc_auc`)
4. Random Forest (tuned via manual grid on a validation split, scoring=`roc_auc`)

## 8. Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.0530 | 0.8737 | 0.1000 | 0.9626 |
| Random Forest (baseline) | 0.8222 | 0.7789 | 0.8000 | 0.9677 |
| Logistic Regression (tuned) | 0.0534 | 0.8737 | 0.1007 | 0.9623 |
| **Random Forest (tuned)** | **0.8222** | **0.7789** | **0.8000** | **0.9677** |

Random Forest confusion matrix (test set, n=56,746):

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 56,635 | 16 |
| **Actual Fraud** | 21 | 74 |

Logistic Regression confusion matrix (tuned):

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 55,181 | 1,470 |
| **Actual Fraud** | 12 | 83 |

## 9. Visualizations
- `outputs/figures/class_imbalance.png` — bar chart of the 99.83%/0.17% split
- `outputs/figures/confusion_matrix_lr_tuned.png`, `confusion_matrix_rf_tuned.png`
- `outputs/figures/roc_curves_comparison.png` — both tuned models overlaid
- `outputs/figures/feature_importance_rf.png` — top 15 features for Random Forest

## 10. Interpretation
Random Forest and Logistic Regression achieve nearly identical ROC-AUC
(0.968 vs 0.962) — at the *ranking* level, both models separate fraud from
legitimate transactions almost equally well. But ROC-AUC alone would be
misleading here: at their respective default decision thresholds, Logistic
Regression's precision (0.053) makes it operationally unusable, while Random
Forest's precision (0.822) makes it deployable. This is a direct illustration
of why the assignment's brief singles out Precision/Recall over a single
aggregate ranking metric, let alone Accuracy (which would show ~99.8% for
*any* model that just predicts "legitimate" every time).

## 11. Business / Statistical Insights
- **False negatives (missed fraud) are the costlier error**: each one is a
  direct, typically unrecoverable financial loss to the card issuer.
- **False positives (false declines) are also costly**, but the cost is
  reputational/customer-experience, not a direct loss — and at Logistic
  Regression's precision, the false-positive volume (1,470 legitimate
  transactions blocked to catch 83 fraud cases) would be operationally
  unacceptable at most institutions.
- Random Forest's balance of the two error types — 16 false positives vs.
  21 false negatives on 56,746 test transactions — is the more defensible
  trade-off for deployment, even though its Recall (0.779) is slightly lower
  than Logistic Regression's (0.874).

## 12. Limitations
- The Random Forest hyperparameter search was a 2-point manual grid on a
  single validation split rather than full k-fold `RandomizedSearchCV`,
  because a full grid search was measured at 70-125 seconds *per fit* on the
  ~227k-row SMOTE-balanced training set in this single-CPU environment,
  making a proper k-fold grid impractical here. The two configurations tested
  represent a reasonable but not exhaustive tuning effort.
- `V1`–`V28` are already anonymized/PCA-transformed by the original dataset
  authors, so no further domain-specific feature engineering was possible or
  attempted — the modeling operates on the features as given.
- This dataset is from September 2013; fraud patterns evolve, so a
  production model would need periodic retraining on more recent data.

## 13. Conclusion
Random Forest is the selected final model for this task: comparable ranking
ability (ROC-AUC) to Logistic Regression, but a vastly more usable precision/recall
trade-off at deployment. The pipeline correctly isolates SMOTE resampling to the
training fold only, avoiding the data-leakage trap the assignment explicitly warns
against, and every metric reported here comes from the actual `run_pipeline.py`
execution on the real 284,807-row dataset.
