# Project 2: Supervised Learning — Fraud Detection Pipeline

## Overview
A binary classification pipeline that flags fraudulent credit-card transactions
in a highly imbalanced dataset, built the way it would need to be built in
production: no data leakage, resampling applied only to the training fold,
and evaluation that ignores Accuracy in favor of Precision, Recall, and ROC-AUC.

## Objective
Build and tune classifiers (Logistic Regression, Random Forest) to identify
fraud, correctly handling a dataset where fraud is only 0.17% of all rows.

## Dataset
**Public dataset used: "Credit Card Fraud Detection" (creditcard.csv)** —
284,807 anonymized European credit-card transactions from September 2013,
492 of which are fraud (0.173%). Features `V1`–`V28` are PCA-transformed
components of the original (confidential) transaction features; `Time` and
`Amount` are the only two untransformed columns. `Class` is the target
(1 = fraud, 0 = legitimate).

This is the standard, widely-used public benchmark for imbalanced fraud
classification (the same dataset referenced in the project's own briefing
slides — "284,807 Transactions ... Fraudulent Rate: 0.17%"). No dataset was
supplied for this project by the user, so this well-known public dataset was
downloaded (via a GitHub mirror of the original Kaggle release) to fulfil the
assignment's requirements. See `data/raw/creditcard.csv`.

## Technologies Used
Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), matplotlib,
seaborn, joblib.

## Project Structure
```
Project_2_Fraud_Detection/
├── data/raw/creditcard.csv
├── src/
│   ├── preprocessing.py   # load, inspect target balance, stratified split
│   ├── models.py          # pipeline builders + hyperparameter tuning
│   └── evaluation.py      # metrics, confusion matrix / ROC plots
├── run_pipeline.py        # end-to-end script (the source of truth for results)
├── outputs/figures/       # class imbalance, confusion matrices, ROC, feature importance
├── outputs/tables/        # model_comparison.csv, pipeline_summary.json
├── models/final_fraud_model.joblib
└── report/Project_2_Report.md
```

## Methodology
1. Load data, inspect class balance (99.83% / 0.17%), check missing values
   and duplicates (1,081 exact duplicate rows found and removed).
2. **Stratified** train/test split (80/20) so both sets keep the same fraud rate.
3. Build an `imblearn.pipeline.Pipeline` so SMOTE is applied **only inside the
   training fold** — never to the test set, and never before the split.
4. Train Logistic Regression (with `StandardScaler`, since it's scale-sensitive)
   and Random Forest (no scaler needed — tree splits are scale-invariant) baselines.
5. Tune Logistic Regression with `RandomizedSearchCV` (`scoring="roc_auc"`,
   3-fold `StratifiedKFold`). Tune Random Forest with a small manual grid on a
   held-out validation split carved from the training data only (see the
   module docstring in `run_pipeline.py` for the compute-time rationale — a
   full k-fold grid search over ~227k rows was measured at 70-125s per fit
   in this single-core environment).
6. Evaluate all four models with Precision, Recall, F1, ROC-AUC, and confusion
   matrices — Accuracy is intentionally never used to pick the final model.

## Key Findings
- **Random Forest (baseline and tuned tied) is the best model**: Precision
  0.822, Recall 0.779, ROC-AUC 0.968 — it catches ~78% of fraud while only
  producing 16 false alarms out of 56,651 legitimate test transactions.
- **Logistic Regression has a severe precision problem**: it catches slightly
  more fraud (Recall 0.874) but at Precision 0.053 — meaning **19 out of every
  20 transactions it flags as fraud are actually legitimate**. In production
  that would mean blocking ~1,470 real customers to catch 83 fraud cases.
- Tuning barely moved either model's test-set numbers here — the untuned
  Random Forest was already close to its ceiling on this task, and the
  Logistic Regression search converged on essentially the same operating point.
- Top predictive features for Random Forest: `V14`, `V10`, `V17`, `V3`, `V12`
  — consistent with published analyses of this dataset.

## How to Run
```bash
cd Project_2_Fraud_Detection
pip install -r requirements.txt
python run_pipeline.py
# or open notebooks/Project_2_Fraud_Detection.ipynb
```

## Results
See `outputs/tables/model_comparison.csv` for the full metrics table and
`outputs/tables/pipeline_summary.json` for every number referenced above,
all produced by the actual run.

## Conclusion
For this task, Random Forest is the clear choice: it gets a materially
better fraud-catch rate *and* a vastly better false-alarm rate than Logistic
Regression. The Logistic Regression model illustrates exactly why Accuracy
and even Recall alone are misleading for imbalanced problems — Precision is
what determines whether the model is actually usable without drowning
customers and support teams in false declines.
