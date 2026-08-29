"""
models.py
----------
Model pipeline construction (imblearn Pipelines with SMOTE applied only
inside the training fold) and hyperparameter tuning for Logistic
Regression and Random Forest.

Why imblearn.pipeline.Pipeline and not sklearn.pipeline.Pipeline?
sklearn's Pipeline expects every step to implement a `transform` method
that only touches X. SMOTE needs to modify BOTH X and y (it creates new
synthetic minority-class rows), which sklearn's Pipeline cannot express.
imblearn's Pipeline natively supports resampling steps via the
`fit_resample` interface, and -- critically -- only ever applies that
resampling to the training fold during cross-validation, never to the
held-out validation/test fold. This is what keeps the pipeline "leak-free".
"""

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

RANDOM_STATE = 42


def build_logistic_regression_pipeline() -> ImbPipeline:
    """
    Logistic Regression pipeline: StandardScaler -> SMOTE -> classifier.
    Scaling matters here because Logistic Regression's regularization
    penalty is sensitive to the huge scale difference between `Amount`
    (up to several thousand) and the PCA components V1-V28 (roughly -30
    to +30) -- without scaling, `Amount` would dominate the objective.
    """
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_random_forest_pipeline() -> ImbPipeline:
    """
    Random Forest pipeline: SMOTE -> classifier (no scaler needed).
    Tree-based splits partition feature space ordinally at each node,
    so they are invariant to monotonic scale transformations -- scaling
    would add computation with no effect on the resulting splits.
    """
    return ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def tune_logistic_regression(pipeline, X_train, y_train, n_iter=8, cv_splits=3):
    """
    Hyperparameter search for Logistic Regression, scored on ROC-AUC
    (not accuracy -- accuracy is misleading on a 99.8%-imbalanced
    target, since predicting "legitimate" for everything would already
    score ~99.8%).
    """
    param_dist = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "smote__k_neighbors": [3, 5],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=n_iter,
        scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(pipeline, X_train, y_train, n_iter=8, cv_splits=3):
    """Hyperparameter search for Random Forest, scored on ROC-AUC."""
    param_dist = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [8, 12],
        "smote__k_neighbors": [3, 5],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=n_iter,
        scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    return search
