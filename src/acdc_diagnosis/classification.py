"""Preprocessing, feature selection and classifiers for the 5-class diagnosis.

Preprocessing: standardisation, concatenation of the principal components that
explain 95% of the variance, then selection of the features whose random forest
importance is above the mean. Each classifier is tuned by grid search with
5-fold stratified cross-validation on the selected features.

Usage:
    preprocessor = build_preprocessor().fit(X_train, y_train)
    model = tune("rf", preprocessor.transform(X_train), y_train)
"""

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]

MODELS = {
    "rf": (
        RandomForestClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        },
    ),
    "mlp": (
        MLPClassifier(max_iter=10000, solver="adam", alpha=1e-4, learning_rate_init=0.01),
        {"hidden_layer_sizes": [(10,), (50,), (100,)]},
    ),
    "svm": (
        SVC(probability=True),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "poly", "rbf", "sigmoid"], "gamma": ["scale", "auto"]},
    ),
    "xgb": (
        XGBClassifier(eval_metric="mlogloss", learning_rate=0.01, colsample_bytree=0.7),
        {"n_estimators": [100, 200, 500, 1000], "max_depth": [1, 3], "subsample": [0.3, 0.5]},
    ),
}

VOTING_MEMBERS = ("rf", "mlp", "xgb")


def cross_validator(seed=None):
    """5-fold stratified, shuffled cross-validation splitter.

    Args:
        seed (int | None): random seed.

    Returns:
        sklearn.model_selection.StratifiedKFold: splitter.
    """
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def build_preprocessor(n_trees=10000, seed=None):
    """Standardisation + PCA (95% variance) concatenation + random forest feature selection.

    Args:
        n_trees (int): number of trees of the random forest used for feature selection.
        seed (int | None): random seed.

    Returns:
        sklearn.pipeline.Pipeline: unfitted preprocessing pipeline.
    """
    selector = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    return Pipeline([
        ("scale", StandardScaler()),
        ("augment", FeatureUnion(
            [("features", "passthrough"), ("pca", PCA(n_components=0.95))],
            verbose_feature_names_out=False,
        )),
        ("select", SelectFromModel(selector, threshold="mean")),
    ])


def tune(name, X, y, seed=None):
    """Grid search one classifier and return the best estimator refitted on X, y.

    Args:
        name (str): one of MODELS.
        X (numpy.ndarray): selected features.
        y (numpy.ndarray): class labels.
        seed (int | None): random seed of the cross-validation and of the model.

    Returns:
        sklearn.base.BaseEstimator: fitted best estimator.
    """
    estimator, grid = MODELS[name]
    estimator = clone(estimator).set_params(random_state=seed)
    search = GridSearchCV(estimator, grid, cv=cross_validator(seed), scoring="accuracy", n_jobs=-1)
    return search.fit(X, y).best_estimator_


def build_voting(X, y, seed=None):
    """Soft-voting ensemble of the tuned random forest, MLP and XGBoost.

    Args:
        X (numpy.ndarray): selected features.
        y (numpy.ndarray): class labels.
        seed (int | None): random seed.

    Returns:
        sklearn.ensemble.VotingClassifier: unfitted ensemble of tuned members.
    """
    members = [(name, tune(name, X, y, seed)) for name in VOTING_MEMBERS]
    return VotingClassifier(members, voting="soft")


def build_model(name, X, y, seed=None):
    """Tuned classifier for a model name, including the voting ensemble.

    Args:
        name (str): one of MODELS or "voting".
        X (numpy.ndarray): selected features.
        y (numpy.ndarray): class labels.
        seed (int | None): random seed.

    Returns:
        sklearn.base.BaseEstimator: classifier ready for cross-validation or fitting.
    """
    return build_voting(X, y, seed) if name == "voting" else tune(name, X, y, seed)
