from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest

def get_models():
    """
    Returns a dictionary of baseline and advanced ML models:
    1. Naive Baseline: Predicts majority class.
    2. Linear ML Baseline: Weighted Logistic Regression.
    3. Balanced Random Forest: Non-linear ensemble with bootstrap balanced class weights.
    4. HistGradientBoosting: Fast gradient-boosted decision trees with class weighting.
    5. Isolation Forest: Unsupervised anomaly detection baseline.
    """
    models = {
        "Naive_Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=150,
            learning_rate=0.08,
            random_state=42
        ),
        "Isolation_Forest": IsolationForest(
            contamination=0.015,
            random_state=42,
            n_jobs=-1
        )
    }
    return models
