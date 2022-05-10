from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from typing import Tuple

def create_pipeline(
    model: str, use_scaler: bool, max_iter: int, n_estimators: int, random_state: int
) -> Tuple[Pipeline, dict]:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if model == "logreg":
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, multi_class='ovr'
                ),
            )
        )
    if model == "randomforest":
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1
                ),
            )
        )

    pipeline = Pipeline(steps)

    if model == 'logreg':
            grid = {
                'classifier__C': [1e-4, 1e-2, 1e-1, 1, 5],
            }
    if model == 'randomforest':
        grid = {
            'classifier__max_depth': [None, 3, 5, 10, 20],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]
        }

    return pipeline, grid