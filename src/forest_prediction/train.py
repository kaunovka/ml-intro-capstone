from pathlib import Path
from joblib import dump

import click
import pandas as pd
import numpy as np

import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--model",
    default="logreg",
    type=click.Choice(["logreg", "randomforest"]),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--random-state",
    default=123,
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    model: str,
    use_scaler: bool,
    max_iter: int,
    logreg_c: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    X = dataset.drop(["Id", "Cover_Type"], axis=1)
    y = dataset["Cover_Type"]

    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if model == "logreg":
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_c, multi_class='ovr'
                ),
            )
        )
    if model == "randomforest":
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                ),
            )
        )

    pipeline = Pipeline(steps)
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    outer_results = {}
    outer_results['accuracy'] = []
    outer_results['roc_auc'] = []
    outer_results['f1'] = []
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

        if model == 'logreg':
            grid = {
                'classifier__C': [1e-4, 1e-2, 1e-1, 1, 5],
            }
        if model == 'randomforest':
            grid = {
                'classifier__n_estimators': [10, 100, 1000, 2000],
                'classifier__max_depth': [None, 3, 5, 10, 20],
            }

        clf = GridSearchCV(pipeline, grid, scoring='accuracy', cv=cv_inner, refit=True)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        best_params = clf.best_params_

        outer_results['accuracy'].append(accuracy_score(y_test, best_model.predict(X_test)))
        outer_results['roc_auc'].append(roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr'))
        outer_results['f1'].append(f1_score(y_test, best_model.predict(X_test), average='micro'))


    accuracy = np.mean(outer_results["accuracy"])
    roc_auc = np.mean(outer_results["roc_auc"])
    f1 = np.mean(outer_results["f1"])

    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"ROC_AUC: {roc_auc}")
    click.echo(f"F1: {f1}")
    click.echo(f"best params: {best_params}")

    dump(best_model, save_model_path)
    click.echo(f"Model saved to {save_model_path}")
