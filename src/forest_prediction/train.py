from pathlib import Path
from joblib import dump

import click
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",  
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
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
    default=100,
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
    "--random-state",
    default=123,
    type=int,
    show_default=True,
)
def train(
        dataset_path: Path, 
        save_model_path: Path,
        use_scaler: bool,
        max_iter: int,
        logreg_c: int,
        random_state: int) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    X = dataset.drop(['Id', 'Cover_Type'], axis=1)
    y = dataset['Cover_Type']

    steps = []
    if (use_scaler):
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_c)))

    pipeline = Pipeline(steps)

    cv_results = cross_validate(pipeline, X, y, cv=3, scoring=['accuracy', 'roc_auc_ovr', 'f1_micro'])
    accuracy = cv_results['test_accuracy'].mean()
    roc_auc = cv_results['test_roc_auc_ovr'].mean()
    f1 = cv_results['test_f1_micro'].mean()

    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("logreg_c", logreg_c)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1", f1)

    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"ROC_AUC: {roc_auc}")
    click.echo(f"F1: {f1}")

    dump(pipeline, save_model_path)
    click.echo(f'Model saved to {save_model_path}')