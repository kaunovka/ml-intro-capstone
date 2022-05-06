from pathlib import Path
from joblib import dump

import click
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

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
def train(dataset_path: Path, save_model_path: Path) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    X = dataset.drop(['Id', 'Cover_Type'], axis=1)
    y = dataset['Cover_Type']
    clf = LogisticRegression(random_state=123)
    cv_results = cross_validate(clf, X, y, cv=3, scoring=['accuracy', 'roc_auc_ovr', 'f1_micro'])
    click.echo(f"Accuracy: {cv_results['test_accuracy']}")
    click.echo(f"ROC_AUC: {cv_results['test_roc_auc_ovr']}")
    click.echo(f"F1: {cv_results['test_f1_micro']}")
    dump(clf, save_model_path)
    click.echo(f'Model saved to {save_model_path}')