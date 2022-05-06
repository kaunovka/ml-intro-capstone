from pathlib import Path

import click
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",  
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def train(dataset_path: Path) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    X = dataset.drop(['Id', 'Cover_Type'], axis=1)
    y = dataset['Cover_Type']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=123)
    clf = LogisticRegression(random_state=123).fit(X_train, y_train)
    accuracy = accuracy_score(y_val, clf.predict(X_val))
    click.echo(f"Accuracy: {accuracy}")