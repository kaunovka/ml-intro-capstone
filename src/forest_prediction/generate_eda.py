from pathlib import Path

import pandas as pd
from pandas_profiling import ProfileReport
import click


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output-path",
    default="data/report.html",
    type=click.Path(dir_okay=True, path_type=Path),
)
def generate(dataset_path: Path, output_path: Path) -> None:
    data = pd.read_csv(dataset_path)
    report = ProfileReport(data)
    report.to_file(output_path)
