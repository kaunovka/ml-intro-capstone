RS School Machine Learning course capstone project.

This project uses [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Usage
1. Clone this repository to your machine.
2. Download [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Screenshots for assingment
8.
![mlflow](https://user-images.githubusercontent.com/47551616/167790322-0952ce27-b3bb-45bd-8159-1e7bc7e55c25.png)
12.

![black_reformatted](https://user-images.githubusercontent.com/47551616/167790437-3d555866-916a-46dc-8bf4-bea590ba52ba.png)
