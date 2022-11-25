import pickle
from typing import TextIO

import click
import pandas as pd
import yaml
from logging import getLogger

from marshmallow_dataclass import class_schema
from yaml.loader import SafeLoader
import logging.config

from .basic_logging import log_conf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .configs import TrainConfig

logging.config.dictConfig(log_conf)

log = getLogger()


def strict_load_yaml(yaml_config: TextIO) -> TrainConfig:
    schema = class_schema(TrainConfig)
    return schema().load(yaml.load(yaml_config, Loader=SafeLoader))


@click.command()
@click.option(
    '--config',
    default='../configs/train.yaml',
    help='YAML with train config.'
)
def train(config):
    log.info("Training model")
    with open(config, 'r') as stream:
        config_yaml = strict_load_yaml(stream)
    data_path = config_yaml.data_path
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, :-1], df[df.columns[-1]]

    test_ratio = float(config_yaml.test_ratio)
    log.info("Splitting data into test and train with ratio %s", test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=1338
    )

    model = GridSearchCV(
        DecisionTreeClassifier(),
        {
            "random_state": [40],
            "min_samples_split": [3, 4, 5],
            "max_depth": [5, 6, 7, 8, 9, 10, 15],
            "min_samples_leaf": [3, 4, 5],
            "criterion": ["gini", "log_loss", "entropy"]
        },
        scoring='f1'
    )
    model.fit(X_train, y_train)
    log.info("Best params:\n%s", model.best_params_)

    predictions_train = model.predict(X_train)
    f1_train = f1_score(y_train, predictions_train)
    log.info("F1 score on train is %s", f1_train)

    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    log.info("F1 score on test is %s", f1)

    output_path = config_yaml.output_path
    log.info("Dumping model to %s", output_path)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    return f1


if __name__ == '__main__':
    train()
