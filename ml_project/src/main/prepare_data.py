import pickle
from typing import TextIO

import click
import pandas as pd
import yaml
from logging import getLogger

from yaml.loader import SafeLoader
import logging.config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from basic_logging import log_conf
from marshmallow_dataclass import class_schema
from configs import PreprocessConfig

logging.config.dictConfig(log_conf)

log = getLogger()


def strict_load_yaml(yaml_config: TextIO) -> PreprocessConfig:
    schema = class_schema(PreprocessConfig)
    return schema().load(yaml.load(yaml_config, Loader=SafeLoader))


@click.command()
@click.option(
    '--config',
    default='../configs/preprocess.yaml',
    help='YAML with input and output paths.'
)
def preprocess(config):
    """Simple program that greets NAME for a total of COUNT times."""
    log.info("Preparing data")
    with open(config, 'r') as stream:
        config_yaml = strict_load_yaml(stream)
    raw_data_path = config_yaml.raw_data_path
    output_path = config_yaml.output_path
    output_path_preparation = config_yaml.output_path_preparation
    input_path_preparation = config_yaml.input_path_preparation

    log.info(
        "raw_data_path = %s, output_path = %s",
        raw_data_path, output_path
    )

    df = pd.read_csv(raw_data_path)

    categorical_features = config_yaml.categorical_features
    quantitative_features = config_yaml.quantitative_features

    log.info("Apply one-hot encoding to %s", categorical_features)
    log.info("Apply StandardScaler to %s", quantitative_features)

    if input_path_preparation is not None:
        with open(input_path_preparation, 'rb') as ct_file:
            ct = pickle.load(ct_file)
        data_transformed = ct.transform(df)
    else:
        ct = ColumnTransformer([
            ("oneHot", OneHotEncoder(), categorical_features),
            ("Standardize", StandardScaler(), quantitative_features)
        ], remainder='passthrough')

        data_transformed = ct.fit_transform(df)
    log.info("Data transformed")
    df_to_save = pd.DataFrame(data_transformed)
    df_to_save.to_csv(output_path, index=False)
    log.info("Data successfully loaded to %s", output_path)

    if output_path_preparation is not None:
        # Just the same, but drop target
        ct = ColumnTransformer([
            ("oneHot", OneHotEncoder(), categorical_features),
            ("Standardize", StandardScaler(), quantitative_features)
        ], remainder='passthrough')
        df.pop(config_yaml.target)
        ct.fit_transform(df)
        with open(output_path_preparation, "wb") as f:
            pickle.dump(ct, f)



if __name__ == '__main__':
    preprocess()
