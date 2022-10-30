import click
import pandas as pd
import yaml
import logging
from logging import getLogger
from yaml.loader import SafeLoader
import logging.config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from basic_logging import log_conf

logging.config.dictConfig(log_conf)

log = getLogger()


@click.command()
@click.option('--config', default='../configs/preprocess.yaml', help='YAML with input and output paths.')
def preprocess(config):
    """Simple program that greets NAME for a total of COUNT times."""
    log.info("Preparing data")
    with open(config, 'r') as stream:
        config_yaml = yaml.load(stream, SafeLoader)
    raw_data_path = config_yaml['raw_data_path']
    output_path = config_yaml['output_path']
    log.info("raw_data_path = %s, output_path = %s", raw_data_path, output_path)

    df = pd.read_csv(raw_data_path)
    all_columns = set(df.columns)
    target_column = config_yaml['target']
    target_set = set()
    target_set.add(target_column)
    X = df[list(all_columns - target_set)]
    y = df[[target_column]]
    categorical_features = config_yaml['categorical_features']
    quantitative_features = config_yaml['quantitative_features']

    log.info("Apply one-hot encoding to %s", categorical_features)
    log.info("Apply StandardScaler to %s", quantitative_features)
    ct = ColumnTransformer([
        ("oneHot", OneHotEncoder(), categorical_features),
        ("Standardize", StandardScaler(), quantitative_features)
    ], remainder='passthrough')

    data_transformed = ct.fit_transform(X, y)
    log.info("Data transformed")
    df = pd.DataFrame(data_transformed)
    df.to_csv(output_path)
    log.info("Data successfully loaded to %s", output_path)


if __name__ == '__main__':
    preprocess()
