import pickle
from logging import getLogger

import pandas as pd

import click
import yaml
from yaml import SafeLoader

from basic_logging import log_conf
import logging.config

logging.config.dictConfig(log_conf)

log = getLogger()


@click.command()
@click.option('--config', default='../configs/predict.yaml', help='YAML with dumped model and output path.')
def predict(config):
    with open(config, 'r') as stream:
        config_yaml = yaml.load(stream, SafeLoader)
    log.info("Predicting")
    model_path = config_yaml['model_path']
    with open(model_path, 'rb') as f_model:
        model = pickle.load(f_model)
    log.info("Loaded model from %s", model_path)
    input_data_path = config_yaml['input_data_path']
    data = pd.read_csv(input_data_path)
    y = model.predict(data)
    log.info("Predicted %s values", len(y))
    result_df = pd.DataFrame(y)
    result_df.to_csv(config_yaml['output_path'], index=False, header=False)
    log.info("Stored results in %s", config_yaml['output_path'])

if __name__ == '__main__':
    predict()
