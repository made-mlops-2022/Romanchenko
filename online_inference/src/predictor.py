import logging
import logging.config
import os

import boto3
import pickle

import pandas as pd

from .basic_logging import log_conf
from .models import InputData

MODEL_PATH = os.getenv("MODEL_PATH", 'models/model.pkl')
PREPARATION_PATH = os.getenv("PREPARATION_PATH", 'models/prepare.pkl')
USE_S3 = (os.getenv("USE_S3", default='False') == 'True')

model = None
model_ready = False
preparation = None

logging.config.dictConfig(log_conf)

log = logging.getLogger()


def load_model():
    log.info('load model called, use_s3=%s', USE_S3)
    log.info(MODEL_PATH)
    log.info(PREPARATION_PATH)
    global model
    global model_ready
    global preparation
    if USE_S3:
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net'
        )
        s3.download_file('mlops-bucket', 'model_v1.pkl', MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f_model:
        model = pickle.load(f_model)
    if USE_S3:
        s3.download_file('mlops-bucket', 'preparation.pkl', PREPARATION_PATH)
    with open(PREPARATION_PATH, 'rb') as f_model:
        preparation = pickle.load(f_model)
        model_ready = True
    log.info('MODEL_READY: %s', model_ready)


def make_prediction(input: InputData):
    log.info("make_prediction executed")
    data = pd.DataFrame(input.__dict__, index=[0])
    log.info(data)
    data_transformed = preparation.transform(data)
    log.info(data_transformed)
    y = model.predict(data_transformed)
    return y


def check_model():
    return model_ready
