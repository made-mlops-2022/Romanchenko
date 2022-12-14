import logging
import logging.config

from fastapi import FastAPI, HTTPException

from .basic_logging import log_conf
from .models import InputData, OutputData
from .predictor import load_model, check_model, make_prediction

app = FastAPI()

logging.config.dictConfig(log_conf)

log = logging.getLogger()


@app.on_event("startup")
def startup_event():
    log.info('startup executed')
    load_model()


@app.post("/predict", response_model=OutputData)
def predict(input_item: InputData):
    log.info("Called predict")
    res = make_prediction(input_item)
    if res:
        return OutputData(prediction=1)
    return OutputData(prediction=0)


@app.get("/health")
def health_check():
    if check_model():
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=400, detail='Service not ready')
