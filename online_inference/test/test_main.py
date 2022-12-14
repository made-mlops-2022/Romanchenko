from time import sleep

import pytest
from fastapi.testclient import TestClient

from ..src.main import app, startup_event

client = TestClient(app)


@pytest.fixture
def wait_startapp():
    startup_event()
    iterations = 100
    for i in range(iterations):
        r = client.get("/health")
        if r.status_code == 200:
            return
        sleep(0.1)
    raise TimeoutError()


def test_predict(wait_startapp):
    data_json = {"age": 69, "sex": 1, "cp": 0, "trestbps": 160, "chol": 234, "fbs": 1, "restecg": 2, "thalach": 131,
            "exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1, "thal": 0}
    response = client.post("/predict", json=data_json)
    assert response.status_code == 200
    assert response.json() == {"prediction": 1}

