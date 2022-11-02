from dataclasses import dataclass
from typing import List


@dataclass
class TrainConfig:
    data_path: str
    output_path: str
    test_ratio: str


@dataclass
class PreprocessConfig:
    raw_data_path: str
    output_path: str
    target: str
    quantitative_features: List[str]
    categorical_features: List[str]


@dataclass
class PredictConfig:
    model_path: str
    output_path: str
    input_data_path: str
