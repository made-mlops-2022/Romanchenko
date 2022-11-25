import os
import unittest

import pandas as pd
import yaml

from ..main.configs import TrainConfig, PredictConfig
from .utils import generate_df
from ..main.train_model import train
from ..main.predict import predict
from sklearn.metrics import f1_score


class TestPredict(unittest.TestCase):
    def test_simple_predict(self):
        train_data_path = "train_data.csv"
        test_data_path = "test_data.csv"
        model_path = 'model.pkl'
        config_file = "TestTrainConfig.yaml"
        output_path = "results.csv"
        config = TrainConfig(
            data_path=train_data_path,
            output_path=model_path,
            test_ratio="0.3"
        )
        with open(config_file, 'w') as f_config:
            yaml.dump(config.__dict__, f_config)

        train_df = generate_df(1000)
        train_df.to_csv(train_data_path, index=False)

        test_df = generate_df(10)
        test_df.iloc[:, :-1].to_csv(test_data_path, index=False)

        train.main(args=["--config", config_file], standalone_mode=False)

        predict_config = PredictConfig(
            model_path=model_path,
            output_path=output_path,
            input_data_path=test_data_path
        )
        predict_config_path = "predict_config.yaml"
        with open(predict_config_path, 'w') as f_config:
            yaml.dump(predict_config.__dict__, f_config)
        predict.main(
            args=["--config", predict_config_path],
            standalone_mode=False
        )

        results = pd.read_csv(output_path, header=None)

        f1 = f1_score(results.iloc[:, 0].values, test_df.iloc[:, -1].values)
        self.assertGreater(f1, 0.7)

        os.remove(train_data_path)
        os.remove(model_path)
        os.remove(config_file)
        os.remove(output_path)
        os.remove(test_data_path)


if __name__ == '__main__':
    unittest.main()
