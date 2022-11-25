import unittest

import yaml
import os

from .utils import generate_df
from ..main.configs import TrainConfig
from ..main.train_model import train


class TestTrain(unittest.TestCase):
    def test_two_features_classification(self):
        input_data_path = "data.csv"
        output_path = 'model.pkl'
        config_file = "TestTrainConfig.yaml"
        config = TrainConfig(
            data_path=input_data_path,
            output_path=output_path,
            test_ratio="0.3"
        )
        with open(config_file, 'w') as f_config:
            yaml.dump(config.__dict__, f_config)

        test_df = generate_df(1000)
        test_df.to_csv(input_data_path, index=False)
        f1_score_test = train.main(
            args=["--config", config_file],
            standalone_mode=False
        )
        self.assertGreater(f1_score_test, 0.7)
        print(os.lstat(output_path))
        os.remove(input_data_path)
        os.remove(output_path)
        os.remove(config_file)


if __name__ == '__main__':
    unittest.main()
