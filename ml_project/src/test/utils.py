import random

import numpy as np
import pandas as pd


def generate_df(size):
    feature1 = [random.choice([0, 1]) for i in range(size)]
    feature2 = feature1.copy()

    y = feature1.copy()
    for i in range(size // 10):
        idx = random.randint(0, size - 1)
        feature1[idx] = 1 - feature1[idx]
    for i in range(size // 10):
        idx = random.randint(0, size - 1)
        feature2[idx] = 1 - feature2[idx]
    arr = np.concatenate(
        [
            np.array(feature1).reshape(-1, 1),
            np.array(feature2).reshape(-1, 1),
            np.array(y).reshape(-1, 1)
        ], axis=1)
    return pd.DataFrame(arr)
