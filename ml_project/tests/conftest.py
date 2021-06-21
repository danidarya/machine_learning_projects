import os
import pytest
from typing import List


@pytest.fixture(scope='session')
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "fake_data.csv")


@pytest.fixture(scope='session')
def target_col():
    return "target"


@pytest.fixture(scope='session')
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture(scope='session')
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]

