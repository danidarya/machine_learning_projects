import os
import pytest
import random
import pandas as pd
from typing import List
from py._path.local import LocalPath
from src.params import (PipelineParams, SplitParams, FeatureParams, TrainParams)
from src.train_and_predict_pipeline import train_pipeline
from faker import Faker

Faker.seed(4321)


def create_fake_data(fake_data_path='tests/fake_data.csv', num=100):
    fake = Faker()

    output = [{"age": fake.pyint(min_value=20, max_value=90),
               "sex": fake.pyint(0, 1),
               "cp": fake.pyint(0, 3),
               "trestbps": fake.pyint(94, 200),
               "chol": fake.pyint(126, 564),
               "fbs": fake.pyint(0, 1),
               "restecg": fake.pyint(0, 1),
               "thalach": fake.pyint(71, 202),
               "exang": fake.pyint(0, 1),
               "oldpeak": fake.pyfloat(0, 6, positive=True),
               "slope": fake.pyint(0, 2),
               "ca": fake.pyint(0, 4),
               "thal": fake.pyint(0, 3),
               "target": fake.pyint(0, 1)} for _ in range(num)]

    df = pd.DataFrame(output)
    df.to_csv(fake_data_path)


@pytest.fixture()
def params(
        tmpdir: LocalPath,
        dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = PipelineParams(
        train_data_path=dataset_path,
        data_for_pred_path=tmpdir.join('fake_data.csv'),
        predictions_path=tmpdir.join('predictions.csv'),
        transformer_path=tmpdir.join('transformer.pkl'),
        model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        split_params=SplitParams(),
        features_params=FeatureParams(
            numerical=numerical_features,
            categorical=categorical_features,
            target=target_col,
        ),
        train_params=TrainParams(model_type="LogisticRegression", C=1, n_jobs=-1, penalty='l2'),
    )
    return params


def test_train_e2e(params):
    real_model_path, metrics = train_pipeline(params)
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
    assert metrics["accuracy"] > 0
