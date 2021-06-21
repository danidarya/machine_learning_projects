from typing import List
import numpy as np
from src.features import process_categorical_features, process_numerical_features
from src.data.make_dataset import read_data

def test_process_categorical_features( dataset_path: str, categorical_features: List[str]):
    data = read_data(dataset_path)
    #print(data)
    df = process_categorical_features(data[categorical_features])
    assert df.shape[1] == 24


def test_process_numerical_features( dataset_path: str, numerical_features: List[str]):
    data = read_data(dataset_path)
    # print(data)
    df = process_numerical_features(data[numerical_features])
    assert df.shape[1] == 5
