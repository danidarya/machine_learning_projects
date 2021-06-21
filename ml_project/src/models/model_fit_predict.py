import pickle
from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from src.params.train_params import TrainParams

SklearnModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(features: pd.DataFrame, target: pd.Series, train_params: TrainParams) -> SklearnModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators, max_depth=train_params.max_depth,
            n_jobs=train_params.n_jobs, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=train_params.C, penalty=train_params.penalty, n_jobs=train_params.n_jobs,
            max_iter=train_params.max_iter, random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def dump_model(model: SklearnModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
