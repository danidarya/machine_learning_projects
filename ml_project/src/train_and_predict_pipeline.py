import logging.config
import click
import json
import pickle
import sys
import pandas as pd

from src.data import read_data, split_train_val_data
from src.params import PipelineParams, read_pipeline_params
from src.models import train_model, predict_model, evaluate_model, dump_model
from src.features import make_features, extract_target, build_transformer

APPLICATION_NAME = "homework_01"

logger = logging.getLogger(APPLICATION_NAME)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("logs/train.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def train_pipeline(params: PipelineParams):
    logger.info(f"Start train with params {params}.")
    data = read_data(params.train_data_path)
    logger.info(f"Data shape is {data.shape}")
    data_train, data_val = split_train_val_data(data, params.split_params)
    logger.info(f"Train data shape is {data_train.shape}")
    logger.info(f"Validation data shape is {data_val.shape}")
    target_train = extract_target(data_train, params.features_params)
    data_train = data_train.drop(columns=['target'])
    transformer = build_transformer(params.features_params)
    transformer.fit(data_train)
    features_train = make_features(transformer, data_train)
    logger.info(f"Train features shape is {features_train.shape}")
    target_val = extract_target(data_val, params.features_params)
    data_val = data_val.drop(columns=['target'])
    features_val = make_features(transformer, data_val)
    logger.info(f"Validation features shape is {features_val.shape}")

    model = train_model(features_train, target_train, params.train_params)
    predicts = predict_model(model, features_val)
    metrics = evaluate_model(predicts, target_val)
    with open(params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metrics are: {metrics}")
    path_to_model = dump_model(model, params.model_path)
    logger.info(f"Model saved at {params.model_path}")
    with open(params.transformer_path, "wb") as tr:
        pickle.dump(transformer, tr)
    logger.info(f"Feature transformer saved at {params.transformer_path}")
    logger.info("Finished.")
    return path_to_model, metrics


def predict_pipeline(params: PipelineParams):
    logger.info(f"Start predict pipeline with params {params}")
    data = pd.read_csv(params.data_for_pred_path)
    logger.info(f"Data shape is {data.shape}")
    with open(params.model_path, 'rb') as m:
        model = pickle.load(m)
    logger.info(f"Model {model} loaded.")
    with open(params.transformer_path, 'rb') as t:
        transformer = pickle.load(t)
    logger.info("Transformer loaded.")
    features = make_features(transformer, data)
    logger.info(f"Features shape is {features.shape}")
    predictions = predict_model(model, features)
    logger.info(f"Predictions shape is {predictions.shape}")
    data["pred_target"] = predictions
    logger.info(f"Predictions saved to {params.predictions_path}")
    data.to_csv(params.predictions_path)
    logger.info("Finished.")


@click.command(name="train_and_predict_pipeline")
@click.argument("train_predict")
@click.argument("config_path")
def pipeline_command(train_predict: str, config_path: str):
    params = read_pipeline_params(config_path)
    if train_predict == "train":
        train_pipeline(params)
    elif train_predict == "predict":
        predict_pipeline(params)


if __name__ == "__main__":
    pipeline_command()
