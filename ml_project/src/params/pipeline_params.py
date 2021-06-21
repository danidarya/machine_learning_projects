from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from .feature_params import FeatureParams
from .split_params import SplitParams
from .train_params import TrainParams


@dataclass
class PipelineParams:
    train_data_path: str
    data_for_pred_path: str
    predictions_path: str
    model_path: str
    transformer_path: str
    metric_path: str
    split_params: SplitParams
    features_params: FeatureParams
    train_params: TrainParams


PipelineParamsSchema = class_schema(PipelineParams)


def read_pipeline_params(path: str) -> PipelineParams:
    with open(path, "r") as input:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(input))
