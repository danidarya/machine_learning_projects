from src.data.make_dataset import read_data, split_train_val_data
from src.params import SplitParams


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) == 100
    assert target_col in data.keys()


def test_split_dataset(dataset_path: str):
    #val_size = 0.25
    splitting_params = SplitParams()
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] == 75
    assert val.shape[0] == 25