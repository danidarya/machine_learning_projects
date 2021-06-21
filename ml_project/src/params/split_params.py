from dataclasses import dataclass


@dataclass()
class SplitParams:
    test_size: float = 0.25
    random_state: int = 123