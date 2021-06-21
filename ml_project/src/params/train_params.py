from dataclasses import dataclass, field
from typing import Optional

@dataclass()
class TrainParams:
    model_type: str
    C: Optional[float]
    penalty: Optional[str]
    max_iter: Optional[int] = 1000
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = 10
    n_jobs: int = -1
    random_state: int = 123