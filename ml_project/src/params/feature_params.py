from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical: List[str]
    numerical: List[str]
    target: Optional[str]