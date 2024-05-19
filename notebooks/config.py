"""Configuration file for research"""

from dataclasses import dataclass


@dataclass
class HackConfig:
    """Configuration metadata"""

    path_raw_train_data: str = "../data/raw/train.json"
    random_seed: int = 4242
    base_model_checkpoint: str = "cointegrated/rubert-tiny2"
    test_size: float = 0.3
