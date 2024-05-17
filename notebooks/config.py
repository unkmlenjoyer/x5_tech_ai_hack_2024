"""Configuration file for research"""

from dataclasses import dataclass


@dataclass
class HackConfig:
    """Configuration metadata"""

    path_raw_train_data: str = "../data/raw/train.json"
    random_seed: int = 4242
