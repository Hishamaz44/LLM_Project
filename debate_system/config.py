"""Config loading. All model IDs live here, never hardcoded elsewhere."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Holds every tunable experiment parameter and all model IDs used in a run.
@dataclass
class Config:
    for_debater_model: str
    against_debater_model: str
    homogeneous_judge_model: str
    heterogeneous_judge_models: list[str]
    rounds: int = 2
    # Each debater writes its argument once per target word length. Length is controlled at
    # generation time (the model composes a complete argument to the budget) rather than by
    # truncating a longer one afterwards, so short arguments are whole, not amputated. The
    # experiment then checks whether judge scores climb as the (length-matched) pair grows.
    length_targets: list[int] = field(default_factory=lambda: [60, 120, 180])
    debater_max_tokens: int = 350
    debater_temperature: float = 0.2
    judge_max_tokens: int = 300
    judge_temperature: float = 0.7

    # Post-init validation: heterogeneous panel size and at least one length target.
    def __post_init__(self):
        if len(self.heterogeneous_judge_models) != 3:
            raise ValueError("heterogeneous_judge_models must contain exactly 3 model IDs")
        if not self.length_targets:
            raise ValueError("length_targets must contain at least one target word count")


# Reads a YAML file from disk and builds a Config from it.
def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(**data)
