"""
Configuration dataclasses for experiment parameters.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    tickers: list[str]
    start_date: str
    end_date: str
    freq: str


@dataclass
class CVConfig:
    h: int
    n_windows: int
    step_size: int
    refit: bool


@dataclass
class TuningConfig:
    num_samples: int
    cpus: int
    gpus: int
    backend: str
    searcher: str | None = None
    scheduler: str | None = None
    tune_objective: str | None = None
    mode: str | None = None
    asha_max_t: int | None = None
    asha_grace_period: int | None = None
    asha_reduction_factor: int | None = None


@dataclass
class RuntimeConfig:
    out_dir: Path
    cuda_visible_devices: str | None
    test: bool

    def __post_init__(self):
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)


@dataclass
class EvaluationConfig:
    metric: str
    reconciliation_method: str


@dataclass
class ExperimentConfig:
    data: DataConfig
    cv: CVConfig
    models: list[str]
    tuning: TuningConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_dict(cls, params: dict) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**params["data"]),
            cv=CVConfig(**params["cv"]),
            models=params["models"],
            tuning=TuningConfig(**params["tuning"]),
            runtime=RuntimeConfig(**params["runtime"]),
            evaluation=EvaluationConfig(**params["evaluation"]),
        )
