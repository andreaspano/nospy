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
    reconciliation_method: str = "BottomUp"


@dataclass
class LLMConfig:
    provider: str = "copilot"
    model: str = "gpt-4o"
    temperature: float = 0.2
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class ExperimentConfig:
    data: DataConfig
    cv: CVConfig
    models: list[str]
    tuning: TuningConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    llm: LLMConfig | None = None

    @classmethod
    def from_dict(cls, params: dict) -> "ExperimentConfig":
        # Always load from the single llm.yaml file, overriding any llm in params
        llm_path = Path(__file__).resolve().parents[1] / "yaml" / "llm.yaml"
        if llm_path.exists():
            with open(llm_path, "r") as f:
                llm_params = yaml.safe_load(f)
        else:
            llm_params = params.get("llm")
        llm_config = LLMConfig(**llm_params) if llm_params else None
        return cls(
            data=DataConfig(**params["data"]),
            cv=CVConfig(**params["cv"]),
            models=params["models"],
            tuning=TuningConfig(**params["tuning"]),
            runtime=RuntimeConfig(**params["runtime"]),
            evaluation=EvaluationConfig(**params["evaluation"]),
            llm=llm_config,
        )
