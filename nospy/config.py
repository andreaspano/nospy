from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    tickers: list[str]
    start_date: str
    end_date: str

    h: int = 5
    n_windows: int = 12
    step_size: int = 1
    freq: str = "B"

    models: list[str] = field(default_factory=lambda: ["AutoNHITS"])
    test: bool = False
    refit: bool = True

    out_dir: Path = Path("./out")

    num_samples: int = 5
    cpus: int = 6
    gpus: int = 1
    backend: str = "ray"

    cuda_visible_devices: str | None = "0"

    evaluation_metric: str = "MAPE"
    reconciliation_method: str = "BottomUp"
