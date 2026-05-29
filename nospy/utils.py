import os
import warnings
from datetime import datetime
from pathlib import Path

import yaml
import torch


class _NoDupSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in YAML: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDupSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_yaml_no_dupes(path: str | Path) -> dict:
    with open(path, "r") as f:
        data = yaml.load(f, Loader=_NoDupSafeLoader)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def setup_environment(cuda_visible_devices: str | None = "0") -> None:
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch version:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())


def make_output_paths(out_dir: Path) -> dict[str, Path]:
    # Use full timestamp including seconds
    tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "cv": run_dir / "cv.csv",
        "metrics": run_dir / "metrics.csv",
        "ranking": run_dir / "ranking.csv",
        "forecast_vs_actuals": run_dir / "forecast_vs_actuals.png",
        "scatter_forecast_vs_actuals": run_dir / "scatter_forecast_vs_actuals.png",
        "run_dir": run_dir,
    }
