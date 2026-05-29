import json
from pathlib import Path

from ray import tune
from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoTFT
from neuralforecast.losses.pytorch import MAE, MAPE

_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "json"


def _load_model_params(model_key: str, test: bool = False) -> dict:
    config_path = _MODEL_CONFIG_DIR / f"{model_key}.json"
    if not config_path.exists():
        raise ValueError(f"Missing config file for model: {model_key}")

    with open(config_path, "r") as f:
        model_cfg = json.load(f)
    fixed_params = model_cfg.get("fixed", {})
    mode_key = "test" if test else "run"
    mode_params = model_cfg.get(mode_key, {})

    merged = {**fixed_params, **mode_params}

    if test:
        return merged

    tuned = {}
    for key, value in merged.items():
        tuned[key] = tune.choice(value) if isinstance(value, list) else value
    return tuned

def get_model_config(model_name: str, test: bool = False) -> dict:
    return _load_model_params(model_name, test=test)


class ModelFactory:
    @staticmethod
    def build(config):
        models = []

        model_map = {
            "autonhits": AutoNHITS,
            "autonbeats": AutoNBEATS,
            "autotft": AutoTFT,
        }

        metric_name = config.evaluation.metric.upper()
        loss_cls = {"MAPE": MAPE, "MAE": MAE}.get(metric_name)
        if loss_cls is None:
            raise ValueError(f"Unsupported evaluation_metric: {metric_name}")

        for model_name in config.models:
            normalized_name = model_name.lower()
            model_cls = model_map.get(normalized_name)
            if model_cls is None:
                raise ValueError(
                    f"Unknown model '{model_name}'. "
                    "Available models: AutoNHITS, AutoNBEATS, AutoTFT."
                )

            model_key = normalized_name.replace("auto", "")
            models.append(
                model_cls(
                    h=config.cv.h,
                    loss=loss_cls(),
                    valid_loss=loss_cls(),
                    config=get_model_config(model_key, config.runtime.test),
                    num_samples=config.tuning.num_samples,
                    cpus=config.tuning.cpus,
                    gpus=config.tuning.gpus,
                    verbose=False,
                    backend=config.tuning.backend,
                )
            )

        if not models:
            raise ValueError("No models selected.")

        return models
