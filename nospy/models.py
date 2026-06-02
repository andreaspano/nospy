import json
import platform
from pathlib import Path

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.sample import Domain
from ray.tune.search.optuna import OptunaSearch
from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoTFT
from neuralforecast.losses.pytorch import MAE, MAPE

from nospy.config import ExperimentConfig

_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "json"


class _AutoWithSchedulerMixin:
    def __init__(
        self,
        *args,
        tune_metric: str = "loss",
        tune_mode: str = "min",
        scheduler_cls=None,
        scheduler_kwargs=None,
        **kwargs,
    ):
        self._tune_metric = tune_metric
        self._tune_mode = tune_mode
        self._scheduler_cls = scheduler_cls
        self._scheduler_kwargs = scheduler_kwargs or {}
        super().__init__(*args, **kwargs)

    def _tune_model(
        self,
        cls_model,
        dataset,
        val_size,
        test_size,
        cpus,
        gpus,
        verbose,
        num_samples,
        search_alg,
        config,
        time_budget,
    ):
        train_fn_with_parameters = tune.with_parameters(
            self._train_tune,
            cls_model=cls_model,
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
        )

        if gpus > 0:
            device_dict = {"gpu": gpus}
        else:
            device_dict = {"cpu": cpus}

        trial_dirname_creator = (
            (lambda trial: f"{trial.trainable_name}_{trial.trial_id}")
            if platform.system() == "Windows"
            else None
        )

        tune_config_kwargs = {
            "metric": self._tune_metric,
            "mode": self._tune_mode,
            "num_samples": num_samples,
            "search_alg": search_alg,
            "trial_dirname_creator": trial_dirname_creator,
            "time_budget_s": time_budget,
        }
        if self._scheduler_cls is not None:
            tune_config_kwargs["scheduler"] = self._scheduler_cls(
                **self._scheduler_kwargs
            )

        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, device_dict),
            run_config=air.RunConfig(callbacks=self.callbacks, verbose=verbose),
            tune_config=tune.TuneConfig(**tune_config_kwargs),
            param_space=config,
        )
        results = tuner.fit()
        return results


class AutoNHITSWithScheduler(_AutoWithSchedulerMixin, AutoNHITS):
    pass


class AutoNBEATSWithScheduler(_AutoWithSchedulerMixin, AutoNBEATS):
    pass


class AutoTFTWithScheduler(_AutoWithSchedulerMixin, AutoTFT):
    pass


def _load_model_params(model_key: str, test: bool = False) -> dict:
    config_path = _MODEL_CONFIG_DIR / f"{model_key}.json"
    if not config_path.exists():
        raise ValueError(f"Missing config file for model: {model_key}")

    with open(config_path, "r") as f:
        model_cfg = json.load(f)
    fixed_params = model_cfg.get("fixed", {})
    mode_key = "test" if test else "run"
    mode_params = model_cfg.get(mode_key, {})

    if test:
        return {**fixed_params, **mode_params}

    tuned = dict(fixed_params)
    for key, value in mode_params.items():
        tuned[key] = tune.choice(value) if isinstance(value, list) else value
    return tuned

_MODEL_MAP = {
    "autonhits": AutoNHITSWithScheduler,
    "autonbeats": AutoNBEATSWithScheduler,
    "autotft": AutoTFTWithScheduler,
}

_LOSS_MAP = {"MAPE": MAPE, "MAE": MAE}


def _build_scheduler(
    tuning,
) -> tuple[type | None, dict]:
    """Return (scheduler_cls, scheduler_kwargs) for the configured scheduler."""
    if tuning.backend != "ray" or (tuning.scheduler or "").lower() != "asha":
        return None, {}

    kwargs: dict = {}
    if tuning.asha_max_t is not None:
        kwargs["max_t"] = tuning.asha_max_t
    if tuning.asha_grace_period is not None:
        kwargs["grace_period"] = tuning.asha_grace_period
    if tuning.asha_reduction_factor is not None:
        kwargs["reduction_factor"] = tuning.asha_reduction_factor
    return ASHAScheduler, kwargs


def _build_search_alg(tuning, model_config: dict) -> OptunaSearch | None:
    """Return an OptunaSearch instance when the config has a tunable search space."""
    has_search_space = any(isinstance(v, Domain) for v in model_config.values())
    if (
        tuning.backend == "ray"
        and (tuning.searcher or "").lower() == "optuna"
        and has_search_space
    ):
        return OptunaSearch()
    return None


def _build_model(
    model_name: str,
    config: ExperimentConfig,
    loss_cls: type,
    scheduler_cls: type | None,
    scheduler_kwargs: dict,
) -> object:
    """Instantiate a single Auto* model from config."""
    normalized = model_name.lower()
    model_cls = _MODEL_MAP.get(normalized)
    if model_cls is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(_MODEL_MAP)}."
        )

    model_key = normalized.replace("auto", "")
    model_config = _load_model_params(model_key, config.runtime.test)
    search_alg = _build_search_alg(config.tuning, model_config)

    tune_metric = config.tuning.tune_objective or "loss"
    tune_mode = config.tuning.mode or "min"

    kwargs = {
        "h": config.cv.h,
        "loss": loss_cls(),
        "valid_loss": loss_cls(),
        "config": model_config,
        "num_samples": config.tuning.num_samples,
        "cpus": config.tuning.cpus,
        "gpus": config.tuning.gpus,
        "verbose": False,
        "backend": config.tuning.backend,
        "scheduler_cls": scheduler_cls,
        "scheduler_kwargs": scheduler_kwargs,
        "tune_metric": tune_metric,
        "tune_mode": tune_mode,
    }
    if search_alg is not None:
        kwargs["search_alg"] = search_alg

    return model_cls(**kwargs)


class ModelFactory:
    @staticmethod
    def build(config: ExperimentConfig) -> list:
        metric_name = config.evaluation.metric.upper()
        loss_cls = _LOSS_MAP.get(metric_name)
        if loss_cls is None:
            raise ValueError(f"Unsupported evaluation metric: {metric_name}")

        scheduler_cls, scheduler_kwargs = _build_scheduler(config.tuning)

        models = [
            _build_model(name, config, loss_cls, scheduler_cls, scheduler_kwargs)
            for name in config.models
        ]

        if not models:
            raise ValueError("No models selected.")

        return models
