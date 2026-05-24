from ray import tune

from neuralforecast.auto import AutoNHITS, AutoNBEATS
from neuralforecast.losses.pytorch import MAE


def get_nhits_config(test: bool = False):
    if test:
        return {
            "input_size": 1,
            "max_steps": 1,
            "learning_rate": 0.01,
            "batch_size": 128,
            "windows_batch_size": 4096,
            "n_pool_kernel_size": [1, 1, 1],
            "n_freq_downsample": [1, 1, 1],
            "scaler_type": "standard",
            "random_seed": 42,
        }

    return {
        "input_size": tune.choice([20, 40, 60, 80]),
        "max_steps": tune.choice([300, 500, 700]),
        "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
        "batch_size": tune.choice([16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
        "windows_batch_size": tune.choice([1024, 2048, 4096, 8192, 16384]),
        "n_pool_kernel_size": tune.choice([[2, 2, 1], [3, 2, 1]]),
        "n_freq_downsample": tune.choice([[8, 4, 1], [4, 2, 1]]),
        "scaler_type": tune.choice(["robust", "standard"]),
        "random_seed": tune.choice([42, 123, 2026]),
    }


def get_nbeats_config(test: bool = False):
    if test:
        return {
            "input_size": 1,
            "max_steps": 1,
            "learning_rate": 0.01,
            "batch_size": 128,
            "windows_batch_size": 128,
            "scaler_type": "standard",
            "random_seed": 42,
            "stack_types": ["identity"],
        }

    return {
        "input_size": tune.choice([20, 40, 60, 80]),
        "max_steps": tune.choice([300, 500, 700]),
        "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
        "batch_size": tune.choice([512, 1024, 2048]),
        "windows_batch_size": tune.choice([1024, 2048]),
        "scaler_type": tune.choice(["robust", "standard"]),
        "random_seed": tune.choice([42, 123, 2026]),
        "stack_types": ["identity"],
    }


class ModelFactory:
    @staticmethod
    def build(config):
        models = []

        for model_name in config.models:
            normalized_name = model_name.lower()

            if normalized_name == "autonhits":
                models.append(
                    AutoNHITS(
                        h=config.h,
                        loss=MAE(),
                        valid_loss=MAE(),
                        config=get_nhits_config(config.test),
                        num_samples=config.num_samples,
                        cpus=config.cpus,
                        gpus=config.gpus,
                        verbose=False,
                        backend=config.backend,
                    )
                )

            elif normalized_name == "autonbeats":
                models.append(
                    AutoNBEATS(
                        h=config.h,
                        loss=MAE(),
                        valid_loss=MAE(),
                        config=get_nbeats_config(config.test),
                        num_samples=config.num_samples,
                        cpus=config.cpus,
                        gpus=config.gpus,
                        verbose=False,
                        backend=config.backend,
                    )
                )

            else:
                raise ValueError(
                    f"Unknown model '{model_name}'. "
                    "Available models: AutoNHITS, AutoNBEATS."
                )

        if not models:
            raise ValueError("No models selected.")

        return models
