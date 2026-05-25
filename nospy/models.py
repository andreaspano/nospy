from ray import tune
from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoTFT
from neuralforecast.losses.pytorch import MAE, MAPE

def get_tft_config(test: bool = False):
    if test:
        return {
            "input_size": 1,
            "max_steps": 1,
            "learning_rate": 0.01,
            "batch_size": 128,
            "windows_batch_size": 128,
            "scaler_type": "standard",
            "random_seed": 42,
        }

    return {
        "input_size": tune.choice([20, 40, 60, 80]),
        "max_steps": tune.choice([300, 500, 700]),
        "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
        "batch_size": tune.choice([512, 1024, 2048]),
        "windows_batch_size": tune.choice([1024, 2048]),
        "scaler_type": tune.choice(["robust", "standard"]),
        "random_seed": tune.choice([42, 123, 2026]),
    }


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
        # quanta storia passata il modello vede
        "input_size": tune.choice([20, 40, 60, 80]),
        "max_steps": tune.choice([300, 500, 700]),
        "learning_rate": tune.choice([1e-3, 5e-4, 1e-4]),
        # In NHITS, this is the number of series for which we compute the loss 
        # and update the model parameters in each training step.
        # batch size troppo grossi rischiano di edere variazioni importanti tra un passo e l'altro, 
        # ma batch size troppo piccoli possono rendere l'addestramento instabile.

        "batch_size": tune.choice([16, 32, 64, 128]), 
        # number of windows processed simultaneously during training. 
        # Higher values can speed up training but require more memory.
        "windows_batch_size": tune.choice([512, 1024, 2048]), 
        
        # => quanto "zoom out" faccio sul passato
         "n_pool_kernel_size": tune.choice([
            [1, 1, 1],
            [2, 2, 1],
            [3, 2, 1],
        ]), 
        
        # => quanto dettaglio/rumore permetto nella previsione
            "n_freq_downsample": tune.choice([
            [1, 1, 1],
            [2, 2, 1],
            [4, 2, 1],
        ]),

   
        "scaler_type": tune.choice(["robust", "standard"]),
        "random_seed": tune.choice([42, 123, 2026]),
        
        # dropout applicato ai layer MLP del modello
        # quanti neuroni vengono "spenti" casualmente durante il training
        "dropout_prob_theta": tune.choice([0.0, 0.1, 0.2]),
        
        "activation": tune.choice(["ReLU", "LeakyReLU", "SELU"]),

        # numero di neuroni nei layer MLP del modello.
        "mlp_units": tune.choice([
            [[128, 128], [128, 128], [128, 128]],
            [[256, 256], [256, 256], [256, 256]],

        ]),
        # ogni 100 step:
        # il training si ferma temporaneamente
        # il modello viene valutato sul validation set
        "val_check_steps": tune.choice([50, 100]),
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



        # Map evaluation_metric to loss class
        metric_name = getattr(config, "evaluation_metric", "MAPE").upper()
        if metric_name == "MAPE":
            loss_cls = MAPE
        elif metric_name == "MAE":
            loss_cls = MAE
        else:
            raise ValueError(f"Unsupported evaluation_metric: {metric_name}")

        for model_name in config.models:
            normalized_name = model_name.lower()

            if normalized_name == "autonhits":
                models.append(
                    AutoNHITS(
                        h=config.h,
                        loss=loss_cls(),
                        valid_loss=loss_cls(),
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
                        loss=loss_cls(),
                        valid_loss=loss_cls(),
                        config=get_nbeats_config(config.test),
                        num_samples=config.num_samples,
                        cpus=config.cpus,
                        gpus=config.gpus,
                        verbose=False,
                        backend=config.backend,
                    )
                )

            elif normalized_name == "autotft":
                models.append(
                    AutoTFT(
                        h=config.h,
                        loss=loss_cls(),
                        valid_loss=loss_cls(),
                        config=get_tft_config(config.test),
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
                    "Available models: AutoNHITS, AutoNBEATS, AutoTFT."
                )

        if not models:
            raise ValueError("No models selected.")

        return models
