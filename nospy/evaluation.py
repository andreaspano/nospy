import pandas as pd
from neuralforecast.losses.numpy import mape


class Evaluator:
    ID_COLUMNS = {"unique_id", "ds", "cutoff", "y"}

    @staticmethod
    def get_model_columns(df_cv: pd.DataFrame) -> list[str]:
        return [
            col for col in df_cv.columns
            if col not in Evaluator.ID_COLUMNS
        ]

    @staticmethod
    def compute_metrics(df_cv: pd.DataFrame) -> pd.DataFrame:
        model_cols = Evaluator.get_model_columns(df_cv)

        if not model_cols:
            raise ValueError("No model prediction columns found in df_cv.")

        results = []

        for model in model_cols:
            global_mape = mape(df_cv["y"].values, df_cv[model].values)

            by_id = (
                df_cv
                .groupby("unique_id")[["y", model]]
                .apply(lambda x: mape(x["y"].values, x[model].values))
                .reset_index(name="MAPE")
            )

            by_id["model"] = model
            results.append(by_id)

            results.append(
                pd.DataFrame(
                    [{
                        "unique_id": "GLOBAL",
                        "MAPE": global_mape,
                        "model": model,
                    }]
                )
            )

        return pd.concat(results, ignore_index=True)[["model", "unique_id", "MAPE"]]

    @staticmethod
    def rank_models(df_metrics: pd.DataFrame) -> pd.DataFrame:
        ranking = (
            df_metrics
            .query("unique_id == 'GLOBAL'")
            .sort_values("MAPE")
            .reset_index(drop=True)
        )

        ranking["rank"] = ranking.index + 1

        return ranking[["rank", "model", "MAPE"]]
