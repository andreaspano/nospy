import pandas as pd
from neuralforecast.losses.numpy import mape

METRIC_FUNCTIONS = {
    "MAPE": mape,
}


class Evaluator:
    ID_COLUMNS = {"unique_id", "ds", "cutoff", "y"}

    @staticmethod
    def get_model_columns(df_cv: pd.DataFrame) -> list[str]:
        return [
            col for col in df_cv.columns
            if col not in Evaluator.ID_COLUMNS
        ]

    @staticmethod
    def compute_metrics(df_cv: pd.DataFrame, metric_name: str = "MAPE") -> pd.DataFrame:
        model_cols = Evaluator.get_model_columns(df_cv)

        if not model_cols:
            raise ValueError("No model prediction columns found in df_cv.")

        if metric_name not in METRIC_FUNCTIONS:
            raise ValueError(f"Metric '{metric_name}' not supported.")
        metric_func = METRIC_FUNCTIONS[metric_name]

        results = []

        for model in model_cols:
            global_metric = metric_func(df_cv["y"].values, df_cv[model].values)

            by_id = (
                df_cv
                .groupby("unique_id")[["y", model]]
                .apply(lambda x: metric_func(x["y"].values, x[model].values))
                .reset_index(name=metric_name)
            )

            by_id["model"] = model
            results.append(by_id)

            results.append(
                pd.DataFrame(
                    [{
                        "unique_id": "GLOBAL",
                        metric_name: global_metric,
                        "model": model,
                    }]
                )
            )

        return pd.concat(results, ignore_index=True)[["model", "unique_id", metric_name]]

    @staticmethod
    def rank_models(df_metrics: pd.DataFrame, metric_name: str = "MAPE") -> pd.DataFrame:
        # Exclude any rows where unique_id is missing, just in case
        df_metrics = df_metrics[df_metrics["unique_id"].notnull()]
        # Rank models for each unique_id (ticket)
        ranking = (
            df_metrics
            .copy()
            .sort_values(["unique_id", metric_name])
        )
        ranking["rank"] = (
            ranking.groupby("unique_id")[metric_name]
            .rank(method="min")
            .astype(int)
        )
        # Reorder columns for clarity
        return ranking[["unique_id", "rank", "model", metric_name]]
