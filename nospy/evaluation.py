"""
Evaluation metrics and ranking.
"""

import pandas as pd
from neuralforecast.losses.numpy import mape

METRIC_FUNCTIONS = {
    "MAPE": mape,
}


class Evaluator:
    ID_COLUMNS = {"unique_id", "ds", "cutoff", "y"}

    @staticmethod
    def get_model_columns(df_cv: pd.DataFrame) -> list[str]:
        return [col for col in df_cv.columns if col not in Evaluator.ID_COLUMNS]

    @staticmethod
    def compute_metrics(df_cv: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        model_cols = Evaluator.get_model_columns(df_cv)

        if not model_cols:
            raise ValueError("No model prediction columns found in df_cv.")

        if metric_name not in METRIC_FUNCTIONS:
            raise ValueError(f"Metric '{metric_name}' not supported.")
        metric_func = METRIC_FUNCTIONS[metric_name]

        results = []

        for model in model_cols:
            by_id_cutoff = (
                df_cv.groupby(["unique_id", "cutoff"])[["y", model]]
                .apply(lambda x: metric_func(x["y"].values, x[model].values))
                .reset_index(name=metric_name)
            )
            by_id_cutoff["model"] = model
            results.append(by_id_cutoff)

        return pd.concat(results, ignore_index=True)[
            ["model", "unique_id", "cutoff", metric_name]
        ]

    @staticmethod
    def add_mean_across_cutoffs(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        mean_rows = df.groupby(["model", "unique_id"])[metric_name].mean().reset_index()
        mean_rows["cutoff"] = "MEAN"
        return pd.concat([df, mean_rows], ignore_index=True)

    @staticmethod
    def rank_models(
        df_metrics: pd.DataFrame, metric_name: str
    ) -> pd.DataFrame:
        df_metrics = df_metrics[df_metrics["unique_id"].notnull()]
        ranking = df_metrics.copy().sort_values(["unique_id", "cutoff", metric_name])
        ranking["rank"] = (
            ranking.groupby(["unique_id", "cutoff"])[metric_name]
            .rank(method="min")
            .astype(int)
        )
        return ranking[["unique_id", "cutoff", "rank", "model", metric_name]]
