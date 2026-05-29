import time

import pandas as pd
from neuralforecast import NeuralForecast


from nospy.plot import save_plots
from nospy.features import build_feature_dataframe

from nospy.data import download_prices, prepare_timeseries
from nospy.evaluation import Evaluator
from nospy.models import ModelFactory
from nospy.utils import make_output_paths, setup_environment


class ForecastExperiment:

    def __init__(self, config):
        self.config = config
        self.output_paths = make_output_paths(config.out_dir)

        self.raw_data = None
        self.ts = None
        self.df_cv = None
        self.df_metrics = None
        self.df_ranking = None
        self.features_df = None

    def load_data(self):
        self.raw_data = download_prices(
            tickers=self.config.tickers,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        return self.raw_data

    def prepare_data(self):
        if self.raw_data is None:
            self.load_data()

        self.ts = prepare_timeseries(self.raw_data)
        return self.ts

    def build_forecaster(self):
        models = ModelFactory.build(self.config)

        return NeuralForecast(
            models=models,
            freq=self.config.freq,
        )

    def add_mib_aggregate_forecast(self, df_cv):
        id_cols = ["unique_id", "ds", "cutoff", "y"]

        model_cols = [
            col for col in df_cv.columns
            if col not in id_cols
        ]

        df_mib = (
            df_cv
            .groupby(["ds", "cutoff"], as_index=False)[["y"] + model_cols]
            .sum()
        )

        df_mib["unique_id"] = "mib"

        df_mib = df_mib[
            ["unique_id", "ds", "cutoff", "y"] + model_cols
        ]

        return pd.concat([df_cv, df_mib], ignore_index=True)

    def run_cross_validation(self):
        if self.ts is None:
            self.prepare_data()

        nf = self.build_forecaster()

        self.df_cv = nf.cross_validation(
            df=self.ts,
            h=self.config.h,
            n_windows=self.config.n_windows,
            step_size=self.config.step_size,
            refit=self.config.refit,
        )

        self.df_cv = self.add_mib_aggregate_forecast(self.df_cv)

        return self.df_cv


    def evaluate(self):
        if self.df_cv is None:
            self.run_cross_validation()

        metric_name = getattr(self.config, "evaluation_metric", "MAPE")

        self.df_metrics = Evaluator.compute_metrics(
            self.df_cv,
            metric_name=metric_name,
        )

        self.df_ranking = Evaluator.rank_models(
            self.df_metrics,
            metric_name=metric_name,
        )

        return self.df_metrics, self.df_ranking

    def save_results(self):
        if self.df_cv is None:
            raise ValueError("No cross-validation results to save.")

        if self.df_metrics is None or self.df_ranking is None:
            self.evaluate()

        # Ensure output directory exists
        run_dir = self.output_paths["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)

        self.df_cv.to_csv(self.output_paths["cv"], index=False)
        self.df_metrics.to_csv(self.output_paths["metrics"], index=False)
        self.df_ranking.to_csv(self.output_paths["ranking"], index=False)

        # Save features
        if self.features_df is not None:
            features_path = run_dir / "features.csv"
            self.features_df.to_csv(features_path, index=False)
            print(f"Features: {features_path}")

        print("Saved results:")
        print(f"CV: {self.output_paths['cv']}")
        print(f"Metrics: {self.output_paths['metrics']}")
        print(f"Ranking: {self.output_paths['ranking']}")


    def save_plots(self):
        save_plots(self.df_cv, self.output_paths)
        print("Saved plots:")
        print(f"Forecast vs Actuals: {self.output_paths.get('forecast_vs_actuals', 'forecast_vs_actuals.png')}")
        print(f"Scatter Forecast vs Actuals: {self.output_paths.get('scatter_forecast_vs_actuals', 'scatter_forecast_vs_actuals.png')}")


    def run(self):
        setup_environment(self.config.cuda_visible_devices)

        self.load_data()
        self.prepare_data()

        self.features_df = build_feature_dataframe(self.ts)
        
        
        self.run_cross_validation()
        self.evaluate()
        self.save_results()
        self.save_plots()

        return self.df_cv, self.df_metrics, self.df_ranking, self.features_df

