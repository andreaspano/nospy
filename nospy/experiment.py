"""
Experiment orchestration for forecasting workflows.
"""

import json

import pandas as pd
from neuralforecast import NeuralForecast

from nospy.config import ExperimentConfig
from nospy.plot import save_plots
from nospy.features import FeaturesCalculator

from nospy.data import download_prices, prepare_timeseries
from nospy.evaluation import Evaluator
from nospy.models import ModelFactory
from nospy.prompt import generate_model_json
from nospy.utils import make_output_paths, setup_environment, silence
from nospy.reconcile import reconcile


class ForecastExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_paths = make_output_paths(config.runtime.out_dir)

        self.raw_data = None
        self.ts = None
        self.df_cv = None
        self.df_metrics = None
        self.df_ranking = None
        self.features_df = None

    def load_data(self) -> pd.DataFrame:
        self.raw_data = download_prices(
            tickers=self.config.data.tickers,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
        )
        return self.raw_data

    def prepare_data(self) -> pd.DataFrame:
        if self.raw_data is None:
            self.load_data()

        self.ts = prepare_timeseries(self.raw_data)
        return self.ts

    def build_forecaster(self) -> NeuralForecast:
        models = ModelFactory.build(self.config)

        return NeuralForecast(
            models=models,
            freq=self.config.data.freq,
        )

    def run_cross_validation(self) -> pd.DataFrame:
        if self.ts is None:
            self.prepare_data()

        nf = self.build_forecaster()

        self.df_cv = nf.cross_validation(
            df=self.ts,
            h=self.config.cv.h,
            n_windows=self.config.cv.n_windows,
            step_size=self.config.cv.step_size,
            refit=self.config.cv.refit,
        )

        return self.df_cv

    def reconcile_forecasts(self) -> None:
        """Reconcile base forecasts using the configured hierarchical method."""
        method = self.config.evaluation.reconciliation_method
        if not method:
            return
        self.df_cv = reconcile(self.df_cv, method=method)

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.df_cv is None:
            self.run_cross_validation()

        metric_name = self.config.evaluation.metric

        self.df_metrics = Evaluator.compute_metrics(
            self.df_cv,
            metric_name=metric_name,
        )
        self.df_metrics = Evaluator.add_mean_across_cutoffs(
            self.df_metrics,
            metric_name=metric_name,
        )

        self.df_ranking = Evaluator.rank_models(
            self.df_metrics,
            metric_name=metric_name,
        )

        return self.df_metrics, self.df_ranking

    def save_results(self) -> None:
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

    def write_forecast_plots(self) -> None:
        id_cols = {"unique_id", "ds", "cutoff", "y"}
        model_cols = [c for c in self.df_cv.columns if c not in id_cols]
        save_plots(self.df_cv, self.output_paths, self.ts, models=model_cols)
        print(f"Saved plots: {self.output_paths['run_dir']}")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        setup_environment(self.config.runtime.cuda_visible_devices)

        self.load_data()
        self.prepare_data()

        run_dir = self.output_paths["run_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build feature summary and regenerate model configs via Copilot
        calc = FeaturesCalculator(self.ts)
        self.features_df = calc.compute_features()
        summary = calc.summarize()
        (run_dir / "features_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        if not self.config.runtime.test:
            for model_name in self.config.models:
                generate_model_json(
                    calc,
                    model_name=model_name,
                    h=self.config.cv.h,
                    config=self.config,
                    llm_config=self.config.llm,
                    out_dir=run_dir,
                )
                print(f"Model config updated: json/{model_name.lower().replace('auto', '')}.json")

        with silence():
            self.run_cross_validation()
        self.reconcile_forecasts()
        self.evaluate()
        self.save_results()
        self.write_forecast_plots()

        return self.df_cv, self.df_metrics, self.df_ranking, self.features_df
