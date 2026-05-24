import time

from neuralforecast import NeuralForecast

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

    def run_cross_validation(self):
        if self.ts is None:
            self.prepare_data()

        nf = self.build_forecaster()

        start = time.time()

        self.df_cv = nf.cross_validation(
            df=self.ts,
            h=self.config.h,
            n_windows=self.config.n_windows,
            step_size=self.config.step_size,
            refit=self.config.refit,
        )

        # Only print elapsed time if needed
        # elapsed = (time.time() - start) / 60
        # print(f"Cross-validation elapsed time: {elapsed:.2f} minutes")

        return self.df_cv

    def evaluate(self):
        if self.df_cv is None:
            self.run_cross_validation()

        self.df_metrics = Evaluator.compute_metrics(self.df_cv)
        self.df_ranking = Evaluator.rank_models(self.df_metrics)

        return self.df_metrics, self.df_ranking

    def save_results(self):
        if self.df_cv is None:
            raise ValueError("No cross-validation results to save.")

        if self.df_metrics is None or self.df_ranking is None:
            self.evaluate()

        self.df_cv.to_csv(self.output_paths["cv"], index=False)
        self.df_metrics.to_csv(self.output_paths["metrics"], index=False)
        self.df_ranking.to_csv(self.output_paths["ranking"], index=False)

        print("Saved results:")
        print(f"CV: {self.output_paths['cv']}")
        print(f"Metrics: {self.output_paths['metrics']}")
        print(f"Ranking: {self.output_paths['ranking']}")

    def run(self):
        setup_environment(self.config.cuda_visible_devices)

        self.load_data()
        self.prepare_data()
        self.run_cross_validation()
        self.evaluate()
        self.save_results()

        return self.df_cv, self.df_metrics, self.df_ranking
