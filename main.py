import time
import argparse
from nospy.config import ExperimentConfig
from nospy.experiment import ForecastExperiment
from nospy.utils import load_yaml_no_dupes


def main():
    parser = argparse.ArgumentParser(description="Run forecasting experiment.")
    parser.add_argument(
        "--config", type=str, default="yaml/run.yaml", help="Path to config YAML file."
    )
    args = parser.parse_args()

    start = time.time()

    params = load_yaml_no_dupes(args.config)

    config = ExperimentConfig.from_dict(params)

    experiment = ForecastExperiment(config)
    df_cv, df_metrics, df_ranking, features_df = experiment.run()

    print("\nModel ranking:")
    print(df_ranking)

    elapsed = time.time() - start
    print(f"\nTotal execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
