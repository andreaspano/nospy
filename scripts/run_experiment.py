


import time
import yaml
import argparse
from nospy.config import ExperimentConfig
from nospy.experiment import ForecastExperiment




def main():
    parser = argparse.ArgumentParser(description="Run forecasting experiment.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file.")
    args = parser.parse_args()

    start = time.time()

    with open(args.config) as f:
        params = yaml.safe_load(f)

    config = ExperimentConfig(**params)

    experiment = ForecastExperiment(config)
    df_cv, df_metrics, df_ranking = experiment.run()

    print("\nModel ranking:")
    print(df_ranking)

    elapsed = time.time() - start
    print(f"\nTotal execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
