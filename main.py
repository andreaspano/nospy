import os
import sys
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only execution (disables GPU).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Comma-separated list of GPU device IDs to use (e.g., '0' or '0,1').",
    )
    # Parse early to set environment variable before any GPU-using imports
    args, _ = parser.parse_known_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Re-parse fully now that environment is set
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
