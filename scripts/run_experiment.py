

import time
import yaml
from nospy.config import ExperimentConfig
from nospy.experiment import ForecastExperiment



def main():

    start = time.time()

    with open("config.yaml") as f:
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
