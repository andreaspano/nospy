# Forecasting Project

A Python module for downloading financial time series data, running NeuralForecast models, cross-validating them, and comparing model performance.  
The project supports **automatic hyperparameter tuning** via Ray Tune + Optuna and can optionally use an **LLM (Copilot or DeepSeek)** to generate or improve the hyperparameter search space based on the statistical features of the input time series.

---

## Overview

- **Data acquisition** – download daily (or other interval) OHLCV prices from Yahoo Finance for a list of tickers.
- **Feature extraction** – compute a rich set of time‑series features (catch22, trend/seasonality, volatility, entropy, etc.) for each series and produce a distributional summary.
- **Model training & cross‑validation** – train any of the supported NeuralForecast models (NHITS, NBEATS, TFT) with a user‑defined hyperparameter search space, using time‑series cross‑validation.
- **Evaluation** – compute metrics (MSE, MAE, MAPE, SMAPE, MASE, RMSSE) across all cutoffs and rank models.
- **LLM‑assisted tuning** – the feature summary can be fed to an LLM (Copilot or DeepSeek) which returns a complete `model.json` file with `fixed`, `run`, and `test` sections tailored to the data.

---

## Installation

From inside the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

---

## Configuration

All experiment settings are defined in a **YAML configuration file** (e.g. `yaml/run.yaml` or `yaml/test.yaml`).  
The configuration is parsed into a set of dataclasses defined in `nospy/config.py`:

| Section | Purpose |
|---------|---------|
| `data` | Ticker list, date range, interval, auto‑adjust |
| `cv` | Cross‑validation parameters (n_windows, step_size, etc.) |
| `tuning` | Ray Tune / Optuna settings (num_samples, cpus, gpus) |
| `runtime` | Output directory, CUDA device |
| `evaluation` | Metric to optimise (e.g. `mse`, `mae`) |
| `llm` | LLM provider (copilot / deepseek), model name, temperature |
| `features` | Whether to include catch22, model‑shape features, etc. |
| `models` | List of model names to train (e.g. `AutoNHITS`, `AutoNBEATS`, `AutoTFT`) |

Example snippet:

```yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  interval: "1d"

cv:
  n_windows: 3
  step_size: 1

tuning:
  num_samples: 20
  cpus: 4
  gpus: 1

models:
  - AutoNHITS
  - AutoNBEATS
```

---

## Running an experiment

```bash
python main.py --config yaml/run.yaml
```

The script will:

1. Download price data for the specified tickers.
2. Prepare the time series (resample, fill missing values, etc.).
3. For each model listed in `models`:
   - Load the corresponding JSON file (`json/<model>.json`) which defines the hyperparameter search space.
   - Run time‑series cross‑validation with Ray Tune + Optuna.
   - Reconcile forecasts (if hierarchical structure is present).
   - Compute evaluation metrics.
4. Rank models according to the chosen metric.
5. Save results to the output directory (`./out/` by default).

---

## Project structure

```text
.
├── json/                     # Per‑model hyperparameter files
│   ├── nhits.json
│   ├── nbeats.json
│   └── tft.json
├── nospy/                    # Main package
│   ├── config.py             # Configuration dataclasses
│   ├── data.py               # Data download & preparation
│   ├── models.py             # Model factory, Ray Tune wrappers
│   ├── evaluation.py         # Metric computation & ranking
│   ├── experiment.py         # Orchestrator class
│   ├── features.py           # Feature extraction & summarisation
│   ├── plot.py               # Forecast visualisation
│   ├── prompt.py             # LLM prompt builder & API integration
│   ├── reconcile.py          # Forecast reconciliation
│   └── utils.py              # YAML loading, environment setup
├── yaml/                     # Experiment configuration files
│   ├── run.yaml
│   └── test.yaml
├── main.py                   # Entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Adding models

1. **Implement the model class** in `nospy/models.py` (or use one of the existing `Auto*WithScheduler` wrappers).
2. **Create a JSON file** under `json/` with the same name as the model key (e.g. `json/my_model.json`).  
   The JSON must contain three sections:
   - `fixed` – parameters that are always applied (scalars).
   - `run` – parameters to be tuned (each value is a JSON array of candidates).
   - `test` – scalar values for quick smoke tests.
3. **Add the model name** to the `models` list in your YAML configuration file.

---

## LLM‑assisted hyperparameter tuning

The project can use an LLM (GitHub Copilot or DeepSeek) to generate or improve the `model.json` file based on the statistical features of the input time series.

### How it works

1. The `FeaturesCalculator` extracts a comprehensive set of features (catch22, trend/seasonality, volatility, entropy, etc.) from the time series.
2. `summarize_features()` produces a distributional summary (quantiles, coverage percentages, etc.).
3. `build_model_prompt()` constructs a detailed prompt that includes:
   - The model schema (allowed parameters, their types, and constraints).
   - The feature summary.
   - Interpretation hints derived from the summary.
   - The current `model.json` (if one exists).
4. The prompt is sent to the LLM API (Copilot or DeepSeek), which returns a complete `model.json` object.
5. The returned JSON is validated, clamped to respect `batch_size` / `windows_batch_size` limits, and written back to `json/<model>.json`.

### Usage

```python
from nospy.features import FeaturesCalculator
from nospy.prompt import generate_model_json
from nospy.config import LLMConfig

calc = FeaturesCalculator(ts_df)
llm_config = LLMConfig(provider="copilot", model="gpt-4o", temperature=0.3)
new_cfg = generate_model_json(calc, model_name="nhits", h=5, llm_config=llm_config)
```

The generated prompt is also saved as `<model>_prompt.txt` in the `json/` directory for inspection.

---

## Outputs

After running an experiment, the output directory (`./out/` by default) contains:

| File | Description |
|------|-------------|
| `cv_results.csv` | Cross‑validation predictions for every cutoff |
| `metrics.csv` | Evaluation metrics per model per cutoff |
| `ranking.csv` | Model ranking based on the chosen metric |
| `forecast_plots/` | PNG plots of forecasts vs. actuals for each model |
| `<model>.json` | Final hyperparameter configuration used |
| `<model>_prompt.txt` | LLM prompt (if LLM‑assisted tuning was used) |

---

## Dependencies

Key packages (see `requirements.txt` for the full list):

- `neuralforecast` – NeuralForecast models
- `ray[tune]` – distributed hyperparameter tuning
- `optuna` – hyperparameter search algorithm
- `pandas`, `numpy` – data manipulation
- `yfinance` – data download
- `openai` – LLM API calls
- `pyyaml` – configuration parsing
- `matplotlib` – plotting
- `catch22` – time‑series feature extraction
- `statsmodels` – statistical tests (ARCH, etc.)
- `scipy` – signal processing (Hurst exponent, etc.)

---

## Tmux (remote sessions)

```bash
tmux new -s forecast
# ... run experiment ...
Ctrl+b d          # detach
tmux attach -t forecast   # re‑attach
```

Use **Tailscale** for secure remote connections.
