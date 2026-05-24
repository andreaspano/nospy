# Forecasting Project

A small Python module for downloading financial time series data, running NeuralForecast models, cross-validating them, and comparing model performance.

## Project structure

```text
forecasting_project/
├── forecasting/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── evaluation.py
│   ├── experiment.py
│   └── utils.py
├── scripts/
│   └── run_experiment.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Install

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

## Run

```bash
python scripts/run_experiment.py
```

Outputs are saved to:

```text
./out/
```

including:

- cross-validation results
- model metrics
- model ranking

## Add more models

Edit `forecasting/models.py` and add model names to the config in `scripts/run_experiment.py`.

Example:

```python
models=["AutoNHITS", "AutoNBEATS"]
```
