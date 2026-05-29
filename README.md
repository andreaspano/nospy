# Forecasting Project

A small Python module for downloading financial time series data, running NeuralForecast models, cross-validating them, and comparing model performance.

## Project structure

```text
nospy/
├── json/
│   ├── nhits.json
│   ├── nbeats.json
│   └── tft.json
├── nospy/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── features.py
│   ├── plot.py
│   └── utils.py
├── yaml/
│   ├── run.yaml
│   └── test.yaml
├── requirements.txt
├── pyproject.toml
├── main.py
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
python main.py --config yaml/run.yaml
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

Edit `nospy/models.py` and add model names to `yaml/run.yaml` (or `yaml/test.yaml`).

Example:

```yaml
models:
	- AutoNHITS
	- AutoNBEATS
```

## Per-model hyperparameters

Each model has its own JSON file under json/. These files define three sections:

- fixed: always applied
- run: lists for tune.choice
- test: fixed values for quick tests

Example (json/nhits.json):

```json
{
	"fixed": {},
	"run": {
		"input_size": [20, 40, 60, 80],
		"learning_rate": [0.001, 0.0005]
	},
	"test": {
		"input_size": 1,
		"learning_rate": 0.01
	}
}
```
