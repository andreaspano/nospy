PYTHON=.venv/bin/python3
PIP=.venv/bin/pip


venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	PYTHONPATH=$(shell pwd) $(PYTHON) scripts/run_experiment.py

freeze:
	$(PIP) freeze > requirements.txt

clean:
	rm -rf .venv
	rm -rf __pycache__

remote:
	./scripts/remote.sh --no-venv 

all: venv run clean	
