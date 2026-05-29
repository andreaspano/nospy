PYTHON=.venv/bin/python3
PIP=.venv/bin/pip


venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt



run:
	PYTHONPATH=$(shell pwd) $(PYTHON) scripts/main.py --config config.yaml


test:
	PYTHONPATH=$(shell pwd) $(PYTHON) scripts/main.py --config test.yaml

freeze:
	$(PIP) freeze > requirements.txt

install:
	$(PIP) install -r requirements.txt

clean:
	rm -rf .venv
	rm -rf __pycache__


all: venv run clean	
