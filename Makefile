PYTHON=.venv/bin/python3
PIP=.venv/bin/pip


venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt





run:
	PYTHONPATH=$(shell pwd) $(PYTHON) main.py --config yaml/run.yaml

sync:
	rsync -avz andrea@mutolo:dev/nospy/out/ ~/dev/nospy/out/

test:
	PYTHONPATH=$(shell pwd) $(PYTHON) main.py --config yaml/test.yaml

freeze:
	$(PIP) freeze > requirements.txt

install:
	$(PIP) install -r requirements.txt

clean:
	rm -rf .venv
	rm -rf __pycache__


all: venv run clean	
