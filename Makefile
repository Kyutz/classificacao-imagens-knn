VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

REQUIREMENTS = numpy scikit-learn scipy joblib matplotlib pandas

SCRIPT = projeto.py

venv:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install $(REQUIREMENTS)


install:
	pip install --upgrade pip
	pip install $(REQUIREMENTS)

run: venv
	$(PYTHON) $(SCRIPT)

clean:
	rm -rf __pycache__ $(VENV_DIR)
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete

.PHONY: install run clean venv