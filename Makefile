PYTHON ?= python3
VENV = .venv
RUFF = $(VENV)/bin/ruff
PYTHON_SOURCES = main.py ui core utils analysis tests scripts

.PHONY: format format-check

$(RUFF): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

format: $(RUFF)
	$(RUFF) format $(PYTHON_SOURCES)

format-check: $(RUFF)
	$(RUFF) format --check $(PYTHON_SOURCES)
