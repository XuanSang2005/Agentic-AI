PY := .venv/bin/python
PIP := .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

api:
	.venv/bin/uvicorn src.api.main:app --reload --port 8000

eval:
	$(PY) -m eval.run_eval

verify-data:
	$(PY) eval/verify_dataset.py

test:
	$(PY) -m pytest tests/ -q

.PHONY: install api eval verify-data test
