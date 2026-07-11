PY := .venv/bin/python
PIP := .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

api:
	.venv/bin/uvicorn src.api.main:app --reload --port 8000

eval:
	$(PY) -m eval.run_eval

stress:
	$(PY) -m eval.stress_queries

bench:
	$(PY) eval/score_hard_benchmark.py

openapi:
	$(PY) -m src.api.export_openapi

deploy-hf:
	$(PY) deploy/deploy_hf.py $(SPACE)

verify-data:
	$(PY) eval/verify_dataset.py

test:
	$(PY) -m pytest tests/ -q

.PHONY: install api eval stress bench openapi deploy-hf verify-data test
