PY ?= .venv/bin/python
.PHONY: test validate data pools notes values baselines analyze reproduce smoke

test:
	$(PY) -m pytest -q

validate:
	$(PY) -m src.schemas.validate instance $(filter-out %/eval_only.jsonl,$(wildcard data/marketbugs/*/*.jsonl)) $(wildcard data/swebench_lite/instances.jsonl)

data: ## build + validate datasets; fails on any invalid row
	$(PY) -m src.gen.build_marketbugs
	$(PY) -m src.swe.build_lite
	$(MAKE) validate

pools:
	$(PY) -m src.cands.build_pools

notes:
	$(PY) -m src.agents.run_notes

values:
	$(PY) -m src.agents.run_values

baselines:
	$(PY) -m src.baselines.run_all

analyze:
	$(PY) -m src.analysis.run_all

reproduce: ## CPU only, rebuilds every headline table from artifacts/
	$(PY) -m src.analysis.run_all --from-cache

smoke: ## G0 GPU smoke test on Modal (costs money)
	modal run src/llm/modal_server.py::smoke
