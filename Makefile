# Convenience targets. Everything here is a thin wrapper over the `biopharma-kg` CLI.
PY ?= python

.PHONY: install install-ml test test-fast smoke run run-offline-core clean

install:            ## core install (no graph ML)
	$(PY) -m pip install -e ".[dev]"

install-ml:         ## core + graph ML + LLM extraction + tests
	$(PY) -m pip install -e ".[ml,llm,dev]"

test:               ## full test suite (offline; ML tests skip if the [ml] extra is missing)
	$(PY) -m pytest -q

test-fast:          ## everything except the slower ML training tests
	$(PY) -m pytest -q --ignore=tests/test_ml.py --ignore=tests/test_ml_run.py

smoke:              ## check the 5 live sources + run a tiny real end-to-end pipeline (needs internet and GDSC2)
	biopharma-kg smoke

run:                ## the full pipeline on real data (needs internet, GDSC2 file, [ml] extra)
	biopharma-kg run-all

run-offline-core:   ## re-run everything that does not need the network or ML, from data already fetched
	biopharma-kg run-all --skip-fetch --skip-ml

clean:
	rm -rf data reports .pytest_cache
