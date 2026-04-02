.PHONY: help run bench

PYTHON ?= python3
DATA_DIR ?= data
CASES_CSV ?= $(DATA_DIR)/generated_test_cases.csv
RESULTS_CSV ?= results/benchmark_results.csv

help:
	@echo "Available targets:"
	@echo "  make run   - Run address correction demo"
	@echo "  make bench - Run benchmark full report"

run:
	$(PYTHON) address_classification.py

bench:
	$(PYTHON) bench.py \
		--data-dir "$(DATA_DIR)" \
		--cases-csv "$(CASES_CSV)" \
		--results-csv "$(RESULTS_CSV)"
