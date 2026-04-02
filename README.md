# Vietnamese Address Classification

Rule-based Vietnamese address parser that extracts **province**, **ward**, and **street** from noisy, abbreviated, or diacritics-missing input — no ML required.

## Quick Start

```bash
# Run the demo
make run

# Run benchmarks
make bench
```

## Requirements

- Python 3.8+
- No external dependencies (stdlib only)

## Project Structure

```
├── address_classification.py   # Core normalizer, fuzzy matcher & AddressCorrector
├── beam_search.py              # Beam-search variant of the corrector
├── template.py                 # Solution interface for pluggable strategies
├── bench.py                    # Benchmark script (latency & accuracy)
├── Makefile                    # Shortcuts for common tasks
├── data/
│   ├── list_province.csv       # Province dictionary
│   ├── list_ward.csv           # Ward dictionary
│   ├── list_street.csv         # Street dictionary
│   └── generated_test_cases.csv
└── results/                    # Benchmark output CSVs
```

## Usage

### As a library

```python
from address_classification import build_corrector

corrector = build_corrector("data")
result = corrector.correct("PhanVăn Trí, Thuần Giao, Hồ Chi Minh City")

print(result["street"])    # street name
print(result["ward"])      # ward name
print(result["province"])  # province name
print(result["ids"])       # matched entity IDs
```

### Beam-search variant

```python
from beam_search import build_beam_corrector

corrector = build_beam_corrector("data")
result = corrector.correct("123 Nguyen Hue, Q1, HCM")
```

### Benchmarks

```bash
# Quick run with make
make bench

# Compare corrector vs beam_search side by side
python3 bench.py \
    --solutions corrector,beam_search \
    --cases-csv ./data/generated_test_cases.csv \
    --samples 100 \
    --warmup 5 \
    --build-runs 1 \
    --results-csv results/benchmark_compare.csv

# Run beam_search only
python3 bench.py \
    --solutions beam_search \
    --cases-csv ./data/generated_test_cases.csv \
    --results-csv results/benchmark_beam.csv
```

#### `bench.py` CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--solutions` | `corrector` | Comma-separated solution IDs to benchmark |
| `--data-dir` | `data/` | Directory containing province/ward/street CSVs |
| `--cases-csv` | `generated_test_cases.csv` | Path to load/save labeled test cases |
| `--results-csv` | `results/benchmark_results.csv` | Path to save benchmark result CSV |
| `--samples` | `5000` | Number of queries for latency measurement |
| `--warmup` | `300` | Warmup queries before measurement starts |
| `--build-runs` | `5` | Number of cold index builds to average |
| `--eval-samples` | `2000` | Synthetic labeled queries for precision/recall |
| `--seed` | `42` | Random seed for reproducibility |
| `--regenerate-cases` | off | Regenerate test cases instead of loading from CSV |

The benchmark reports **precision/recall/F1** per level, **entry-level accuracy**, **latency stats** (mean, p50, p95, p99), and checks acceptance criteria (max < 100ms, avg < 10ms per request).

## Available Solutions

| ID | Description |
|----|-------------|
| `corrector` | Default fuzzy-match corrector |
| `corrector_no_number` | Same corrector with standalone numbers stripped first |
| `beam_search` | Beam-search–based corrector |
