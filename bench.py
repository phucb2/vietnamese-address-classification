"""
Benchmark script for AddressCorrector indexing and inference latency.

Use this when you need repeatable performance numbers for:
- Index build time (cold build of AddressCorrector)
- Query latency (warm runtime correction speed)
"""

import argparse
import csv
import os
import random
import statistics
import time
from typing import Dict, List, Tuple

from address_classification import (
    load_provinces,
    load_streets,
    load_wards,
    normalize_text,
)
from template import available_solution_ids, build_solution, parse_solution_ids


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def generate_queries(corrector, sample_size: int, seed: int) -> List[str]:
    rng = random.Random(seed)

    province_names = [p.name for p in corrector.provinces.values()]
    ward_names = [w.name for w in corrector.wards.values()]
    street_names = [s.name for s in corrector.streets.values()] if corrector.streets else []

    if not province_names or not ward_names:
        raise ValueError("Dataset is empty: need at least provinces and wards.")

    def maybe_typo(s: str) -> str:
        # Add a small typo chance so benchmark includes some fuzzy path usage.
        if len(s) < 5 or rng.random() > 0.25:
            return s
        i = rng.randint(1, len(s) - 2)
        return s[:i] + s[i + 1 :]

    queries = []
    for _ in range(sample_size):
        province = maybe_typo(rng.choice(province_names))
        ward = maybe_typo(rng.choice(ward_names))

        if street_names and rng.random() < 0.85:
            street = maybe_typo(rng.choice(street_names))
            q = f"{street}, {ward}, {province}"
        else:
            q = f"{ward}, {province}"

        # Randomly normalize spacing/punctuation shapes in input.
        if rng.random() < 0.2:
            q = q.replace(", ", ",")
        if rng.random() < 0.1:
            q = q.replace(" ", "")
        queries.append(q)

    return queries


def _abbreviate_common(s: str, rng: random.Random) -> str:
    """
    Inject common Vietnamese address abbreviations in normalized forms.
    """
    rules = [
        ("thanh pho ho chi minh", ["tphcm", "tp hcm", "tp.hcm"]),
        ("ho chi minh", ["hcm"]),
        ("ha noi", ["hn"]),
        ("da nang", ["dn"]),
        ("thanh pho", ["tp", "t.p"]),
        ("phuong", ["p", "p."]),
        ("xa", ["x", "x."]),
        ("duong", ["d", "d.", "dg", "dg."]),
    ]
    out = s
    for src, candidates in rules:
        if src in out and rng.random() < 0.45:
            out = out.replace(src, rng.choice(candidates))
    return out


def _inject_typo(s: str, rng: random.Random) -> str:
    """
    Inject one typo: drop/swap/duplicate character.
    """
    if len(s) < 4:
        return s
    op = rng.choice(["drop", "swap", "dup"])
    i = rng.randint(1, len(s) - 2)
    if op == "drop":
        return s[:i] + s[i + 1 :]
    if op == "swap" and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    return s[:i] + s[i] + s[i:]


def load_catalog(data_dir: str) -> Dict[str, object]:
    province_path = os.path.join(data_dir, "list_province.csv")
    ward_path = os.path.join(data_dir, "list_ward.csv")
    street_path = os.path.join(data_dir, "list_street.csv")
    provinces = load_provinces(province_path)
    wards = load_wards(ward_path)
    streets = load_streets(street_path) if os.path.exists(street_path) else {}
    return {"provinces": provinces, "wards": wards, "streets": streets}


def generate_labeled_queries(catalog: Dict[str, object], sample_size: int, seed: int) -> List[Dict[str, str]]:
    """
    Build synthetic test cases with expected labels and noisy query text.
    """
    rng = random.Random(seed)
    provinces = catalog["provinces"]
    wards = list(catalog["wards"].values())
    streets = list(catalog["streets"].values()) if catalog["streets"] else []
    if not wards:
        raise ValueError("Dataset is empty: need wards with province mapping.")

    rows: List[Dict[str, str]] = []
    for _ in range(sample_size):
        ward = rng.choice(wards)
        province = provinces[ward.province_id]
        street = rng.choice(streets) if streets and rng.random() < 0.85 else None

        street_text = normalize_text(street.name)["spaced"] if street else ""
        ward_text = normalize_text(ward.name)["spaced"]
        province_text = normalize_text(province.name)["spaced"]

        parts = [p for p in [street_text, ward_text, province_text] if p]
        query = ", ".join(parts)
        noise_tags = []

        if rng.random() < 0.75:
            query = _abbreviate_common(query, rng)
            noise_tags.append("abbr")
        if rng.random() < 0.70:
            query = _inject_typo(query, rng)
            noise_tags.append("typo")
        if rng.random() < 0.25:
            query = query.replace(", ", ",")
            noise_tags.append("tight-comma")
        if rng.random() < 0.15:
            query = query.replace(" ", "")
            noise_tags.append("no-space")

        rows.append(
            {
                "query": query,
                "expected_street_id": street.id if street else "",
                "expected_street": street.name if street else "",
                "expected_ward_id": ward.id,
                "expected_ward": ward.name,
                "expected_province_id": province.id,
                "expected_province": province.name,
                "noise": "|".join(noise_tags) if noise_tags else "clean",
            }
        )
    return rows


def evaluate_precision_recall(
    solution, cases: List[Dict[str, str]]
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
    """
    Compute precision/recall per level and return row-wise predictions.
    """
    metrics = {
        "street": {"tp": 0, "fp": 0, "fn": 0},
        "ward": {"tp": 0, "fp": 0, "fn": 0},
        "province": {"tp": 0, "fp": 0, "fn": 0},
    }
    exported_rows: List[Dict[str, str]] = []

    for case in cases:
        out = solution.correct(case["query"])
        pred_ids = out["ids"]

        expected = {
            "street": case["expected_street_id"] or None,
            "ward": case["expected_ward_id"] or None,
            "province": case["expected_province_id"] or None,
        }
        predicted = {
            "street": pred_ids.get("street_id"),
            "ward": pred_ids.get("ward_id"),
            "province": pred_ids.get("province_id"),
        }

        for lvl in ("street", "ward", "province"):
            exp = expected[lvl]
            pred = predicted[lvl]
            if pred is not None:
                if exp is not None and pred == exp:
                    metrics[lvl]["tp"] += 1
                else:
                    metrics[lvl]["fp"] += 1
            if exp is not None and pred != exp:
                metrics[lvl]["fn"] += 1

        exported_rows.append(
            {
                **case,
                "pred_street_id": predicted["street"] or "",
                "pred_street": out["street"] or "",
                "pred_ward_id": predicted["ward"] or "",
                "pred_ward": out["ward"] or "",
                "pred_province_id": predicted["province"] or "",
                "pred_province": out["province"] or "",
                "street_correct": str(predicted["street"] == expected["street"]),
                "ward_correct": str(predicted["ward"] == expected["ward"]),
                "province_correct": str(predicted["province"] == expected["province"]),
            }
        )

    report: Dict[str, Dict[str, float]] = {}
    for lvl, m in metrics.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        report[lvl] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return report, exported_rows


def save_cases_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_cases_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def benchmark_build(solution_id: str, data_dir: str, runs: int) -> List[float]:
    durations_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = build_solution(solution_id, data_dir)
        end = time.perf_counter()
        durations_ms.append((end - start) * 1000.0)
    return durations_ms


def benchmark_latency(solution, queries: List[str], warmup: int) -> List[float]:
    warmup = max(0, min(warmup, len(queries)))

    for i in range(warmup):
        _ = solution.correct(queries[i])

    latencies_ms = []
    for q in queries[warmup:]:
        start = time.perf_counter()
        _ = solution.correct(q)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
    return latencies_ms


def print_stats(name: str, values_ms: List[float]) -> None:
    vals = sorted(values_ms)
    mean_v = statistics.fmean(vals) if vals else 0.0
    print(f"\n{name}")
    print("-" * len(name))
    print(f"count: {len(vals)}")
    print(f"mean : {mean_v:.4f} ms")
    print(f"min  : {vals[0]:.4f} ms" if vals else "min  : 0.0000 ms")
    print(f"p50  : {percentile(vals, 50):.4f} ms")
    print(f"p95  : {percentile(vals, 95):.4f} ms")
    print(f"p99  : {percentile(vals, 99):.4f} ms")
    print(f"max  : {vals[-1]:.4f} ms" if vals else "max  : 0.0000 ms")


def evaluate_acceptance(latencies_ms: List[float]) -> Dict[str, object]:
    """
    Acceptance criteria:
    - Maximum time per request < 0.1s (100ms)
    - Average time per request < 0.01s (10ms)
    """
    if not latencies_ms:
        return {
            "max_ms": 0.0,
            "avg_ms": 0.0,
            "max_ok": True,
            "avg_ok": True,
            "all_ok": True,
        }

    max_ms = max(latencies_ms)
    avg_ms = statistics.fmean(latencies_ms)
    max_ok = max_ms < 100.0
    avg_ok = avg_ms < 10.0
    return {
        "max_ms": max_ms,
        "avg_ms": avg_ms,
        "max_ok": max_ok,
        "avg_ok": avg_ok,
        "all_ok": max_ok and avg_ok,
    }


def compute_entry_accuracy(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Accuracy = correct entry / total entry
    where entry levels are: street, ward, province.
    """
    if not rows:
        return {
            "correct_entries": 0.0,
            "total_entries": 0.0,
            "accuracy": 0.0,
        }

    levels = ("street", "ward", "province")
    correct = 0
    for row in rows:
        for lvl in levels:
            if row.get(f"{lvl}_correct", "").lower() == "true":
                correct += 1

    total_entries = len(rows) * len(levels)
    accuracy = (correct / total_entries) if total_entries > 0 else 0.0
    return {
        "correct_entries": float(correct),
        "total_entries": float(total_entries),
        "accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple address solutions with one interface."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Path to data directory containing list_province.csv/list_ward.csv/list_street.csv",
    )
    parser.add_argument(
        "--build-runs",
        type=int,
        default=5,
        help="Number of cold index builds to benchmark.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic address queries for latency benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=300,
        help="Number of warmup queries before latency measurement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic synthetic query generation.",
    )
    parser.add_argument(
        "--solutions",
        default="corrector",
        help=(
            "Comma-separated solution ids to benchmark. "
            f"Available: {', '.join(available_solution_ids())}"
        ),
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=2000,
        help="Number of synthetic labeled queries for precision/recall evaluation.",
    )
    parser.add_argument(
        "--cases-csv",
        default=os.path.join(os.path.dirname(__file__), "generated_test_cases.csv"),
        help="Path to load/save labeled test cases CSV.",
    )
    parser.add_argument(
        "--results-csv",
        default=os.path.join(os.path.dirname(__file__), "results", "benchmark_results.csv"),
        help="Path to save benchmark predictions/results CSV.",
    )
    parser.add_argument(
        "--regenerate-cases",
        action="store_true",
        help="Regenerate labeled test cases and overwrite --cases-csv.",
    )
    args = parser.parse_args()

    if args.build_runs <= 0:
        raise ValueError("--build-runs must be > 0")
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.eval_samples <= 0:
        raise ValueError("--eval-samples must be > 0")
    solution_ids = parse_solution_ids(args.solutions)
    invalid = [s for s in solution_ids if s not in set(available_solution_ids())]
    if invalid:
        raise ValueError(
            f"Unknown solution ids: {invalid}. Available: {available_solution_ids()}"
        )

    print("AddressCorrector benchmark configuration")
    print("---------------------------------------")
    print(f"data_dir   : {args.data_dir}")
    print(f"build_runs : {args.build_runs}")
    print(f"samples    : {args.samples}")
    print(f"warmup     : {args.warmup}")
    print(f"seed       : {args.seed}")
    print(f"eval_samples: {args.eval_samples}")
    print(f"cases_csv  : {args.cases_csv}")
    print(f"results_csv: {args.results_csv}")
    print(f"solutions  : {', '.join(solution_ids)}")
    print(f"regenerate_cases: {args.regenerate_cases}")
    catalog = load_catalog(args.data_dir)
    eval_cases = load_cases_csv(args.cases_csv)
    cases_loaded = bool(eval_cases)

    if args.regenerate_cases or not cases_loaded:
        eval_cases = generate_labeled_queries(catalog, sample_size=args.eval_samples, seed=args.seed + 1)
        save_cases_csv(args.cases_csv, eval_cases)
        print(
            f"\nTest cases source: generated ({len(eval_cases)} rows) "
            f"and saved to {args.cases_csv}"
        )
    else:
        print(f"\nTest cases source: loaded {len(eval_cases)} rows from {args.cases_csv}")

    # Reuse labeled cases for latency too (no random generation each run).
    queries = [r["query"] for r in eval_cases if r.get("query")]
    if not queries:
        raise ValueError(f"No query rows found in {args.cases_csv}")
    if args.samples < len(queries):
        queries = queries[:args.samples]
    elif args.samples > len(queries):
        # Keep deterministic behavior: repeat from the start if user requests more samples than loaded cases.
        repeats = (args.samples + len(queries) - 1) // len(queries)
        queries = (queries * repeats)[:args.samples]

    # Sanity ensure normalization path is valid and avoid dead-code assumptions.
    _ = normalize_text(queries[0])
    for solution_id in solution_ids:
        print(f"\n=== Solution: {solution_id} ===")
        build_times = benchmark_build(solution_id, args.data_dir, runs=args.build_runs)
        print_stats("Index Build Time", build_times)

        solution = build_solution(solution_id, args.data_dir)
        latencies = benchmark_latency(solution, queries, warmup=args.warmup)
        print_stats("Correction Latency", latencies)

        total_s = sum(latencies) / 1000.0
        throughput = (len(latencies) / total_s) if total_s > 0 else 0.0
        print(f"\nthroughput: {throughput:.2f} queries/sec")

        report, rows = evaluate_precision_recall(solution, eval_cases)
        for row in rows:
            row["solution_id"] = solution_id
        result_path = args.results_csv
        if len(solution_ids) > 1:
            root, ext = os.path.splitext(args.results_csv)
            result_path = f"{root}_{solution_id}{ext or '.csv'}"
        save_cases_csv(result_path, rows)
        entry_acc = compute_entry_accuracy(rows)
        acceptance = evaluate_acceptance(latencies)

        print("\nPrecision/Recall Report")
        print("-----------------------")
        for lvl in ("street", "ward", "province"):
            m = report[lvl]
            print(
                f"{lvl:9s} "
                f"precision={m['precision']:.4f} "
                f"recall={m['recall']:.4f} "
                f"f1={m['f1']:.4f} "
                f"(tp={int(m['tp'])}, fp={int(m['fp'])}, fn={int(m['fn'])})"
            )
        print(f"\nSaved test cases to : {args.cases_csv}")
        print(f"Saved benchmark results to: {result_path}")

        print("\nAccuracy Report")
        print("---------------")
        print(
            "accuracy = correct_entry / total_entry "
            f"= {int(entry_acc['correct_entries'])}/{int(entry_acc['total_entries'])} "
            f"= {entry_acc['accuracy']:.4f}"
        )

        print("\nAcceptance Criteria")
        print("-------------------")
        print(
            f"Maximum time/request < 0.1s : "
            f"{'PASS' if acceptance['max_ok'] else 'FAIL'} "
            f"(max={acceptance['max_ms']:.4f} ms)"
        )
        print(
            f"Average time/request < 0.01s: "
            f"{'PASS' if acceptance['avg_ok'] else 'FAIL'} "
            f"(avg={acceptance['avg_ms']:.4f} ms)"
        )
        print(f"Overall acceptance: {'PASS' if acceptance['all_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
