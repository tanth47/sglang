#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from benchmark.dspark_profile_artifacts import (
        add_provenance_args,
        artifact_records,
        provenance_from_args,
    )
    from benchmark.dspark_tuned_miss_inputs import (
        A8W8_CONFIG_TOKEN,
        BF16_CONFIG_NAME,
        write_a8w8_untuned,
        write_bf16_untuned,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark.dspark_profile_artifacts import (
        add_provenance_args,
        artifact_records,
        provenance_from_args,
    )
    from benchmark.dspark_tuned_miss_inputs import (
        A8W8_CONFIG_TOKEN,
        BF16_CONFIG_NAME,
        write_a8w8_untuned,
        write_bf16_untuned,
    )


ShapeKey = tuple[str, int, int, int]
Shape = tuple[int, int, int]


def config_class(config: str) -> str | None:
    config_name = Path(config).name
    if A8W8_CONFIG_TOKEN in config_name:
        return "a8w8"
    if config_name == BF16_CONFIG_NAME:
        return "bf16"
    return None


def parse_runs(value: str | None) -> set[str] | None:
    if value is None:
        return None
    runs = {item.strip() for item in value.split(",") if item.strip()}
    return runs or None


def read_miss_counter(
    path: Path,
    *,
    runs: set[str] | None = None,
    max_m: int | None = None,
) -> Counter[ShapeKey]:
    counter: Counter[ShapeKey] = Counter()
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if runs is not None and row.get("run") not in runs:
                continue
            klass = config_class(row["config"])
            if klass is None:
                continue
            m = int(row["m"])
            if max_m is not None and m > max_m:
                continue
            key = (klass, m, int(row["n"]), int(row["k"]))
            counter[key] += int(row.get("count", "1"))
    return counter


def sort_key(key: ShapeKey) -> tuple[str, int, int, int]:
    klass, m, n, k = key
    return (klass, n, k, m)


def split_candidate_sets(
    candidate: Counter[ShapeKey],
    baseline: Counter[ShapeKey],
) -> dict[str, list[ShapeKey]]:
    candidate_keys = set(candidate)
    baseline_keys = set(baseline)
    return {
        "overlap": sorted(candidate_keys & baseline_keys, key=sort_key),
        "candidate_only": sorted(candidate_keys - baseline_keys, key=sort_key),
        "baseline_only": sorted(baseline_keys - candidate_keys, key=sort_key),
    }


def shapes_for_class(keys: list[ShapeKey], klass: str) -> list[Shape]:
    return [(m, n, k) for key_klass, m, n, k in keys if key_klass == klass]


def write_full_rows(
    path: Path,
    keys: list[ShapeKey],
    *,
    candidate: Counter[ShapeKey],
    baseline: Counter[ShapeKey],
    row_class: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "config_class",
                "M",
                "N",
                "K",
                "candidate_count",
                "baseline_count",
                "class",
            ]
        )
        for key in keys:
            klass, m, n, k = key
            writer.writerow(
                [klass, m, n, k, candidate.get(key, 0), baseline.get(key, 0), row_class]
            )


def write_outputs(
    output_dir: Path,
    *,
    candidate: Counter[ShapeKey],
    baseline: Counter[ShapeKey],
    splits: dict[str, list[ShapeKey]],
    max_m_label: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "overlap_full": f"miss_overlap_{max_m_label}_full.csv",
        "candidate_only_full": f"miss_candidate_only_{max_m_label}_full.csv",
        "baseline_only_full": f"miss_baseline_only_{max_m_label}_full.csv",
        "a8w8_overlap": f"aiter_a8w8_bpreshuffle_overlap_tier1_{max_m_label}.csv",
        "bf16_overlap": f"aiter_bf16_overlap_tier1_{max_m_label}.csv",
        "a8w8_candidate_only": f"aiter_a8w8_candidate_only_tier2_{max_m_label}.csv",
        "bf16_candidate_only": f"aiter_bf16_candidate_only_tier2_{max_m_label}.csv",
    }
    write_full_rows(
        output_dir / files["overlap_full"],
        splits["overlap"],
        candidate=candidate,
        baseline=baseline,
        row_class="overlap",
    )
    write_full_rows(
        output_dir / files["candidate_only_full"],
        splits["candidate_only"],
        candidate=candidate,
        baseline=baseline,
        row_class="candidate_only",
    )
    write_full_rows(
        output_dir / files["baseline_only_full"],
        splits["baseline_only"],
        candidate=candidate,
        baseline=baseline,
        row_class="baseline_only",
    )
    write_a8w8_untuned(
        output_dir / files["a8w8_overlap"],
        shapes_for_class(splits["overlap"], "a8w8"),
    )
    write_bf16_untuned(
        output_dir / files["bf16_overlap"],
        shapes_for_class(splits["overlap"], "bf16"),
    )
    write_a8w8_untuned(
        output_dir / files["a8w8_candidate_only"],
        shapes_for_class(splits["candidate_only"], "a8w8"),
    )
    write_bf16_untuned(
        output_dir / files["bf16_candidate_only"],
        shapes_for_class(splits["candidate_only"], "bf16"),
    )
    return files


def build_summary(
    *,
    candidate_input: Path,
    baseline_input: Path,
    candidate_label: str,
    baseline_label: str,
    candidate_runs: set[str] | None,
    baseline_runs: set[str] | None,
    max_m: int | None,
    candidate: Counter[ShapeKey],
    baseline: Counter[ShapeKey],
    splits: dict[str, list[ShapeKey]],
    files: dict[str, str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "candidate_input": str(candidate_input),
        "baseline_input": str(baseline_input),
        "candidate_label": candidate_label,
        "baseline_label": baseline_label,
        "candidate_runs": sorted(candidate_runs) if candidate_runs else None,
        "baseline_runs": sorted(baseline_runs) if baseline_runs else None,
        "max_m": max_m,
        "candidate_shapes": len(candidate),
        "baseline_shapes": len(baseline),
        "candidate_events": sum(candidate.values()),
        "baseline_events": sum(baseline.values()),
        "overlap_shapes": len(splits["overlap"]),
        "candidate_only_shapes": len(splits["candidate_only"]),
        "baseline_only_shapes": len(splits["baseline_only"]),
        "overlap_candidate_events": sum(candidate[key] for key in splits["overlap"]),
        "overlap_baseline_events": sum(baseline[key] for key in splits["overlap"]),
        "candidate_only_events": sum(
            candidate[key] for key in splits["candidate_only"]
        ),
        "baseline_only_events": sum(baseline[key] for key in splits["baseline_only"]),
        "files": files,
    }
    for klass in ("a8w8", "bf16"):
        candidate_keys = {key for key in candidate if key[0] == klass}
        baseline_keys = {key for key in baseline if key[0] == klass}
        summary[klass] = {
            "candidate_shapes": len(candidate_keys),
            "baseline_shapes": len(baseline_keys),
            "overlap_shapes": len(candidate_keys & baseline_keys),
            "candidate_only_shapes": len(candidate_keys - baseline_keys),
            "baseline_only_shapes": len(baseline_keys - candidate_keys),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two DSpark tuned-miss profiles and build tiered AITer "
            "tuning candidate CSVs. The overlap set is intended as tier 1; "
            "candidate-only rows are tier 2."
        )
    )
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--baseline-input", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-runs")
    parser.add_argument("--baseline-runs")
    parser.add_argument(
        "--max-m",
        type=int,
        default=1536,
        help="Only include miss rows with M <= this value. Use -1 for no limit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()

    max_m = None if args.max_m < 0 else args.max_m
    max_m_label = "mall" if max_m is None else f"m{max_m}"
    candidate_runs = parse_runs(args.candidate_runs)
    baseline_runs = parse_runs(args.baseline_runs)
    candidate = read_miss_counter(
        args.candidate_input, runs=candidate_runs, max_m=max_m
    )
    baseline = read_miss_counter(args.baseline_input, runs=baseline_runs, max_m=max_m)
    splits = split_candidate_sets(candidate, baseline)
    files = write_outputs(
        args.output_dir,
        candidate=candidate,
        baseline=baseline,
        splits=splits,
        max_m_label=max_m_label,
    )
    summary = build_summary(
        candidate_input=args.candidate_input,
        baseline_input=args.baseline_input,
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
        candidate_runs=candidate_runs,
        baseline_runs=baseline_runs,
        max_m=max_m,
        candidate=candidate,
        baseline=baseline,
        splits=splits,
        files=files,
    )
    summary["artifacts"] = {
        "inputs": artifact_records(
            {
                "candidate": args.candidate_input,
                "baseline": args.baseline_input,
            }
        ),
        "outputs": artifact_records(
            {label: args.output_dir / name for label, name in files.items()}
        ),
    }
    summary["provenance"] = provenance_from_args(args)
    summary_output = args.summary_output or (args.output_dir / "summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
