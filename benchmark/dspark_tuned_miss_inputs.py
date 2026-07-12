#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from benchmark.dspark_profile_artifacts import (
        add_provenance_args,
        artifact_records,
        provenance_from_args,
    )
except ModuleNotFoundError:
    from dspark_profile_artifacts import (
        add_provenance_args,
        artifact_records,
        provenance_from_args,
    )

A8W8_CONFIG_TOKEN = "a8w8_blockscale_bpreshuffle"
BF16_CONFIG_NAME = "bf16_tuned_gemm.csv"


def parse_runs(value: str | None) -> set[str] | None:
    if value is None:
        return None
    runs = {item.strip() for item in value.split(",") if item.strip()}
    return runs or None


def read_tuned_miss_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def filter_shapes(
    rows: list[dict[str, str]],
    *,
    runs: set[str] | None,
    max_m: int | None,
) -> dict[str, list[tuple[int, int, int]]]:
    a8w8: set[tuple[int, int, int]] = set()
    bf16: set[tuple[int, int, int]] = set()
    for row in rows:
        if runs is not None and row.get("run") not in runs:
            continue
        m = int(row["m"])
        if max_m is not None and m > max_m:
            continue
        shape = (m, int(row["n"]), int(row["k"]))
        config_name = Path(row["config"]).name
        if A8W8_CONFIG_TOKEN in config_name:
            a8w8.add(shape)
        elif config_name == BF16_CONFIG_NAME:
            bf16.add(shape)
    return {
        "a8w8": sorted(a8w8, key=lambda item: (item[1], item[2], item[0])),
        "bf16": sorted(bf16, key=lambda item: (item[1], item[2], item[0])),
    }


def write_a8w8_untuned(path: Path, shapes: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["M", "N", "K"])
        writer.writerows(shapes)


def write_bf16_untuned(path: Path, shapes: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
        )
        for m, n, k in shapes:
            writer.writerow(
                [
                    m,
                    n,
                    k,
                    "False",
                    "torch.bfloat16",
                    "torch.bfloat16",
                    "False",
                    "False",
                ]
            )


def build_summary(
    *,
    input_path: Path,
    runs: set[str] | None,
    max_m: int | None,
    shapes: dict[str, list[tuple[int, int, int]]],
) -> dict[str, Any]:
    return {
        "input": str(input_path),
        "runs": sorted(runs) if runs is not None else None,
        "max_m": max_m,
        "a8w8_shape_count": len(shapes["a8w8"]),
        "bf16_shape_count": len(shapes["bf16"]),
        "a8w8_shapes": [{"m": m, "n": n, "k": k} for m, n, k in shapes["a8w8"]],
        "bf16_shapes": [{"m": m, "n": n, "k": k} for m, n, k in shapes["bf16"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build AITer untuned GEMM CSV inputs from DSpark launch-report "
            "tuned-miss rows."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--runs",
        help="Optional comma-separated run labels to include, for example p9c,p9a.",
    )
    parser.add_argument(
        "--max-m",
        type=int,
        default=64,
        help="Only include tuned misses with M <= this value. Use -1 for no limit.",
    )
    parser.add_argument("--a8w8-output", type=Path, required=True)
    parser.add_argument("--bf16-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    add_provenance_args(parser)
    args = parser.parse_args()

    max_m = None if args.max_m < 0 else args.max_m
    runs = parse_runs(args.runs)
    rows = read_tuned_miss_rows(args.input)
    shapes = filter_shapes(rows, runs=runs, max_m=max_m)
    write_a8w8_untuned(args.a8w8_output, shapes["a8w8"])
    write_bf16_untuned(args.bf16_output, shapes["bf16"])

    summary = build_summary(
        input_path=args.input,
        runs=runs,
        max_m=max_m,
        shapes=shapes,
    )
    summary["artifacts"] = {
        "inputs": artifact_records({"misses": args.input}),
        "outputs": artifact_records(
            {
                "a8w8": args.a8w8_output,
                "bf16": args.bf16_output,
            }
        ),
    }
    summary["provenance"] = provenance_from_args(args)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
