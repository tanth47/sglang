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

Shape = tuple[int, int, int]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        return list(reader.fieldnames or []), list(reader)


def shape_key(row: dict[str, str]) -> Shape:
    return (int(row["M"]), int(row["N"]), int(row["K"]))


def row_key(row: dict[str, str]) -> tuple[str, str, int, int, int]:
    return (
        row.get("gfx", ""),
        row.get("cu_num", ""),
        int(row["M"]),
        int(row["N"]),
        int(row["K"]),
    )


def read_input_shapes(path: Path) -> set[Shape]:
    _fieldnames, rows = read_csv_rows(path)
    return {shape_key(row) for row in rows}


def parse_err_ratio(row: dict[str, str]) -> float | None:
    value = row.get("errRatio", "").strip()
    if not value:
        return None
    return float(value)


def build_strict_overlay(
    *,
    base_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    input_shapes: set[Shape],
    err_ratio_max: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    selected_by_key: dict[tuple[str, str, int, int, int], dict[str, str]] = {}
    selected_shapes: set[Shape] = set()
    rejected_rows: list[dict[str, Any]] = []

    for row in candidate_rows:
        shape = shape_key(row)
        if shape not in input_shapes:
            continue
        err_ratio = parse_err_ratio(row)
        if err_ratio is None or err_ratio > err_ratio_max:
            rejected_rows.append(
                {
                    "m": shape[0],
                    "n": shape[1],
                    "k": shape[2],
                    "errRatio": err_ratio,
                    "kernelId": row.get("kernelId"),
                    "libtype": row.get("libtype"),
                    "reason": (
                        "missing_errRatio"
                        if err_ratio is None
                        else "errRatio_above_threshold"
                    ),
                }
            )
            continue
        selected_by_key[row_key(row)] = row
        selected_shapes.add(shape)

    output_rows: list[dict[str, str]] = []
    replaced = 0
    for row in base_rows:
        key = row_key(row)
        selected = selected_by_key.pop(key, None)
        if selected is None:
            output_rows.append(row)
        else:
            output_rows.append(selected)
            replaced += 1

    appended = len(selected_by_key)
    output_rows.extend(selected_by_key.values())
    missing_shapes = sorted(
        input_shapes - selected_shapes, key=lambda item: (item[1], item[2], item[0])
    )

    summary = {
        "err_ratio_max": err_ratio_max,
        "base_rows": len(base_rows),
        "candidate_rows": len(candidate_rows),
        "input_shape_count": len(input_shapes),
        "selected_shape_count": len(selected_shapes),
        "rejected_candidate_row_count": len(rejected_rows),
        "missing_or_rejected_shape_count": len(missing_shapes),
        "replaced_rows": replaced,
        "appended_rows": appended,
        "output_rows": len(output_rows),
        "selected_shapes": [
            {"m": m, "n": n, "k": k}
            for m, n, k in sorted(
                selected_shapes, key=lambda item: (item[1], item[2], item[0])
            )
        ],
        "missing_or_rejected_shapes": [
            {"m": m, "n": n, "k": k} for m, n, k in missing_shapes
        ],
        "rejected_candidate_rows": rejected_rows,
    }
    return output_rows, summary


def write_csv_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict tuned-config overlay by admitting only candidate rows "
            "that exactly match workload shapes and pass an errRatio threshold."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--input-shapes", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--err-ratio-max", type=float, default=0.0001)
    add_provenance_args(parser)
    args = parser.parse_args()

    base_fieldnames, base_rows = read_csv_rows(args.base_config)
    candidate_fieldnames, candidate_rows = read_csv_rows(args.candidate_config)
    if base_fieldnames != candidate_fieldnames:
        raise ValueError(
            "base and candidate configs must use the same CSV header: "
            f"{base_fieldnames} != {candidate_fieldnames}"
        )

    output_rows, summary = build_strict_overlay(
        base_rows=base_rows,
        candidate_rows=candidate_rows,
        input_shapes=read_input_shapes(args.input_shapes),
        err_ratio_max=args.err_ratio_max,
    )
    summary.update(
        {
            "base_config": str(args.base_config),
            "candidate_config": str(args.candidate_config),
            "input_shapes": str(args.input_shapes),
            "output_config": str(args.output_config),
        }
    )

    write_csv_rows(args.output_config, base_fieldnames, output_rows)
    summary["artifacts"] = {
        "inputs": artifact_records(
            {
                "base_config": args.base_config,
                "candidate_config": args.candidate_config,
                "input_shapes": args.input_shapes,
            }
        ),
        "outputs": artifact_records({"output_config": args.output_config}),
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
