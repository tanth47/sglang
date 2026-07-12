import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark import dspark_strict_tuned_config as strict
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


FIELDNAMES = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "libtype",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]


def row(m, n, k, *, gfx="gfx950", kernel="1", err="0.0"):
    return {
        "gfx": gfx,
        "cu_num": "256",
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "libtype": "ck",
        "kernelId": kernel,
        "splitK": "0",
        "us": "1.0",
        "kernelName": "kernel",
        "tflops": "1.0",
        "bw": "1.0",
        "errRatio": err,
    }


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class TestDSparkStrictTunedConfig(CustomTestCase):
    def test_build_strict_overlay(self):
        base = [
            row(8, 6144, 3072, kernel="base"),
            row(16, 6144, 3072, kernel="keep"),
        ]
        candidate = [
            row(8, 6144, 3072, kernel="selected-replace", err="0.0"),
            row(24, 6144, 3072, kernel="selected-append", err="0.0001"),
            row(32, 6144, 3072, kernel="rejected", err="0.02"),
            row(40, 6144, 3072, kernel="neighbor", err="0.0"),
        ]

        output, summary = strict.build_strict_overlay(
            base_rows=base,
            candidate_rows=candidate,
            input_shapes={
                (8, 6144, 3072),
                (24, 6144, 3072),
                (32, 6144, 3072),
            },
            err_ratio_max=0.0001,
        )

        self.assertEqual(summary["selected_shape_count"], 2)
        self.assertEqual(summary["missing_or_rejected_shape_count"], 1)
        self.assertEqual(summary["replaced_rows"], 1)
        self.assertEqual(summary["appended_rows"], 1)
        self.assertEqual(
            [item["kernelId"] for item in output],
            ["selected-replace", "keep", "selected-append"],
        )
        self.assertEqual(
            summary["rejected_candidate_rows"][0]["reason"], "errRatio_above_threshold"
        )

    def test_cli_writes_config_and_summary_with_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base_path = root / "base.csv"
            candidate_path = root / "candidate.csv"
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            summary_path = root / "summary.json"

            write_rows(base_path, [row(8, 6144, 3072, kernel="base")])
            write_rows(
                candidate_path,
                [
                    row(8, 6144, 3072, kernel="selected", err="0.0"),
                    row(9, 6144, 3072, kernel="neighbor", err="0.0"),
                ],
            )
            input_path.write_text("M,N,K\n8,6144,3072\n", encoding="utf-8")

            argv = [
                "dspark_strict_tuned_config.py",
                "--base-config",
                str(base_path),
                "--candidate-config",
                str(candidate_path),
                "--input-shapes",
                str(input_path),
                "--output-config",
                str(output_path),
                "--summary-output",
                str(summary_path),
                "--metadata",
                "image=sglang:test",
            ]
            with mock.patch.object(sys, "argv", argv):
                strict.main()

            rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["kernelId"], "selected")
            self.assertEqual(summary["selected_shape_count"], 1)
            self.assertIn("sha256", summary["artifacts"]["inputs"]["base_config"])
            self.assertIn("sha256", summary["artifacts"]["outputs"]["output_config"])
            self.assertEqual(summary["provenance"]["metadata"]["image"], "sglang:test")


if __name__ == "__main__":
    unittest.main()
