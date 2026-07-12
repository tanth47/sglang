import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark import dspark_tuned_miss_inputs as inputs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkTunedMissInputs(CustomTestCase):
    def test_filter_and_write_tuning_inputs(self):
        rows = [
            {
                "run": "p9c",
                "config": "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                "m": "8",
                "n": "6144",
                "k": "4096",
                "count": "4",
            },
            {
                "run": "p9a",
                "config": "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                "m": "64",
                "n": "6144",
                "k": "3072",
                "count": "4",
            },
            {
                "run": "p8a",
                "config": "/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                "m": "128",
                "n": "6144",
                "k": "3072",
                "count": "4",
            },
            {
                "run": "p9c",
                "config": "/tmp/aiter_configs/bf16_tuned_gemm.csv",
                "m": "32",
                "n": "3072",
                "k": "6144",
                "count": "4",
            },
        ]

        shapes = inputs.filter_shapes(rows, runs={"p9c", "p9a"}, max_m=64)

        self.assertEqual(shapes["a8w8"], [(64, 6144, 3072), (8, 6144, 4096)])
        self.assertEqual(shapes["bf16"], [(32, 3072, 6144)])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a8w8_path = root / "a8w8.csv"
            bf16_path = root / "bf16.csv"

            inputs.write_a8w8_untuned(a8w8_path, shapes["a8w8"])
            inputs.write_bf16_untuned(bf16_path, shapes["bf16"])

            self.assertEqual(
                list(csv.reader(a8w8_path.open("r", encoding="utf-8"))),
                [
                    ["M", "N", "K"],
                    ["64", "6144", "3072"],
                    ["8", "6144", "4096"],
                ],
            )
            self.assertEqual(
                list(csv.reader(bf16_path.open("r", encoding="utf-8"))),
                [
                    [
                        "M",
                        "N",
                        "K",
                        "bias",
                        "dtype",
                        "outdtype",
                        "scaleAB",
                        "bpreshuffle",
                    ],
                    [
                        "32",
                        "3072",
                        "6144",
                        "False",
                        "torch.bfloat16",
                        "torch.bfloat16",
                        "False",
                        "False",
                    ],
                ],
            )

    def test_build_summary(self):
        summary = inputs.build_summary(
            input_path=Path("misses.csv"),
            runs={"p9a", "p9c"},
            max_m=64,
            shapes={
                "a8w8": [(8, 6144, 4096)],
                "bf16": [(32, 3072, 6144)],
            },
        )

        encoded = json.dumps(summary, sort_keys=True)
        self.assertIn("a8w8_shape_count", encoded)
        self.assertEqual(summary["runs"], ["p9a", "p9c"])
        self.assertEqual(summary["a8w8_shape_count"], 1)
        self.assertEqual(summary["bf16_shapes"][0]["k"], 6144)

    def test_cli_summary_records_artifacts_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            misses = root / "misses.csv"
            a8w8_output = root / "a8w8.csv"
            bf16_output = root / "bf16.csv"
            summary_output = root / "summary.json"
            misses.write_text(
                "\n".join(
                    [
                        "run,config,m,n,k,count",
                        "p9c,/tmp/a8w8_blockscale_bpreshuffle_tuned_gemm.csv,8,6144,4096,4",
                        "p9c,/tmp/bf16_tuned_gemm.csv,16,3072,6144,2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "dspark_tuned_miss_inputs.py",
                "--input",
                str(misses),
                "--a8w8-output",
                str(a8w8_output),
                "--bf16-output",
                str(bf16_output),
                "--summary-output",
                str(summary_output),
                "--metadata",
                "git_commit=abc123",
                "--env-key",
                "DSPARK_TEST_ENV",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(
                os.environ, {"DSPARK_TEST_ENV": "present"}
            ):
                inputs.main()

            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertIn("sha256", summary["artifacts"]["inputs"]["misses"])
            self.assertIn("sha256", summary["artifacts"]["outputs"]["a8w8"])
            self.assertEqual(summary["provenance"]["metadata"]["git_commit"], "abc123")
            self.assertEqual(summary["provenance"]["env"]["DSPARK_TEST_ENV"], "present")


if __name__ == "__main__":
    unittest.main()
