import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark import dspark_miss_overlap_candidates as overlap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


FIELDNAMES = ["run", "config", "m", "n", "k", "count"]


def miss_row(run, config, m, n, k, count=4):
    return {
        "run": run,
        "config": config,
        "m": str(m),
        "n": str(n),
        "k": str(k),
        "count": str(count),
    }


def write_misses(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class TestDSparkMissOverlapCandidates(CustomTestCase):
    def test_read_and_split_uses_config_name_and_filters_m(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate_path = root / "candidate.csv"
            baseline_path = root / "baseline.csv"
            write_misses(
                candidate_path,
                [
                    miss_row(
                        "p17",
                        "/candidate/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                        64,
                        6144,
                        3072,
                    ),
                    miss_row("p17", "/candidate/bf16_tuned_gemm.csv", 128, 256, 6144),
                    miss_row(
                        "p17",
                        "/candidate/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                        2048,
                        6144,
                        3072,
                    ),
                ],
            )
            write_misses(
                baseline_path,
                [
                    miss_row(
                        "p18",
                        "/baseline/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                        64,
                        6144,
                        3072,
                    ),
                    miss_row(
                        "p18",
                        "/baseline/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                        96,
                        6144,
                        4096,
                    ),
                ],
            )

            candidate = overlap.read_miss_counter(candidate_path, max_m=1536)
            baseline = overlap.read_miss_counter(baseline_path, max_m=1536)
            splits = overlap.split_candidate_sets(candidate, baseline)

            self.assertEqual(splits["overlap"], [("a8w8", 64, 6144, 3072)])
            self.assertEqual(splits["candidate_only"], [("bf16", 128, 256, 6144)])
            self.assertEqual(splits["baseline_only"], [("a8w8", 96, 6144, 4096)])

    def test_write_outputs_and_summary(self):
        candidate = overlap.Counter(
            {
                ("a8w8", 64, 6144, 3072): 4,
                ("bf16", 128, 256, 6144): 8,
            }
        )
        baseline = overlap.Counter(
            {
                ("a8w8", 64, 6144, 3072): 4,
                ("a8w8", 96, 6144, 4096): 4,
            }
        )
        splits = overlap.split_candidate_sets(candidate, baseline)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = overlap.write_outputs(
                root,
                candidate=candidate,
                baseline=baseline,
                splits=splits,
                max_m_label="m1536",
            )
            summary = overlap.build_summary(
                candidate_input=Path("candidate.csv"),
                baseline_input=Path("baseline.csv"),
                candidate_label="p17",
                baseline_label="p18",
                candidate_runs={"p17"},
                baseline_runs={"p18"},
                max_m=1536,
                candidate=candidate,
                baseline=baseline,
                splits=splits,
                files=files,
            )

            self.assertEqual(summary["overlap_shapes"], 1)
            self.assertEqual(summary["candidate_only_events"], 8)
            self.assertEqual(summary["baseline_only_events"], 4)
            self.assertEqual(summary["a8w8"]["overlap_shapes"], 1)
            self.assertEqual(summary["bf16"]["candidate_only_shapes"], 1)
            json.dumps(summary, sort_keys=True)

            a8w8_overlap = list(
                csv.reader((root / files["a8w8_overlap"]).open("r", encoding="utf-8"))
            )
            self.assertEqual(a8w8_overlap, [["M", "N", "K"], ["64", "6144", "3072"]])
            bf16_tier2 = list(
                csv.reader(
                    (root / files["bf16_candidate_only"]).open("r", encoding="utf-8")
                )
            )
            self.assertEqual(bf16_tier2[0][0:3], ["M", "N", "K"])
            self.assertEqual(bf16_tier2[1][0:3], ["128", "256", "6144"])

    def test_cli_summary_records_artifacts_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate_path = root / "candidate.csv"
            baseline_path = root / "baseline.csv"
            output_dir = root / "out"
            summary_path = root / "summary.json"
            write_misses(
                candidate_path,
                [
                    miss_row(
                        "p17",
                        "/candidate/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                        64,
                        6144,
                        3072,
                    )
                ],
            )
            write_misses(
                baseline_path,
                [
                    miss_row(
                        "p18",
                        "/baseline/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
                        64,
                        6144,
                        3072,
                    )
                ],
            )

            argv = [
                "dspark_miss_overlap_candidates.py",
                "--candidate-input",
                str(candidate_path),
                "--baseline-input",
                str(baseline_path),
                "--output-dir",
                str(output_dir),
                "--summary-output",
                str(summary_path),
                "--metadata",
                "run_id=p17p18",
            ]
            with mock.patch.object(sys, "argv", argv):
                overlap.main()

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["overlap_shapes"], 1)
            self.assertIn("sha256", summary["artifacts"]["inputs"]["candidate"])
            self.assertIn("sha256", summary["artifacts"]["outputs"]["a8w8_overlap"])
            self.assertEqual(summary["provenance"]["metadata"]["run_id"], "p17p18")


if __name__ == "__main__":
    unittest.main()
