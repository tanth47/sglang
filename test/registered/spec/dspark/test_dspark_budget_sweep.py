import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmark import dspark_budget_sweep as sweep
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkBudgetSweep(CustomTestCase):
    def test_parse_frac_list(self):
        self.assertEqual(sweep.parse_frac_list("0.25, .5,1"), [0.25, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            sweep.parse_frac_list("0")
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            sweep.parse_frac_list("1.5")

    def test_build_sweep_runs_includes_auto(self):
        with tempfile.TemporaryDirectory() as td:
            runs = sweep.build_sweep_runs(
                output_dir=Path(td),
                run_prefix="glm52",
                fracs=[0.25, 1.0],
                include_auto=True,
            )

            self.assertEqual(
                [run.label for run in runs],
                ["glm52_auto", "glm52_frac_0p25", "glm52_frac_1"],
            )
            self.assertEqual([run.frac for run in runs], [None, 0.25, 1.0])
            self.assertEqual(runs[1].output.name, "glm52_frac_0p25.jsonl")
            self.assertEqual(
                runs[1].info_summary_output.name,
                "glm52_frac_0p25_info_summary.json",
            )

    def test_collect_command_sets_forced_budget_only_for_frac_runs(self):
        args = SimpleNamespace(
            harness=Path("benchmark/dspark_accuracy_harness.py"),
            base_url="http://127.0.0.1:30000",
            prompts=Path("/tmp/prompts.jsonl"),
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            concurrency=4,
            timeout_s=600,
            retries=1,
            retry_sleep_s=5.0,
            limit=16,
            start_idx=None,
            end_idx=None,
            sampling_seed=None,
            allow_nondeterministic_sampling=False,
            ignore_eos=True,
            extra_collect_arg=[],
            extra_info_summary_arg=[],
        )
        run = sweep.SweepRun(
            label="glm52_frac_0p5",
            frac=0.5,
            output=Path("/tmp/out.jsonl"),
            server_info_output=Path("/tmp/info.json"),
            manifest_output=Path("/tmp/manifest.json"),
            info_summary_output=Path("/tmp/info_summary.json"),
        )

        cmd = sweep.build_collect_command(args, run)

        self.assertIn("--dspark-clear-info-records", cmd)
        self.assertIn("--no-print-records", cmd)
        self.assertEqual(cmd[cmd.index("--limit") + 1], "16")
        self.assertEqual(cmd[cmd.index("--dspark-force-budget-frac") + 1], "0.5")

        auto_run = sweep.SweepRun(
            label="glm52_auto",
            frac=None,
            output=Path("/tmp/auto.jsonl"),
            server_info_output=Path("/tmp/auto_info.json"),
            manifest_output=Path("/tmp/auto_manifest.json"),
            info_summary_output=Path("/tmp/auto_info_summary.json"),
        )
        auto_cmd = sweep.build_collect_command(args, auto_run)
        self.assertNotIn("--dspark-force-budget-frac", auto_cmd)

        info_cmd = sweep.build_info_summary_command(args, run)
        self.assertEqual(info_cmd[2], "info-summary")
        self.assertEqual(info_cmd[info_cmd.index("--input") + 1], "/tmp/info.json")
        self.assertEqual(
            info_cmd[info_cmd.index("--summary-output") + 1],
            "/tmp/info_summary.json",
        )

    def test_report_command_passes_extra_report_args(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(
                report=Path("benchmark/dspark_perf_report.py"),
                output_dir=root,
                run_prefix="glm52",
                extra_report_arg=[
                    "--baseline-label",
                    "target",
                    "--min-speedup-vs-baseline",
                    "1.05",
                    "--require-sps-table",
                ],
            )
            runs = [
                sweep.SweepRun(
                    label="target",
                    frac=None,
                    output=root / "target.jsonl",
                    server_info_output=root / "target_info.json",
                    manifest_output=root / "target_manifest.json",
                    info_summary_output=root / "target_info_summary.json",
                ),
                sweep.SweepRun(
                    label="dspark",
                    frac=None,
                    output=root / "dspark.jsonl",
                    server_info_output=root / "dspark_info.json",
                    manifest_output=root / "dspark_manifest.json",
                    info_summary_output=root / "dspark_info_summary.json",
                ),
            ]

            cmd = sweep.build_report_command(args, runs)

            self.assertIn("--manifest-run", cmd)
            self.assertEqual(cmd[-5:], args.extra_report_arg)
            self.assertEqual(cmd[cmd.index("--baseline-label") + 1], "target")
            self.assertIn("--require-sps-table", cmd)

    def test_dry_run_does_not_create_output_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = root / "prompts.jsonl"
            prompts.write_text('{"idx": 0, "prompt": "hello"}\n', encoding="utf-8")
            output_dir = root / "dry-run-output"
            argv = [
                "dspark_budget_sweep.py",
                "--base-url",
                "http://127.0.0.1:30000",
                "--prompts",
                str(prompts),
                "--output-dir",
                str(output_dir),
                "--run-prefix",
                "glm52",
                "--fracs",
                "0.5",
                "--no-include-auto",
                "--dry-run",
            ]

            with mock.patch.object(sys, "argv", argv):
                sweep.main()

            self.assertFalse(output_dir.exists())

    def test_command_manifest_records_specs_and_log_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(
                base_url="http://127.0.0.1:30000",
                prompts=Path("/tmp/prompts.jsonl"),
                output_dir=root,
                run_prefix="glm52",
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                concurrency=1,
                limit=None,
                start_idx=None,
                end_idx=None,
            )
            run = sweep.SweepRun(
                label="glm52_auto",
                frac=None,
                output=root / "out.jsonl",
                server_info_output=root / "server_info.json",
                manifest_output=root / "manifest.json",
                info_summary_output=root / "summary.json",
            )
            specs = [
                sweep.CommandSpec("glm52_auto_collect", ["python", "collect.py"]),
                sweep.CommandSpec("perf_report", ["python", "report.py"]),
            ]

            manifest = sweep.command_manifest(args, [run], specs)

            self.assertEqual(
                manifest["command_log_dir"], str(root / "glm52_command_logs")
            )
            self.assertEqual(manifest["commands"], [spec.command for spec in specs])
            self.assertEqual(manifest["command_specs"][0]["label"], specs[0].label)
            self.assertEqual(manifest["command_results"], [])
            json.dumps(manifest, sort_keys=True)

    def test_run_command_writes_log_and_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "cmd.log"
            spec = sweep.CommandSpec(
                label="hello",
                command=[
                    sys.executable,
                    "-c",
                    "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)",
                ],
            )

            result = sweep.run_command(spec, dry_run=False, log_path=log_path)

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.label, "hello")
            self.assertEqual(result.log_path, str(log_path))
            self.assertGreaterEqual(result.elapsed_s, 0.0)
            self.assertIn("stdout-line", log_path.read_text(encoding="utf-8"))
            self.assertIn("stderr-line", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
