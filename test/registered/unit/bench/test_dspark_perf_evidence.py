import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_tool():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "benchmark" / "dspark_perf_evidence.py"
    spec = importlib.util.spec_from_file_location("dspark_perf_evidence", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDSparkPerfEvidence(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_tool()

    def test_summarize_dspark_info_computes_gpu_timing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dspark_info.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": "compact",
                        "gamma": 7,
                        "records": [
                            {
                                "step_gpu_ms": 10.0,
                                "draft_gpu_ms": 2.0,
                                "target_verify_gpu_ms": 5.0,
                            },
                            {
                                "step_gpu_ms": 14.0,
                                "draft_gpu_ms": 4.0,
                                "target_verify_gpu_ms": 6.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = self.tool.summarize_dspark_info(path)

        self.assertEqual(summary["records"], 2)
        self.assertAlmostEqual(summary["timing_ms"]["step_gpu_ms"]["mean"], 12.0)
        self.assertAlmostEqual(summary["timing_ms"]["draft_gpu_ms"]["mean"], 3.0)
        self.assertAlmostEqual(
            summary["timing_ms"]["target_verify_gpu_ms"]["mean"], 5.5
        )
        self.assertAlmostEqual(summary["mean_unattributed_gpu_ms"], 3.5)

    def test_profile_trace_idle_fraction_uses_idle_counter_durations(self):
        events = [
            {"ph": "C", "name": "Idle", "ts": 0, "args": {"Idle": 1}},
            {"ph": "C", "name": "Idle", "ts": 40, "args": {"Idle": 0}},
            {"ph": "C", "name": "Idle", "ts": 100, "args": {"Idle": 1}},
            {"ph": "C", "name": "Idle", "ts": 120, "args": {"Idle": 0}},
        ]

        summary = self.tool.summarize_trace_idle(events)

        self.assertEqual(summary["idle_counter_events"], 4)
        self.assertAlmostEqual(summary["observed_time_us"], 120.0)
        self.assertAlmostEqual(summary["idle_time_us"], 60.0)
        self.assertAlmostEqual(summary["idle_fraction"], 0.5)

    def test_command_summarize_writes_manifest_and_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            accuracy = root / "accuracy.json"
            accuracy.write_text(
                json.dumps(
                    {
                        "requests": 4,
                        "ok_requests": 4,
                        "error_requests": 0,
                        "aggregate_accept_rate": 0.5,
                        "aggregate_accept_length": 4.5,
                        "verdict": {"passed": True, "failures": []},
                    }
                ),
                encoding="utf-8",
            )
            serving = root / "serving.jsonl"
            serving.write_text(
                json.dumps(
                    {
                        "completed": 4,
                        "request_throughput": 1.25,
                        "output_throughput": 128.0,
                        "mean_ttft_ms": 12.0,
                        "mean_tpot_ms": 3.0,
                        "p99_itl_ms": 6.0,
                        "accept_length": 4.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dspark = root / "server_info.json"
            dspark.write_text(
                json.dumps(
                    {
                        "internal_states": [
                            {
                                "dspark_info_record": {
                                    "records": [
                                        {
                                            "step_gpu_ms": 10.0,
                                            "draft_gpu_ms": 2.0,
                                            "target_verify_gpu_ms": 5.0,
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profile_dir = root / "profile"
            profile_dir.mkdir()
            (profile_dir / "trace.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "C", "name": "Idle", "ts": 0, "args": {"Idle": 1}},
                            {"ph": "C", "name": "Idle", "ts": 10, "args": {"Idle": 0}},
                            {"ph": "C", "name": "Idle", "ts": 20, "args": {"Idle": 0}},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            args = SimpleNamespace(
                out_dir=str(root / "out"),
                label="unit",
                node="node-a",
                gpu_ids="0,1,2,3",
                image="image-a",
                server_command_file=None,
                server_env_json=None,
                accuracy_summary=[str(accuracy)],
                serving_output=[str(serving)],
                dspark_info_json=[str(dspark)],
                profile_dir=[str(profile_dir)],
                notes=["note-a"],
                max_profile_json_bytes=1024 * 1024,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.tool.command_summarize(args)

            manifest = json.loads(
                (root / "out" / "dspark_perf_manifest.json").read_text()
            )
            report = (root / "out" / "dspark_perf_report.md").read_text()

        self.assertEqual(manifest["accuracy"][0]["ok_requests"], 4)
        self.assertAlmostEqual(
            manifest["serving"][0]["latest"]["output_throughput"], 128.0
        )
        self.assertEqual(manifest["dspark"][0]["records"], 1)
        self.assertAlmostEqual(
            manifest["profiles"][0]["trace_summaries"][0]["idle_fraction"], 0.5
        )
        self.assertIn("aggregate AR", report)
        self.assertIn("mean step GPU ms", report)


if __name__ == "__main__":
    unittest.main()
