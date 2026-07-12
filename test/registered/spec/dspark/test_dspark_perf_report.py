import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark import dspark_perf_report as report
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestDSparkPerfReport(CustomTestCase):
    def test_manifest_run_path_map_artifacts_and_gates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            collect = root / "collect.jsonl"
            server_info = root / "server_info.json"
            manifest = root / "manifest.json"
            trace_summary = root / "trace_summary.json"
            sps_table = root / "sps.json"
            sts = root / "sts.json"

            _write_jsonl(
                collect,
                [
                    {
                        "ok": True,
                        "elapsed_s": 1.0,
                        "completion_tokens": 4,
                        "meta_info": {
                            "completion_tokens": 4,
                            "spec_verify_ct": 2,
                            "spec_num_correct_drafts": 4,
                            "spec_num_proposed_drafts": 8,
                            "spec_accept_histogram": [0, 1, 1],
                            "spec_accept_length": 2.0,
                            "spec_accept_rate": 0.5,
                        },
                    }
                ],
            )
            _write_json(
                server_info,
                {
                    "verify_num_draft_tokens": 4,
                    "mode": "compact",
                    "records": [
                        {
                            "mode": "compact",
                            "bs": 2,
                            "num_verify_tokens": 5,
                            "reqs": [{"verify_len": 1}, {"verify_len": 4}],
                        }
                    ],
                },
            )
            _write_json(
                manifest,
                {
                    "summary": {
                        "run_label": "compact",
                        "elapsed_s": 2.0,
                        "output": "/artifacts/collect.jsonl",
                        "server_info_output": "/artifacts/server_info.json",
                    },
                    "artifacts": {
                        "collect_output": {"path": "/artifacts/collect.jsonl"},
                        "server_info": {"path": "/artifacts/server_info.json"},
                    },
                },
            )
            _write_json(
                trace_summary,
                {
                    "cuda_graph_records": 0,
                    "eager_records": 3,
                    "compact_records": 2,
                    "non_uniform_verify_lens_records": 1,
                    "saved_verify_tokens": 4,
                },
            )
            _write_json(sps_table, {"bias_seconds": 1.0})
            _write_json(sts, {"temperatures": [1.0]})

            path_maps = report.parse_path_maps([f"/artifacts={root}"])
            run = report.run_input_from_manifest("compact", manifest)
            record = report.build_record(run, elapsed_overrides={}, path_maps=path_maps)
            evidence = report.build_evidence(
                SimpleNamespace(
                    trace_summary=Path("/artifacts/trace_summary.json"),
                    sps_table=Path("/artifacts/sps.json"),
                    sps_manifest=None,
                    sts_calibration=Path("/artifacts/sts.json"),
                ),
                path_maps=path_maps,
            )
            verdict = report.build_verdict(
                [record],
                evidence,
                SimpleNamespace(
                    min_ok_requests=1,
                    min_ar=0.5,
                    min_al=2.0,
                    require_compact=True,
                    require_non_uniform_verify_lens=True,
                    min_saved_verify_tokens=1,
                    require_sps_table=True,
                    require_sts_calibration=True,
                    expect_target_verify_eager=True,
                    expect_target_verify_graph=False,
                ),
            )

            self.assertEqual(record["ok_requests"], 1)
            self.assertEqual(record["ar"], 0.5)
            self.assertEqual(record["al"], 2.0)
            self.assertEqual(record["throughput_tokens_s"], 2.0)
            self.assertEqual(record["compact_records"], 1)
            self.assertEqual(record["non_uniform_verify_lens_records"], 1)
            self.assertEqual(record["saved_verify_tokens"], 3)
            self.assertEqual(record["throughput_completion_tokens"], 4)
            self.assertTrue(evidence["artifacts"]["sps_table"]["exists"])
            self.assertIn("sha256", evidence["artifacts"]["sts_calibration"])
            self.assertTrue(verdict["passed"], verdict["failures"])

    def test_server_info_saved_tokens_uses_verify_lens_not_graph_padding(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            server_info = root / "server_info.json"
            _write_json(
                server_info,
                {
                    "verify_num_draft_tokens": 4,
                    "mode": "compact",
                    "records": [
                        {
                            "mode": "compact",
                            "bs": 2,
                            "num_verify_tokens": 8,
                            "verify_tokens_graph_key": 8,
                            "reqs": [{"verify_len": 1}, {"verify_len": 4}],
                        }
                    ],
                },
            )

            info = report.summarize_server_info(server_info, path_maps=[])

            self.assertEqual(info["full_verify_token_sum"], 8)
            self.assertEqual(info["scheduled_verify_token_sum"], 5)
            self.assertEqual(info["graph_token_sum"], 8)
            self.assertEqual(info["saved_verify_tokens"], 3)
            self.assertEqual(info["graph_padding_tokens"], 3)
            self.assertEqual(info["padded_graph_records"], 1)

    def test_manifest_resume_window_uses_new_request_tokens_for_throughput(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            collect = root / "collect.jsonl"
            manifest = root / "manifest.json"
            _write_jsonl(
                collect,
                [
                    {"ok": True, "completion_tokens": 10},
                    {"ok": True, "completion_tokens": 10},
                    {"ok": True, "completion_tokens": 10},
                ],
            )
            _write_json(
                manifest,
                {
                    "summary": {
                        "output": str(collect),
                        "elapsed_s_new_requests": 2.0,
                        "completion_tokens_new_requests": 10,
                    },
                    "artifacts": {
                        "collect_output": {"path": str(collect)},
                    },
                },
            )

            run = report.run_input_from_manifest("resumed", manifest)
            record = report.build_record(run, elapsed_overrides={}, path_maps=[])

            self.assertEqual(record["completion_tokens"], 30)
            self.assertEqual(record["throughput_completion_tokens"], 10)
            self.assertEqual(record["elapsed_s"], 2.0)
            self.assertEqual(record["throughput_tokens_s"], 5.0)
            self.assertEqual(
                record["throughput_basis"], "manifest_summary_elapsed_s_new_requests"
            )

    def test_output_helpers_create_parent_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jsonl_path = root / "nested" / "records.jsonl"
            text_path = root / "nested" / "report.md"

            report.write_jsonl(jsonl_path, [{"label": "dspark"}])
            report.write_text(text_path, "hello")

            self.assertTrue(jsonl_path.exists())
            self.assertEqual(
                json.loads(jsonl_path.read_text(encoding="utf-8"))["label"], "dspark"
            )
            self.assertEqual(text_path.read_text(encoding="utf-8"), "hello\n")

    def test_verdict_fails_low_acceptance_and_missing_artifacts(self):
        record = {
            "label": "bad",
            "ok_requests": 0,
            "ar": 0.1,
            "al": 1.0,
            "compact_records": 0,
            "non_uniform_verify_lens_records": 0,
            "saved_verify_tokens": 0,
        }
        verdict = report.build_verdict(
            [record],
            {"artifacts": {}, "trace_summary": None},
            SimpleNamespace(
                min_ok_requests=1,
                min_ar=0.5,
                min_al=2.0,
                require_compact=True,
                require_non_uniform_verify_lens=True,
                min_saved_verify_tokens=1,
                require_sps_table=True,
                require_sts_calibration=True,
                expect_target_verify_eager=True,
                expect_target_verify_graph=False,
            ),
        )

        self.assertFalse(verdict["passed"])
        self.assertGreaterEqual(len(verdict["failures"]), 8)

    def test_verdict_uses_trace_summary_for_evidence_level_gates(self):
        records = [
            {
                "label": "full_budget",
                "ok_requests": 8,
                "ar": 0.5,
                "al": 4.0,
                "compact_records": 1,
                "non_uniform_verify_lens_records": 0,
                "saved_verify_tokens": 0,
            }
        ]
        verdict = report.build_verdict(
            records,
            {
                "artifacts": {
                    "sps_table": {"exists": True},
                    "sts_calibration": {"exists": True},
                },
                "trace_summary": {
                    "compact_records": 8,
                    "non_uniform_verify_lens_records": 3,
                    "saved_verify_tokens": 17,
                    "cuda_graph_records": 0,
                    "eager_records": 8,
                },
            },
            SimpleNamespace(
                min_ok_requests=8,
                min_ar=0.5,
                min_al=4.0,
                require_compact=True,
                require_non_uniform_verify_lens=True,
                min_saved_verify_tokens=1,
                require_sps_table=True,
                require_sts_calibration=True,
                expect_target_verify_eager=True,
                expect_target_verify_graph=False,
            ),
        )

        self.assertTrue(verdict["passed"], verdict["failures"])

    def test_verdict_gates_non_greedy_seeded_coverage(self):
        records = [
            {
                "label": "dspark",
                "ok_requests": 8,
                "ar": 0.5,
                "al": 4.0,
            }
        ]
        args = SimpleNamespace(
            min_ok_requests=8,
            min_ar=0.5,
            min_al=4.0,
            min_speedup_vs_baseline=None,
            require_compact=False,
            require_non_uniform_verify_lens=False,
            require_non_greedy=True,
            require_seeded_sampling=True,
            require_non_greedy_accept_coverage=True,
            require_no_skipped=True,
            min_saved_verify_tokens=None,
            require_sps_table=False,
            require_sts_calibration=False,
            require_trace_verdict=False,
            expect_target_verify_eager=False,
            expect_target_verify_graph=False,
        )

        passing = report.build_verdict(
            records,
            {
                "artifacts": {},
                "trace_summary": {
                    "non_greedy_records": 4,
                    "seeded_sampling_records": 4,
                    "non_greedy_accept_covered_records": 4,
                    "non_greedy_accept_uncovered_records": 0,
                    "skipped_records": 0,
                },
            },
            args,
        )
        self.assertTrue(passing["passed"], passing["failures"])

        failing = report.build_verdict(
            records,
            {
                "artifacts": {},
                "trace_summary": {
                    "non_greedy_records": 0,
                    "seeded_sampling_records": 0,
                    "non_greedy_accept_covered_records": 0,
                    "non_greedy_accept_uncovered_records": 0,
                    "skipped_records": 1,
                },
            },
            args,
        )
        self.assertFalse(failing["passed"])
        self.assertTrue(
            any("no non-greedy records" in failure for failure in failing["failures"])
        )
        self.assertTrue(
            any(
                "no seeded sampling records" in failure
                for failure in failing["failures"]
            )
        )
        self.assertTrue(
            any("skipped records" in failure for failure in failing["failures"])
        )

    def test_acceptance_gates_ignore_target_only_baseline(self):
        records = [
            {
                "label": "target",
                "ok_requests": 16,
                "ar": None,
                "al": None,
                "spec_metric_rows": 0,
            },
            {
                "label": "dspark",
                "ok_requests": 16,
                "ar": 0.55,
                "al": 4.5,
                "spec_metric_rows": 16,
            },
        ]
        verdict = report.build_verdict(
            records,
            {"artifacts": {}, "trace_summary": None},
            SimpleNamespace(
                baseline_label="target",
                min_ok_requests=16,
                min_ar=0.5,
                min_al=4.0,
                min_speedup_vs_baseline=None,
                require_compact=False,
                require_non_uniform_verify_lens=False,
                min_saved_verify_tokens=None,
                require_sps_table=False,
                require_sts_calibration=False,
                require_trace_verdict=False,
                expect_target_verify_eager=False,
                expect_target_verify_graph=False,
            ),
        )

        self.assertTrue(verdict["passed"], verdict["failures"])

    def test_markdown_includes_trace_evidence(self):
        markdown = report.render_markdown(
            [{"label": "dspark", "ok_requests": 1, "requests": 1}],
            evidence={
                "trace_summary": {
                    "verdict": {"passed": True},
                    "records": 3,
                    "compact_records": 2,
                    "non_uniform_verify_lens_records": 1,
                    "saved_verify_tokens": 4,
                    "scheduled_verify_token_ratio": 0.5,
                    "eager_records": 3,
                    "cuda_graph_records": 0,
                }
            },
        )

        self.assertIn("Trace evidence", markdown)
        self.assertIn("compact=2", markdown)
        self.assertIn("scheduled_ratio=0.500", markdown)

    def test_speedup_annotation_and_trace_verdict_gate(self):
        records = [
            {
                "label": "target",
                "ok_requests": 16,
                "ar": None,
                "al": None,
                "throughput_tokens_s": 100.0,
            },
            {
                "label": "dspark",
                "ok_requests": 16,
                "ar": 0.55,
                "al": 4.5,
                "throughput_tokens_s": 125.0,
                "compact_records": 8,
                "non_uniform_verify_lens_records": 4,
                "saved_verify_tokens": 32,
            },
        ]
        report.annotate_speedups(records, baseline_label="target")

        self.assertEqual(records[0]["speedup_vs_baseline"], 1.0)
        self.assertEqual(records[1]["speedup_vs_baseline"], 1.25)
        markdown = report.render_markdown(records)
        self.assertIn("speedup", markdown)
        self.assertIn("1.250x", markdown)

        verdict = report.build_verdict(
            records,
            {
                "artifacts": {},
                "trace_summary": {
                    "verdict": {"passed": True},
                    "compact_records": 8,
                    "non_uniform_verify_lens_records": 4,
                    "saved_verify_tokens": 32,
                },
            },
            SimpleNamespace(
                baseline_label="target",
                min_ok_requests=16,
                min_ar=None,
                min_al=None,
                min_speedup_vs_baseline=1.2,
                require_compact=True,
                require_non_uniform_verify_lens=True,
                min_saved_verify_tokens=1,
                require_sps_table=False,
                require_sts_calibration=False,
                require_trace_verdict=True,
                expect_target_verify_eager=False,
                expect_target_verify_graph=False,
            ),
        )
        self.assertTrue(verdict["passed"], verdict["failures"])

    def test_speedup_gate_requires_baseline_label(self):
        verdict = report.build_verdict(
            [
                {
                    "label": "dspark",
                    "ok_requests": 1,
                    "ar": 0.5,
                    "al": 2.0,
                    "speedup_vs_baseline": None,
                }
            ],
            {"artifacts": {}, "trace_summary": None},
            SimpleNamespace(
                baseline_label=None,
                min_ok_requests=None,
                min_ar=None,
                min_al=None,
                min_speedup_vs_baseline=1.0,
                require_compact=False,
                require_non_uniform_verify_lens=False,
                min_saved_verify_tokens=None,
                require_sps_table=False,
                require_sts_calibration=False,
                require_trace_verdict=False,
                expect_target_verify_eager=False,
                expect_target_verify_graph=False,
            ),
        )

        self.assertFalse(verdict["passed"])
        self.assertTrue(
            any(
                "--min-speedup-vs-baseline requires" in failure
                for failure in verdict["failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
