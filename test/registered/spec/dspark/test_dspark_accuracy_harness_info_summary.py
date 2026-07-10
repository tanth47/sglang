import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_harness():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "benchmark" / "dspark_accuracy_harness.py"
    module_name = "_test_dspark_accuracy_harness"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDSparkAccuracyHarnessInfoSummary(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = _load_harness()

    def _run_info_summary(self, payload, **kwargs):
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            args = Namespace(
                input=f.name,
                summary_output=None,
                require_records_min=kwargs.get("require_records_min"),
                require_compact=kwargs.get("require_compact", False),
                require_non_uniform_verify_lens=kwargs.get(
                    "require_non_uniform_verify_lens", False
                ),
                require_padded_graph=kwargs.get("require_padded_graph", False),
                require_trimmed_verify_tokens=kwargs.get(
                    "require_trimmed_verify_tokens", False
                ),
                min_saved_verify_tokens=kwargs.get("min_saved_verify_tokens"),
                fail_on_verdict=kwargs.get("fail_on_verdict", False),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.harness.command_info_summary(args)
            return json.loads(output.getvalue())

    def test_info_summary_counts_compact_non_uniform_and_graph_padding(self):
        payload = {
            "internal_states": [
                {
                    "dspark_info_record": {
                        "mode": "compact",
                        "verify_num_draft_tokens": 8,
                        "records": [
                            {
                                "bs": 2,
                                "num_verify_tokens": 10,
                                "verify_tokens_graph_key": 16,
                                "step_gpu_ms": 3.0,
                                "target_verify_gpu_ms": 2.0,
                                "reqs": [
                                    {"rid": "a", "verify_len": 8},
                                    {"rid": "b", "verify_len": 2},
                                ],
                            }
                        ],
                    }
                }
            ]
        }

        summary = self._run_info_summary(
            payload,
            require_records_min=1,
            require_compact=True,
            require_non_uniform_verify_lens=True,
            require_padded_graph=True,
            require_trimmed_verify_tokens=True,
            min_saved_verify_tokens=6,
            fail_on_verdict=True,
        )

        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["request_rows"], 2)
        self.assertEqual(summary["compact_records"], 1)
        self.assertEqual(summary["non_uniform_verify_lens_records"], 1)
        self.assertEqual(summary["padded_graph_records"], 1)
        self.assertEqual(summary["full_verify_token_sum"], 16)
        self.assertEqual(summary["scheduled_verify_token_sum"], 10)
        self.assertEqual(summary["saved_verify_tokens"], 6)
        self.assertEqual(summary["graph_padding_tokens"], 6)
        self.assertEqual(summary["verify_len_histogram"], {"2": 1, "8": 1})
        self.assertEqual(summary["graph_key_counts"], {"16": 1})
        self.assertEqual(summary["timing_ms"]["step_gpu"]["count"], 1)
        self.assertTrue(summary["verdict"]["passed"])

    def test_info_summary_gate_fails_without_non_uniform_verify_lens(self):
        payload = {
            "dspark_info_record": {
                "mode": "compact",
                "verify_num_draft_tokens": 8,
                "records": [
                    {
                        "bs": 2,
                        "num_verify_tokens": 16,
                        "verify_tokens_graph_key": 16,
                        "reqs": [
                            {"rid": "a", "verify_len": 8},
                            {"rid": "b", "verify_len": 8},
                        ],
                    }
                ],
            }
        }

        with self.assertRaises(SystemExit):
            self._run_info_summary(
                payload,
                require_non_uniform_verify_lens=True,
                fail_on_verdict=True,
            )

    def test_collect_manifest_hashes_artifacts_and_filters_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            prompts = tmpdir / "prompts.jsonl"
            collect_output = tmpdir / "collect.jsonl"
            server_info = tmpdir / "server_info.json"
            prompts.write_text('{"idx": 0, "text": "hello"}\n', encoding="utf-8")
            collect_output.write_text('{"idx": 0, "ok": true}\n', encoding="utf-8")
            server_info.write_text('{"internal_states": []}\n', encoding="utf-8")
            args = Namespace(
                base_url="http://127.0.0.1:30000",
                prompts=str(prompts),
                output=str(collect_output),
                server_info_output=str(server_info),
                manifest_output=str(tmpdir / "manifest.json"),
                run_label="dspark-smoke",
                dspark_force_budget_frac=0.35,
            )

            with patch.dict(
                "os.environ",
                {
                    "HF_TOKEN": "should-not-be-recorded",
                    "HIP_VISIBLE_DEVICES": "0,1,2,3",
                    "SGLANG_RAGGED_VERIFY_MODE": "compact",
                },
            ):
                manifest = self.harness.build_collect_manifest(
                    args, {"ok_requests": 1}
                )

            self.assertEqual(
                manifest["schema"],
                "sglang-dspark-accuracy-harness-manifest-v1",
            )
            self.assertEqual(manifest["command"], "collect")
            self.assertEqual(manifest["args"]["run_label"], "dspark-smoke")
            self.assertEqual(
                manifest["artifacts"]["prompts"]["sha256"],
                self.harness.sha256_file(prompts),
            )
            self.assertEqual(
                manifest["artifacts"]["collect_output"]["sha256"],
                self.harness.sha256_file(collect_output),
            )
            self.assertEqual(
                manifest["artifacts"]["server_info"]["sha256"],
                self.harness.sha256_file(server_info),
            )
            self.assertEqual(
                manifest["environment"]["SGLANG_RAGGED_VERIFY_MODE"], "compact"
            )
            self.assertEqual(
                manifest["environment"]["HIP_VISIBLE_DEVICES"], "0,1,2,3"
            )
            self.assertNotIn("HF_TOKEN", manifest["environment"])


if __name__ == "__main__":
    unittest.main()
