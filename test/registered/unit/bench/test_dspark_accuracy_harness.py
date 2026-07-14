import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_harness():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "benchmark" / "dspark_accuracy_harness.py"
    spec = importlib.util.spec_from_file_location(
        "dspark_accuracy_harness", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    idx: int,
    output_ids,
    text: str,
    completion_tokens: int,
    verify_ct: int,
    correct: int,
    proposed: int,
    accept_length: float,
    accept_rate: float,
):
    request = {
        "prompt_sha256": f"prompt-{idx}",
        "payload_sha256": f"payload-{idx}",
    }
    return {
        "idx": idx,
        "ok": True,
        "text": text,
        "output_ids": output_ids,
        "request": request,
        "meta_info": {
            "completion_tokens": completion_tokens,
            "spec_verify_ct": verify_ct,
            "spec_num_correct_drafts": correct,
            "spec_num_proposed_drafts": proposed,
            "spec_accept_length": accept_length,
            "spec_accept_rate": accept_rate,
        },
    }


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class TestDSparkAccuracyHarness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = _load_harness()

    def test_summarize_run_aggregates_acceptance_metrics(self):
        rows = [
            _row(
                idx=0,
                output_ids=[1, 2, 3],
                text="a",
                completion_tokens=8,
                verify_ct=2,
                correct=3,
                proposed=4,
                accept_length=4.0,
                accept_rate=0.75,
            ),
            _row(
                idx=1,
                output_ids=[4, 5],
                text="b",
                completion_tokens=6,
                verify_ct=2,
                correct=1,
                proposed=4,
                accept_length=3.0,
                accept_rate=0.25,
            ),
            {"idx": 2, "ok": False, "error": "boom"},
        ]

        summary = self.harness.summarize_run(rows)

        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["ok_requests"], 2)
        self.assertEqual(summary["error_requests"], 1)
        self.assertEqual(summary["spec_metric_rows"], 2)
        self.assertAlmostEqual(summary["aggregate_accept_length"], 3.5)
        self.assertAlmostEqual(summary["aggregate_accept_rate"], 0.5)
        self.assertEqual(summary["accept_rate_buckets"]["<0.3"], 1)
        self.assertEqual(summary["accept_rate_buckets"]["0.7-0.85"], 1)

    def test_set_internal_state_posts_dspark_controls(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse({"updated": True})

        with patch.object(self.harness.urllib.request, "urlopen", fake_urlopen):
            response = self.harness.set_internal_state(
                "http://localhost:30000",
                {
                    "dspark_force_budget_frac": 0.25,
                    "dspark_clear_info_records": True,
                },
                7,
            )

        self.assertEqual(captured["url"], "http://localhost:30000/set_internal_state")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(
            captured["body"],
            {
                "server_args": {
                    "dspark_force_budget_frac": 0.25,
                    "dspark_clear_info_records": True,
                }
            },
        )
        self.assertEqual(response, {"updated": True})

    def test_set_internal_state_rejects_failed_response(self):
        def fake_urlopen(req, timeout):
            return _FakeResponse([{"updated": True}, {"updated": False}])

        with patch.object(self.harness.urllib.request, "urlopen", fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, "set_internal_state rejected"):
                self.harness.set_internal_state(
                    "http://localhost:30000",
                    {"dspark_force_budget_frac": 0.25},
                    7,
                )

    def test_compare_reports_token_mismatch_and_threshold_failure(self):
        target_rows = [
            _row(
                idx=0,
                output_ids=[1, 2, 3],
                text="same",
                completion_tokens=3,
                verify_ct=0,
                correct=0,
                proposed=0,
                accept_length=0.0,
                accept_rate=0.0,
            ),
            _row(
                idx=1,
                output_ids=[4, 5, 6],
                text="target",
                completion_tokens=3,
                verify_ct=0,
                correct=0,
                proposed=0,
                accept_length=0.0,
                accept_rate=0.0,
            ),
        ]
        spec_rows = [
            _row(
                idx=0,
                output_ids=[1, 2, 3],
                text="same",
                completion_tokens=3,
                verify_ct=1,
                correct=1,
                proposed=2,
                accept_length=3.0,
                accept_rate=0.5,
            ),
            _row(
                idx=1,
                output_ids=[4, 9, 6],
                text="spec",
                completion_tokens=3,
                verify_ct=1,
                correct=0,
                proposed=2,
                accept_length=1.0,
                accept_rate=0.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "target.jsonl"
            spec = Path(td) / "spec.jsonl"
            summary_path = Path(td) / "summary.json"
            self.harness.write_jsonl(target, target_rows)
            self.harness.write_jsonl(spec, spec_rows)
            args = SimpleNamespace(
                target=str(target),
                spec=str(spec),
                summary_output=str(summary_path),
                max_mismatch_examples=5,
                require_request_signature_match=True,
                min_token_exact_rate=1.0,
                fail_on_verdict=False,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.harness.command_compare(args)
            summary = json.loads(summary_path.read_text())

        self.assertEqual(summary["token_comparable_requests"], 2)
        self.assertEqual(summary["token_exact_matches"], 1)
        self.assertEqual(summary["token_mismatches"], 1)
        self.assertFalse(summary["verdict"]["passed"])
        self.assertIn("token exact match rate", summary["verdict"]["failures"][0])
        self.assertEqual(summary["mismatch_examples"][0]["first_mismatch"], 1)

    def test_compare_warns_when_token_mismatch_is_not_a_gate(self):
        target_rows = [
            _row(
                idx=0,
                output_ids=[1, 2, 3],
                text="target",
                completion_tokens=3,
                verify_ct=0,
                correct=0,
                proposed=0,
                accept_length=0.0,
                accept_rate=0.0,
            )
        ]
        spec_rows = [
            _row(
                idx=0,
                output_ids=[1, 9, 3],
                text="spec",
                completion_tokens=3,
                verify_ct=1,
                correct=1,
                proposed=2,
                accept_length=3.0,
                accept_rate=0.5,
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "target.jsonl"
            spec = Path(td) / "spec.jsonl"
            summary_path = Path(td) / "summary.json"
            self.harness.write_jsonl(target, target_rows)
            self.harness.write_jsonl(spec, spec_rows)
            args = SimpleNamespace(
                target=str(target),
                spec=str(spec),
                summary_output=str(summary_path),
                max_mismatch_examples=5,
                require_request_signature_match=True,
                min_token_exact_rate=None,
                fail_on_verdict=False,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.harness.command_compare(args)
            summary = json.loads(summary_path.read_text())

        self.assertTrue(summary["verdict"]["passed"])
        self.assertEqual(summary["token_exact_match_rate"], 0.0)
        self.assertIn("token outputs differ", summary["verdict"]["warnings"][0])


if __name__ == "__main__":
    unittest.main()
