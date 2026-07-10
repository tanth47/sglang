import os
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_verify_planner import (
    DSparkVerifyPlanner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args():
    return types.SimpleNamespace(
        speculative_dspark_align_verify_tokens_to_graph_tier=False,
        speculative_dspark_confidence_sts_path=None,
    )


def _draft_model(*, with_confidence_head=True):
    return types.SimpleNamespace(
        confidence_head=object() if with_confidence_head else None
    )


class TestDSparkVerifyPlanner(CustomTestCase):
    def test_static_mode_allows_backend_without_ragged_verify_graph_support(self):
        with patch.dict(os.environ, {"SGLANG_RAGGED_VERIFY_MODE": "static"}):
            planner = DSparkVerifyPlanner(
                draft_model=_draft_model(with_confidence_head=False),
                gamma=7,
                model_runner=types.SimpleNamespace(
                    attn_backend=types.SimpleNamespace(
                        supports_ragged_verify_graph=False
                    )
                ),
                device=torch.device("cpu"),
                tp_rank=0,
                server_args=_server_args(),
                verify_num_draft_tokens=8,
            )

        self.assertEqual(planner.mode_value, "static")
        self.assertFalse(planner.schedules_verify_budget)

    def test_compact_mode_rejects_backend_without_ragged_verify_graph_support(self):
        with patch.dict(os.environ, {"SGLANG_RAGGED_VERIFY_MODE": "compact"}):
            with self.assertRaisesRegex(
                ValueError,
                "does not advertise supports_ragged_verify_graph",
            ):
                DSparkVerifyPlanner(
                    draft_model=_draft_model(),
                    gamma=7,
                    model_runner=types.SimpleNamespace(
                        attn_backend=types.SimpleNamespace(
                            supports_ragged_verify_graph=False
                        )
                    ),
                    device=torch.device("cpu"),
                    tp_rank=0,
                    server_args=_server_args(),
                    verify_num_draft_tokens=8,
                )


if __name__ == "__main__":
    unittest.main()
