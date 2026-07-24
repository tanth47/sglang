import types
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _planner() -> DSparkVerifyPlanner:
    planner = object.__new__(DSparkVerifyPlanner)
    planner._align_verify_tokens_to_graph_tier = True
    planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
    planner._schedule_cfg = DSparkScheduleConfig(gamma=7)
    planner._budget_planner = object()
    planner._dynamic_graph_tier = True
    planner.verify_num_draft_tokens = 8
    planner.model_runner = types.SimpleNamespace(
        decode_cuda_graph_runner=types.SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=[8, 16],
            max_bs=8,
        ),
        attn_backend=types.SimpleNamespace(use_dsa=False, dsa_index_topk=None),
    )
    planner.server_args = types.SimpleNamespace(tp_size=1)
    return planner


class TestS2AcceptanceContract(CustomTestCase):
    def test_observed_layout_keeps_sps_and_execution_lens(self):
        planner = _planner()
        physical = torch.tensor([8, 8], dtype=torch.int32)
        logical = torch.tensor([2, 2], dtype=torch.int32)

        with (
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute_with_sps_budget",
                return_value=(physical, logical),
            ) as dual_schedule,
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError(
                    "dual telemetry must not run a second top-k"
                ),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(None, 1),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner.is_hip",
                return_value=False,
            ),
        ):
            layout = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([100, 200], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=torch.ones((2, 7), dtype=torch.float32),
                budget=7,
                collect_sps_verify_lens=True,
            )

        self.assertIsNotNone(layout)
        self.assertTrue(torch.equal(layout.verify_lens, physical))
        self.assertTrue(torch.equal(layout.sps_verify_lens, logical))
        dual_schedule.assert_called_once_with(
            confidence=mock.ANY,
            execution_budget=14,
            sps_budget=7,
            cfg=planner._schedule_cfg,
            exact_execution=True,
        )

    def test_unobserved_layout_keeps_legacy_single_output_path(self):
        planner = _planner()
        physical = torch.tensor([8, 8], dtype=torch.int32)

        with (
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                return_value=physical,
            ) as schedule,
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute_with_sps_budget",
                side_effect=AssertionError("telemetry path must stay disabled"),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(None, 1),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner.is_hip",
                return_value=False,
            ),
        ):
            layout = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([100, 200], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=torch.ones((2, 7), dtype=torch.float32),
                budget=7,
            )

        self.assertIsNotNone(layout)
        self.assertIsNone(layout.sps_verify_lens)
        schedule.assert_called_once_with(
            confidence=mock.ANY,
            budget=14,
            cfg=planner._schedule_cfg,
            exact_budget=True,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
