import os
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_components.dspark_verify_planner import (
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args():
    return types.SimpleNamespace(
        speculative_dspark_align_verify_tokens_to_graph_tier=False,
        speculative_dspark_confidence_sts_path=None,
        speculative_dspark_min_verify_len=None,
        speculative_dspark_max_verify_len=None,
        speculative_dspark_survival_eps=1e-6,
    )


def _draft_model(*, with_confidence_head=True):
    return types.SimpleNamespace(
        confidence_head=object() if with_confidence_head else None
    )


class TestDSparkVerifyPlanner(CustomTestCase):
    def test_compute_confidence_hook_without_raw_out_fills_buffer_from_stash(self):
        gamma = 3
        raw = torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
        head = types.SimpleNamespace(_last_confidence_raw=None)

        class LegacyHookDraftModel:
            confidence_head = head

            def compute_confidence(self, *, anchor_tokens, sampled_tokens, x_post_hc):
                del anchor_tokens, sampled_tokens, x_post_hc
                head._last_confidence_raw = raw
                return torch.sigmoid(raw)

        planner = object.__new__(DSparkVerifyPlanner)
        planner.draft_model = LegacyHookDraftModel()
        planner._confidence_head = head
        planner.gamma = gamma
        raw_out = torch.empty_like(raw)

        confidence = planner.compute_confidence_tensor(
            draft_hidden=None,
            anchor_tokens=torch.tensor([10, 20]),
            draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]]),
            confidence_tap=torch.empty(2, gamma, 4),
            raw_out=raw_out,
        )

        torch.testing.assert_close(confidence, torch.sigmoid(raw))
        torch.testing.assert_close(raw_out, raw)

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

    def test_server_args_configure_schedule_knobs(self):
        server_args = _server_args()
        server_args.speculative_dspark_min_verify_len = 2
        server_args.speculative_dspark_max_verify_len = 5
        server_args.speculative_dspark_survival_eps = 0.05
        with patch.dict(os.environ, {"SGLANG_RAGGED_VERIFY_MODE": "static"}):
            planner = DSparkVerifyPlanner(
                draft_model=_draft_model(with_confidence_head=False),
                gamma=7,
                model_runner=types.SimpleNamespace(),
                device=torch.device("cpu"),
                tp_rank=0,
                server_args=server_args,
                verify_num_draft_tokens=8,
            )

        self.assertEqual(planner._schedule_cfg.min_verify_len, 2)
        self.assertEqual(planner._schedule_cfg.max_verify_len, 5)
        self.assertEqual(planner._schedule_cfg.survival_eps, 0.05)

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

    def test_compact_backend_check_can_defer_until_backend_init(self):
        with patch.dict(os.environ, {"SGLANG_RAGGED_VERIFY_MODE": "static"}):
            planner = DSparkVerifyPlanner(
                draft_model=_draft_model(with_confidence_head=False),
                gamma=7,
                model_runner=types.SimpleNamespace(),
                device=torch.device("cpu"),
                tp_rank=0,
                server_args=_server_args(),
                verify_num_draft_tokens=8,
            )

        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._require_compact_backend_support(allow_missing=True)

        with self.assertRaisesRegex(ValueError, "current backend <missing>"):
            planner.validate_attention_backend_support()

        planner.model_runner.attn_backend = types.SimpleNamespace(
            supports_ragged_verify_graph=True
        )
        planner.validate_attention_backend_support()

    def test_compute_budget_sync_preserves_prefix_lens_for_resolved_budget(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = object()
        planner.model_runner = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(
                req_generation=torch.tensor([10, 11, 12], dtype=torch.int64)
            )
        )
        seen = {}

        def budget_from_resolved(
            *, resolved, req_pool_indices_cpu, current_seq_lens_cpu
        ):
            seen["resolved_seq_lens"] = resolved.seq_lens
            seen["current_seq_lens_cpu"] = current_seq_lens_cpu
            seen["req_pool_indices_cpu"] = req_pool_indices_cpu
            return 123

        planner._budget_from_resolved = budget_from_resolved
        prefix_lens = torch.tensor([5, 9], dtype=torch.int64)
        req_pool_indices = torch.tensor([0, 2], dtype=torch.int64)

        budget = planner.compute_budget_sync(
            confidence=torch.ones(2, 7),
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
        )

        self.assertEqual(budget, 123)
        torch.testing.assert_close(seen["resolved_seq_lens"], prefix_lens)
        torch.testing.assert_close(seen["current_seq_lens_cpu"], prefix_lens)
        torch.testing.assert_close(seen["req_pool_indices_cpu"], req_pool_indices)

    def test_tp_verify_lens_broadcast_syncs_graph_num_tokens(self):
        class FakeBudgetPlanner:
            forced_budget_frac = None

            def observe_budget_step(self, *, num_requests, budget):
                del num_requests, budget

        class FakeBroadcastGroup:
            def broadcast(self, tensor, src):
                del src
                if tensor.numel() == 1:
                    tensor.fill_(16)
                else:
                    tensor.copy_(
                        torch.tensor([8, 8], dtype=tensor.dtype, device=tensor.device)
                    )

        planner = object.__new__(DSparkVerifyPlanner)
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._dynamic_graph_tier = True
        planner._align_verify_tokens_to_graph_tier = False
        planner._schedule_cfg = DSparkScheduleConfig(gamma=7, min_verify_len=1)
        planner._budget_planner = FakeBudgetPlanner()
        planner.verify_num_draft_tokens = 8
        planner.server_args = types.SimpleNamespace(tp_size=2)
        planner.model_runner = types.SimpleNamespace(
            decode_cuda_graph_runner=types.SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[4, 8, 16],
                max_bs=2,
            )
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify_planner."
            "verify_lens_broadcast_group",
            return_value=(FakeBroadcastGroup(), 2),
        ), patch(
            "sglang.srt.speculative.dspark_components.dspark_verify_planner."
            "ScheduleVerifyLensTopk.execute",
            return_value=torch.tensor([1, 1], dtype=torch.int32),
        ):
            layout = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([8, 8], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=torch.ones(2, 7),
                budget=2,
            )

        self.assertIsNotNone(layout)
        self.assertEqual(layout.graph_num_tokens, 16)
        torch.testing.assert_close(
            layout.verify_lens, torch.tensor([8, 8], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
