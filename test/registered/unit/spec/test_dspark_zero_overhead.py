import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


class _FakeBroadcastGroup:
    def __init__(self, payload: torch.Tensor, *, rank_in_group: int = 1):
        self.rank_in_group = rank_in_group
        self.payload = payload
        self.calls = 0
        self.ranks = [0, 1]
        self.cpu_group = object()

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.calls += 1
        self.assert_source = src
        if self.rank_in_group != src:
            tensor.copy_(self.payload)


class TestDSparkPlannerZeroOverhead(unittest.TestCase):
    def test_tp_dynamic_tier_is_deterministic_without_cpu_collective(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner.verify_num_draft_tokens = 4
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(spec_verify_tier_num_tokens=-1, batch_size=lambda: 2)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("pure TP must not coordinate on CPU"),
            ),
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(batch=batch, local_tier_num_tokens=3)

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_dp_attention_keeps_cpu_tier_coordination(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(spec_verify_tier_num_tokens=-1)

        def broadcast(tensor, *, src, group):
            self.assertEqual(src, 0)
            self.assertIs(group, group_ref.cpu_group)
            tensor.fill_(3)

        group_ref = group
        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=broadcast,
            ) as tier_broadcast,
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(batch=batch, local_tier_num_tokens=-1)

        self.assertEqual(batch.spec_verify_tier_num_tokens, 3)
        tier_broadcast.assert_called_once()
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=3)

    def test_conservative_tp_tier_preserves_non_decode_zero(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.verify_num_draft_tokens = 4
        batch = SimpleNamespace(batch_size=lambda: 2)
        self.assertEqual(
            planner._conservative_tp_verify_tier(batch=batch, local_tier_num_tokens=0),
            0,
        )

        empty_batch = SimpleNamespace(batch_size=lambda: 0)
        self.assertEqual(
            planner._conservative_tp_verify_tier(
                batch=empty_batch, local_tier_num_tokens=-1
            ),
            0,
        )

    def test_verify_all_skips_tp_tier_collective(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner._is_verify_all = True
        planner._budget_planner = SimpleNamespace(forced_budget_frac=None)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(spec_verify_tier_num_tokens=-1)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("verify-all must not coordinate TP tier"),
            ),
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(batch=batch, local_tier_num_tokens=8)

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_verify_all_forced_budget_uses_conservative_tp_tier(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner.verify_num_draft_tokens = 4
        planner._is_verify_all = True
        planner._budget_planner = SimpleNamespace(forced_budget_frac=0.5)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(spec_verify_tier_num_tokens=-1, batch_size=lambda: 2)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("pure TP must not coordinate on CPU"),
            ),
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(batch=batch, local_tier_num_tokens=-1)

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_dp_layout_uses_only_dp_global_tier_hint(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._align_verify_tokens_to_graph_tier = False
        planner._budget_planner = object()
        planner._dynamic_graph_tier = False
        planner._is_verify_all = False
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._schedule_cfg = DSparkScheduleConfig(gamma=3)
        planner._uniform_layout_cache = {}
        planner.verify_num_draft_tokens = 4
        planner.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[4, 8],
                max_bs=8,
            )
        )
        planner.server_args = SimpleNamespace(tp_size=2)
        source_lens = torch.tensor([1, 2], dtype=torch.int32)
        group = _FakeBroadcastGroup(source_lens)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
        ):
            local_graph_tiers = {
                planner.schedule_layout(
                    req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                    prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                    device=torch.device("cpu"),
                    confidence=None,
                    budget=None,
                    global_num_reqs=2,
                    dp_tier_num_tokens=None,
                    tp_tier_num_tokens=local_tier,
                ).graph_num_tokens
                for local_tier in (3, 5)
            }
            gathered_layout = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                global_num_reqs=2,
                dp_tier_num_tokens=3,
                tp_tier_num_tokens=5,
            )

        self.assertEqual(local_graph_tiers, {8})
        self.assertEqual(gathered_layout.graph_num_tokens, 4)

    def _eager_tp_layout_stays_device_side(
        self, *, mode: RaggedVerifyMode
    ) -> RaggedVerifyLayout:
        planner = object.__new__(DSparkVerifyPlanner)
        planner._align_verify_tokens_to_graph_tier = False
        planner._budget_planner = object()
        planner._dynamic_graph_tier = False
        planner._is_verify_all = False
        planner._ragged_verify_mode = mode
        planner._schedule_cfg = DSparkScheduleConfig(gamma=3)
        planner._uniform_layout_cache = {}
        planner.verify_num_draft_tokens = 4
        planner.model_runner = SimpleNamespace(decode_cuda_graph_runner=None)
        planner.server_args = SimpleNamespace(tp_size=2)
        source_lens = torch.tensor([1, 2], dtype=torch.int32)
        group = _FakeBroadcastGroup(source_lens)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens",
                side_effect=AssertionError("pure TP materialized verify lengths"),
            ),
        ):
            result = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
            )

        self.assertTrue(torch.equal(result.verify_lens, source_lens))
        self.assertIsNone(result.verify_lens_cpu)
        self.assertEqual(result.graph_num_tokens, 8)
        return result

    def test_cap_accept_tp_fallback_stays_device_side(self):
        layout = self._eager_tp_layout_stays_device_side(
            mode=RaggedVerifyMode.CAP_ACCEPT
        )
        self.assertIsNone(layout.total_verify_tokens)

    def test_compact_eager_tp_fallback_supplies_consumer_allocation_total(self):
        layout = self._eager_tp_layout_stays_device_side(mode=RaggedVerifyMode.COMPACT)
        self.assertEqual(int(layout.verify_lens.sum()), 3)
        self.assertEqual(layout.total_verify_tokens, layout.graph_num_tokens)

        kernel = mock.MagicMock()
        kernel.__getitem__.return_value = mock.Mock()
        verify_input = DFlashVerifyInput(
            draft_token=torch.ones(8, dtype=torch.int64),
            positions=torch.arange(8, dtype=torch.int64),
            draft_token_num=4,
            ragged_verify_layout=layout,
        )
        with mock.patch(
            "sglang.srt.speculative.dflash_info.create_flashinfer_kv_indices_triton",
            kernel,
        ):
            kv_indices, _, _, _ = verify_input.generate_attn_arg_prefill(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                paged_kernel_lens=torch.tensor([10, 20], dtype=torch.int32),
                paged_kernel_lens_sum=30,
                req_to_token=torch.zeros((2, 32), dtype=torch.int32),
            )

        self.assertEqual(kv_indices.numel(), 30 + layout.graph_num_tokens)

    def test_verify_all_to_forced_budget_keeps_confidence_relay_warm(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = SimpleNamespace(forced_budget_frac=None)
        planner._is_verify_all = True
        worker = object.__new__(DSparkWorkerV2)
        worker._verify_planner = planner
        worker._observers = SimpleNamespace(needs_budget_telemetry=False)

        confidence = torch.ones(1)
        self.assertTrue(planner.needs_confidence_publication)
        self.assertTrue(worker._should_publish_confidence(confidence))

        planner._budget_planner.forced_budget_frac = 0.5
        self.assertTrue(planner.needs_confidence_publication)
        self.assertTrue(worker._should_publish_confidence(confidence))

        planner._budget_planner.forced_budget_frac = None
        planner._is_verify_all = False
        self.assertTrue(planner.needs_confidence_publication)

        planner._budget_planner = None
        self.assertFalse(planner.needs_confidence_publication)
