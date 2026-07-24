import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
)
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
        batch = SimpleNamespace(
            spec_verify_tier_num_tokens=-1, batch_size=lambda: 2
        )

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
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=3
            )

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
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=-1
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 3)
        tier_broadcast.assert_called_once()
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=3)

    def test_conservative_tp_tier_preserves_non_decode_zero(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.verify_num_draft_tokens = 4
        batch = SimpleNamespace(batch_size=lambda: 2)
        self.assertEqual(
            planner._conservative_tp_verify_tier(
                batch=batch, local_tier_num_tokens=0
            ),
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
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=8
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_verify_all_forced_budget_uses_conservative_tp_tier(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner.verify_num_draft_tokens = 4
        planner._is_verify_all = True
        planner._budget_planner = SimpleNamespace(forced_budget_frac=0.5)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(
            spec_verify_tier_num_tokens=-1, batch_size=lambda: 2
        )

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
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=-1
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

