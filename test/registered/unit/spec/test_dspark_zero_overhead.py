import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBroadcastGroup:
    def __init__(self, payload: torch.Tensor, *, rank_in_group: int = 1):
        self.rank_in_group = rank_in_group
        self.payload = payload
        self.calls = 0

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.calls += 1
        self.assert_source = src
        if self.rank_in_group != src:
            tensor.copy_(self.payload)


class TestDSparkPlannerZeroOverhead(unittest.TestCase):
    def test_tp_non_source_broadcasts_without_local_confidence(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = object()
        planner.verify_num_draft_tokens = 4
        source_lens = torch.tensor([2, 3], dtype=torch.int32)
        group = _FakeBroadcastGroup(source_lens)

        with mock.patch.object(
            ScheduleVerifyLensTopk,
            "execute",
            side_effect=AssertionError("non-source rank must not schedule"),
        ):
            verify_lens = planner._schedule_verify_lens(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                broadcast_group=group,
                broadcast_group_size=2,
            )

        self.assertTrue(torch.equal(verify_lens, source_lens))
        self.assertEqual(group.calls, 1)
        self.assertEqual(group.assert_source, 0)

    def test_tp_source_missing_readiness_broadcasts_full_verify(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = object()
        planner.verify_num_draft_tokens = 4
        group = _FakeBroadcastGroup(
            torch.empty(0, dtype=torch.int32), rank_in_group=0
        )

        with mock.patch.object(
            ScheduleVerifyLensTopk,
            "execute",
            side_effect=AssertionError("missing readiness must use full verify"),
        ):
            verify_lens = planner._schedule_verify_lens(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                broadcast_group=group,
                broadcast_group_size=2,
            )

        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([4, 4], dtype=torch.int32))
        )
        self.assertEqual(group.calls, 1)

    def test_dynamic_compact_tp_layout_stays_device_side(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._align_verify_tokens_to_graph_tier = False
        planner._budget_planner = object()
        planner._dynamic_graph_tier = True
        planner._is_verify_all = False
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._schedule_cfg = DSparkScheduleConfig(gamma=3)
        planner._uniform_layout_cache = {}
        planner.verify_num_draft_tokens = 4
        planner.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[4, 8, 16],
                max_bs=8,
            )
        )
        planner.server_args = SimpleNamespace(tp_size=2)
        source_lens = torch.tensor([1, 2], dtype=torch.int32)
        group = _FakeBroadcastGroup(source_lens)
        layout = object()

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens",
                side_effect=AssertionError("dynamic compact graph materialized lengths"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens_device",
                return_value=layout,
            ) as assemble,
        ):
            result = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
            )

        self.assertIs(result, layout)
        self.assertTrue(
            torch.equal(assemble.call_args.kwargs["verify_lens"], source_lens)
        )
        self.assertEqual(assemble.call_args.kwargs["graph_num_tokens"], 8)

    def test_uniform_cache_is_keyed_by_physical_dp_tier(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._is_verify_all = True
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._schedule_cfg = DSparkScheduleConfig(gamma=3)
        planner._budget_planner = SimpleNamespace(forced_budget_frac=None)
        planner._uniform_layout_cache = {}
        planner.verify_num_draft_tokens = 4
        planner.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[4, 8, 16],
                max_bs=8,
            )
        )
        planner.server_args = SimpleNamespace(tp_size=1)
        base = dict(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
            device=torch.device("cpu"),
            confidence=None,
            budget=None,
            global_num_reqs=4,
        )
        layouts = [object(), object()]
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner."
            "uniform_ragged_layout",
            side_effect=layouts,
        ) as build_layout:
            first = planner.schedule_layout(**base, dp_tier_num_tokens=8)
            self.assertIs(
                planner.schedule_layout(**base, dp_tier_num_tokens=8), first
            )
            second = planner.schedule_layout(**base, dp_tier_num_tokens=16)

        self.assertIs(first, layouts[0])
        self.assertIs(second, layouts[1])
        self.assertEqual(build_layout.call_count, 2)
        self.assertEqual(
            [call.kwargs["tier_num_tokens"] for call in build_layout.call_args_list],
            [8, 16],
        )


class _FakeTargetWorker:
    def __init__(self):
        self.model_runner = SimpleNamespace(attn_backend=SimpleNamespace())

    def forward_batch_generation(self, **kwargs):
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.empty(0)),
            can_run_cuda_graph=False,
        )


class TestDSparkDeviceOnlyLengths(unittest.TestCase):
    def _run_draft(self, *, seq_lens_cpu):
        seen = {}
        gamma = 4
        bs = 2

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["seq_lens_cpu"] = forward_batch.seq_lens_cpu
                seen["seq_lens_sum"] = forward_batch.seq_lens_sum
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=torch.empty((bs * gamma, 16))
                    ),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=None if seq_lens_cpu is None else int(seq_lens_cpu.sum()),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            global_num_tokens=None,
        )
        proposer._run_forward(
            batch=batch,
            draft_input=SimpleNamespace(
                bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
                reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
                reserved_seq_lens_sum=46,
            ),
            verify_window=SimpleNamespace(
                positions_2d=torch.arange(bs * gamma).view(bs, gamma),
                verify_cache_loc_2d=torch.arange(bs * gamma).view(bs, gamma),
            ),
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, 16),
        )
        return seen

    def test_draft_ignores_reserved_host_bound_without_cpu_mirror(self):
        seen = self._run_draft(seq_lens_cpu=None)
        self.assertIsNone(seen["seq_lens_cpu"])
        self.assertIsNone(seen["seq_lens_sum"])

    def test_draft_retains_eager_cpu_mirror_fallback(self):
        seen = self._run_draft(
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32)
        )
        self.assertEqual(seen["seq_lens_cpu"].tolist(), [14, 24])
        self.assertEqual(seen["seq_lens_sum"], 38)

    def test_target_verify_ignores_reserved_bound_without_cpu_mirror(self):
        target_worker = _FakeTargetWorker()
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            gamma=3,
            verify_num_draft_tokens=4,
            model_runner=target_worker.model_runner,
            kv_injector=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=None,
        )
        seen = {}

        def capture_prepare(_verify_input, verify_batch, _target_worker):
            seen["seq_lens_cpu"] = verify_batch.seq_lens_cpu
            seen["seq_lens_sum"] = verify_batch.seq_lens_sum
            return SimpleNamespace(), False

        with mock.patch.object(
            DFlashVerifyInput, "prepare_for_verify", new=capture_prepare
        ):
            executor.run_non_compact(
                batch=batch,
                draft_input=SimpleNamespace(
                    reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
                    reserved_seq_lens_sum=46,
                ),
                verify_ids_2d=torch.ones((2, 4), dtype=torch.int64),
                verify_window=SimpleNamespace(
                    positions_2d=torch.arange(8).view(2, 4),
                    verify_cache_loc=torch.arange(8),
                ),
                sampling_info=None,
            )

        self.assertIsNone(seen["seq_lens_cpu"])
        self.assertIsNone(seen["seq_lens_sum"])
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)


if __name__ == "__main__":
    unittest.main()
