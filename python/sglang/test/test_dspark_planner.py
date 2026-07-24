import math
import types
import unittest
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
    HostConfidenceBudgetPlanner,
    VerifyBudgetDecision,
    compute_verify_token_budget,
    resolve_host_seq_lens_upper_bound,
    uniform_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
)


def _flat_sps_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=64,
    )


def _survival() -> torch.Tensor:
    return torch.tensor(
        [
            [0.9, 0.8, 0.7],
            [0.9, 0.8, 0.7],
        ],
        dtype=torch.float32,
    )


def _uniform_cache_planner() -> DSparkVerifyPlanner:
    planner = object.__new__(DSparkVerifyPlanner)
    planner._align_verify_tokens_to_graph_tier = False
    planner._budget_planner = types.SimpleNamespace(forced_budget_frac=None)
    planner._dynamic_graph_tier = True
    planner._is_verify_all = True
    planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
    planner._schedule_cfg = DSparkScheduleConfig(gamma=3)
    planner._uniform_layout_cache = {}
    planner.verify_num_draft_tokens = 4
    planner.model_runner = types.SimpleNamespace(
        decode_cuda_graph_runner=types.SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=[4, 8, 16],
            max_bs=8,
        ),
        attn_backend=types.SimpleNamespace(use_dsa=False, dsa_index_topk=None),
    )
    planner.server_args = types.SimpleNamespace(tp_size=1)
    return planner


class _FakeBroadcastGroup:
    def __init__(self, *, rank_in_group: int, source_payloads=()):
        self.rank_in_group = rank_in_group
        self.source_payloads = list(source_payloads)
        self.calls = []
        self.ranks = [0, 1]
        self.cpu_group = object()

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if self.rank_in_group != src:
            tensor.copy_(self.source_payloads[len(self.calls)])
        self.calls.append((tensor.clone(), src))


class TestDSparkPlanner(unittest.TestCase):
    def test_tp_verify_tier_uses_source_cpu_control(self):
        planner = _uniform_cache_planner()
        planner._is_verify_all = False
        planner.server_args.tp_size = 2
        group = _FakeBroadcastGroup(rank_in_group=1)
        batch = types.SimpleNamespace(spec_verify_tier_num_tokens=-1)

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

    def test_verify_all_skips_tp_tier_collective(self):
        planner = _uniform_cache_planner()
        planner.server_args.tp_size = 2
        group = _FakeBroadcastGroup(rank_in_group=1)
        batch = types.SimpleNamespace(spec_verify_tier_num_tokens=-1)

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

    def test_verify_all_uniform_layout_cache_reuses_layout_and_none(self):
        for cached_layout in (object(), None):
            with self.subTest(cached_layout=cached_layout):
                planner = _uniform_cache_planner()
                schedule_kwargs = dict(
                    req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                    prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                    device=torch.device("cpu"),
                    confidence=None,
                    budget=None,
                    global_num_reqs=4,
                    dp_tier_num_tokens=8,
                )
                with (
                    mock.patch(
                        "sglang.srt.speculative.dspark_components.dspark_planner."
                        "uniform_ragged_layout",
                        return_value=cached_layout,
                    ) as build_layout,
                    mock.patch.object(
                        planner,
                        "_schedule_verify_lens",
                        side_effect=AssertionError(
                            "verify-all cache must bypass top-k scheduling"
                        ),
                    ),
                ):
                    first = planner.schedule_layout(**schedule_kwargs)
                    second = planner.schedule_layout(**schedule_kwargs)

                self.assertIs(first, cached_layout)
                self.assertIs(second, cached_layout)
                build_layout.assert_called_once_with(
                    bs=2,
                    device=torch.device("cpu"),
                    verify_num_draft_tokens=4,
                    ragged_verify_mode=RaggedVerifyMode.COMPACT,
                    model_runner=planner.model_runner,
                    tier_num_reqs=4,
                    tier_num_tokens=8,
                    carry_sps_verify_lens=False,
                )

    def test_verify_all_cache_separates_s2_and_telemetry_tiers(self):
        planner = _uniform_cache_planner()
        cached_layouts = [object(), object(), object()]
        base_kwargs = dict(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
            device=torch.device("cpu"),
            confidence=None,
            budget=None,
            global_num_reqs=4,
        )
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner."
            "uniform_ragged_layout",
            side_effect=cached_layouts,
        ) as build_layout:
            first = planner.schedule_layout(
                **base_kwargs,
                dp_tier_num_tokens=8,
            )
            self.assertIs(
                planner.schedule_layout(
                    **base_kwargs,
                    dp_tier_num_tokens=8,
                ),
                first,
            )
            second = planner.schedule_layout(
                **base_kwargs,
                dp_tier_num_tokens=16,
            )
            third = planner.schedule_layout(
                **base_kwargs,
                dp_tier_num_tokens=16,
                collect_sps_verify_lens=True,
            )

        self.assertIs(first, cached_layouts[0])
        self.assertIs(second, cached_layouts[1])
        self.assertIs(third, cached_layouts[2])
        self.assertEqual(build_layout.call_count, 3)
        self.assertEqual(
            [
                (
                    call.kwargs["tier_num_reqs"],
                    call.kwargs["tier_num_tokens"],
                    call.kwargs["carry_sps_verify_lens"],
                )
                for call in build_layout.call_args_list
            ],
            [(4, 8, False), (4, 16, False), (4, 16, True)],
        )

    def test_uniform_layout_uses_token_tier_and_carries_sps_lens(self):
        planner = _uniform_cache_planner()
        cached_layout = object()

        with mock.patch.object(
            RaggedVerifyLayout,
            "from_verify_lens",
            return_value=cached_layout,
        ) as assemble:
            layout = uniform_ragged_layout(
                bs=2,
                device=torch.device("cpu"),
                verify_num_draft_tokens=4,
                ragged_verify_mode=RaggedVerifyMode.COMPACT,
                model_runner=planner.model_runner,
                tier_num_reqs=4,
                tier_num_tokens=8,
                carry_sps_verify_lens=True,
            )

        self.assertIs(layout, cached_layout)
        self.assertEqual(assemble.call_args.kwargs["verify_lens_cpu"], [4, 4])
        self.assertEqual(assemble.call_args.kwargs["graph_num_tokens_floor"], 8)
        self.assertTrue(
            torch.equal(
                assemble.call_args.kwargs["sps_verify_lens"],
                torch.tensor([4, 4], dtype=torch.int32),
            )
        )

    def test_uniform_cache_respects_non_uniform_schedule_policies(self):
        planner = _uniform_cache_planner()
        planner._schedule_cfg = DSparkScheduleConfig(
            gamma=3,
            sps_target_accept_length=2.0,
        )
        self.assertFalse(planner._can_cache_uniform_layout(dp_tier_num_tokens=None))

        planner._schedule_cfg = DSparkScheduleConfig(
            gamma=3,
            sps_target_accept_length=2.0,
            sps_dry_run=True,
        )
        self.assertTrue(planner._can_cache_uniform_layout(dp_tier_num_tokens=None))

        planner._budget_planner.forced_budget_frac = 0.5
        self.assertFalse(planner._can_cache_uniform_layout(dp_tier_num_tokens=None))
        planner._budget_planner.forced_budget_frac = None

        with mock.patch.object(
            planner,
            "_uses_rocm_dsa_graph_safe_policy",
            return_value=True,
        ):
            self.assertFalse(planner._can_cache_uniform_layout(dp_tier_num_tokens=None))

    def test_host_upper_bound_avoids_prefix_device_read(self):
        device_prefix = types.SimpleNamespace(
            shape=(1,),
            device=types.SimpleNamespace(type="cuda"),
        )
        host_upper_bound = torch.tensor([2044], dtype=torch.int32)

        self.assertEqual(
            resolve_host_seq_lens_upper_bound(
                prefix_lens=device_prefix,
                host_seq_lens_upper_bound=host_upper_bound,
            ),
            [2044],
        )
        self.assertIsNone(
            resolve_host_seq_lens_upper_bound(
                prefix_lens=device_prefix,
                host_seq_lens_upper_bound=None,
            )
        )

    def test_graph_safe_cap_uses_conservative_host_upper_bound(self):
        planner = _uniform_cache_planner()
        planner._align_verify_tokens_to_graph_tier = True
        planner._is_verify_all = False
        planner.model_runner.attn_backend = types.SimpleNamespace(
            use_dsa=True,
            dsa_index_topk=2048,
        )
        planner.model_runner.decode_cuda_graph_runner.capture_num_tokens = list(
            range(1, 17)
        )
        physical = torch.tensor([4], dtype=torch.int32)

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner.is_hip",
            return_value=True,
        ):
            adjusted, adjusted_budget = planner._adjust_rocm_dsa_graph_safe_sps_layout(
                prefix_lens=torch.tensor([1900], dtype=torch.int64),
                host_seq_lens_upper_bound=torch.tensor([2044], dtype=torch.int32),
                verify_lens=physical,
                budget=3,
                global_num_reqs=None,
                dp_tier_num_tokens=None,
            )
            boundary_fallback, fallback_budget = (
                planner._adjust_rocm_dsa_graph_safe_sps_layout(
                    prefix_lens=torch.tensor([1900], dtype=torch.int64),
                    host_seq_lens_upper_bound=torch.tensor([2047], dtype=torch.int32),
                    verify_lens=physical,
                    budget=3,
                    global_num_reqs=None,
                    dp_tier_num_tokens=None,
                )
            )

        self.assertTrue(torch.equal(adjusted, torch.tensor([3], dtype=torch.int32)))
        self.assertEqual(adjusted_budget, 2)
        self.assertIs(boundary_fallback, physical)
        self.assertEqual(fallback_budget, 3)

    def test_regime_neutral_graph_never_caps_from_host_reservation(self):
        planner = _uniform_cache_planner()
        planner._align_verify_tokens_to_graph_tier = True
        planner._is_verify_all = False
        planner.model_runner.attn_backend = types.SimpleNamespace(
            use_dsa=True,
            dsa_index_topk=2048,
            supports_dsa_target_verify_post_topk_graph=True,
        )
        planner.model_runner.decode_cuda_graph_runner.capture_num_tokens = list(
            range(1, 17)
        )
        physical = torch.tensor([4], dtype=torch.int32)

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_planner.is_hip",
            return_value=True,
        ):
            adjusted, adjusted_budget = planner._adjust_rocm_dsa_graph_safe_sps_layout(
                prefix_lens=torch.tensor([1900], dtype=torch.int64),
                host_seq_lens_upper_bound=torch.tensor([2044], dtype=torch.int32),
                verify_lens=physical,
                budget=3,
                global_num_reqs=None,
                dp_tier_num_tokens=None,
            )

        self.assertIs(adjusted, physical)
        self.assertEqual(adjusted_budget, 3)

    def test_tp_non_source_enters_broadcast_when_readiness_is_missing(self):
        planner = _uniform_cache_planner()
        planner._is_verify_all = False
        planner._budget_planner = object()
        source_physical = torch.tensor([2, 3], dtype=torch.int32)
        source_logical = torch.tensor([1, 2], dtype=torch.int32)
        group = _FakeBroadcastGroup(
            rank_in_group=1,
            source_payloads=(source_physical, source_logical),
        )

        with (
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute_with_sps_budget",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
        ):
            physical, logical = planner._schedule_verify_lens(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                sps_budget=None,
                collect_sps_verify_lens=True,
                broadcast_group=group,
                broadcast_group_size=2,
            )

        self.assertTrue(torch.equal(physical, source_physical))
        self.assertTrue(torch.equal(logical, source_logical))
        self.assertEqual(len(group.calls), 2)
        self.assertEqual([src for _, src in group.calls], [0, 0])

    def test_tp_source_missing_readiness_broadcasts_full_verify(self):
        planner = _uniform_cache_planner()
        planner._is_verify_all = False
        planner._budget_planner = object()
        group = _FakeBroadcastGroup(rank_in_group=0)

        with mock.patch.object(
            ScheduleVerifyLensTopk,
            "execute",
            side_effect=AssertionError("missing readiness must use full verify"),
        ):
            physical, logical = planner._schedule_verify_lens(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                sps_budget=None,
                collect_sps_verify_lens=True,
                broadcast_group=group,
                broadcast_group_size=2,
            )

        expected = torch.tensor([4, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(physical, expected))
        self.assertTrue(torch.equal(logical, expected))
        self.assertEqual(len(group.calls), 2)

    def test_tp_layout_uses_deterministic_max_physical_tier(self):
        planner = _uniform_cache_planner()
        planner._is_verify_all = False
        planner._budget_planner = object()
        source_physical = torch.tensor([1, 2], dtype=torch.int32)
        group = _FakeBroadcastGroup(
            rank_in_group=1,
            source_payloads=(source_physical,),
        )
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
                tp_tier_num_tokens=3,
            )

        self.assertIs(result, layout)
        self.assertTrue(
            torch.equal(
                assemble.call_args.kwargs["verify_lens"],
                source_physical,
            )
        )
        self.assertEqual(assemble.call_args.kwargs["graph_num_tokens"], 4)

        unavailable_group = _FakeBroadcastGroup(
            rank_in_group=1,
            source_payloads=(source_physical,),
        )
        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(unavailable_group, 2),
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens_device",
                return_value=layout,
            ) as unavailable_assemble,
        ):
            result = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=0,
                tp_tier_num_tokens=-1,
            )

        self.assertIs(result, layout)
        self.assertEqual(
            unavailable_assemble.call_args.kwargs["graph_num_tokens"], 8
        )

    def test_sps_target_accept_length_default_keeps_sps_argmax(self):
        decision = compute_verify_token_budget(
            history_survival_probs=_survival(),
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3),
        )

        self.assertEqual(decision.budget, 6)
        self.assertAlmostEqual(decision.predicted_theta, 6.8)

    def test_sps_target_accept_length_caps_budget(self):
        decision = compute_verify_token_budget(
            history_survival_probs=_survival(),
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
        )

        # tau_star = num_reqs + cumulative survival. The first budget reaching
        # 2.0 expected accepted tokens per request is 3.
        self.assertEqual(decision.budget, 3)
        self.assertAlmostEqual(decision.predicted_theta, 4.6)

    def test_sps_target_accept_length_rejects_unreachable_value(self):
        with self.assertRaisesRegex(ValueError, "sps_target_accept_length"):
            DSparkScheduleConfig(gamma=3, sps_target_accept_length=10.0).validate()

    def test_sps_target_accept_length_updates_additive_prediction(self):
        table = SpsAdditiveCostTable(
            bias_seconds=0.1,
            bs_probes=[1, 2],
            alpha_seconds=[0.01, 0.02],
            m_probes=[1, 8],
            theta_seconds=[0.001, 0.008],
        )

        decision = compute_verify_token_budget(
            history_survival_probs=_survival(),
            sps_table=table,
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
        )

        self.assertEqual(decision.budget, 3)
        self.assertTrue(
            math.isclose(
                decision.predicted_step_seconds,
                table.step_time(num_reqs=2, budget=3),
                rel_tol=1e-6,
            )
        )

    def test_sps_target_accept_length_rejects_negative_value(self):
        with self.assertRaisesRegex(ValueError, "sps_target_accept_length"):
            DSparkScheduleConfig(gamma=3, sps_target_accept_length=-1).validate()

    def test_sps_min_schedule_batch_size_rejects_non_positive_value(self):
        with self.assertRaisesRegex(ValueError, "sps_min_schedule_batch_size"):
            DSparkScheduleConfig(gamma=3, sps_min_schedule_batch_size=0).validate()

    def test_sps_min_schedule_batch_size_uses_full_budget_below_floor(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(
                gamma=3,
                sps_target_accept_length=2.0,
                sps_min_schedule_batch_size=3,
            ),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens_cpu(
            accept_lens_cpu=torch.tensor([3, 3], dtype=torch.int32)
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 6)

    def test_sps_min_schedule_batch_size_allows_trim_at_floor(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(
                gamma=3,
                sps_target_accept_length=2.0,
                sps_min_schedule_batch_size=2,
            ),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens_cpu(
            accept_lens_cpu=torch.tensor([3, 3], dtype=torch.int32)
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 3)

    def test_sps_target_accept_length_cold_start_uses_full_budget(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
            model_runner=None,
            relay_lag_steps=1024,
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 6)

    def test_sps_target_accept_length_allows_trim_after_healthy_acceptance(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens_cpu(
            accept_lens_cpu=torch.tensor([3, 3], dtype=torch.int32)
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 3)

    def test_sps_target_accept_length_protects_cap_trimmed_blocks(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens_cpu(
            accept_lens_cpu=torch.tensor([3, 3], dtype=torch.int32),
            cap_trim_lens_cpu=torch.tensor([1, 0], dtype=torch.int32),
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 6)

    def test_runtime_reset_drops_generation_keyed_state(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner._carry_confidence = torch.ones((1, 2, 3))
        planner._carry_generation = torch.ones((1, 2), dtype=torch.int64)
        planner._carry_pos = 7
        planner._accept_len_ewma = 2.5
        planner._accept_guard_cooldown = 4
        planner.last_decision = VerifyBudgetDecision(budget=3)

        planner.reset_runtime_state()

        self.assertIsNone(planner._carry_confidence)
        self.assertIsNone(planner._carry_generation)
        self.assertEqual(planner._carry_pos, 0)
        self.assertIsNone(planner._accept_len_ewma)
        self.assertEqual(planner._accept_guard_cooldown, 0)
        self.assertIsNone(planner.last_decision)

    def test_sps_target_accept_length_protects_low_observed_acceptance(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens_cpu(
            accept_lens_cpu=torch.tensor([1, 1], dtype=torch.int32)
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 6)

    def test_sps_dry_run_records_planned_budget_but_uses_full_verify(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(
                gamma=3, sps_target_accept_length=2.0, sps_dry_run=True
            ),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens_cpu(
            accept_lens_cpu=torch.tensor([3, 3], dtype=torch.int32)
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )
        decision = planner.take_last_decision()

        self.assertEqual(budget, 6)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.budget, 3)
        self.assertTrue(decision.dry_run)


if __name__ == "__main__":
    unittest.main()
