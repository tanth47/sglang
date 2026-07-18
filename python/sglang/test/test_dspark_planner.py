import math
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    compute_verify_token_budget,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
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


class TestDSparkPlanner(unittest.TestCase):
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

    def test_sps_target_accept_length_unreachable_keeps_sps_argmax(self):
        decision = compute_verify_token_budget(
            history_survival_probs=_survival(),
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=10.0),
        )

        self.assertEqual(decision.budget, 6)
        self.assertAlmostEqual(decision.predicted_theta, 6.8)

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
        planner.observe_accept_lens(accept_lens=torch.tensor([3, 3], dtype=torch.int32))

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
        planner.observe_accept_lens(accept_lens=torch.tensor([3, 3], dtype=torch.int32))

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
        planner.observe_accept_lens(accept_lens=torch.tensor([3, 3], dtype=torch.int32))

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
        planner.observe_accept_lens(
            accept_lens=torch.tensor([3, 3], dtype=torch.int32),
            cap_trim_lens=torch.tensor([1, 0], dtype=torch.int32),
        )

        budget = planner.compute_budget(
            confidence=_survival(),
            generation=torch.ones(2, dtype=torch.int64),
            current_generation=torch.ones(2, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(2, dtype=torch.int64),
        )

        self.assertEqual(budget, 6)

    def test_sps_target_accept_length_protects_low_observed_acceptance(self):
        planner = HostConfidenceBudgetPlanner(
            sps_table=_flat_sps_table(),
            cfg=DSparkScheduleConfig(gamma=3, sps_target_accept_length=2.0),
            model_runner=None,
            relay_lag_steps=1024,
        )
        planner.observe_accept_lens(accept_lens=torch.tensor([1, 1], dtype=torch.int32))

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
        planner.observe_accept_lens(accept_lens=torch.tensor([3, 3], dtype=torch.int32))

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
