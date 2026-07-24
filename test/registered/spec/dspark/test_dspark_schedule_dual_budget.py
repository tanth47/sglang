import unittest
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_components.kernels import dspark_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestScheduleVerifyLensDualBudget(CustomTestCase):
    def test_public_api_returns_physical_then_sps_lens(self):
        confidence = torch.tensor(
            [[0.95, 0.8, 0.4, 0.1], [0.9, 0.7, 0.6, 0.2]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=4, survival_eps=0.25)

        physical, sps = dspark_schedule.ScheduleVerifyLensTopk.execute_with_sps_budget(
            confidence=confidence,
            execution_budget=5,
            sps_budget=2,
            cfg=cfg,
            exact_execution=True,
        )

        expected_physical = dspark_schedule.schedule_verify_lens_topk(
            confidence=confidence, budget=5, cfg=cfg, exact_budget=True
        )
        expected_sps = dspark_schedule.schedule_verify_lens_topk(
            confidence=confidence, budget=2, cfg=cfg
        )
        self.assertTrue(torch.equal(physical, expected_physical))
        self.assertTrue(torch.equal(sps, expected_sps))

    def test_single_budget_execute_does_not_enter_dual_path(self):
        confidence = torch.full((2, 3), 0.8, dtype=torch.float32)
        cfg = DSparkScheduleConfig(gamma=3)

        with mock.patch.object(
            dspark_schedule, "schedule_verify_lens_topk_with_sps_budget"
        ) as dual_schedule:
            actual = dspark_schedule.ScheduleVerifyLensTopk.execute(
                confidence=confidence, budget=2, cfg=cfg
            )

        dual_schedule.assert_not_called()
        expected = dspark_schedule.schedule_verify_lens_topk(
            confidence=confidence, budget=2, cfg=cfg
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_nondefault_min_max_use_union_window_and_one_ranking(self):
        # Exact execution ranks positions [1, 4); legacy SPS ranks [0, 5).
        survival = torch.tensor(
            [
                [0.99, 0.01, 0.01, 0.01, 0.98, 0.0],
                [0.80, 0.70, 0.60, 0.40, 0.90, 0.0],
            ],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(
            gamma=6, min_verify_len=2, max_verify_len=5, survival_eps=0.5
        )

        with mock.patch.object(
            dspark_schedule,
            "_value_independent_descending_order",
            wraps=dspark_schedule._value_independent_descending_order,
        ) as ranking:
            physical, sps = (
                dspark_schedule.schedule_verify_lens_topk_with_sps_budget_from_survival(
                    survival_probs=survival,
                    execution_budget=4,
                    sps_budget=2,
                    cfg=cfg,
                    exact_execution=True,
                )
            )

        self.assertEqual(ranking.call_count, 1)
        self.assertTrue(torch.equal(physical, torch.tensor([3, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(sps, torch.tensor([4, 2], dtype=torch.int32)))
        expected_physical = dspark_schedule.schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=4, cfg=cfg, exact_budget=True
        )
        expected_sps = dspark_schedule.schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=2, cfg=cfg
        )
        self.assertTrue(torch.equal(physical, expected_physical))
        self.assertTrue(torch.equal(sps, expected_sps))

    def test_exact_execution_backfills_while_sps_keeps_legacy_filter(self):
        survival = torch.tensor(
            [[0.90, float("nan"), float("nan")], [0.80, 0.70, 0.60]],
            dtype=torch.float32,
        )
        cfg = DSparkScheduleConfig(gamma=3, survival_eps=0.25)

        physical, sps = (
            dspark_schedule.schedule_verify_lens_topk_with_sps_budget_from_survival(
                survival_probs=survival,
                execution_budget=5,
                sps_budget=6,
                cfg=cfg,
                exact_execution=True,
            )
        )

        self.assertTrue(torch.equal(physical, torch.tensor([3, 4], dtype=torch.int32)))
        self.assertTrue(torch.equal(sps, torch.tensor([2, 4], dtype=torch.int32)))

    def test_seeded_dual_outputs_match_independent_schedules(self):
        generator = torch.Generator().manual_seed(20260724)
        for trial in range(128):
            num_requests = int(torch.randint(1, 6, (), generator=generator).item())
            gamma = int(torch.randint(1, 8, (), generator=generator).item())
            min_verify_len = int(
                torch.randint(0, gamma + 1, (), generator=generator).item()
            )
            max_verify_len = int(
                torch.randint(
                    max(min_verify_len, 1),
                    gamma + 2,
                    (),
                    generator=generator,
                ).item()
            )
            survival = torch.rand(
                num_requests, gamma, dtype=torch.float32, generator=generator
            )
            if trial % 3 == 0:
                survival = (survival * 4).round() / 4
            if trial % 11 == 0:
                survival[trial % num_requests, trial % gamma] = float("nan")
            cfg = DSparkScheduleConfig(
                gamma=gamma,
                min_verify_len=min_verify_len,
                max_verify_len=max_verify_len,
                survival_eps=(0.0, 0.1, 0.5)[trial % 3],
            )
            execution_budget = int(
                torch.randint(
                    0, num_requests * gamma + 3, (), generator=generator
                ).item()
            )
            sps_budget = int(
                torch.randint(
                    0, num_requests * gamma + 3, (), generator=generator
                ).item()
            )
            exact_execution = trial % 2 == 0

            physical, sps = (
                dspark_schedule.schedule_verify_lens_topk_with_sps_budget_from_survival(
                    survival_probs=survival,
                    execution_budget=execution_budget,
                    sps_budget=sps_budget,
                    cfg=cfg,
                    exact_execution=exact_execution,
                )
            )
            expected_physical = dspark_schedule.schedule_verify_lens_topk_from_survival(
                survival_probs=survival,
                budget=execution_budget,
                cfg=cfg,
                exact_budget=exact_execution,
            )
            expected_sps = dspark_schedule.schedule_verify_lens_topk_from_survival(
                survival_probs=survival, budget=sps_budget, cfg=cfg
            )
            with self.subTest(trial=trial):
                self.assertTrue(torch.equal(physical, expected_physical))
                self.assertTrue(torch.equal(sps, expected_sps))


if __name__ == "__main__":
    unittest.main()
