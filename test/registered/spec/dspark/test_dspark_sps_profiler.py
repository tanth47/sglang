import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sglang.benchmark.dspark_sps_profiler as profiler
from sglang.benchmark.dspark_sps_profiler import (
    SPS_RECORD_SOURCE,
    LoadInfo,
    ServerContext,
    SpsRow,
    build_request_count_sweep,
    build_table_from_summaries,
    count_aligned_steps,
    postprocess_round,
    resolve_cuda_graph_max_bs,
    round_summary_dict,
    run_one_round,
    validate_sweep_against_server,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_load_info() -> LoadInfo:
    return LoadInfo(
        num_requests=4, max_new_tokens=1200, wall_seconds=1.0, reached_target=True
    )


def make_rows(
    *,
    num_rows: int = 30,
    num_running_reqs: int = 4,
    num_verify_tokens: int = 32,
    step_time: float = 0.01,
    first_forward_ct: int = 0,
) -> list[SpsRow]:
    return [
        SpsRow(
            forward_ct=first_forward_ct + index,
            num_running_reqs=num_running_reqs,
            num_verify_tokens=num_verify_tokens,
            step_time=step_time,
        )
        for index in range(num_rows)
    ]


def make_context(**overrides) -> ServerContext:
    values = dict(
        base_url="http://localhost:30000",
        tokenizer_path="dummy",
        tp_size=4,
        dp_size=1,
        verify_num_draft_tokens=8,
        simulate_acc_len=1.0,
        cuda_graph_max_bs=128,
        skip_max_running_requests_threshold=float("inf"),
        skip_token_capacity_threshold=float("inf"),
        record_source=SPS_RECORD_SOURCE,
    )
    values.update(overrides)
    return ServerContext(**values)


def build_table_from_rounds(rounds, max_batch_tokens):
    summaries = [
        round_summary_dict(outcome=outcome, repeat=repeat)
        for repeat, outcome in enumerate(rounds)
    ]
    return build_table_from_summaries(
        summaries=summaries,
        max_batch_tokens=max_batch_tokens,
        offdiag=False,
    )


class TestPostprocessRound(CustomTestCase):
    def test_single_rank_round_builds_probe_from_median_step_time(self):
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01)],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)
        self.assertEqual(outcome.match_fraction, 1.0)

    def test_round_warmup_steps_are_dropped_from_timing(self):
        slow_head = make_rows(num_rows=8, step_time=0.5, first_forward_ct=0)
        steady_tail = make_rows(num_rows=20, step_time=0.01, first_forward_ct=8)
        outcome = postprocess_round(
            rank_rows=[slow_head + steady_tail],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)

    def test_off_target_batch_rows_are_filtered_out(self):
        ramp = make_rows(num_rows=10, num_running_reqs=2, num_verify_tokens=16)
        steady = make_rows(num_rows=30, first_forward_ct=10, step_time=0.02)
        outcome = postprocess_round(
            rank_rows=[ramp + steady],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertAlmostEqual(outcome.match_fraction, 1.0)

    def test_offdiag_warmup_verify_token_drift_is_ignored(self):
        warmup = make_rows(
            num_rows=8,
            num_running_reqs=1,
            num_verify_tokens=5,
            step_time=0.5,
            first_forward_ct=0,
        )
        steady = make_rows(
            num_rows=20,
            num_running_reqs=1,
            num_verify_tokens=6,
            step_time=0.01,
            first_forward_ct=8,
        )
        outcome = postprocess_round(
            rank_rows=[warmup + steady],
            batch_size_per_rank=1,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
            frac=0.75,
        )
        self.assertEqual(outcome.batch_tokens, 6)
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)

    def test_offdiag_underfilled_steps_are_filtered_from_timing(self):
        warmup = make_rows(
            num_rows=8,
            num_running_reqs=1,
            num_verify_tokens=5,
            step_time=0.5,
            first_forward_ct=0,
        )
        underfilled = make_rows(
            num_rows=10,
            num_running_reqs=1,
            num_verify_tokens=5,
            step_time=0.5,
            first_forward_ct=8,
        )
        steady = make_rows(
            num_rows=16,
            num_running_reqs=1,
            num_verify_tokens=6,
            step_time=0.01,
            first_forward_ct=18,
        )
        outcome = postprocess_round(
            rank_rows=[warmup + underfilled + steady],
            batch_size_per_rank=1,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
            frac=0.75,
        )
        self.assertEqual(outcome.batch_tokens, 6)
        self.assertEqual(outcome.num_steady_steps, 16)
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)
        self.assertAlmostEqual(outcome.budget_match_fraction, 16 / 26)

    def test_offdiag_steady_verify_token_drift_raises(self):
        warmup = make_rows(
            num_rows=8,
            num_running_reqs=1,
            num_verify_tokens=5,
            first_forward_ct=0,
        )
        steady_a = make_rows(
            num_rows=10,
            num_running_reqs=1,
            num_verify_tokens=6,
            first_forward_ct=8,
        )
        steady_b = make_rows(
            num_rows=10,
            num_running_reqs=1,
            num_verify_tokens=7,
            first_forward_ct=18,
        )
        with self.assertRaisesRegex(RuntimeError, "steady steps"):
            postprocess_round(
                rank_rows=[warmup + steady_a + steady_b],
                batch_size_per_rank=1,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
                frac=0.75,
            )

    def test_mid_round_instability_raises(self):
        head = make_rows(num_rows=15)
        gap = make_rows(
            num_rows=40, num_running_reqs=3, num_verify_tokens=24, first_forward_ct=15
        )
        tail = make_rows(num_rows=15, first_forward_ct=55)
        with self.assertRaisesRegex(RuntimeError, "unstable mid-round"):
            postprocess_round(
                rank_rows=[head + gap + tail],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_round_that_never_stabilizes_raises(self):
        rows = make_rows(num_rows=50, num_running_reqs=3, num_verify_tokens=24)
        rows += make_rows(num_rows=2, first_forward_ct=50)
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[rows],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )


class TestPostprocessRoundCrossRank(CustomTestCase):
    def test_two_uniform_ranks_average_their_step_times(self):
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01), make_rows(step_time=0.03)],
            batch_size_per_rank=4,
            dp_size=2,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertEqual(outcome.batch_size_per_rank, 4)
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertEqual(len(outcome.per_rank_median_step_time), 2)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[0], 0.01)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[1], 0.03)

    def test_rank_with_no_new_records_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no new decode-step records"):
            postprocess_round(
                rank_rows=[make_rows(), []],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_disjoint_forward_ct_ranges_raise(self):
        with self.assertRaisesRegex(RuntimeError, "no common forward_ct"):
            postprocess_round(
                rank_rows=[
                    make_rows(first_forward_ct=0),
                    make_rows(first_forward_ct=1000),
                ],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_cross_rank_verify_token_mismatch_raises(self):
        with self.assertRaisesRegex(RuntimeError, "num_verify_tokens"):
            postprocess_round(
                rank_rows=[make_rows(), make_rows(num_verify_tokens=40)],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_rank_count_mismatch_raises(self):
        with self.assertRaisesRegex(RuntimeError, "DP ranks"):
            postprocess_round(
                rank_rows=[make_rows()],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )


class TestTableAssembly(CustomTestCase):
    def test_repeats_take_the_median_per_batch_tokens(self):
        rounds = [
            postprocess_round(
                rank_rows=[make_rows(step_time=step_time)],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )
            for step_time in (0.01, 0.02, 0.04)
        ]
        table = build_table_from_rounds(rounds=rounds, max_batch_tokens=None)
        self.assertEqual(table.sample_batch_tokens, [32])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 50.0)

    def test_probes_are_sorted_by_batch_tokens(self):
        rounds = [
            postprocess_round(
                rank_rows=[
                    make_rows(
                        num_running_reqs=batch_size,
                        num_verify_tokens=batch_size * 8,
                    )
                ],
                batch_size_per_rank=batch_size,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )
            for batch_size in (8, 2, 4)
        ]
        table = build_table_from_rounds(rounds=rounds, max_batch_tokens=None)
        self.assertEqual(table.sample_batch_tokens, [16, 32, 64])

    def test_offdiag_small_m_sweep_keeps_m_axis(self):
        summaries = [
            {
                "batch_size": 1,
                "batch_size_per_rank": 1,
                "frac": 0.25,
                "batch_tokens": 2,
                "steps_per_sec": 10.0,
            },
            {
                "batch_size": 1,
                "batch_size_per_rank": 1,
                "frac": 1.0,
                "batch_tokens": 8,
                "steps_per_sec": 9.0,
            },
            {
                "batch_size": 4,
                "batch_size_per_rank": 4,
                "frac": 0.25,
                "batch_tokens": 11,
                "steps_per_sec": 8.0,
            },
            {
                "batch_size": 4,
                "batch_size_per_rank": 4,
                "frac": 1.0,
                "batch_tokens": 32,
                "steps_per_sec": 7.0,
            },
        ]
        table = build_table_from_summaries(
            summaries=summaries,
            max_batch_tokens=None,
            offdiag=True,
        )

        self.assertGreater(len(table.m_probes), 1)
        self.assertIn(32, table.m_probes)


class TestSweepHelpers(CustomTestCase):
    def test_request_count_sweep_tapers_and_hits_the_max(self):
        sweep = build_request_count_sweep(100)
        self.assertEqual(sweep[:4], [1, 2, 4, 8])
        self.assertEqual(sweep[-1], 100)
        self.assertIn(64, sweep)

    def test_request_count_sweep_rejects_non_positive_max(self):
        with self.assertRaises(ValueError):
            build_request_count_sweep(0)

    def test_sweep_beyond_captured_cuda_graphs_raises(self):
        with self.assertRaisesRegex(ValueError, "cuda graphs"):
            validate_sweep_against_server(
                context=make_context(cuda_graph_max_bs=64),
                batch_sizes=[8, 128],
            )

    def test_sweep_within_captured_cuda_graphs_passes(self):
        validate_sweep_against_server(
            context=make_context(cuda_graph_max_bs=64, dp_size=2),
            batch_sizes=[8, 64],
        )

    def test_resolve_cuda_graph_max_bs_prefers_captured_list(self):
        internal_state = {
            "cuda_graph_config": {"decode": {"bs": [1, 2, 160], "max_bs": 128}}
        }
        self.assertEqual(resolve_cuda_graph_max_bs(internal_state=internal_state), 160)

    def test_resolve_cuda_graph_max_bs_handles_missing_config(self):
        self.assertIsNone(resolve_cuda_graph_max_bs(internal_state={}))


class TestRunOneRoundForcedBudget(CustomTestCase):
    def _settings(self):
        return profiler.RoundSettings(
            input_len=4,
            temperature=0.0,
            min_steady_steps=1,
            min_steady_seconds=0.0,
            round_timeout_seconds=1.0,
            ramp_token_slack=0,
        )

    def test_forced_budget_is_reset_after_success(self):
        calls = []
        load_thread = Mock()
        load_thread.is_alive.return_value = False
        outcome = object()

        with (
            patch.object(
                profiler,
                "set_forced_budget_frac",
                side_effect=lambda *, base_url, frac: calls.append(frac),
            ),
            patch.object(profiler, "flush_cache"),
            patch.object(profiler, "fetch_rank_rows", return_value=[[]]),
            patch.object(profiler, "start_load", return_value=load_thread),
            patch.object(profiler, "wait_for_aligned_steps", return_value=True),
            patch.object(profiler, "abort_all_requests"),
            patch.object(profiler, "postprocess_round", return_value=outcome),
        ):
            result = run_one_round(
                context=make_context(),
                vocab_size=128,
                batch_size_per_rank=1,
                settings=self._settings(),
                rng=random.Random(0),
                frac=0.5,
            )

        self.assertIs(result, outcome)
        self.assertEqual(calls, [0.5, None])

    def test_forced_budget_is_reset_after_exception(self):
        calls = []
        load_thread = Mock()
        load_thread.is_alive.return_value = False

        with (
            patch.object(
                profiler,
                "set_forced_budget_frac",
                side_effect=lambda *, base_url, frac: calls.append(frac),
            ),
            patch.object(profiler, "flush_cache"),
            patch.object(profiler, "fetch_rank_rows", return_value=[[]]),
            patch.object(profiler, "start_load", return_value=load_thread),
            patch.object(
                profiler,
                "wait_for_aligned_steps",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_one_round(
                    context=make_context(),
                    vocab_size=128,
                    batch_size_per_rank=1,
                    settings=self._settings(),
                    rng=random.Random(0),
                    frac=0.5,
                )

        self.assertEqual(calls, [0.5, None])


class TestCountAlignedSteps(CustomTestCase):
    def test_counts_common_cts_at_target_batch(self):
        rows_a = make_rows(num_rows=10)
        rows_b = make_rows(num_rows=8, first_forward_ct=2)
        self.assertEqual(
            count_aligned_steps(rank_rows=[rows_a, rows_b], batch_size_per_rank=4),
            8,
        )

    def test_zero_when_any_rank_is_empty(self):
        self.assertEqual(
            count_aligned_steps(rank_rows=[make_rows(), []], batch_size_per_rank=4),
            0,
        )

    def test_off_target_steps_are_not_counted(self):
        rows = make_rows(num_rows=10, num_running_reqs=3)
        self.assertEqual(
            count_aligned_steps(rank_rows=[rows], batch_size_per_rank=4), 0
        )


class TestMinSteadySteps(CustomTestCase):
    def test_min_steady_steps_rejects_thin_probes(self):
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[make_rows(num_rows=20)],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=32,
                load_info=make_load_info(),
            )


class TestManifestArtifacts(CustomTestCase):
    def test_write_manifest_records_profile_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table_path = root / "sps.json"
            records_path = root / "sps.records.jsonl"
            rounds_path = root / "sps.rounds.jsonl"
            plot_path = root / "sps.plot.png"
            manifest_path = root / "sps.json.manifest.json"
            table_path.write_text('{"schema":"test"}\n', encoding="utf-8")
            records_path.write_text('{"forward_ct":1}\n', encoding="utf-8")
            rounds_path.write_text('{"batch_tokens":8}\n', encoding="utf-8")
            outcome = postprocess_round(
                rank_rows=[make_rows()],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

            profiler.write_manifest(
                manifest_path=manifest_path,
                table_path=table_path,
                records_path=records_path,
                rounds_path=rounds_path,
                plot_path=plot_path,
                context=make_context(),
                batch_sizes=[4],
                settings=profiler.RoundSettings(
                    input_len=16,
                    temperature=1.0,
                    min_steady_steps=16,
                    min_steady_seconds=1.0,
                    round_timeout_seconds=60.0,
                ),
                repeats=1,
                rounds=[outcome],
                fracs=None,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest["artifacts"]
            self.assertEqual(manifest["records_jsonl"], records_path.name)
            self.assertEqual(manifest["rounds_jsonl"], rounds_path.name)
            for key in ("table", "records_jsonl", "rounds_jsonl"):
                self.assertTrue(artifacts[key]["exists"])
                self.assertGreater(artifacts[key]["size_bytes"], 0)
                self.assertRegex(artifacts[key]["sha256"], r"^[0-9a-f]{64}$")
            self.assertFalse(artifacts["plot"]["exists"])
            self.assertNotIn("sha256", artifacts["plot"])

    def test_refresh_manifest_artifacts_after_fit_adds_table_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table_path = root / "sps.json"
            records_path = root / "sps.records.jsonl"
            rounds_path = root / "sps.rounds.jsonl"
            plot_path = root / "sps.plot.png"
            manifest_path = root / "sps.json.manifest.json"
            records_path.write_text('{"forward_ct":1}\n', encoding="utf-8")
            rounds_path.write_text('{"batch_tokens":8}\n', encoding="utf-8")
            outcome = postprocess_round(
                rank_rows=[make_rows()],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )
            profiler.write_manifest(
                manifest_path=manifest_path,
                table_path=table_path,
                records_path=records_path,
                rounds_path=rounds_path,
                plot_path=plot_path,
                context=make_context(),
                batch_sizes=[4],
                settings=profiler.RoundSettings(
                    input_len=16,
                    temperature=1.0,
                    min_steady_steps=16,
                    min_steady_seconds=1.0,
                    round_timeout_seconds=60.0,
                ),
                repeats=1,
                rounds=[outcome],
                fracs=None,
            )
            before = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(before["artifacts"]["table"]["exists"])

            table_path.write_text('{"schema":"test"}\n', encoding="utf-8")
            self.assertTrue(profiler.refresh_manifest_artifacts(out=str(table_path)))

            after = json.loads(manifest_path.read_text(encoding="utf-8"))
            table = after["artifacts"]["table"]
            self.assertTrue(table["exists"])
            self.assertEqual(table["size_bytes"], table_path.stat().st_size)
            self.assertRegex(table["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
