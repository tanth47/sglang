import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.ragged_verify import (
    DSA_TARGET_VERIFY_BATCH_MIXED_REGIONS_REJECT,
    DSA_TARGET_VERIFY_POST_TOPK_ABOVE_CAPTURE_REJECT,
    DSA_TARGET_VERIFY_POST_TOPK_CAPTURE_MISMATCH_REJECT,
    DSA_TARGET_VERIFY_POST_TOPK_GRAPH,
    DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT,
    DSA_TARGET_VERIFY_PRE_TOPK_GRAPH,
    DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT,
    RaggedVerifyLayout,
    build_ragged_target_verify_geometry,
    classify_dsa_target_verify_graph_regime,
    classify_dsa_target_verify_graph_reject_reason,
    expand_target_verify_page_table,
    is_static_full_verify_layout,
    required_padded_verify_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [8, 16, 24, 32, 64]

# The backend capability checks (supports_ragged_verify_graph) live in
# test_ragged_verify_backend_capability.py: importing the backend modules
# pulls GPU-only wheels, which fail to import on the CPU runners.


class TestRaggedTargetVerifyGeometry(unittest.TestCase):
    def test_mixed_verify_lens_geometry(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33])
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12])
        self.assertEqual(geometry.cu_seqlens_k.tolist(), [0, 18, 39, 72])
        self.assertEqual(geometry.max_seq_len_q, 8)

    def test_geometry_dtypes_are_int32(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int64)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_k.dtype, torch.int32)


class TestTargetVerifyPageTableContract(unittest.TestCase):
    def test_nonuniform_verify_lens_expand_request_rows_in_token_order(self):
        page_table = torch.tensor(
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32
        )
        verify_lens = torch.tensor([3, 1, 2], dtype=torch.int32)

        expanded = expand_target_verify_page_table(
            page_table=page_table,
            verify_lens=verify_lens,
            output_num_tokens=6,
        )

        self.assertEqual(
            expanded.tolist(),
            [
                [10, 11, 12],
                [10, 11, 12],
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32],
                [30, 31, 32],
            ],
        )

    def test_different_layouts_keep_the_same_graph_tier_shape(self):
        capture_lens = torch.ones((32,), dtype=torch.int32)
        capture_rows = torch.arange(32, dtype=torch.int32).view(32, 1)
        capture_page_table = expand_target_verify_page_table(
            page_table=capture_rows,
            verify_lens=capture_lens,
            output_num_tokens=32,
        )

        replay = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        ).padded_to_bucket(padded_bs=32)
        replay_rows = torch.arange(32, dtype=torch.int32).view(32, 1)
        replay_page_table = expand_target_verify_page_table(
            page_table=replay_rows,
            verify_lens=replay.verify_lens,
            output_num_tokens=replay.graph_num_tokens,
        )

        self.assertEqual(capture_page_table.shape, replay_page_table.shape)
        self.assertEqual(replay_page_table[:12, 0].tolist(), [0] * 8 + [1] + [2] * 3)
        self.assertEqual(int(replay.verify_lens.sum()), 32)
        self.assertLessEqual(int(replay.verify_lens.max()), 8)

    def test_page_rows_must_match_verify_lens(self):
        with self.assertRaisesRegex(ValueError, "page-table rows"):
            expand_target_verify_page_table(
                page_table=torch.zeros((2, 4), dtype=torch.int32),
                verify_lens=torch.ones((3,), dtype=torch.int32),
                output_num_tokens=3,
            )


class TestDsaTargetVerifyGraphRegime(unittest.TestCase):
    def test_pre_topk_window_uses_default_regime(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[1000, 1200],
            verify_lens_cpu=[8, 1],
            dsa_index_topk=2048,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_PRE_TOPK_GRAPH)

    def test_post_topk_window_stays_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2048, 3000],
            verify_lens_cpu=[1, 8],
            dsa_index_topk=2048,
        )
        self.assertIsNone(regime)

    def test_post_topk_window_rejects_capture_seq_len_mismatch(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2048, 3000],
            verify_lens_cpu=[1, 8],
            dsa_index_topk=2048,
            post_topk_capture_seq_len=4096,
        )
        self.assertIsNone(regime)

    def test_post_topk_window_can_use_exact_capture_contract(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[4096],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_capture_seq_len=4096,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_POST_TOPK_GRAPH)

    def test_full_block_post_topk_window_stays_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2080],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
        )
        self.assertIsNone(regime)

    def test_post_topk_guard_keeps_near_boundary_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2080],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
        )
        self.assertIsNone(regime)

    def test_post_topk_guard_keeps_far_post_window_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2112],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
        )
        self.assertIsNone(regime)

    def test_post_topk_guard_rejects_far_post_window_capture_mismatch(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2112],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
            post_topk_capture_seq_len=4096,
        )
        self.assertIsNone(regime)

    def test_post_topk_guard_allows_exact_capture_contract(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[4096],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
            post_topk_capture_seq_len=4096,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_POST_TOPK_GRAPH)

    def test_post_topk_rejects_above_capture_contract(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[4097],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_capture_seq_len=4096,
        )
        self.assertIsNone(regime)

    def test_boundary_token_stays_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2047],
            verify_lens_cpu=[1],
            dsa_index_topk=2048,
        )
        self.assertIsNone(regime)

    def test_mixed_transition_window_stays_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2044],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
        )
        self.assertIsNone(regime)

    def test_zero_verify_padding_does_not_force_mixed(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2050, 1],
            verify_lens_cpu=[4, 0],
            dsa_index_topk=2048,
        )
        self.assertIsNone(regime)

    def test_reject_reason_is_none_for_graphable_pre_topk(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[1000, 1200],
            verify_lens_cpu=[8, 1],
            dsa_index_topk=2048,
        )
        self.assertIsNone(reason)

    def test_reject_reason_splits_post_topk_without_capture_contract(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[2112, 3000],
            verify_lens_cpu=[1, 8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
        )
        self.assertEqual(reason, DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT)

    def test_reject_reason_splits_post_topk_above_capture_contract(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[4097],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
            post_topk_capture_seq_len=4096,
        )
        self.assertEqual(reason, DSA_TARGET_VERIFY_POST_TOPK_ABOVE_CAPTURE_REJECT)

    def test_reject_reason_splits_post_topk_capture_seq_len_mismatch(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[2112],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
            post_topk_capture_seq_len=4096,
        )
        self.assertEqual(reason, DSA_TARGET_VERIFY_POST_TOPK_CAPTURE_MISMATCH_REJECT)

    def test_reject_reason_keeps_true_transition_separate(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[2044],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
        )
        self.assertEqual(reason, DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT)

    def test_reject_reason_splits_batch_mixed_regions(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[1000, 3000],
            verify_lens_cpu=[8, 8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
        )
        self.assertEqual(reason, DSA_TARGET_VERIFY_BATCH_MIXED_REGIONS_REJECT)

    def test_reject_reason_is_none_for_graphable_post_topk(self):
        reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[4096],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
            post_topk_capture_seq_len=4096,
        )
        self.assertIsNone(reason)

    def test_post_topk_capture_contract_does_not_admit_mixed_transition(self):
        kwargs = dict(
            seq_lens_cpu=[4096, 2044],
            verify_lens_cpu=[3, 4],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
            post_topk_capture_seq_len=4096,
        )
        self.assertIsNone(classify_dsa_target_verify_graph_regime(**kwargs))
        self.assertEqual(
            classify_dsa_target_verify_graph_reject_reason(**kwargs),
            DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT,
        )


class TestDsaTargetVerifyGraphAdmission(unittest.TestCase):
    def test_unified_shape_support_does_not_bypass_topk_regime_guard(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[4], device=_DEVICE, grid=[4]
        )
        cases = (
            (2112, DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT),
            (2046, DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT),
        )

        for seq_len, expected_reason in cases:
            with self.subTest(seq_len=seq_len):
                runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
                runner.ragged_verify_mode = True
                runner.model_runner = SimpleNamespace(is_draft_worker=False)
                runner.attn_backend = SimpleNamespace(
                    use_dsa=True,
                    dsa_index_topk=2048,
                    supports_unified_dsa_target_verify_graph=True,
                    supports_dsa_target_verify_post_topk_graph=False,
                )
                runner.num_tokens_per_req = 8
                runner._logged_graph_reject_keys = set()
                runner._dsa_target_verify_post_topk_graph_enabled_for_bs = mock.Mock(
                    return_value=False
                )
                runner._can_run_ragged_verify_graph = mock.Mock(return_value=True)
                forward_batch = SimpleNamespace(
                    replace_embeds=None,
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    batch_size=1,
                    seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                    seq_lens_cpu=[seq_len],
                    spec_info=SimpleNamespace(ragged_verify_layout=layout),
                )

                with mock.patch(
                    "sglang.srt.model_executor.runner."
                    "decode_cuda_graph_runner.is_hip",
                    return_value=True,
                ):
                    self.assertFalse(runner.can_run_graph(forward_batch))

                self.assertEqual(runner.last_graph_reject_reason, expected_reason)
                runner._can_run_ragged_verify_graph.assert_not_called()

    def test_idle_target_verify_capture_without_layout_falls_back_eager(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
        runner.model_runner = SimpleNamespace(is_draft_worker=False)
        runner.attn_backend = SimpleNamespace(use_dsa=True)
        runner._log_graph_reject = mock.Mock()
        forward_batch = SimpleNamespace(
            replace_embeds=None,
            forward_mode=ForwardMode.IDLE,
            spec_info=None,
        )

        self.assertFalse(runner.can_run_graph(forward_batch))
        runner._log_graph_reject.assert_called_once_with(
            forward_batch, "missing_ragged_layout"
        )


class TestPaddedRaggedVerifyGeometry(unittest.TestCase):
    def test_required_slots_preserve_live_lens_and_bound_dummy_lens(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        self.assertEqual(raw.graph_num_tokens, 32)
        required_slots = required_padded_verify_slots(raw, num_tokens_per_req=8)
        self.assertEqual(required_slots, 6)
        padded = raw.padded_to_bucket(padded_bs=required_slots)
        self.assertEqual(padded.verify_lens[:3].tolist(), [8, 1, 3])
        self.assertLessEqual(max(padded.verify_lens.tolist()), 8)
        self.assertEqual(int(padded.qo_indptr_device[-1]), 32)

    def test_padded_layout_decoupled_slots_spread_slack(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        padded = raw.padded_to_bucket(padded_bs=6)
        self.assertEqual(padded.bs, 6)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 3, 7, 7, 6])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 32)

    def test_padded_layout_budget_tier_below_uniform(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )
        self.assertEqual(raw.graph_num_tokens, 16)
        padded = raw.padded_to_bucket(padded_bs=3)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 7])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 16)

    def test_padded_layout_zero_len_pad_rows(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )
        self.assertEqual(raw.graph_num_tokens, 16)
        padded = raw.padded_to_bucket(padded_bs=8)
        self.assertEqual(padded.verify_lens.tolist(), [8, 8, 0, 0, 0, 0, 0, 0])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 16)


class TestStaticFullVerifyLayout(unittest.TestCase):
    def test_full_width_layout_is_static_even_when_graph_tier_rounds_up(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8, 8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )

        self.assertEqual(layout.graph_num_tokens, 32)
        self.assertTrue(is_static_full_verify_layout(layout, num_tokens_per_req=8))

    def test_single_full_width_layout_is_static(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )

        self.assertTrue(is_static_full_verify_layout(layout, num_tokens_per_req=8))

    def test_non_full_width_layout_stays_ragged(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 7],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )

        self.assertFalse(is_static_full_verify_layout(layout, num_tokens_per_req=8))


class TestCaptureVerifyLens(unittest.TestCase):
    def test_small_tier_one_token_rows(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(num_tokens=8, num_slots=8, num_draft_tokens=8)
        self.assertEqual(lens, [1] * 8)

    def test_large_tier_spreads_within_window(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(
            num_tokens=1024, num_slots=128, num_draft_tokens=8
        )
        self.assertEqual(sum(lens), 1024)
        self.assertEqual(lens, [8] * 128)

    def test_uneven_tier_rows_stay_legal(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(num_tokens=24, num_slots=5, num_draft_tokens=8)
        self.assertEqual(sum(lens), 24)
        self.assertTrue(all(1 <= v <= 8 for v in lens))

    def test_rejects_overpacked_tier(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        with self.assertRaises(ValueError):
            build_capture_verify_lens(num_tokens=64, num_slots=4, num_draft_tokens=8)
        with self.assertRaises(ValueError):
            build_capture_verify_lens(num_tokens=4, num_slots=8, num_draft_tokens=8)


class TestCompactTargetVerifyExecution(unittest.TestCase):
    def test_mixed_region_graph_reject_runs_one_full_batch_target_forward(self):
        reject_reason = classify_dsa_target_verify_graph_reject_reason(
            seq_lens_cpu=[1000, 3000],
            verify_lens_cpu=[8, 8],
            dsa_index_topk=2048,
            post_topk_guard_tokens=64,
        )
        self.assertEqual(reject_reason, DSA_TARGET_VERIFY_BATCH_MIXED_REGIONS_REJECT)

        graph_runner = mock.Mock()
        graph_runner.can_run_graph.return_value = False
        attn_backend = mock.Mock()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                decode_cuda_graph_runner=graph_runner,
                attn_backend=attn_backend,
            ),
            forward_batch_generation=mock.Mock(),
        )
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 8
        executor.model_runner = object()
        executor.target_worker = target_worker
        executor.verify_epilogue = None
        executor._verify_backend_self_adds_seq_lens_cache = True

        logits_output = SimpleNamespace(
            next_token_logits=torch.empty((16, 4)),
            hidden_states=torch.empty((16, 4)),
        )
        target_worker.forward_batch_generation.return_value = SimpleNamespace(
            logits_output=logits_output,
            can_run_cuda_graph=False,
            model_forward_calls=1,
            cuda_graph_reject_reason=reject_reason,
            cuda_graph_reject_details={"graph_regime": "batch_mixed_regions"},
        )
        executor._compact_outputs_to_strided = mock.Mock(
            return_value=(
                torch.empty((16, 4)),
                torch.empty((16, 4)),
            )
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8], device=_DEVICE, grid=_GRID
        )
        ragged_window = SimpleNamespace(
            verify_ids=torch.zeros((16,), dtype=torch.int64),
            positions=torch.arange(16, dtype=torch.int64),
            verify_cache_loc=torch.arange(16, dtype=torch.int64),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([1000, 3000], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([1000, 3000], dtype=torch.int64),
            seq_lens_sum=4000,
            forward_mode=ForwardMode.DECODE,
        )
        full_forward_batch = SimpleNamespace(batch_size=2)

        def init_full_forward_batch(prepared_batch, _model_runner):
            self.assertIs(prepared_batch, batch)
            self.assertEqual(prepared_batch.input_ids.numel(), 16)
            self.assertIs(prepared_batch.spec_info.ragged_verify_layout, layout)
            return full_forward_batch

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_verify."
                "BuildRaggedVerifyWindow.execute",
                return_value=ragged_window,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_verify."
                "apply_logits_adjustments_strided"
            ),
            mock.patch(
                "sglang.srt.speculative.dflash_info.ForwardBatch.init_new",
                side_effect=init_full_forward_batch,
            ),
        ):
            result, _ = executor.run_compact(
                batch=batch,
                layout=layout,
                draft_block_ids=torch.zeros((2, 1), dtype=torch.int64),
                draft_tokens=torch.zeros((2, 7), dtype=torch.int64),
                bs=2,
                device="cpu",
                sampling_info=None,
            )

        self.assertFalse(result.can_run_cuda_graph)
        self.assertEqual(result.target_forward_calls, 1)
        self.assertEqual(result.cuda_graph_reject_reason, reject_reason)
        graph_runner.can_run_graph.assert_called_once_with(full_forward_batch)
        graph_runner.load_batch.assert_not_called()
        attn_backend.init_forward_metadata.assert_called_once_with(full_forward_batch)
        target_worker.forward_batch_generation.assert_called_once_with(
            batch=None,
            forward_batch=full_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )


if __name__ == "__main__":
    unittest.main()
