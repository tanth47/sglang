import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
    _encoder_lens_has_payload,
    _multimodal_inputs_have_payload,
    _tensor_has_payload,
)
from sglang.srt.speculative.ragged_verify import (
    DSA_TARGET_VERIFY_GROUPED_PARTIAL_REJECT,
    DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT,
    DSA_TARGET_VERIFY_PRE_TOPK_GRAPH,
    DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT,
    DsaTargetVerifyGraphGroup,
    RaggedVerifyLayout,
)


def _logits_output(**kwargs):
    return SimpleNamespace(**kwargs)


class TestGroupedTargetVerifyPayloadGuards(unittest.TestCase):
    def test_empty_tensor_fields_are_not_payload(self):
        self.assertFalse(_tensor_has_payload(None))
        self.assertFalse(_tensor_has_payload(torch.empty(0, dtype=torch.int64)))
        self.assertTrue(_tensor_has_payload(torch.zeros(1, dtype=torch.int64)))
        self.assertTrue(_tensor_has_payload(object()))

    def test_zero_encoder_lens_are_language_only(self):
        self.assertFalse(
            _encoder_lens_has_payload(torch.zeros(4, dtype=torch.int64), None)
        )
        self.assertFalse(_encoder_lens_has_payload(None, [0, 0, 0]))
        self.assertTrue(
            _encoder_lens_has_payload(torch.tensor([0, 2], dtype=torch.int64), None)
        )
        self.assertTrue(_encoder_lens_has_payload(None, [0, 2]))

    def test_none_multimodal_list_is_language_only(self):
        self.assertFalse(_multimodal_inputs_have_payload(None))
        self.assertFalse(_multimodal_inputs_have_payload([None, None]))
        self.assertTrue(_multimodal_inputs_have_payload([None, object()]))
        self.assertTrue(_multimodal_inputs_have_payload(object()))


class TestGroupedTargetVerifyAdmission(unittest.TestCase):
    def _executor(self):
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 8
        executor.gamma = 7
        executor._simulate_acc_len = 0.0
        executor._simulated_correct_drafts_buf = None
        executor.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(use_dsa=True, dsa_index_topk=2048)
            )
        )
        executor.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(dp_size=1, enable_dp_attention=False),
            decode_cuda_graph_runner=SimpleNamespace(
                capture_num_tokens=[8, 16, 24, 32]
            ),
        )
        return executor

    def _batch(self, seq_lens):
        bs = len(seq_lens)
        return SimpleNamespace(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=torch.empty(0, dtype=torch.int64),
            multimodal_inputs=[None] * bs,
            inner_idle_batch=None,
        )

    def test_window_transition_mixed_batch_can_run_grouped_partial(self):
        executor = self._executor()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8, 8],
            device=torch.device("cpu"),
            grid=[8, 16, 24, 32],
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.is_hip",
            return_value=True,
        ):
            groups = executor._grouped_dsa_target_verify_groups(
                batch=self._batch([2106, 1836, 2531]),
                layout=layout,
                bs=3,
            )

        self.assertIsNotNone(groups)
        self.assertTrue(
            any(
                group.graph_regime == DSA_TARGET_VERIFY_PRE_TOPK_GRAPH
                for group in groups
            )
        )
        self.assertTrue(
            any(
                group.reject_reason == DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT
                for group in groups
            )
        )
        self.assertTrue(
            any(
                group.reject_reason == DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT
                for group in groups
            )
        )

    def test_all_window_transition_batch_stays_ungrouped(self):
        executor = self._executor()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8],
            device=torch.device("cpu"),
            grid=[8, 16, 24, 32],
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.is_hip",
            return_value=True,
        ):
            groups = executor._grouped_dsa_target_verify_groups(
                batch=self._batch([2106]),
                layout=layout,
                bs=1,
            )

        self.assertIsNone(groups)

    def test_all_post_topk_batch_stays_ungrouped_without_capture_contract(self):
        executor = self._executor()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8],
            device=torch.device("cpu"),
            grid=[8, 16, 24, 32],
        )

        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.is_hip",
            return_value=True,
        ):
            groups = executor._grouped_dsa_target_verify_groups(
                batch=self._batch([2531, 2600]),
                layout=layout,
                bs=2,
            )

        self.assertIsNone(groups)

    def test_grouped_partial_scatter_preserves_original_request_order(self):
        executor = self._executor()
        executor.verify_epilogue = None
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8, 8],
            device=torch.device("cpu"),
            grid=[8, 16, 24, 32],
        )
        groups = [
            DsaTargetVerifyGraphGroup(
                indices=(0, 2), graph_regime=DSA_TARGET_VERIFY_PRE_TOPK_GRAPH
            ),
            DsaTargetVerifyGraphGroup(
                indices=(1,), reject_reason=DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT
            ),
        ]
        draft_block_ids = torch.zeros((3, 1), dtype=torch.int64)
        draft_tokens = torch.zeros((3, 8), dtype=torch.int64)

        def make_group_batch(**kwargs):
            return SimpleNamespace(group_indices=tuple(kwargs["indices_cpu"]))

        def run_ragged(*, batch, **_kwargs):
            return SimpleNamespace(
                group_indices=batch.group_indices,
                logits_output=_logits_output(next_token_logits=None),
                can_run_cuda_graph=len(batch.group_indices) > 1,
            )

        def compact_outputs_to_strided(*, target_verify, **_kwargs):
            values = []
            for req_idx in target_verify.group_indices:
                values.extend(req_idx * 100 + slot for slot in range(8))
            logits = torch.tensor(values, dtype=torch.float32).view(-1, 1)
            hidden = logits + 1000
            return logits, hidden

        with (
            mock.patch.object(
                executor, "_grouped_dsa_target_verify_groups", return_value=groups
            ),
            mock.patch.object(
                executor, "_make_group_batch", side_effect=make_group_batch
            ),
            mock.patch.object(executor, "_run_ragged", side_effect=run_ragged),
            mock.patch.object(
                executor,
                "_compact_outputs_to_strided",
                side_effect=compact_outputs_to_strided,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_verify."
                "BuildRaggedVerifyWindow.execute",
                return_value=object(),
            ),
        ):
            result = executor._run_grouped_compact_if_supported(
                batch=SimpleNamespace(),
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=3,
                device="cpu",
                sampling_info=None,
            )

        self.assertIsNotNone(result)
        target_verify, hidden = result
        expected = torch.tensor(
            [req_idx * 100 + slot for req_idx in range(3) for slot in range(8)],
            dtype=torch.float32,
        ).view(-1, 1)
        torch.testing.assert_close(
            target_verify.logits_output.next_token_logits, expected
        )
        torch.testing.assert_close(hidden, expected + 1000)
        self.assertEqual(
            target_verify.cuda_graph_reject_reason,
            DSA_TARGET_VERIFY_GROUPED_PARTIAL_REJECT,
        )
        self.assertEqual(
            target_verify.cuda_graph_reject_details["graph_group_count"], 1
        )
        self.assertEqual(
            target_verify.cuda_graph_reject_details["eager_group_count"], 1
        )

    def test_grouped_partial_can_precompute_accept_in_original_order(self):
        executor = self._executor()
        executor.verify_epilogue = SimpleNamespace(
            begin_step=lambda *_args, **_kwargs: None
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8, 8],
            device=torch.device("cpu"),
            grid=[8, 16, 24, 32],
        )
        groups = [
            DsaTargetVerifyGraphGroup(
                indices=(0, 2), graph_regime=DSA_TARGET_VERIFY_PRE_TOPK_GRAPH
            ),
            DsaTargetVerifyGraphGroup(
                indices=(1,), reject_reason=DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT
            ),
        ]
        draft_block_ids = torch.tensor([[10], [20], [30]], dtype=torch.int64)
        draft_tokens = torch.tensor(
            [
                [11, 12, 13, 14, 15, 16, 17],
                [21, 22, 23, 24, 25, 26, 27],
                [31, 32, 33, 34, 35, 36, 37],
            ],
            dtype=torch.int64,
        )
        verify_ids_2d = torch.cat([draft_block_ids[:, :1], draft_tokens], dim=1)
        expected_correct = torch.tensor([2, 0, 6], dtype=torch.int64)
        expected_bonus = torch.tensor([92, 90, 96], dtype=torch.int64)
        target_predict = torch.zeros((3, 8), dtype=torch.int64)
        for req_idx, correct_len in enumerate(expected_correct.tolist()):
            for slot in range(correct_len):
                target_predict[req_idx, slot] = verify_ids_2d[req_idx, slot + 1]
            target_predict[req_idx, correct_len] = expected_bonus[req_idx]

        vocab = int(target_predict.max().item()) + 1
        full_logits_by_req = torch.full((3, 8, vocab), -1000.0)
        for req_idx in range(3):
            for slot in range(8):
                token_id = target_predict[req_idx, slot]
                full_logits_by_req[req_idx, slot, token_id] = 1.0

        def make_group_batch(**kwargs):
            return SimpleNamespace(group_indices=tuple(kwargs["indices_cpu"]))

        def run_ragged(*, batch, **_kwargs):
            return SimpleNamespace(
                group_indices=batch.group_indices,
                logits_output=_logits_output(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        def compact_outputs_to_strided(*, target_verify, **_kwargs):
            logits = torch.cat(
                [
                    full_logits_by_req[req_idx]
                    for req_idx in target_verify.group_indices
                ],
                dim=0,
            )
            hidden = torch.zeros((logits.shape[0], 1), dtype=torch.float32)
            return logits, hidden

        with (
            mock.patch.object(
                executor, "_grouped_dsa_target_verify_groups", return_value=groups
            ),
            mock.patch.object(
                executor, "_make_group_batch", side_effect=make_group_batch
            ),
            mock.patch.object(executor, "_run_ragged", side_effect=run_ragged),
            mock.patch.object(
                executor,
                "_compact_outputs_to_strided",
                side_effect=compact_outputs_to_strided,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_verify."
                "BuildRaggedVerifyWindow.execute",
                return_value=object(),
            ),
        ):
            target_verify, _hidden = executor._run_grouped_compact_if_supported(
                batch=SimpleNamespace(),
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=3,
                device="cpu",
                sampling_info=None,
                fold_accept=True,
                verify_ids_2d=verify_ids_2d,
                draft_block=SimpleNamespace(greedy_mask=None),
                draft_input=SimpleNamespace(),
                prefix_lens=torch.tensor([100, 200, 300], dtype=torch.int64),
            )

        accept = target_verify.precomputed_accept
        self.assertIsNotNone(accept)
        torch.testing.assert_close(accept.correct_len, expected_correct)
        torch.testing.assert_close(accept.bonus, expected_bonus)
        torch.testing.assert_close(
            accept.commit_lens, torch.tensor([3, 1, 7], dtype=torch.int32)
        )
        torch.testing.assert_close(
            accept.new_seq_lens, torch.tensor([103, 201, 307], dtype=torch.int64)
        )
        expected_out = draft_tokens.clone()
        expected_out = torch.cat(
            [expected_out, torch.zeros((3, 1), dtype=torch.int64)], dim=1
        )
        expected_out[0, 2] = 92
        expected_out[1, 0] = 90
        expected_out[2, 6] = 96
        torch.testing.assert_close(accept.out_tokens, expected_out)
        self.assertFalse(target_verify.can_run_cuda_graph)
        self.assertTrue(target_verify.cuda_graph_reject_details["precomputed_accept"])


if __name__ == "__main__":
    unittest.main()
