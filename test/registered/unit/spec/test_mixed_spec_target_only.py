import types
import unittest
from array import array
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import (  # noqa: E402
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.managers.overlap_utils import resolve_forward_inputs  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import (  # noqa: E402
    Scheduler,
    _supports_mixed_spec_verify_requests,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.mem_cache.common import release_kv_cache  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.eagle_worker_common import (  # noqa: E402
    finish_eagle_verify,
)
from sglang.srt.speculative.mixed_spec_info import (  # noqa: E402
    EAGLE_VERIFY_WIDTH,
    MixedSpecBatchInfo,
    MixedSpecMode,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SpecAlgorithm:
    @staticmethod
    def is_none():
        return False


class TestMixedSpecBatchInfo(unittest.TestCase):
    def test_target_only_layout_is_authoritative(self):
        info = MixedSpecBatchInfo.target_only([3, 5], decode_bs=2)

        self.assertIs(info.mode, MixedSpecMode.TARGET_ONLY)
        self.assertEqual(info.query_lens, (3, 5, 1, 1))
        self.assertEqual(info.query_start_loc, (0, 3, 8, 9, 10))
        self.assertEqual(info.batch_size, 4)
        self.assertEqual(info.num_tokens, 10)
        self.assertEqual(list(info.prefill_indices), [0, 1])
        self.assertEqual(list(info.decode_indices), [2, 3])
        self.assertFalse(info.is_decode_index(1))
        self.assertTrue(info.is_decode_index(2))
        self.assertTrue(info.is_decode_index(3))

    def test_target_only_layout_rejects_empty_decode_partition(self):
        with self.assertRaisesRegex(ValueError, "at least one decode request"):
            MixedSpecBatchInfo.target_only([3], decode_bs=0)

    def test_verify_layout_and_output_partitions_are_authoritative(self):
        info = MixedSpecBatchInfo.verify([3, 5], verify_bs=2)

        self.assertIs(info.mode, MixedSpecMode.VERIFY)
        self.assertEqual(info.query_lens, (3, 5, 6, 6))
        self.assertEqual(info.query_start_loc, (0, 3, 8, 14, 20))
        self.assertEqual(info.prefill_num_tokens, 8)
        self.assertEqual(info.verify_num_tokens, 12)
        self.assertEqual(info.target_logit_row_indices, (2, 7, *range(8, 20)))
        self.assertEqual(info.target_output_rows, 14)
        self.assertEqual(
            info.causal_context_lens([0, 0, 10, 20]),
            (
                1,
                2,
                3,
                1,
                2,
                3,
                4,
                5,
                11,
                12,
                13,
                14,
                15,
                16,
                21,
                22,
                23,
                24,
                25,
                26,
            ),
        )

        prefill, verify = info.split_target_outputs(torch.arange(14))
        self.assertTrue(torch.equal(prefill, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(verify, torch.arange(2, 14)))

        prefill_tokens, verify_tokens = info.split_flattened_tokens(torch.arange(20))
        self.assertTrue(torch.equal(prefill_tokens, torch.arange(8)))
        self.assertTrue(torch.equal(verify_tokens, torch.arange(8, 20)))


class TestPrepareMixedSpecTargetOnly(unittest.TestCase):
    def test_selects_reserved_tail_slot_without_precommitting_request_kv(self):
        reqs = [
            types.SimpleNamespace(kv_committed_len=3),
            types.SimpleNamespace(kv_committed_len=5),
        ]
        req_to_token = torch.tensor(
            [
                [100, 101, 102, 103, 104, 105, 106, 107],
                [200, 201, 202, 203, 204, 205, 206, 207],
            ],
            dtype=torch.int64,
        )
        bonus_tokens = torch.tensor([41, 42], dtype=torch.int32)
        batch = ScheduleBatch(
            reqs=reqs,
            device="cpu",
            enable_overlap=False,
            spec_algorithm=_SpecAlgorithm(),
            spec_info=types.SimpleNamespace(bonus_tokens=bonus_tokens),
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
            req_pool_indices=torch.tensor([1, 0], dtype=torch.int64),
            seq_lens=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int64),
            orig_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        )

        with (
            patch(
                "sglang.srt.managers.schedule_batch.get_server_args",
                return_value=types.SimpleNamespace(),
            ),
            patch("sglang.srt.speculative.spec_utils.spec_prepare_for_decode"),
        ):
            running_input_ids = batch.prepare_for_mixed_spec_target_only()

        self.assertIs(running_input_ids, bonus_tokens)
        self.assertTrue(
            torch.equal(batch.out_cache_loc, torch.tensor([203, 105]))
        )
        self.assertTrue(torch.equal(batch.seq_lens, torch.tensor([4, 6])))
        self.assertTrue(torch.equal(batch.seq_lens_cpu, torch.tensor([4, 6])))
        self.assertTrue(
            torch.equal(batch.orig_seq_lens, torch.tensor([4, 6], dtype=torch.int32))
        )
        self.assertEqual([req.kv_committed_len for req in reqs], [3, 5])


class TestPrepareMixedSpecVerify(unittest.TestCase):
    def test_stages_six_reserved_rows_without_committing_request_kv(self):
        reqs = [
            types.SimpleNamespace(kv_committed_len=3),
            types.SimpleNamespace(kv_committed_len=5),
        ]
        req_to_token = torch.tensor(
            [
                [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211],
            ],
            dtype=torch.int64,
        )
        draft_token = torch.arange(12, dtype=torch.int64)
        verify_input = types.SimpleNamespace(
            draft_token=draft_token,
            draft_token_num=EAGLE_VERIFY_WIDTH,
        )
        batch = ScheduleBatch(
            reqs=reqs,
            device="cpu",
            enable_overlap=False,
            spec_algorithm=_SpecAlgorithm(),
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
            req_pool_indices=torch.tensor([1, 0], dtype=torch.int64),
            seq_lens=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int64),
            orig_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        )

        running_input_ids, prefix_lens = batch.prepare_for_mixed_spec_verify(
            verify_input
        )

        self.assertIs(running_input_ids, draft_token)
        self.assertEqual(prefix_lens, [3, 5])
        self.assertTrue(
            torch.equal(
                batch.out_cache_loc,
                torch.tensor(
                    [203, 204, 205, 206, 207, 208, 105, 106, 107, 108, 109, 110]
                ),
            )
        )
        self.assertTrue(torch.equal(batch.seq_lens, torch.tensor([9, 11])))
        self.assertTrue(torch.equal(batch.seq_lens_cpu, torch.tensor([9, 11])))
        self.assertTrue(
            torch.equal(batch.orig_seq_lens, torch.tensor([9, 11], dtype=torch.int32))
        )
        self.assertEqual([req.kv_committed_len for req in reqs], [3, 5])

    def test_composes_exact_heterogeneous_target_forward_accounting(self):
        model_config = types.SimpleNamespace(is_encoder_decoder=False)
        prefill_sampling_info = MagicMock()
        running_sampling_info = MagicMock()
        prefill_batch = ScheduleBatch(
            reqs=[types.SimpleNamespace(), types.SimpleNamespace()],
            device="cpu",
            enable_overlap=False,
            spec_algorithm=_SpecAlgorithm(),
            model_config=model_config,
            sampling_info=prefill_sampling_info,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int64),
            orig_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            out_cache_loc=torch.arange(8, dtype=torch.int64),
            input_ids=torch.arange(8, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[3, 5],
            extend_num_tokens=8,
            extend_logprob_start_lens=[0, 0],
            is_prefill_only=False,
            return_logprob=False,
            has_grammar=False,
            return_hidden_states=False,
        )

        req_to_token = torch.arange(4 * 40, dtype=torch.int64).reshape(4, 40)
        running_reqs = [
            types.SimpleNamespace(kv_committed_len=10),
            types.SimpleNamespace(kv_committed_len=20),
        ]
        running_batch = ScheduleBatch(
            reqs=running_reqs,
            device="cpu",
            enable_overlap=False,
            spec_algorithm=_SpecAlgorithm(),
            model_config=model_config,
            sampling_info=running_sampling_info,
            req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
            req_pool_indices=torch.tensor([2, 3], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([2, 3], dtype=torch.int64),
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int64),
            orig_seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            is_prefill_only=False,
            return_logprob=False,
            has_grammar=False,
            return_hidden_states=False,
        )
        verify_input = types.SimpleNamespace(
            draft_token=torch.arange(100, 112, dtype=torch.int64),
            draft_token_num=EAGLE_VERIFY_WIDTH,
        )

        info = prefill_batch.mix_with_running_verify(running_batch, verify_input)

        self.assertIs(info.mode, MixedSpecMode.VERIFY)
        self.assertEqual(tuple(prefill_batch.extend_lens), (3, 5, 6, 6))
        self.assertEqual(prefill_batch.extend_num_tokens, 20)
        self.assertEqual(prefill_batch.prefix_lens, [0, 0, 10, 20])
        self.assertTrue(
            torch.equal(prefill_batch.seq_lens, torch.tensor([3, 5, 16, 26]))
        )
        self.assertTrue(
            torch.equal(
                prefill_batch.input_ids,
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, *range(100, 112)]),
            )
        )
        self.assertEqual(prefill_batch.out_cache_loc.numel(), 20)
        self.assertIsNone(prefill_batch.spec_info)
        self.assertEqual([req.kv_committed_len for req in running_reqs], [10, 20])
        prefill_sampling_info.merge_batch.assert_called_once_with(
            running_sampling_info
        )


class TestMixedSpecVerifyLogits(unittest.TestCase):
    def test_keeps_prefill_last_rows_and_every_verify_row(self):
        info = MixedSpecBatchInfo.verify([3, 5], verify_bs=2)
        hidden_states = torch.arange(40, dtype=torch.float32).reshape(20, 2)
        before_norm = hidden_states + 100
        aux_hidden_states = [hidden_states + 200, hidden_states + 300]
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.MIXED,
            extend_seq_lens=torch.tensor(info.query_lens, dtype=torch.int32),
            mixed_spec_info=info,
        )

        (
            pruned_states,
            pruned_before_norm,
            aux_pruned_states,
            sample_indices,
            input_logprob_indices,
            token_to_seq_idx,
        ) = LogitsProcessor._get_pruned_states(
            None,
            hidden_states,
            before_norm,
            aux_hidden_states,
            metadata,
        )

        expected = torch.tensor(info.target_logit_row_indices)
        self.assertTrue(torch.equal(pruned_states, hidden_states[expected]))
        self.assertTrue(torch.equal(pruned_before_norm, before_norm[expected]))
        self.assertTrue(
            torch.equal(aux_pruned_states[0], aux_hidden_states[0][expected])
        )
        self.assertTrue(
            torch.equal(aux_pruned_states[1], aux_hidden_states[1][expected])
        )
        self.assertIsNone(sample_indices)
        self.assertIsNone(input_logprob_indices)
        self.assertEqual(token_to_seq_idx, [])


class TestMixedSpecInputResolution(unittest.TestCase):
    def test_direct_bonus_tokens_are_appended_to_prefill_tokens(self):
        batch = ScheduleBatch(
            reqs=[],
            device="cpu",
            enable_overlap=False,
            prefill_input_ids_cpu=torch.tensor([1, 2, 3], dtype=torch.int64),
            mix_running_indices=None,
            mix_running_input_ids=torch.tensor([9, 10], dtype=torch.int32),
        )

        resolve_forward_inputs(batch, MagicMock())

        self.assertTrue(
            torch.equal(batch.input_ids, torch.tensor([1, 2, 3, 9, 10]))
        )
        self.assertEqual(batch.input_ids.dtype, torch.int64)
        self.assertIsNone(batch.prefill_input_ids_cpu)
        self.assertIsNone(batch.mix_running_input_ids)


class TestMixedSpecRequestScope(unittest.TestCase):
    @staticmethod
    def _batch(*top_ks, has_grammar=False):
        return types.SimpleNamespace(
            has_grammar=has_grammar,
            reqs=[
                types.SimpleNamespace(
                    sampling_params=types.SimpleNamespace(top_k=top_k)
                )
                for top_k in top_ks
            ],
        )

    def test_accepts_only_grammar_free_greedy_requests(self):
        self.assertTrue(
            _supports_mixed_spec_verify_requests(
                self._batch(1, 1), self._batch(1)
            )
        )
        self.assertFalse(
            _supports_mixed_spec_verify_requests(
                self._batch(1, has_grammar=True), self._batch(1)
            )
        )
        self.assertFalse(
            _supports_mixed_spec_verify_requests(
                self._batch(1), self._batch(1, 8)
            )
        )


class TestMixedSpecCommit(unittest.TestCase):
    def test_only_running_partition_commits_one_consumed_bonus_token(self):
        prefill_req = types.SimpleNamespace(kv_committed_len=8)
        decode_reqs = [
            types.SimpleNamespace(kv_committed_len=20),
            types.SimpleNamespace(kv_committed_len=30),
        ]
        batch = ScheduleBatch(
            reqs=[prefill_req, *decode_reqs],
            decoding_reqs=decode_reqs,
            mixed_spec_info=MixedSpecBatchInfo.target_only([8], decode_bs=2),
            seq_lens_cpu=torch.tensor([8, 21, 31], dtype=torch.int64),
        )

        for i, req in enumerate(batch.reqs):
            SchedulerBatchResultProcessor._commit_mixed_spec_target_only(
                batch, i, req
            )

        self.assertEqual(prefill_req.kv_committed_len, 8)
        self.assertEqual([req.kv_committed_len for req in decode_reqs], [21, 31])

    def test_commit_rejects_a_stale_sequence_watermark(self):
        decode_req = types.SimpleNamespace(kv_committed_len=20)
        batch = ScheduleBatch(
            reqs=[decode_req],
            decoding_reqs=[decode_req],
            mixed_spec_info=MixedSpecBatchInfo.target_only([], decode_bs=1),
            seq_lens_cpu=torch.tensor([23], dtype=torch.int64),
        )

        with self.assertRaises(AssertionError):
            SchedulerBatchResultProcessor._commit_mixed_spec_target_only(
                batch, 0, decode_req
            )
        self.assertEqual(decode_req.kv_committed_len, 20)


class TestMixedSpecVerifySettlement(unittest.TestCase):
    @staticmethod
    def _req(base_len):
        req = types.SimpleNamespace(
            kv_committed_len=base_len,
            is_retracted=False,
            grammar=None,
            spec_verify_ct=0,
            spec_num_correct_drafts=0,
            spec_num_block_accept_tokens=0,
            spec_num_cap_tokens=0,
        )
        req.finished = lambda: False
        req.update_spec_correct_drafts_histogram = MagicMock()
        req.update_spec_cap_lens_histogram = MagicMock()
        return req

    def test_forced_accept_lengths_publish_only_after_rejoin_then_commit(self):
        model_config = types.SimpleNamespace(is_encoder_decoder=False)
        prefill_req = self._req(8)
        decode_reqs = [self._req(20 + i) for i in range(EAGLE_VERIFY_WIDTH)]

        prefill_draft = MagicMock()
        decode_draft = MagicMock()
        prefill_batch = ScheduleBatch(
            reqs=[prefill_req],
            device="cpu",
            enable_overlap=False,
            spec_algorithm=_SpecAlgorithm(),
            model_config=model_config,
            sampling_info=MagicMock(),
            req_pool_indices=torch.tensor([0]),
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            seq_lens_cpu=torch.tensor([8]),
            orig_seq_lens=torch.tensor([8], dtype=torch.int32),
            input_ids=torch.tensor([1]),
            out_cache_loc=torch.tensor([100]),
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[8],
            prefix_lens=[0],
            extend_num_tokens=8,
            extend_logprob_start_lens=[0],
            return_logprob=False,
            has_grammar=False,
            is_prefill_only=False,
        )
        base_lens = torch.tensor(
            [req.kv_committed_len for req in decode_reqs], dtype=torch.int64
        )
        running_batch = ScheduleBatch(
            reqs=decode_reqs,
            device="cpu",
            enable_overlap=False,
            spec_algorithm=_SpecAlgorithm(),
            model_config=model_config,
            sampling_info=MagicMock(),
            req_pool_indices=torch.arange(1, 1 + EAGLE_VERIFY_WIDTH),
            req_pool_indices_cpu=torch.arange(1, 1 + EAGLE_VERIFY_WIDTH),
            seq_lens=base_lens.clone(),
            seq_lens_cpu=base_lens.clone(),
            orig_seq_lens=base_lens.to(torch.int32),
            input_ids=None,
            out_cache_loc=None,
            forward_mode=ForwardMode.DECODE,
            return_logprob=False,
            has_grammar=False,
            is_prefill_only=False,
        )
        prefill_batch.mixed_spec_running_batch = running_batch

        accept_lens = torch.arange(1, EAGLE_VERIFY_WIDTH + 1, dtype=torch.int32)
        decode_result = GenerationBatchResult(
            next_token_ids=torch.arange(
                EAGLE_VERIFY_WIDTH * EAGLE_VERIFY_WIDTH, dtype=torch.int64
            ),
            accept_lens=accept_lens,
            speculative_num_draft_tokens=EAGLE_VERIFY_WIDTH,
            next_draft_input=decode_draft,
            new_seq_lens=base_lens + accept_lens,
        )
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([9]),
            next_draft_input=prefill_draft,
            new_seq_lens=torch.tensor([8]),
            mixed_spec_decode_result=decode_result,
            mixed_spec_info=MixedSpecBatchInfo.verify(
                [8], verify_bs=EAGLE_VERIFY_WIDTH
            ),
        )

        Scheduler._materialize_mixed_spec_result(None, prefill_batch, result)

        # Rejoin publishes accepted batch-local lengths and next draft state,
        # but request-visible KV is still transactional until result processing.
        self.assertTrue(
            torch.equal(
                prefill_batch.seq_lens,
                torch.cat([torch.tensor([8]), base_lens + accept_lens]),
            )
        )
        self.assertEqual(
            [req.kv_committed_len for req in decode_reqs], base_lens.tolist()
        )
        self.assertEqual(result.mixed_spec_prefill_batch.reqs, [prefill_req])
        self.assertEqual(result.mixed_spec_decode_batch.reqs, decode_reqs)
        prefill_draft.merge_batch.assert_called_once_with(decode_draft)

        fake_processor = types.SimpleNamespace(
            model_worker=types.SimpleNamespace(
                on_verify_complete_cpu=MagicMock()
            ),
            advance_grammar_fsm=lambda _result, _batch: None,
        )
        accepted = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
            fake_processor, decode_result, result.mixed_spec_decode_batch
        )

        self.assertEqual([len(tokens) for tokens in accepted], list(range(1, 7)))
        self.assertEqual(
            [req.kv_committed_len for req in decode_reqs],
            (base_lens + accept_lens).tolist(),
        )
        self.assertEqual([req.spec_verify_ct for req in decode_reqs], [1] * 6)

    def test_shared_verify_finisher_clears_rejected_state_for_lengths_one_to_six(
        self,
    ):
        base_lens = torch.arange(10, 16, dtype=torch.int64)
        accept_lens = torch.arange(1, 7, dtype=torch.int32)
        predict = torch.arange(36, dtype=torch.int64)
        accept_index = torch.arange(36, dtype=torch.int64).reshape(6, 6)
        clear_unaccepted = MagicMock()
        allocator = types.SimpleNamespace(
            get_kvcache=lambda: types.SimpleNamespace(
                clear_unaccepted_c128_draft_states=clear_unaccepted
            )
        )
        batch = ScheduleBatch(
            reqs=[self._req(int(x)) for x in base_lens],
            device="cpu",
            spec_algorithm=_SpecAlgorithm(),
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=base_lens.clone(),
            req_pool_indices=torch.arange(6),
            return_logprob=False,
        )
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.zeros((36, 8)),
            hidden_states=torch.arange(36, dtype=torch.float32).unsqueeze(1),
        )

        def fill_bonus(accept_tokens, lens, bonus, stride, bs):
            rows = accept_tokens.reshape(bs, stride)
            bonus.copy_(rows[torch.arange(bs), lens.to(torch.int64) - 1])

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_common.eagle_sample",
                return_value=(predict, accept_lens, accept_index),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_common.fill_bonus_tokens_func",
                side_effect=fill_bonus,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_common.commit_mamba_states_after_verify"
            ) as commit_mamba,
        ):
            result = finish_eagle_verify(
                batch,
                verify_input=types.SimpleNamespace(),
                logits_output=logits_output,
                target_worker=MagicMock(),
                token_to_kv_pool_allocator=allocator,
                topk=1,
                num_steps=5,
                num_draft_tokens=6,
                device="cpu",
                can_run_cuda_graph=False,
                finalize_tree_path=True,
            )

        self.assertTrue(torch.equal(result.new_seq_lens, base_lens + accept_lens))
        self.assertTrue(torch.equal(result.accept_lens, accept_lens))
        self.assertTrue(
            torch.equal(
                result.next_draft_input.bonus_tokens,
                torch.tensor([0, 7, 14, 21, 28, 35], dtype=torch.int32),
            )
        )
        clear_unaccepted.assert_called_once()
        clear_args = clear_unaccepted.call_args.args
        self.assertTrue(torch.equal(clear_args[1], base_lens))
        self.assertTrue(torch.equal(clear_args[2], accept_lens))
        self.assertEqual(clear_args[3], 6)
        commit_mamba.assert_called_once()
        self.assertEqual(
            [req.kv_committed_len for req in batch.reqs], base_lens.tolist()
        )

    def test_mixed_result_cpu_copy_recurses_into_decode_partition(self):
        event = MagicMock()
        decode_result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=torch.zeros((6, 8)),
                hidden_states=torch.arange(6, dtype=torch.float32).unsqueeze(1),
            ),
            next_token_ids=torch.arange(6),
            accept_lens=torch.tensor([3], dtype=torch.int32),
        )
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=torch.zeros((1, 8)),
                hidden_states=torch.ones((1, 1)),
            ),
            next_token_ids=torch.tensor([7]),
            mixed_spec_decode_result=decode_result,
            copy_done=event,
        )

        result.copy_to_cpu(return_logprob=False, return_hidden_states=True)

        self.assertIs(decode_result.copy_done, event)
        self.assertTrue(result.next_token_ids.is_cpu)
        self.assertTrue(decode_result.next_token_ids.is_cpu)
        self.assertTrue(decode_result.accept_lens.is_cpu)
        self.assertEqual(event.record.call_count, 2)


class TestMixedSpecVerifyLifecycleBoundaries(unittest.TestCase):
    @staticmethod
    def _real_req(
        *,
        rid="mixed-boundary",
        prompt_len=3,
        output_ids=(),
        max_new_tokens=128,
        eos_token_ids=frozenset(),
    ):
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0,
        )
        sampling_params.normalize(None)
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", range(1, prompt_len + 1)),
            sampling_params=sampling_params,
            eos_token_ids=set(eos_token_ids),
            vocab_size=1024,
        )
        req.output_ids = array("q", output_ids)
        req.kv_committed_len = prompt_len + len(output_ids)
        return req

    @staticmethod
    def _resolve(req, tokens):
        result = GenerationBatchResult(
            next_token_ids=torch.tensor(tokens, dtype=torch.int64),
            accept_lens=torch.tensor([len(tokens)], dtype=torch.int32),
            speculative_num_draft_tokens=EAGLE_VERIFY_WIDTH,
        )
        processor = types.SimpleNamespace(
            model_worker=types.SimpleNamespace(
                on_verify_complete_cpu=MagicMock()
            ),
            advance_grammar_fsm=lambda _result, _batch: None,
        )
        batch = types.SimpleNamespace(reqs=[req])
        accepted = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
            processor, result, batch
        )
        return accepted[0], result

    @staticmethod
    def _release_and_capture(req, *, allocated_len):
        req.req_pool_idx = 0
        req.kv = ReqKvInfo(
            kv_allocated_len=allocated_len,
            swa_evicted_seqlen=0,
        )
        req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(128, dtype=torch.int64).reshape(1, 128),
            free=MagicMock(),
        )
        allocator = types.SimpleNamespace(free=MagicMock())
        tree_cache = types.SimpleNamespace(
            cache_finished_req=MagicMock(),
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )
        server_args = types.SimpleNamespace(
            page_size=1,
            speculative_algorithm="EAGLE",
            strip_thinking_cache=False,
        )
        with (
            patch(
                "sglang.srt.managers.schedule_batch.get_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.srt.mem_cache.common.get_server_args",
                return_value=server_args,
            ),
        ):
            release_kv_cache(req, tree_cache)
        return tree_cache, allocator, req_to_token_pool

    def test_eos_at_every_verify_position_limits_radix_visible_watermark(self):
        for eos_position in range(EAGLE_VERIFY_WIDTH):
            with self.subTest(eos_position=eos_position):
                req = self._real_req(output_ids=[40, 41], eos_token_ids={99})
                base_len = req.kv_committed_len
                tokens = [50, 51, 52, 53, 54, 55]
                tokens[eos_position] = 99

                accepted, _ = self._resolve(req, tokens)
                req.output_ids.extend(accepted)
                req.update_finish_state(len(accepted))

                self.assertEqual(accepted, tokens)
                self.assertEqual(req.kv_committed_len, base_len + len(tokens))
                self.assertEqual(req.finished_len, 2 + eos_position + 1)
                server_args = types.SimpleNamespace(strip_thinking_cache=False)
                with patch(
                    "sglang.srt.managers.schedule_batch.get_server_args",
                    return_value=server_args,
                ):
                    self.assertEqual(
                        req.effective_kv_committed_len(),
                        base_len + eos_position + 1,
                    )

    def test_max_new_tokens_and_context_clip_trim_speculative_tail(self):
        max_req = self._real_req(output_ids=[40, 41], max_new_tokens=3)
        base_len = max_req.kv_committed_len
        accepted, _ = self._resolve(max_req, [50, 51, 52, 53, 54, 55])
        max_req.output_ids.extend(accepted)
        max_req.update_finish_state(len(accepted))

        context_req = self._real_req(prompt_len=7, max_new_tokens=100)
        fake_scheduler = types.SimpleNamespace(
            max_new_tokens_limit=None,
            max_req_len=10,
            max_total_num_tokens=128,
            page_size=1,
        )
        Scheduler.init_req_max_new_tokens(fake_scheduler, context_req)
        context_base = context_req.kv_committed_len
        context_accepted, _ = self._resolve(
            context_req, [60, 61, 62, 63, 64, 65]
        )
        context_req.output_ids.extend(context_accepted)
        context_req.update_finish_state(len(context_accepted))

        server_args = types.SimpleNamespace(strip_thinking_cache=False)
        with patch(
            "sglang.srt.managers.schedule_batch.get_server_args",
            return_value=server_args,
        ):
            self.assertEqual(max_req.finished_len, 3)
            self.assertEqual(max_req.effective_kv_committed_len(), base_len + 1)
            self.assertEqual(context_req.sampling_params.max_new_tokens, 2)
            self.assertEqual(context_req.finished_len, 2)
            self.assertEqual(
                context_req.effective_kv_committed_len(), context_base + 2
            )

    def test_abort_drops_inflight_acceptance_and_releases_full_reserve(self):
        req = self._real_req(output_ids=[40, 41])
        base_len = req.kv_committed_len
        req.to_finish = FINISH_ABORT()

        accepted, _ = self._resolve(req, [50, 51, 52, 53, 54, 55])
        req.output_ids.extend(accepted)
        req.update_finish_state(len(accepted))

        self.assertEqual(accepted, [])
        self.assertEqual(req.kv_committed_len, base_len)
        self.assertEqual(list(req.output_ids), [40, 41])
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)

        allocated_len = base_len + 2 * EAGLE_VERIFY_WIDTH
        tree_cache, allocator, req_to_token_pool = self._release_and_capture(
            req, allocated_len=allocated_len
        )
        self.assertEqual(
            tree_cache.cache_finished_req.call_args.kwargs["kv_len_to_handle"],
            base_len,
        )
        self.assertTrue(
            torch.equal(
                allocator.free.call_args.args[0],
                torch.arange(base_len, allocated_len),
            )
        )
        req_to_token_pool.free.assert_called_once_with(req)
        self.assertIsNone(req.kv)

    def test_retracted_logical_result_never_republishes_tokens(self):
        req = self._real_req(output_ids=[40, 41])
        base_len = req.kv_committed_len
        req.is_retracted = True
        allocator = types.SimpleNamespace(
            free_group_begin=MagicMock(),
            free_group_end=MagicMock(),
        )
        metrics_reporter = types.SimpleNamespace(
            num_generated_tokens=0,
            forward_ct_decode=0,
            update_spec_metrics=MagicMock(),
            report_decode_stats=MagicMock(),
        )
        output_streamer = types.SimpleNamespace(stream_output=MagicMock())
        processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=False,
            enable_overlap_mlx=False,
            server_args=types.SimpleNamespace(enable_metrics=False),
            model_config=types.SimpleNamespace(think_end_id=None),
            token_to_kv_pool_allocator=allocator,
            tree_cache=None,
            hisparse_coordinator=None,
            req_to_token_pool=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=metrics_reporter,
            draft_worker=None,
            model_worker=types.SimpleNamespace(
                on_verify_complete_cpu=MagicMock()
            ),
            logprob_result_processor=None,
            output_streamer=output_streamer,
            abort_request=lambda *_args, **_kwargs: None,
        )
        batch = ScheduleBatch(
            reqs=[req],
            spec_algorithm=_SpecAlgorithm(),
            forward_mode=ForwardMode.DECODE,
            return_logprob=False,
            return_hidden_states=False,
            has_grammar=False,
        )
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=torch.zeros((EAGLE_VERIFY_WIDTH, 8))
            ),
            next_token_ids=torch.tensor([50, 51, 52, 53, 54, 55]),
            accept_lens=torch.tensor([EAGLE_VERIFY_WIDTH], dtype=torch.int32),
            speculative_num_draft_tokens=EAGLE_VERIFY_WIDTH,
        )

        processor.process_batch_result_decode(batch, result)

        self.assertEqual(req.kv_committed_len, base_len)
        self.assertEqual(list(req.output_ids), [40, 41])
        allocator.free_group_begin.assert_called_once()
        allocator.free_group_end.assert_called_once()
        output_streamer.stream_output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
