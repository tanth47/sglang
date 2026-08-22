import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import resolve_forward_inputs  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import (  # noqa: E402
    _supports_mixed_spec_target_only_requests,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.speculative.mixed_spec_info import (  # noqa: E402
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
            _supports_mixed_spec_target_only_requests(
                self._batch(1, 1), self._batch(1)
            )
        )
        self.assertFalse(
            _supports_mixed_spec_target_only_requests(
                self._batch(1, has_grammar=True), self._batch(1)
            )
        )
        self.assertFalse(
            _supports_mixed_spec_target_only_requests(
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


if __name__ == "__main__":
    unittest.main()
