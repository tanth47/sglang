import torch

from sglang.srt.speculative.dspark_components.dspark_info import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_verify_epilogue import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeSwaPool:
    def __init__(self):
        self.full_to_swa_index_mapping = torch.arange(128, dtype=torch.int64) + 1000

    def set_swa_key_buffer_radix_fused_norm_rope(self):
        raise AssertionError("marker method should not be called by folds_commit")


class _FakeDraftModel:
    def __init__(self):
        self.calls = []

    def write_target_hidden_kv(self, **kwargs):
        self.calls.append(kwargs)


class TestDsparkVerifyEpilogueCommitWindow(CustomTestCase):
    def _make_epilogue(self):
        pool = _FakeSwaPool()
        draft_model = _FakeDraftModel()
        epilogue = DsparkVerifyEpilogue(
            max_bs=3,
            verify_num_draft_tokens=4,
            device=torch.device("cpu"),
            commit_ctx=CommitInjectCtx(
                draft_model=draft_model,
                block_pos_offsets=torch.arange(4, dtype=torch.int64),
                resolve_pool=lambda: pool,
                resolve_req_to_token=lambda: torch.full(
                    (3, 32), -999, dtype=torch.int64
                ),
            ),
        )
        epilogue.strided_hidden = torch.arange(24, dtype=torch.float32).view(12, 2)
        verify_window = VerifyWindow(
            positions_2d=torch.tensor(
                [[100, 101, 102, 103], [200, 201, 202, 203], [300, 301, 302, 303]],
                dtype=torch.int64,
            ),
            verify_cache_loc=torch.tensor(
                [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
                dtype=torch.int64,
            ),
            verify_cache_loc_2d=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
                dtype=torch.int64,
            ),
        )
        return epilogue, draft_model, verify_window

    def test_folded_commit_reuses_staged_verify_window_and_masks_by_verify_len(self):
        epilogue, draft_model, verify_window = self._make_epilogue()
        verify_lens = torch.tensor([4, 3, 1], dtype=torch.int64)
        commit_lens = torch.tensor([0, 2, 4], dtype=torch.int32)

        epilogue.begin_step(verify_lens, armed=True, verify_window=verify_window)
        epilogue._commit_inject(
            commit_lens=commit_lens,
            verify_lens=verify_lens,
            seq_lens=torch.tensor([1, 2, 3], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            bs=3,
        )

        self.assertEqual(len(draft_model.calls), 1)
        call = draft_model.calls[0]
        torch.testing.assert_close(call["main_hidden"], epilogue.strided_hidden)
        torch.testing.assert_close(
            call["positions"], verify_window.positions_2d.reshape(-1)
        )
        torch.testing.assert_close(
            call["swa_loc"],
            torch.tensor(
                [-1, -1, -1, -1, 1020, 1021, -1, -1, 1030, -1, -1, -1],
                dtype=torch.int32,
            ),
        )

    def test_folded_commit_gate_zero_disables_injection_locs(self):
        epilogue, draft_model, verify_window = self._make_epilogue()
        verify_lens = torch.tensor([4, 4, 4], dtype=torch.int64)
        commit_lens = torch.tensor([4, 4, 4], dtype=torch.int32)

        epilogue.begin_step(verify_lens, armed=True, verify_window=verify_window)
        epilogue.begin_step(verify_lens, armed=False, verify_window=verify_window)
        epilogue._commit_inject(
            commit_lens=commit_lens,
            verify_lens=verify_lens,
            seq_lens=torch.tensor([1, 2, 3], dtype=torch.int64),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            bs=3,
        )

        call = draft_model.calls[0]
        torch.testing.assert_close(
            call["swa_loc"], torch.full((12,), -1, dtype=torch.int32)
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
