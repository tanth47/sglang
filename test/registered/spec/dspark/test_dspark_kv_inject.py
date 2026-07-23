import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeDraftModel:
    def __init__(self):
        self.calls = []

    def write_target_hidden_kv(self, **kwargs):
        self.calls.append(kwargs)


class FakeFusedPool:
    def __init__(self, full_to_swa_index_mapping):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def set_swa_key_buffer_radix_fused_norm_rope(self, *args, **kwargs):
        raise AssertionError("The fake draft model should receive the pool.")


def make_injector(*, draft_model, pool, req_to_token=None, stride=3):
    if req_to_token is None:
        req_to_token = torch.zeros((1, 8), dtype=torch.int64)
    return TargetHiddenKvInjector(
        draft_model=draft_model,
        draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
        model_runner=SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        ),
        device=torch.device("cpu"),
        verify_num_draft_tokens=stride,
        block_pos_offsets=torch.arange(stride, dtype=torch.int64),
    )


class TestTargetHiddenKvInjector(unittest.TestCase):
    def test_generic_target_hidden_path_forwards_commit_lens(self):
        draft_model = FakeDraftModel()
        pool = object()
        injector = make_injector(draft_model=draft_model, pool=pool)

        target_hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
        cache_loc = torch.tensor([10, 11, 12], dtype=torch.int64)
        cache_loc_2d = cache_loc.view(1, 3)
        positions = torch.tensor([5, 6, 7], dtype=torch.int64)
        commit_lens = torch.tensor([2], dtype=torch.int32)

        path = injector.inject_target_hidden(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )

        self.assertEqual(path, "generic_prefix_valid")
        self.assertEqual(len(draft_model.calls), 1)
        call = draft_model.calls[0]
        self.assertIs(call["pool"], pool)
        torch.testing.assert_close(call["target_hidden"], target_hidden)
        torch.testing.assert_close(call["cache_loc"], cache_loc)
        torch.testing.assert_close(call["cache_loc_2d"], cache_loc_2d)
        torch.testing.assert_close(call["positions"], positions)
        torch.testing.assert_close(call["commit_lens"], commit_lens)

    def test_ragged_fused_path_builds_masked_swa_layout(self):
        draft_model = FakeDraftModel()
        req_to_token = torch.stack(
            [
                torch.arange(0, 10, dtype=torch.int64),
                torch.arange(10, 20, dtype=torch.int64),
            ]
        )
        full_to_swa = torch.arange(128, dtype=torch.int64) + 100
        pool = FakeFusedPool(full_to_swa)
        injector = make_injector(
            draft_model=draft_model,
            pool=pool,
            req_to_token=req_to_token,
            stride=3,
        )
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([1, 0], dtype=torch.int64),
            seq_lens=torch.tensor([2, 4], dtype=torch.int64),
        )
        hidden_strided = torch.arange(12, dtype=torch.float32).view(6, 2)
        commit_lens = torch.tensor([2, 0], dtype=torch.int32)

        expected_positions = torch.tensor([2, 3, 4, 4, 5, 6], dtype=torch.int64)
        expected_swa_loc = torch.tensor(
            [112, 113, -1, -1, -1, -1], dtype=torch.int32
        )
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_kv_inject."
            "BuildCommitInjectLayout.execute",
            return_value=SimpleNamespace(
                swa_loc=expected_swa_loc, positions=expected_positions
            ),
        ) as build_layout:
            path = injector.inject_ragged(
                batch=batch,
                layout=None,
                hidden_strided=hidden_strided,
                commit_lens=commit_lens,
                bs=2,
            )

        self.assertEqual(path, "ragged_swa_fused_layout")
        layout_kwargs = build_layout.call_args.kwargs
        torch.testing.assert_close(
            layout_kwargs["req_pool_indices"], batch.req_pool_indices
        )
        torch.testing.assert_close(layout_kwargs["req_to_token"], req_to_token)
        torch.testing.assert_close(layout_kwargs["prefix_lens"], batch.seq_lens)
        torch.testing.assert_close(layout_kwargs["full_to_swa_mapping"], full_to_swa)
        torch.testing.assert_close(layout_kwargs["commit_lens"], commit_lens)
        self.assertEqual(layout_kwargs["stride"], 3)
        self.assertEqual(len(draft_model.calls), 1)
        call = draft_model.calls[0]
        self.assertIs(call["pool"], pool)
        torch.testing.assert_close(call["main_hidden"], hidden_strided)
        torch.testing.assert_close(call["positions"], expected_positions)
        torch.testing.assert_close(call["swa_loc"], expected_swa_loc)


if __name__ == "__main__":
    unittest.main()
