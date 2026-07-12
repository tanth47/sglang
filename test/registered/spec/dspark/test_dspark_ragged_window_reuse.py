import types

import torch

from sglang.srt.speculative.dspark_components.dspark_info import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.kernels import (
    compact_layout as _compact_layout_mod,
)
from sglang.srt.speculative.dspark_components.kernels.build_ragged_verify_window import (
    build_ragged_verify_window_from_strided,
)
from sglang.srt.speculative.dspark_components.kernels.commit_inject_layout import (
    build_commit_inject_layout_from_window,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_OLD_COMPACT_LAYOUT_IMPL = _compact_layout_mod._KERNEL_IMPL


def setUpModule():
    _compact_layout_mod._KERNEL_IMPL = "torch"


def tearDownModule():
    _compact_layout_mod._KERNEL_IMPL = _OLD_COMPACT_LAYOUT_IMPL


class TestDSparkRaggedWindowReuse(CustomTestCase):
    def test_compacts_existing_strided_verify_window_with_padding(self):
        verify_lens = torch.tensor([1, 3, 2], dtype=torch.int32)
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens,
            graph_num_tokens=8,
            total_verify_tokens=6,
        )
        positions_2d = torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33],
            ],
            dtype=torch.int64,
        )
        cache_2d = torch.tensor(
            [
                [100, 101, 102, 103],
                [200, 201, 202, 203],
                [300, 301, 302, 303],
            ],
            dtype=torch.int64,
        )
        verify_ids_2d = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            dtype=torch.int64,
        )
        verify_window = VerifyWindow(
            positions_2d=positions_2d,
            verify_cache_loc=cache_2d.reshape(-1),
            verify_cache_loc_2d=cache_2d,
        )

        ragged = build_ragged_verify_window_from_strided(
            layout=layout,
            verify_ids_2d=verify_ids_2d,
            verify_window=verify_window,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            ragged.positions,
            torch.tensor([10, 20, 21, 22, 30, 31, 0, 0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            ragged.verify_cache_loc,
            torch.tensor([100, 200, 201, 202, 300, 301, 0, 0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            ragged.verify_ids,
            torch.tensor([1, 5, 6, 7, 9, 10, 0, 0], dtype=torch.int64),
        )

    def test_commit_inject_layout_from_verify_window_masks_uncommitted_slots(self):
        cache_2d = torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33],
            ],
            dtype=torch.int64,
        )
        positions_2d = torch.tensor(
            [
                [100, 101, 102, 103],
                [200, 201, 202, 203],
                [300, 301, 302, 303],
            ],
            dtype=torch.int64,
        )
        full_to_swa = torch.arange(128, dtype=torch.int64) + 1000
        commit_lens = torch.tensor([0, 2, 4], dtype=torch.int32)

        layout = build_commit_inject_layout_from_window(
            cache_loc_2d=cache_2d,
            positions_2d=positions_2d,
            full_to_swa_mapping=full_to_swa,
            commit_lens=commit_lens,
            stride=4,
        )

        torch.testing.assert_close(
            layout.positions,
            torch.tensor(
                [100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303],
                dtype=torch.int64,
            ),
        )
        torch.testing.assert_close(
            layout.swa_loc,
            torch.tensor(
                [-1, -1, -1, -1, 1020, 1021, -1, -1, 1030, 1031, 1032, 1033],
                dtype=torch.int32,
            ),
        )

    def test_ragged_injector_reuses_verify_window_for_non_swa_pool(self):
        class _DraftModel:
            def __init__(self):
                self.calls = []

            def write_target_hidden_kv(self, **kwargs):
                self.calls.append(kwargs)

        draft_model = _DraftModel()
        draft_runner = types.SimpleNamespace(token_to_kv_pool=object())
        model_runner = types.SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.full((2, 16), -999, dtype=torch.int64)
            ),
        )
        injector = TargetHiddenKvInjector(
            draft_model=draft_model,
            draft_model_runner=draft_runner,
            model_runner=model_runner,
            device=torch.device("cpu"),
            verify_num_draft_tokens=4,
            block_pos_offsets=torch.arange(4, dtype=torch.int64),
        )
        verify_window = VerifyWindow(
            positions_2d=torch.tensor(
                [[100, 101, 102, 103], [200, 201, 202, 203]], dtype=torch.int64
            ),
            verify_cache_loc=torch.tensor(
                [10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int64
            ),
            verify_cache_loc_2d=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int64
            ),
        )
        hidden = torch.arange(16, dtype=torch.float32).view(8, 2)

        injector.inject_ragged(
            batch=types.SimpleNamespace(
                seq_lens=torch.tensor([1, 2], dtype=torch.int64),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            ),
            layout=None,
            hidden_strided=hidden,
            commit_lens=torch.tensor([1, 3], dtype=torch.int32),
            bs=2,
            verify_window=verify_window,
        )

        self.assertEqual(len(draft_model.calls), 1)
        call = draft_model.calls[0]
        torch.testing.assert_close(call["target_hidden"], hidden)
        torch.testing.assert_close(
            call["positions"], verify_window.positions_2d.reshape(-1)
        )
        torch.testing.assert_close(call["cache_loc"], verify_window.verify_cache_loc)
        torch.testing.assert_close(
            call["cache_loc_2d"], verify_window.verify_cache_loc_2d
        )
        torch.testing.assert_close(
            call["commit_lens"], torch.tensor([1, 3], dtype=torch.int32)
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
