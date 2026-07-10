import torch

from sglang.srt.speculative.dspark_components.dspark_info import VerifyWindow
from sglang.srt.speculative.dspark_components.kernels import (
    compact_layout as _compact_layout_mod,
)
from sglang.srt.speculative.dspark_components.kernels.build_ragged_verify_window import (
    build_ragged_verify_window_from_strided,
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


if __name__ == "__main__":
    import unittest

    unittest.main()
