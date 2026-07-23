import unittest

import torch

from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
    accept_greedy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkAcceptSafety(unittest.TestCase):
    def test_zero_verify_len_masks_dummy_row_bonus(self):
        candidates = torch.tensor(
            [
                [0, 1, 2],
                [0, 3, 4],
                [0, 5, 6],
            ],
            dtype=torch.int64,
        )
        target_tokens = torch.tensor(
            [
                [1, 2, 9],
                [3, 4, 17],
                [5, 6, 23],
            ],
            dtype=torch.int64,
        )
        target_logits = torch.full((9, 32), -1000.0)
        target_logits.scatter_(1, target_tokens.reshape(-1, 1), 1000.0)

        correct_len, bonus, cap_trim_lens = accept_greedy(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=3,
            cutoff_verify_lens=torch.tensor([3, 0, 2], dtype=torch.int32),
        )

        self.assertEqual(correct_len.tolist(), [2, -1, 1])
        self.assertEqual(bonus.tolist(), [9, 0, 6])
        self.assertEqual(cap_trim_lens.tolist(), [0, 3, 1])


if __name__ == "__main__":
    unittest.main()
