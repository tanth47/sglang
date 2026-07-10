import torch

from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from sglang.srt.speculative.dspark_components.dspark_draft import DsparkDraftSampler
from sglang.srt.speculative.dspark_components.kernels.finalize_accept_lens import (
    finalize_accept_lens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingMarkovHead:
    def __init__(self, gamma: int):
        self.gamma = int(gamma)
        self.base_logits = None
        self.first_prev_tokens = None
        self.hidden_states = None

    def sample_block(self, base_logits, first_prev_tokens, hidden_states, sampler):
        del sampler
        self.base_logits = base_logits.detach().clone()
        self.first_prev_tokens = first_prev_tokens.detach().clone()
        self.hidden_states = hidden_states.detach().clone()
        step_offsets = torch.arange(
            1, self.gamma + 1, dtype=first_prev_tokens.dtype
        ).unsqueeze(0)
        draft_tokens = first_prev_tokens.unsqueeze(1) + step_offsets
        corrected_logits = torch.zeros(
            base_logits.shape[0],
            self.gamma,
            base_logits.shape[-1],
            dtype=base_logits.dtype,
        )
        return draft_tokens, corrected_logits


class _RecordingDraftModel:
    def __init__(self, gamma: int, vocab_size: int):
        self.markov_head = _RecordingMarkovHead(gamma)
        self.hidden_for_logits = None
        self.vocab_size = int(vocab_size)

    def compute_base_logits(self, hidden_for_logits):
        self.hidden_for_logits = hidden_for_logits.detach().clone()
        rows = hidden_for_logits.shape[0]
        logits = torch.zeros(rows, self.vocab_size, dtype=hidden_for_logits.dtype)
        return logits, None


class TestDSparkSlotContract(CustomTestCase):
    def test_folded_sampler_uses_slot_zero_as_anchor_and_draft_slots_for_hidden(self):
        gamma = 3
        bs = 2
        hidden_size = 4
        model = _RecordingDraftModel(gamma=gamma, vocab_size=16)
        sampler = DsparkDraftSampler(
            model=model,
            gamma=gamma,
            max_bs=bs,
            device=torch.device("cpu"),
        )
        hidden_3d = torch.arange(
            bs * (gamma + 1) * hidden_size, dtype=torch.float32
        ).view(bs, gamma + 1, hidden_size)
        ids_2d = torch.tensor(
            [
                [100, 11, 12, 13],
                [200, 21, 22, 23],
            ],
            dtype=torch.int64,
        )

        sampler(hidden_3d.reshape(bs * (gamma + 1), hidden_size), ids_2d.reshape(-1))

        expected_draft_hidden = hidden_3d[:, 1:, :].contiguous()
        torch.testing.assert_close(
            model.hidden_for_logits,
            expected_draft_hidden.reshape(bs * gamma, hidden_size),
        )
        torch.testing.assert_close(
            model.markov_head.hidden_states,
            expected_draft_hidden,
        )
        torch.testing.assert_close(
            model.markov_head.first_prev_tokens,
            ids_2d[:, 0],
        )
        expected_tokens = torch.tensor(
            [
                [101, 102, 103],
                [201, 202, 203],
            ],
            dtype=torch.int64,
        )
        torch.testing.assert_close(
            sampler.out[: bs * gamma].view(bs, gamma),
            expected_tokens,
        )

    def test_greedy_accept_compares_draft_slots_and_finalize_commits_bonus(self):
        candidates = torch.tensor(
            [
                [99, 10, 11, 12],
                [88, 20, 21, 22],
                [77, 30, 31, 32],
            ],
            dtype=torch.int64,
        )
        target_predict = torch.tensor(
            [
                [10, 11, 7, 0],
                [5, 20, 21, 22],
                [30, 31, 32, 33],
            ],
            dtype=torch.int64,
        )

        correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        finalized = finalize_accept_lens(
            correct_len=correct_len,
            cap_trim_lens=torch.zeros(3, dtype=torch.int32),
            prefix_lens=torch.tensor([30, 40, 50], dtype=torch.int64),
        )

        torch.testing.assert_close(
            correct_len, torch.tensor([2, 0, 3], dtype=correct_len.dtype)
        )
        torch.testing.assert_close(bonus, torch.tensor([7, 5, 33], dtype=torch.int64))
        torch.testing.assert_close(
            finalized.commit_lens, torch.tensor([3, 1, 4], dtype=torch.int32)
        )
        torch.testing.assert_close(
            finalized.new_seq_lens, torch.tensor([33, 41, 54], dtype=torch.int64)
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
