import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkDraftAnchorLayout(CustomTestCase):
    def _run_forward_layout_case(
        self,
        *,
        gamma: int,
        sample_from_anchor: bool,
        expected_first_hidden_slot: int,
    ):
        seen = {}
        bs = 2
        draft_query_width = gamma if sample_from_anchor else gamma + 1
        verify_width = gamma + 1
        hidden_size = 8

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["input_ids"] = forward_batch.input_ids.clone()
                seen["positions"] = forward_batch.positions.clone()
                seen["out_cache_loc"] = forward_batch.out_cache_loc.clone()
                hidden = torch.arange(
                    bs * draft_query_width * hidden_size, dtype=torch.float32
                ).view(bs * draft_query_width, hidden_size)
                return SimpleNamespace(
                    logits_output=SimpleNamespace(hidden_states=hidden),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_sum=30,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * verify_width, dtype=torch.int64).view(
                bs, verify_width
            ),
            verify_cache_loc_2d=torch.arange(
                100, 100 + bs * verify_width, dtype=torch.int64
            ).view(bs, verify_width),
        )

        out = proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, hidden_size),
        )

        self.assertEqual(tuple(out.draft_block_ids.shape), (bs, draft_query_width))
        self.assertEqual(out.draft_block_ids[:, 0].tolist(), [7, 8])
        self.assertEqual(seen["input_ids"].numel(), bs * draft_query_width)
        expected_positions = (
            verify_window.positions_2d[:, :draft_query_width].reshape(-1).tolist()
        )
        self.assertEqual(seen["positions"].tolist(), expected_positions)
        expected_cache_locs = (
            verify_window.verify_cache_loc_2d[:, :draft_query_width]
            .reshape(-1)
            .tolist()
        )
        self.assertEqual(seen["out_cache_loc"].tolist(), expected_cache_locs)
        self.assertEqual(tuple(out.draft_hidden_3d.shape), (bs, gamma, hidden_size))
        self.assertEqual(tuple(out.raw_hidden.shape), (bs * gamma, hidden_size))
        first = expected_first_hidden_slot * hidden_size
        self.assertEqual(
            out.raw_hidden[0].tolist(), list(range(first, first + hidden_size))
        )

    def test_legacy_draft_forward_crops_anchor_hidden(self):
        self._run_forward_layout_case(
            gamma=4,
            sample_from_anchor=False,
            expected_first_hidden_slot=1,
        )

    def test_anchor_sampled_draft_forward_keeps_anchor_hidden(self):
        self._run_forward_layout_case(
            gamma=4,
            sample_from_anchor=True,
            expected_first_hidden_slot=0,
        )

    def test_folded_sampler_rejects_incomplete_draft_query_blocks(self):
        class FakeModel:
            def __init__(self):
                self.markov_head = SimpleNamespace()

            def compute_base_logits(self, hidden_states):
                vocab_size = 16
                logits = torch.zeros((hidden_states.shape[0], vocab_size))
                return logits, None

        sampler = DsparkDraftSampler(
            model=FakeModel(),
            gamma=4,
            sample_from_anchor=False,
            max_bs=2,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(RuntimeError, "draft query"):
            sampler(torch.empty((8, 8)), torch.empty((8,), dtype=torch.int64))

    def test_folded_sampler_uses_layout_aligned_hidden_slots(self):
        gamma = 3
        bs = 2
        hidden_size = 4
        vocab_size = 8

        layouts = ((False, 4, 1), (True, 3, 0))
        for (
            sample_from_anchor,
            draft_query_width,
            expected_first_hidden_slot,
        ) in layouts:
            with self.subTest(
                sample_from_anchor=sample_from_anchor,
                draft_query_width=draft_query_width,
            ):
                seen = {}

                class FakeMarkovHead:
                    def sample_block(
                        self,
                        base_logits,
                        *,
                        first_prev_tokens,
                        hidden_states,
                        sampler,
                    ):
                        del sampler
                        seen["first_prev_tokens"] = first_prev_tokens.clone()
                        seen["hidden_states"] = hidden_states.clone()
                        return (
                            torch.zeros((bs, gamma), dtype=torch.int64),
                            base_logits,
                        )

                class FakeModel:
                    def __init__(self):
                        self.markov_head = FakeMarkovHead()

                    def compute_base_logits(self, hidden_states):
                        seen["hidden_for_logits"] = hidden_states.clone()
                        return (
                            torch.zeros(
                                (hidden_states.shape[0], vocab_size),
                                dtype=torch.float32,
                            ),
                            None,
                        )

                sampler = DsparkDraftSampler(
                    model=FakeModel(),
                    gamma=gamma,
                    sample_from_anchor=sample_from_anchor,
                    max_bs=bs,
                    device=torch.device("cpu"),
                )
                hidden = torch.arange(
                    bs * draft_query_width * hidden_size, dtype=torch.float32
                ).view(bs * draft_query_width, hidden_size)
                input_ids = torch.arange(
                    10, 10 + bs * draft_query_width, dtype=torch.int64
                )

                sampler(hidden, input_ids)

                hidden_3d = hidden.view(bs, draft_query_width, hidden_size)
                expected_hidden = hidden_3d[
                    :,
                    expected_first_hidden_slot : expected_first_hidden_slot + gamma,
                    :,
                ]
                self.assertEqual(
                    seen["first_prev_tokens"].tolist(),
                    input_ids.view(bs, draft_query_width)[:, 0].tolist(),
                )
                self.assertTrue(torch.equal(seen["hidden_states"], expected_hidden))
                self.assertTrue(
                    torch.equal(
                        seen["hidden_for_logits"],
                        expected_hidden.reshape(bs * gamma, hidden_size),
                    )
                )


if __name__ == "__main__":
    unittest.main()
