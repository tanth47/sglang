import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.models.dspark import GatedMarkovHead, RNNHead, VanillaMarkov
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_lm_head(*, vocab_size: int, per_partition: int):
    return SimpleNamespace(
        org_vocab_size=vocab_size,
        tp_size=1,
        num_embeddings_per_partition=per_partition,
        num_embeddings_padded=per_partition,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=vocab_size,
        ),
    )


def _tp1_group():
    return SimpleNamespace(world_size=1)


def _env(**overrides):
    values = {
        "SGLANG_DSPARK_OPT_MARKOV_W2_BF16": "0",
        "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD": "0",
    }
    values.update(overrides)
    return patch.dict(os.environ, values)


def _argmax_sampler(logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(logits, dim=-1)


class TestDSparkMarkovW2TpShard(CustomTestCase):
    def _make_vanilla_pair(self, *, vocab_size: int = 11, markov_rank: int = 4):
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD="0"):
            full = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD="1"):
            shard = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
        shard.load_state_dict(full.state_dict())
        return full, shard

    def test_bf16_w2_projects_back_to_float_logits(self):
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_BF16="1"):
            head = VanillaMarkov(vocab_size=13, markov_rank=5)

        latent = torch.randn(3, 5)
        bias = head.project_bias(latent)

        self.assertEqual(head.markov_w2.weight.dtype, torch.bfloat16)
        self.assertEqual(bias.dtype, torch.float32)
        self.assertEqual(bias.shape, (3, 13))

    def test_vanilla_sharded_step_matches_full_vocab_path_with_padding(self):
        torch.manual_seed(0)
        full, shard = self._make_vanilla_pair()
        vocab_size = full.vocab_size
        per_partition = vocab_size + 3

        base_full = torch.randn(5, vocab_size)
        base_local = F.pad(base_full, (0, per_partition - vocab_size))
        token_ids = torch.randint(0, vocab_size, (5,))

        expected = full.apply_step_logits(
            base_full, token_ids=token_ids, hidden_states=None
        )
        with patch("sglang.srt.models.dspark.get_attention_tp_group", _tp1_group):
            shard.configure_tp_shard(
                lm_head=_fake_lm_head(
                    vocab_size=vocab_size, per_partition=per_partition
                )
            )
            actual = shard.apply_step_logits(
                base_local, token_ids=token_ids, hidden_states=None
            )

        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected)

    def test_gated_sharded_step_matches_full_vocab_path_with_padding(self):
        torch.manual_seed(1)
        vocab_size, markov_rank, hidden_size = 11, 4, 7
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD="0"):
            full = GatedMarkovHead(
                vocab_size=vocab_size,
                markov_rank=markov_rank,
                hidden_size=hidden_size,
            )
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD="1"):
            shard = GatedMarkovHead(
                vocab_size=vocab_size,
                markov_rank=markov_rank,
                hidden_size=hidden_size,
            )
        shard.load_state_dict(full.state_dict())
        per_partition = vocab_size + 2

        base_full = torch.randn(3, vocab_size)
        base_local = F.pad(base_full, (0, per_partition - vocab_size))
        token_ids = torch.randint(0, vocab_size, (3,))
        hidden = torch.randn(3, hidden_size)

        expected = full.apply_step_logits(
            base_full, token_ids=token_ids, hidden_states=hidden
        )
        with patch("sglang.srt.models.dspark.get_attention_tp_group", _tp1_group):
            shard.configure_tp_shard(
                lm_head=_fake_lm_head(
                    vocab_size=vocab_size, per_partition=per_partition
                )
            )
            actual = shard.apply_step_logits(
                base_local, token_ids=token_ids, hidden_states=hidden
            )

        torch.testing.assert_close(actual, expected)

    def test_rnn_sharded_sampling_matches_full_vocab_path_with_padding(self):
        torch.manual_seed(2)
        vocab_size, markov_rank, hidden_size = 13, 5, 7
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD="0"):
            full = RNNHead(
                vocab_size=vocab_size,
                markov_rank=markov_rank,
                hidden_size=hidden_size,
            )
        with _env(SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD="1"):
            shard = RNNHead(
                vocab_size=vocab_size,
                markov_rank=markov_rank,
                hidden_size=hidden_size,
            )
        shard.load_state_dict(full.state_dict())
        per_partition = vocab_size + 1

        base_full = torch.randn(4, 3, vocab_size)
        base_local = F.pad(base_full, (0, per_partition - vocab_size))
        anchor_tokens = torch.randint(0, vocab_size, (4,))
        hidden = torch.randn(4, 3, hidden_size)

        expected_tokens, expected_logits = full.sample_block(
            base_full,
            first_prev_tokens=anchor_tokens,
            hidden_states=hidden,
            sampler=_argmax_sampler,
        )
        with patch("sglang.srt.models.dspark.get_attention_tp_group", _tp1_group):
            shard.configure_tp_shard(
                lm_head=_fake_lm_head(
                    vocab_size=vocab_size, per_partition=per_partition
                )
            )
            actual_tokens, actual_logits = shard.sample_block(
                base_local,
                first_prev_tokens=anchor_tokens,
                hidden_states=hidden,
                sampler=_argmax_sampler,
            )

        self.assertTrue(torch.equal(actual_tokens, expected_tokens))
        torch.testing.assert_close(actual_logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
