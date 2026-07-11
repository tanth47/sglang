import types
import unittest

import torch

from sglang.srt.managers.overlap_utils import ConfidenceRelay
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestConfidenceRelayStats(CustomTestCase):
    def _batch(self):
        return types.SimpleNamespace(
            spec_info=types.SimpleNamespace(
                future_indices=torch.tensor([0, 2], dtype=torch.int64)
            ),
            req_pool_indices=torch.tensor([0, 2], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0, 2], dtype=torch.int64),
        )

    def test_resolve_records_miss_and_direct_hit(self):
        pool = types.SimpleNamespace(req_generation=torch.arange(4, dtype=torch.int64))
        relay = ConfidenceRelay(
            device=torch.device("cpu"),
            req_pool_size=4,
            pool=pool,
        )

        self.assertIsNone(
            relay.resolve(self._batch(), stream=None, publish_ready=None)
        )
        stats = relay.snapshot_stats()
        self.assertEqual(stats.attempts, 1)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.last_status, "uninitialized")

        relay.scatter(
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
            seq_lens=torch.tensor([11, 17], dtype=torch.int64),
        )
        resolved = relay.resolve(self._batch(), stream=None, publish_ready=None)

        self.assertIsNotNone(resolved)
        torch.testing.assert_close(
            resolved.confidence,
            torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            resolved.generation,
            torch.tensor([0, 2], dtype=torch.int64),
        )
        torch.testing.assert_close(
            resolved.seq_lens,
            torch.tensor([11, 17], dtype=torch.int64),
        )
        stats = relay.snapshot_stats()
        self.assertEqual(stats.attempts, 2)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.last_status, "direct")

    def test_resolve_omits_seq_lens_when_not_published(self):
        pool = types.SimpleNamespace(req_generation=torch.arange(4, dtype=torch.int64))
        relay = ConfidenceRelay(
            device=torch.device("cpu"),
            req_pool_size=4,
            pool=pool,
        )
        relay.scatter(
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
        )
        resolved = relay.resolve(self._batch(), stream=None, publish_ready=None)

        self.assertIsNotNone(resolved)
        self.assertIsNone(resolved.seq_lens)


if __name__ == "__main__":
    unittest.main()
