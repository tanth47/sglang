import unittest

import torch

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")


class TestSpsVerifyLensTelemetry(unittest.TestCase):
    def test_host_constructor_keeps_logical_lens_out_of_physical_geometry(self):
        baseline = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8], device=_DEVICE, grid=[16]
        )
        sps_verify_lens = torch.tensor([3, 2], dtype=torch.int32)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8],
            device=_DEVICE,
            grid=[16],
            sps_verify_lens=sps_verify_lens,
        )

        self.assertIs(layout.sps_verify_lens, sps_verify_lens)
        self.assertTrue(torch.equal(layout.verify_lens, baseline.verify_lens))
        self.assertTrue(torch.equal(layout.qo_indptr_device, baseline.qo_indptr_device))
        self.assertEqual(layout.graph_num_tokens, baseline.graph_num_tokens)
        self.assertEqual(layout.total_verify_tokens, baseline.total_verify_tokens)

    def test_device_constructor_threads_logical_lens_without_host_physical_lens(self):
        sps_verify_lens = torch.tensor([3, 2], dtype=torch.int32)
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([8, 8], dtype=torch.int32),
            graph_num_tokens=16,
            sps_verify_lens=sps_verify_lens,
        )

        self.assertIs(layout.sps_verify_lens, sps_verify_lens)
        self.assertIsNone(layout.verify_lens_cpu)
        self.assertIsNone(layout.total_verify_tokens)

    def test_padding_drops_logical_lens_from_physical_dummy_layout(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8],
            device=_DEVICE,
            grid=[16, 32],
            graph_num_tokens_floor=32,
            sps_verify_lens=torch.tensor([3, 2], dtype=torch.int32),
        )

        padded = raw.padded_to_bucket(padded_bs=4)

        self.assertTrue(
            torch.equal(raw.sps_verify_lens, torch.tensor([3, 2], dtype=torch.int32))
        )
        self.assertIsNone(padded.sps_verify_lens)
        self.assertEqual(padded.verify_lens.tolist(), [8, 8, 8, 8])

    def test_absent_sidecar_does_not_leak_to_ephemeral_layout(self):
        RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8],
            device=_DEVICE,
            grid=[8],
            sps_verify_lens=torch.tensor([3], dtype=torch.int32),
        )

        ephemeral = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1] * 8,
            device=_DEVICE,
            grid=[8],
        )

        self.assertIsNone(ephemeral.sps_verify_lens)

    def test_sidecar_shape_must_match_physical_layout(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.tensor([8, 8], dtype=torch.int32),
                graph_num_tokens=16,
                sps_verify_lens=torch.tensor([3], dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
