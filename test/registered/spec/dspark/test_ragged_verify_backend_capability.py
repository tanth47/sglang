"""Backend opt-in flags for the ragged-verify graphs.

Runs in the GPU suite because importing the backend modules pulls GPU-only
wheels (sgl_kernel) at module scope, which fail to import on CPU runners.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


class TestRaggedVerifyGraphCapability(CustomTestCase):
    @staticmethod
    def _rocm_regime_neutral_graph_kwargs():
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend

        return {
            "ragged_graph_requested": True,
            "use_fused_topk": True,
            "dsa_topk_backend": DSATopKBackend.SGL_KERNEL,
            "dsa_decode_impl": "tilelang",
            "kv_cache_dtype": torch.bfloat16,
            "page_size": 64,
            "dsa_index_topk": 2048,
            "hisparse_enabled": False,
            "gfx95_supported": True,
        }

    def test_rocm_regime_neutral_target_verify_graph_supported_route(self):
        from sglang.srt.layers.attention.dsa_backend import (
            _supports_rocm_regime_neutral_target_verify_graph,
        )

        self.assertTrue(
            _supports_rocm_regime_neutral_target_verify_graph(
                **self._rocm_regime_neutral_graph_kwargs()
            )
        )

    def test_rocm_regime_neutral_target_verify_graph_rejects_unsafe_routes(self):
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
        from sglang.srt.layers.attention.dsa_backend import (
            _supports_rocm_regime_neutral_target_verify_graph,
        )

        unsafe_routes = {
            "ragged graph disabled": ("ragged_graph_requested", False),
            "fused top-k disabled": ("use_fused_topk", False),
            "non-SGL top-k backend": (
                "dsa_topk_backend",
                DSATopKBackend.TORCH,
            ),
            "non-TileLang decode": ("dsa_decode_impl", "fa3"),
            "non-bfloat16 KV cache": ("kv_cache_dtype", torch.float16),
            "unsupported page size": ("page_size", 32),
            "unsupported index top-k": ("dsa_index_topk", 1024),
            "HiSparse enabled": ("hisparse_enabled", True),
            "non-gfx95 device": ("gfx95_supported", False),
        }

        for route, (argument, unsafe_value) in unsafe_routes.items():
            with self.subTest(route=route):
                kwargs = self._rocm_regime_neutral_graph_kwargs()
                kwargs[argument] = unsafe_value
                self.assertFalse(
                    _supports_rocm_regime_neutral_target_verify_graph(**kwargs)
                )

    def test_base_backend_defaults_false(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)
        self.assertIsNone(
            AttentionBackend.required_ragged_verify_slots(
                object(),
                forward_batch=SimpleNamespace(batch_size=2),
                ragged_layout=object(),
                num_tokens_per_req=8,
            )
        )

    def test_unified_dsa_slot_requirement_is_logical_batch_size(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )

        resolve_slots = DeepseekSparseAttnBackend.required_ragged_verify_slots
        forward_batch = SimpleNamespace(batch_size=16)
        self.assertEqual(
            resolve_slots(
                SimpleNamespace(supports_unified_dsa_target_verify_graph=True),
                forward_batch=forward_batch,
                ragged_layout=object(),
                num_tokens_per_req=8,
            ),
            16,
        )
        self.assertIsNone(
            resolve_slots(
                SimpleNamespace(supports_unified_dsa_target_verify_graph=False),
                forward_batch=forward_batch,
                ragged_layout=object(),
                num_tokens_per_req=8,
            )
        )

    def test_dsa_eager_target_verify_rehydrates_cpu_lengths(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )

        forward_batch = SimpleNamespace(
            seq_lens=torch.tensor([17, 29], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )
        actual = DeepseekSparseAttnBackend._target_verify_seq_lens_cpu(
            forward_batch
        )

        self.assertEqual(actual, [17, 29])
        self.assertEqual(forward_batch.seq_lens_cpu.tolist(), [17, 29])
        self.assertEqual(forward_batch.seq_lens_sum, 46)

    def test_ragged_implementing_backends_declare_the_flag(self):
        """Every backend with a ragged-verify metadata path must opt in; a
        dropped flag silently disables ragged graphs for that backend (the
        runner falls back to eager with no other test going red)."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        for backend in (
            TRTLLMHAAttnBackend,
            DeepseekV4AttnBackend,
            FlashAttentionBackend,
        ):
            with self.subTest(backend=backend.__name__):
                self.assertTrue(backend.supports_ragged_verify_graph)

    def test_dsa_topk_transform_selects_query_cu_seqlens(self):
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
            TopkTransformMethod,
        )
        from sglang.srt.layers.attention.dsa_backend import DSAIndexerMetadata

        class FakeTopKBackend:
            def topk_transform(self, **kwargs):
                self.kwargs = kwargs
                return kwargs["logits"]

        logits = torch.zeros((3, 5))
        dsa_cu_seqlens_q = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        logical_cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32)

        for page_table_is_token_expanded in (True, False):
            with self.subTest(
                page_table_is_token_expanded=page_table_is_token_expanded
            ):
                attn_metadata = SimpleNamespace(
                    page_table_is_token_expanded=page_table_is_token_expanded,
                    dsa_cu_seqlens_q=dsa_cu_seqlens_q,
                    cu_seqlens_q=logical_cu_seqlens_q,
                    topk_indices_offset=torch.tensor([0], dtype=torch.int32),
                    dsa_extend_seq_lens_list=[],
                    dsa_seqlens_expanded=torch.tensor([5, 5, 5], dtype=torch.int32),
                )
                topk_backend = FakeTopKBackend()
                metadata = DSAIndexerMetadata(
                    attn_metadata=attn_metadata,
                    topk_transform_method=TopkTransformMethod.PAGED,
                    topk_backend=topk_backend,
                )

                self.assertIs(metadata.topk_transform(logits, topk=2), logits)
                actual = topk_backend.kwargs["cu_seqlens_q_topk"]
                if page_table_is_token_expanded:
                    torch.testing.assert_close(
                        actual, dsa_cu_seqlens_q[: logits.shape[0] + 1]
                    )
                else:
                    self.assertIs(actual, logical_cu_seqlens_q)

    def test_dsa_cuda_graph_metadata_key_separates_modes_and_tiers(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = SimpleNamespace(
            supports_unified_dsa_target_verify_graph=True,
            speculative_num_draft_tokens=8,
        )
        resolve_key = DeepseekSparseAttnBackend._cuda_graph_metadata_key
        decode_key = resolve_key(backend, 16, ForwardMode.DECODE, None)
        draft_key = resolve_key(backend, 16, ForwardMode.DRAFT_EXTEND_V2, None)
        target_full_key = resolve_key(backend, 16, ForwardMode.TARGET_VERIFY, None)
        target_16_key = resolve_key(
            backend,
            16,
            ForwardMode.TARGET_VERIFY,
            SimpleNamespace(
                ragged_verify_layout=SimpleNamespace(
                    graph_num_tokens=16, total_verify_tokens=16
                )
            ),
        )
        target_112_key = resolve_key(
            backend,
            16,
            ForwardMode.TARGET_VERIFY,
            SimpleNamespace(
                ragged_verify_layout=SimpleNamespace(
                    graph_num_tokens=112, total_verify_tokens=112
                )
            ),
        )

        self.assertEqual(decode_key[1:], (16, 16))
        self.assertEqual(draft_key[1:], (16, 128))
        self.assertEqual(target_full_key[1:], (16, 128))
        self.assertEqual(target_16_key[1:], (16, 16))
        self.assertEqual(target_112_key[1:], (112, 112))
        self.assertEqual(
            len(
                {
                    decode_key,
                    draft_key,
                    target_full_key,
                    target_16_key,
                    target_112_key,
                }
            ),
            5,
        )

    def test_unified_target_verify_lens_stays_device_only(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        backend = SimpleNamespace(
            supports_unified_dsa_target_verify_graph=True,
            speculative_num_draft_tokens=8,
        )
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([3, 2], dtype=torch.int32),
            graph_num_tokens=8,
        )
        spec_info = SimpleNamespace(ragged_verify_layout=layout)

        with mock.patch(
            "sglang.srt.layers.attention.dsa_backend." "materialize_verify_lens_cpu",
            side_effect=AssertionError("unexpected D2H verify-lens materialization"),
        ):
            verify_lens, physical_lens_cpu, total_tokens = (
                DeepseekSparseAttnBackend._target_verify_lens_for_graph(
                    backend,
                    bs=8,
                    seq_lens=torch.ones(8, dtype=torch.int32),
                    spec_info=spec_info,
                )
            )

        self.assertEqual(total_tokens, 8)
        self.assertEqual(int(verify_lens.sum()), 8)
        self.assertEqual(physical_lens_cpu, [1] * 8)

        large_layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.full((15,), 8, dtype=torch.int32),
            graph_num_tokens=128,
        )
        with mock.patch(
            "sglang.srt.layers.attention.dsa_backend." "materialize_verify_lens_cpu",
            side_effect=AssertionError("unexpected D2H verify-lens materialization"),
        ):
            large_lens, large_physical_cpu, large_total = (
                DeepseekSparseAttnBackend._target_verify_lens_for_graph(
                    backend,
                    bs=16,
                    seq_lens=torch.ones(16, dtype=torch.int32),
                    spec_info=SimpleNamespace(ragged_verify_layout=large_layout),
                )
            )

        self.assertEqual(large_total, 128)
        self.assertEqual(int(large_lens.sum()), 128)
        self.assertEqual(large_physical_cpu, [8] * 16)

    def test_mtp_precompute_uses_mode_aware_tuple_metadata_key(self):
        from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
            DeepseekSparseAttnBackendMTPPrecomputeMixin,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        for forward_mode, selected_helper in (
            (ForwardMode.DECODE, "_precompute_decode_mode"),
            (ForwardMode.TARGET_VERIFY, "_precompute_target_verify_mode"),
        ):
            with self.subTest(forward_mode=forward_mode):
                metadata_key = (forward_mode.name, 2, 16)
                graph_metadata = SimpleNamespace(page_table_1=torch.empty((2, 1)))
                expected = object()
                decode_helper = mock.Mock(return_value=expected)
                target_helper = mock.Mock(return_value=expected)
                backend = SimpleNamespace(
                    decode_cuda_graph_metadata={metadata_key: graph_metadata},
                    _cuda_graph_metadata_key=mock.Mock(return_value=metadata_key),
                    _precompute_decode_mode=decode_helper,
                    _precompute_target_verify_mode=target_helper,
                )

                actual = DeepseekSparseAttnBackendMTPPrecomputeMixin._precompute_replay_metadata(
                    backend,
                    bs=2,
                    req_pool_indices=torch.tensor([0, 1]),
                    seq_lens=torch.tensor([8, 9]),
                    seq_lens_cpu=[8, 9],
                    forward_mode=forward_mode,
                )

                self.assertIs(actual, expected)
                backend._cuda_graph_metadata_key.assert_called_once_with(
                    2, forward_mode, None
                )
                helper = getattr(backend, selected_helper)
                helper.assert_called_once()
                self.assertIs(helper.call_args.args[-1], graph_metadata)

    def test_static_full_runtime_layout_reuses_physical_token_tier_key(self):
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            resolve_graph_spec_info,
        )
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        backend = SimpleNamespace(
            supports_unified_dsa_target_verify_graph=True,
            speculative_num_draft_tokens=8,
        )
        runtime_layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8], device=torch.device("cpu"), grid=[8]
        )
        runtime_spec_info = resolve_graph_spec_info(
            SimpleNamespace(
                spec_info=SimpleNamespace(ragged_verify_layout=runtime_layout)
            ),
            num_tokens_per_req=8,
            preserve_static_full_layout=True,
        )
        capture_layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1] * 8, device=torch.device("cpu"), grid=[8]
        )
        capture_spec_info = SimpleNamespace(ragged_verify_layout=capture_layout)

        self.assertIs(runtime_spec_info.ragged_verify_layout, runtime_layout)
        resolve_key = DeepseekSparseAttnBackend._cuda_graph_metadata_key
        runtime_key = resolve_key(
            backend, 1, ForwardMode.TARGET_VERIFY, runtime_spec_info
        )
        capture_key = resolve_key(
            backend, 8, ForwardMode.TARGET_VERIFY, capture_spec_info
        )

        self.assertEqual(runtime_key, capture_key)
        self.assertEqual(runtime_key[1:], (8, 8))


if __name__ == "__main__":
    unittest.main()
