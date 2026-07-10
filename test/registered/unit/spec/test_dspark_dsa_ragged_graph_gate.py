from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DeepseekSparseAttnBackend = type("DeepseekSparseAttnBackend", (), {})
OtherBackend = type("OtherBackend", (), {})


def _runner(backend, *, verify_width=8):
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.attn_backend = backend
    runner.num_tokens_per_bs = verify_width
    return runner


def _layout(*, verify_lens, graph_num_tokens=None, total_verify_tokens=None):
    if graph_num_tokens is None:
        graph_num_tokens = sum(verify_lens)
    if total_verify_tokens is None:
        total_verify_tokens = sum(verify_lens)
    return SimpleNamespace(
        verify_lens=torch.tensor(verify_lens, dtype=torch.int32),
        verify_lens_cpu=list(verify_lens),
        graph_num_tokens=graph_num_tokens,
        total_verify_tokens=total_verify_tokens,
    )


def test_dsa_ragged_graph_allows_full_width_layout():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 8]),
    )


def test_dsa_ragged_graph_rejects_non_uniform_compact_layout():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert not runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 1], graph_num_tokens=16),
    )


def test_dsa_ragged_graph_rejects_token_tier_padding():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert not runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 8], graph_num_tokens=24, total_verify_tokens=16),
    )


def test_non_dsa_backend_keeps_generic_ragged_graph_path():
    runner = _runner(OtherBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 1], graph_num_tokens=16),
    )


def test_full_width_target_verify_without_layout_is_graph_eligible():
    runner = _runner(DeepseekSparseAttnBackend(), verify_width=8)
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.empty(16, dtype=torch.int64),
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
    )

    assert runner._is_full_width_target_verify_without_layout(forward_batch)


def test_partial_target_verify_without_layout_is_not_graph_eligible():
    runner = _runner(DeepseekSparseAttnBackend(), verify_width=8)
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.empty(9, dtype=torch.int64),
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
    )

    assert not runner._is_full_width_target_verify_without_layout(forward_batch)


def _can_run_runner(*, verify_width=8):
    runner = _runner(DeepseekSparseAttnBackend(), verify_width=verify_width)
    runner.ragged_verify_mode = True
    runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = False
    runner.disable_padding = False
    runner.max_bs = 4
    runner.enable_pdmux = False
    runner.is_encoder_decoder = False
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.enable_two_batch_overlap = False
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
    )
    return runner


def _target_verify_batch(*, bs, num_tokens):
    return SimpleNamespace(
        replace_embeds=None,
        batch_size=bs,
        input_ids=torch.empty(num_tokens, dtype=torch.int64),
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        global_num_tokens_cpu=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        spec_info=SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL),
    )


def test_compact_mode_admits_full_width_target_verify_without_layout():
    runner = _can_run_runner()

    assert runner.can_run_graph(_target_verify_batch(bs=2, num_tokens=16))


def test_compact_mode_rejects_partial_target_verify_without_layout():
    runner = _can_run_runner()

    assert not runner.can_run_graph(_target_verify_batch(bs=2, num_tokens=9))
