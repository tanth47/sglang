from types import SimpleNamespace

import torch

import sglang.srt.model_executor.runner.decode_cuda_graph_runner as decode_cgr
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.transform_index import (
    transform_index_page_table_prefill_ref,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend as RealDeepseekSparseAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dspark_components.kernels.padded_to_bucket import (
    PadVerifyLensWithinRows,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DeepseekSparseAttnBackend = type(
    "DeepseekSparseAttnBackend", (), {"supports_ragged_verify_graph": True}
)
NoRaggedGraphBackend = type(
    "NoRaggedGraphBackend", (), {"supports_ragged_verify_graph": False}
)
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


def test_dsa_ragged_graph_allows_non_uniform_compact_layout():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert runner._attn_backend_supports_layout_ragged_verify_graph(
        forward_batch,
        _layout(verify_lens=[8, 1], graph_num_tokens=16),
    )


def test_dsa_ragged_graph_allows_token_tier_padding():
    runner = _runner(DeepseekSparseAttnBackend())
    forward_batch = SimpleNamespace(batch_size=2)

    assert runner._attn_backend_supports_layout_ragged_verify_graph(
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
    runner.capture_num_tokens = [8, 16, 24, 32]
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


def _target_verify_batch(*, bs, num_tokens, layout=None):
    return SimpleNamespace(
        replace_embeds=None,
        batch_size=bs,
        input_ids=torch.empty(num_tokens, dtype=torch.int64),
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        global_num_tokens_cpu=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        spec_info=SimpleNamespace(
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
        ),
    )


def test_compact_mode_admits_full_width_target_verify_without_layout():
    runner = _can_run_runner()

    assert runner.can_run_graph(_target_verify_batch(bs=2, num_tokens=16))


def test_compact_mode_synthesizes_layout_for_full_width_target_verify():
    runner = _can_run_runner()
    layout = runner._ragged_verify_layout(_target_verify_batch(bs=2, num_tokens=16))

    assert layout is not None
    assert layout.verify_lens_cpu == [8, 8]
    assert layout.graph_num_tokens == 16
    assert layout.is_full_width is True


def test_compact_mode_rejects_partial_target_verify_without_layout():
    runner = _can_run_runner()

    assert not runner.can_run_graph(_target_verify_batch(bs=2, num_tokens=9))


def test_compact_mode_admits_non_uniform_dsa_ragged_layout():
    runner = _can_run_runner()
    layout = _layout(verify_lens=[8, 1], graph_num_tokens=16)

    assert runner.can_run_graph(
        _target_verify_batch(bs=2, num_tokens=16, layout=layout)
    )


def test_compact_mode_rejects_backend_without_ragged_graph_support():
    runner = _can_run_runner()
    runner.attn_backend = NoRaggedGraphBackend()
    layout = _layout(verify_lens=[8, 1], graph_num_tokens=16)

    assert not runner.can_run_graph(
        _target_verify_batch(bs=2, num_tokens=16, layout=layout)
    )


def test_compact_mode_admits_token_tier_padded_dsa_ragged_layout():
    runner = _can_run_runner()
    layout = _layout(verify_lens=[8, 8], graph_num_tokens=24, total_verify_tokens=16)

    assert runner.can_run_graph(
        _target_verify_batch(bs=2, num_tokens=24, layout=layout)
    )


def _graph_gate_model_runner(
    *,
    backend="dsa",
    is_dspark=True,
    is_draft_worker=False,
):
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: is_dspark),
        server_args=SimpleNamespace(target_verify_attention_backend=lambda: backend),
    )


def test_dspark_dsa_target_verify_graph_disabled_on_hip_topk_broadcast(monkeypatch):
    monkeypatch.setattr(decode_cgr, "is_hip", lambda: True)

    with envs.SGLANG_DSA_TOPK_BROADCAST.override(
        True
    ), envs.SGLANG_DSPARK_ALLOW_HIP_DSA_TARGET_VERIFY_GRAPH.override(False):
        assert decode_cgr.dspark_dsa_target_verify_cuda_graph_unsafe_on_hip(
            _graph_gate_model_runner()
        )


def test_dspark_dsa_target_verify_graph_can_be_experimentally_enabled_on_hip(
    monkeypatch,
):
    monkeypatch.setattr(decode_cgr, "is_hip", lambda: True)

    with envs.SGLANG_DSA_TOPK_BROADCAST.override(
        True
    ), envs.SGLANG_DSPARK_ALLOW_HIP_DSA_TARGET_VERIFY_GRAPH.override(True):
        assert not decode_cgr.dspark_dsa_target_verify_cuda_graph_unsafe_on_hip(
            _graph_gate_model_runner()
        )


def test_dspark_dsa_target_verify_graph_not_disabled_without_hip(monkeypatch):
    monkeypatch.setattr(decode_cgr, "is_hip", lambda: False)

    with envs.SGLANG_DSA_TOPK_BROADCAST.override(True):
        assert not decode_cgr.dspark_dsa_target_verify_cuda_graph_unsafe_on_hip(
            _graph_gate_model_runner()
        )


def test_dspark_dsa_target_verify_graph_not_disabled_for_draft_worker(monkeypatch):
    monkeypatch.setattr(decode_cgr, "is_hip", lambda: True)

    with envs.SGLANG_DSA_TOPK_BROADCAST.override(True):
        assert not decode_cgr.dspark_dsa_target_verify_cuda_graph_unsafe_on_hip(
            _graph_gate_model_runner(is_draft_worker=True)
        )


def test_dspark_dsa_target_verify_graph_not_disabled_for_non_dsa(monkeypatch):
    monkeypatch.setattr(decode_cgr, "is_hip", lambda: True)

    with envs.SGLANG_DSA_TOPK_BROADCAST.override(True):
        assert not decode_cgr.dspark_dsa_target_verify_cuda_graph_unsafe_on_hip(
            _graph_gate_model_runner(backend="triton")
        )


def test_dspark_dsa_target_verify_graph_not_disabled_for_non_dspark(monkeypatch):
    monkeypatch.setattr(decode_cgr, "is_hip", lambda: True)

    with envs.SGLANG_DSA_TOPK_BROADCAST.override(True):
        assert not decode_cgr.dspark_dsa_target_verify_cuda_graph_unsafe_on_hip(
            _graph_gate_model_runner(is_dspark=False)
        )


def test_dsa_ragged_metadata_key_uses_token_tier():
    backend = object.__new__(RealDeepseekSparseAttnBackend)
    backend.device = torch.device("cpu")
    backend.speculative_num_draft_tokens = 8
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[8, 1],
        device=torch.device("cpu"),
        grid=[16],
        num_draft_tokens=8,
    )
    spec_info = SimpleNamespace(ragged_verify_layout=layout)

    assert backend._cuda_graph_metadata_key(
        16, ForwardMode.TARGET_VERIFY, spec_info
    ) == ("target_verify_ragged", 16)


def test_dsa_ragged_graph_max_q_len_uses_static_width():
    backend = object.__new__(RealDeepseekSparseAttnBackend)
    backend.speculative_num_draft_tokens = 8

    assert backend._ragged_verify_max_q_len_for_cuda_graph() == 8


def test_dsa_ragged_metadata_padding_caps_each_row():
    backend = object.__new__(RealDeepseekSparseAttnBackend)
    backend.device = torch.device("cpu")
    backend.speculative_num_draft_tokens = 8
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[1] * 31,
        device=torch.device("cpu"),
        grid=[64],
        graph_num_tokens_floor=64,
        num_draft_tokens=8,
    )

    padded = backend._ragged_verify_lens_for_cuda_graph(layout=layout, bs=32)

    assert padded.shape == (32,)
    assert int(padded.sum().item()) == 64
    assert int(padded.max().item()) <= 8


def test_dsa_ragged_metadata_padding_prefers_dummy_rows():
    backend = object.__new__(RealDeepseekSparseAttnBackend)
    backend.device = torch.device("cpu")
    backend.speculative_num_draft_tokens = 8
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[8, 1],
        device=torch.device("cpu"),
        grid=[24],
        graph_num_tokens_floor=24,
        num_draft_tokens=8,
    )

    padded = backend._ragged_verify_lens_for_cuda_graph(layout=layout, bs=3)

    assert padded.tolist() == [8, 8, 8]


def test_dsa_ragged_metadata_pads_device_only_layout_to_graph_total():
    backend = object.__new__(RealDeepseekSparseAttnBackend)
    backend.device = torch.device("cpu")
    backend.speculative_num_draft_tokens = 8
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([8, 1], dtype=torch.int32),
        graph_num_tokens=16,
    )

    padded = backend._ragged_verify_lens_for_cuda_graph(layout=layout, bs=2)

    assert layout.total_verify_tokens is None
    assert padded.tolist() == [8, 8]


def test_pad_verify_lens_within_rows_fills_real_rows_when_no_dummy_rows():
    padded = PadVerifyLensWithinRows.execute(
        verify_lens=torch.tensor([8, 1], dtype=torch.int32),
        graph_num_tokens=16,
        bs=2,
        padded_bs=2,
        max_verify_len=8,
    )

    assert padded.tolist() == [8, 8]


def test_pad_verify_lens_within_rows_rejects_over_capacity():
    try:
        PadVerifyLensWithinRows.execute(
            verify_lens=torch.tensor([8, 8], dtype=torch.int32),
            graph_num_tokens=25,
            bs=2,
            padded_bs=3,
            max_verify_len=8,
        )
    except ValueError as exc:
        assert "cannot fit" in str(exc)
    else:
        raise AssertionError("expected over-capacity graph tier to be rejected")


def test_dsa_expanded_verify_page_table_uses_token_level_extend_lens():
    page_table = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)
    topk = torch.tensor(
        [
            [0, 2, -1],
            [1, 3, -1],
            [2, 4, -1],
            [3, 5, -1],
        ],
        dtype=torch.int64,
    )

    result = transform_index_page_table_prefill_ref(
        page_table=page_table,
        topk_indices=topk,
        extend_lens_cpu=[1, 1, 1, 1],
    )

    expected = torch.gather(page_table.to(result.dtype), dim=1, index=topk.clamp(min=0))
    expected[topk < 0] = -1
    assert torch.equal(result, expected)


def test_dsa_expanded_verify_page_table_rejects_request_level_extend_lens():
    page_table = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)
    topk = torch.zeros((4, 3), dtype=torch.int64)

    try:
        transform_index_page_table_prefill_ref(
            page_table=page_table,
            topk_indices=topk,
            extend_lens_cpu=[1, 3],
        )
    except AssertionError:
        return
    raise AssertionError("request-level extend lengths must not match token rows")
