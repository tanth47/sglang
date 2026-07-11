import types

import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dspark_components.dspark_info import VerifyWindow
from sglang.srt.speculative.dspark_components.kernels.accept_greedy import (
    accept_greedy,
    accept_greedy_triton,
)
from sglang.srt.speculative.dspark_components.kernels.build_ragged_verify_window import (
    build_ragged_verify_window,
    build_ragged_verify_window_from_strided_torch,
    build_ragged_verify_window_from_strided_triton,
    build_ragged_verify_window_triton,
)
from sglang.srt.speculative.dspark_components.kernels.build_out_tokens import (
    build_out_tokens,
    build_out_tokens_triton,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    cap_correct_len,
    cap_correct_len_triton,
)
from sglang.srt.speculative.dspark_components.kernels.commit_inject_layout import (
    build_commit_inject_layout,
    build_commit_inject_layout_from_window,
    build_commit_inject_layout_from_window_triton,
    build_commit_inject_layout_triton,
)
from sglang.srt.speculative.dspark_components.kernels.finalize_accept_lens import (
    finalize_accept_lens,
    finalize_accept_lens_triton,
)
from sglang.srt.speculative.dspark_components.kernels.scatter_compact_to_strided import (
    scatter_compact_to_strided,
    scatter_compact_to_strided_triton,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSpark compact kernels need CUDA/ROCm"
)

GAMMA = 5
VERIFY_TOKENS = GAMMA + 1
CTX = 64
VOCAB = 200
POOL_REQS = 64
POOL_LEN = 64
FULL_SLOTS = 4096


def _ragged_window_fixtures(bs, graph_num_tokens, device):
    verify_lens = torch.randint(
        1, VERIFY_TOKENS + 1, (bs,), dtype=torch.int32, device=device
    )
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
    )
    seq_lens = torch.randint(1, 20, (bs,), dtype=torch.int64, device=device)
    num_reqs = bs + 3
    req_pool_indices = torch.randperm(num_reqs, device=device)[:bs].to(torch.int64)
    req_to_token = torch.randint(
        0, 1_000_000, (num_reqs, CTX), dtype=torch.int32, device=device
    )
    batch = types.SimpleNamespace(seq_lens=seq_lens, req_pool_indices=req_pool_indices)
    model_runner = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token)
    )
    draft_block_ids = torch.randint(
        0, 129280, (bs, GAMMA), dtype=torch.int64, device=device
    )
    draft_tokens = torch.randint(
        0, 129280, (bs, GAMMA), dtype=torch.int64, device=device
    )
    return layout, batch, model_runner, draft_block_ids, draft_tokens


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("pad", ["tight", "bucket"])
def test_build_ragged_verify_window_triton_matches_torch(bs, pad):
    device = torch.device("cuda")
    graph_num_tokens = bs * VERIFY_TOKENS if pad == "tight" else (bs + 3) * VERIFY_TOKENS
    layout, batch, model_runner, draft_block_ids, draft_tokens = (
        _ragged_window_fixtures(bs, graph_num_tokens, device)
    )
    ref = build_ragged_verify_window(
        batch=batch,
        layout=layout,
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        bs=bs,
        device=device,
        verify_num_draft_tokens=VERIFY_TOKENS,
        model_runner=model_runner,
    )
    got = build_ragged_verify_window_triton(
        batch=batch,
        layout=layout,
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        bs=bs,
        device=device,
        verify_num_draft_tokens=VERIFY_TOKENS,
        model_runner=model_runner,
    )
    assert torch.equal(got.positions, ref.positions)
    assert torch.equal(got.verify_cache_loc, ref.verify_cache_loc)
    assert torch.equal(got.verify_ids, ref.verify_ids)


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("pad", ["exact", "bucket"])
def test_build_ragged_verify_window_from_strided_triton_matches_torch(bs, pad):
    device = torch.device("cuda")
    verify_lens = torch.randint(
        1, VERIFY_TOKENS + 1, (bs,), dtype=torch.int32, device=device
    )
    total = int(verify_lens.sum().item())
    graph_num_tokens = total if pad == "exact" else bs * VERIFY_TOKENS
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens,
        graph_num_tokens=graph_num_tokens,
    )
    positions_2d = torch.randint(
        0, 1 << 20, (bs, VERIFY_TOKENS), dtype=torch.int64, device=device
    )
    cache_2d = torch.randint(
        0, 1 << 20, (bs, VERIFY_TOKENS), dtype=torch.int64, device=device
    )
    verify_ids_2d = torch.randint(
        0, 129280, (bs, VERIFY_TOKENS), dtype=torch.int64, device=device
    )
    verify_window = VerifyWindow(
        positions_2d=positions_2d,
        verify_cache_loc=cache_2d.reshape(-1),
        verify_cache_loc_2d=cache_2d,
    )
    ref = build_ragged_verify_window_from_strided_torch(
        layout=layout,
        verify_ids_2d=verify_ids_2d,
        verify_window=verify_window,
        device=device,
    )
    got = build_ragged_verify_window_from_strided_triton(
        layout=layout,
        verify_ids_2d=verify_ids_2d,
        verify_window=verify_window,
        device=device,
    )
    assert torch.equal(got.positions, ref.positions)
    assert torch.equal(got.verify_cache_loc, ref.verify_cache_loc)
    assert torch.equal(got.verify_ids, ref.verify_ids)


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("dim", [16, 4096])
@pytest.mark.parametrize("pad", ["exact", "bucket"])
def test_scatter_compact_to_strided_triton_matches_torch(bs, dim, pad):
    device = torch.device("cuda")
    verify_lens = torch.randint(
        1, VERIFY_TOKENS + 1, (bs,), dtype=torch.int32, device=device
    )
    total = int(verify_lens.sum().item())
    graph_num_tokens = total if pad == "exact" else bs * VERIFY_TOKENS
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
    )
    compact = torch.randn(graph_num_tokens, dim, dtype=torch.bfloat16, device=device)
    ref = scatter_compact_to_strided(
        compact=compact,
        layout=layout,
        fill_value=0.0,
        verify_num_draft_tokens=VERIFY_TOKENS,
    )
    got = scatter_compact_to_strided_triton(
        compact=compact,
        layout=layout,
        fill_value=0.0,
        verify_num_draft_tokens=VERIFY_TOKENS,
    )
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


@pytest.mark.parametrize(
    "cfg",
    [
        DSparkScheduleConfig(gamma=GAMMA),
        DSparkScheduleConfig(gamma=GAMMA, min_verify_len=0),
        DSparkScheduleConfig(gamma=GAMMA, min_verify_len=2),
        DSparkScheduleConfig(gamma=GAMMA, min_verify_len=1, max_verify_len=3),
    ],
)
def test_schedule_verify_lens_topk_triton_matches_torch_for_floor_configs(cfg):
    device = torch.device("cuda")
    confidence = torch.rand(8, GAMMA, device=device)
    for budget in (0, 1, 7, 1000):
        ref = schedule_verify_lens_topk(
            confidence=confidence,
            budget=budget,
            cfg=cfg,
        )
        got = schedule_verify_lens_topk_triton(
            confidence=confidence,
            budget=budget,
            cfg=cfg,
        )
        assert torch.equal(got, ref)


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("with_cutoff", [True, False])
def test_accept_greedy_triton_matches_torch_with_compact_cutoff(bs, with_cutoff):
    device = torch.device("cuda")
    candidates = torch.randint(
        0, VOCAB, (bs, VERIFY_TOKENS), dtype=torch.int64, device=device
    )
    target_logits = torch.randn(bs * VERIFY_TOKENS, VOCAB, device=device)
    cutoff = None
    if with_cutoff:
        cutoff = torch.randint(
            1, VERIFY_TOKENS + 1, (bs,), dtype=torch.int32, device=device
        )

    ref_correct, ref_bonus, ref_trim = accept_greedy(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=VERIFY_TOKENS,
        cutoff_verify_lens=cutoff,
    )
    got_correct, got_bonus, got_trim = accept_greedy_triton(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=VERIFY_TOKENS,
        cutoff_verify_lens=cutoff,
    )
    assert torch.equal(got_correct, ref_correct)
    assert torch.equal(got_bonus, ref_bonus)
    assert torch.equal(got_trim, ref_trim)


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("correct_dtype", [torch.int32, torch.int64])
def test_cap_correct_len_triton_matches_torch(bs, correct_dtype):
    device = torch.device("cuda")
    verify_lens = torch.randint(
        1, VERIFY_TOKENS + 1, (bs,), dtype=torch.int32, device=device
    )
    correct_len = (torch.arange(bs, device=device) % (VERIFY_TOKENS + 1)).to(
        correct_dtype
    )
    ref_capped, ref_trim = cap_correct_len(
        correct_len=correct_len,
        verify_lens=verify_lens,
    )
    got_capped, got_trim = cap_correct_len_triton(
        correct_len=correct_len,
        verify_lens=verify_lens,
    )
    assert got_capped.dtype == ref_capped.dtype
    assert got_trim.dtype == ref_trim.dtype
    assert torch.equal(got_capped, ref_capped)
    assert torch.equal(got_trim, ref_trim)


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64])
def test_finalize_accept_lens_triton_matches_torch(bs, prefix_dtype):
    device = torch.device("cuda")
    correct_len = torch.randint(0, VERIFY_TOKENS + 1, (bs,), device=device).to(
        torch.int32
    )
    cap_trim_lens = torch.randint(0, 4, (bs,), device=device).to(torch.int64)
    prefix_lens = torch.randint(1, 4000, (bs,), device=device).to(prefix_dtype)

    ref = finalize_accept_lens(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )
    got = finalize_accept_lens_triton(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )
    assert torch.equal(got.commit_lens, ref.commit_lens)
    assert torch.equal(got.new_seq_lens, ref.new_seq_lens)
    assert torch.equal(got.cap_trim_lens, ref.cap_trim_lens)
    assert got.commit_lens.dtype == torch.int32
    assert got.new_seq_lens.dtype == prefix_dtype
    assert got.cap_trim_lens.dtype == torch.int32


@pytest.mark.parametrize("bs", [2, 8])
@pytest.mark.parametrize("correct_dtype", [torch.int32, torch.int64])
def test_build_out_tokens_triton_matches_torch(bs, correct_dtype):
    device = torch.device("cuda")
    draft_tokens = torch.randint(
        0, 129280, (bs, GAMMA), dtype=torch.int64, device=device
    )
    bonus = torch.randint(0, 129280, (bs,), dtype=torch.int64, device=device)
    correct_len = (torch.arange(bs, device=device) % (GAMMA + 1)).to(correct_dtype)
    ref = build_out_tokens(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=VERIFY_TOKENS,
        gamma=GAMMA,
    )
    got = build_out_tokens_triton(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=VERIFY_TOKENS,
        gamma=GAMMA,
    )
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)


def _commit_inject_inputs(bs, device):
    req_to_token = torch.randint(
        0, FULL_SLOTS, (POOL_REQS, POOL_LEN), device=device, dtype=torch.int64
    )
    full_to_swa = torch.randint(
        0, 1 << 20, (FULL_SLOTS,), device=device, dtype=torch.int64
    )
    req_pool_indices = torch.randperm(POOL_REQS, device=device, dtype=torch.int64)[:bs]
    prefix_lens = torch.randint(
        1, POOL_LEN - VERIFY_TOKENS, (bs,), device=device, dtype=torch.int64
    )
    block_pos_offsets = torch.arange(VERIFY_TOKENS, device=device, dtype=torch.int64)
    commit_lens = torch.randint(
        0, VERIFY_TOKENS + 1, (bs,), device=device, dtype=torch.int32
    )
    return (
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa,
        commit_lens,
    )


def test_commit_inject_layout_triton_matches_torch_and_masks_edges():
    device = torch.device("cuda")
    (
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa,
        _,
    ) = _commit_inject_inputs(2, device)
    commit_lens = torch.tensor([0, VERIFY_TOKENS], device=device, dtype=torch.int32)

    ref = build_commit_inject_layout(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=VERIFY_TOKENS,
    )
    got = build_commit_inject_layout_triton(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=VERIFY_TOKENS,
    )
    assert torch.equal(got.swa_loc, ref.swa_loc)
    assert torch.equal(got.positions, ref.positions)
    swa_2d = got.swa_loc.view(2, VERIFY_TOKENS)
    assert bool((swa_2d[0] == -1).all())
    assert bool((swa_2d[1] >= 0).all())


def test_commit_inject_layout_from_window_triton_matches_legacy_layout():
    device = torch.device("cuda")
    (
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa,
        _,
    ) = _commit_inject_inputs(3, device)
    commit_lens = torch.tensor(
        [0, VERIFY_TOKENS // 2, VERIFY_TOKENS], device=device, dtype=torch.int32
    )
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:VERIFY_TOKENS]
    cache_loc_2d = req_to_token[req_pool_indices.view(-1, 1), positions_2d]

    ref = build_commit_inject_layout(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=VERIFY_TOKENS,
    )
    ref_from_window = build_commit_inject_layout_from_window(
        cache_loc_2d=cache_loc_2d,
        positions_2d=positions_2d,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=VERIFY_TOKENS,
    )
    got = build_commit_inject_layout_from_window_triton(
        cache_loc_2d=cache_loc_2d,
        positions_2d=positions_2d,
        full_to_swa_mapping=full_to_swa,
        commit_lens=commit_lens,
        stride=VERIFY_TOKENS,
    )
    assert torch.equal(ref_from_window.swa_loc, ref.swa_loc)
    assert torch.equal(ref_from_window.positions, ref.positions)
    assert torch.equal(got.swa_loc, ref.swa_loc)
    assert torch.equal(got.positions, ref.positions)
    swa_2d = got.swa_loc.view(3, VERIFY_TOKENS)
    assert bool((swa_2d[0] == -1).all())
    assert bool((swa_2d[1, : VERIFY_TOKENS // 2] >= 0).all())
    assert bool((swa_2d[1, VERIFY_TOKENS // 2 :] == -1).all())
    assert bool((swa_2d[2] >= 0).all())
