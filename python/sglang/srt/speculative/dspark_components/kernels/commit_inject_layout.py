from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_COMMIT_INJECT_LAYOUT.get()


class CommitInjectLayoutResult(msgspec.Struct):
    swa_loc: torch.Tensor
    positions: torch.Tensor


class BuildCommitInjectLayout:
    @classmethod
    def execute(cls, *args, **kwargs) -> CommitInjectLayoutResult:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout_triton(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )


class BuildCommitInjectLayoutFromWindow:
    @classmethod
    def execute(cls, *args, **kwargs) -> CommitInjectLayoutResult:
        cache_loc_2d = kwargs.get("cache_loc_2d")
        if _KERNEL_IMPL == "torch" or cache_loc_2d.device.type == "cpu":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        cache_loc_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout_from_window(
            cache_loc_2d=cache_loc_2d,
            positions_2d=positions_2d,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )

    @classmethod
    def triton(
        cls,
        *,
        cache_loc_2d: torch.Tensor,
        positions_2d: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout_from_window_triton(
            cache_loc_2d=cache_loc_2d,
            positions_2d=positions_2d,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )


def build_commit_inject_layout(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    from sglang.srt.speculative.triton_ops.cache_locs import (
        assign_extend_cache_locs_func,
    )

    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device

    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(dtype=torch.int64)

    cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + stride,
        batch_size=bs,
        draft_token_num=stride,
        device=device,
    ).to(dtype=torch.int64)
    swa_loc = full_to_swa_mapping[cache_loc].to(torch.int32)

    col = torch.arange(stride, device=device).view(1, -1)
    committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
    swa_loc = torch.where(committed, swa_loc, torch.full_like(swa_loc, -1))

    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


def build_commit_inject_layout_from_window(
    *,
    cache_loc_2d: torch.Tensor,
    positions_2d: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    device = cache_loc_2d.device
    cache_loc = cache_loc_2d[:, :stride].reshape(-1).to(torch.int64)
    positions = positions_2d[:, :stride].reshape(-1).to(torch.int64)
    swa_loc = full_to_swa_mapping[cache_loc].to(torch.int32)

    col = torch.arange(stride, device=device).view(1, -1)
    committed = col < commit_lens.to(device=device, dtype=torch.long).view(-1, 1)
    committed = committed.reshape(-1)
    swa_loc = torch.where(committed, swa_loc, torch.full_like(swa_loc, -1))

    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


@triton.jit
def _commit_inject_layout_kernel(
    req_pool_ptr,
    req_to_token_ptr,
    prefix_lens_ptr,
    block_pos_offsets_ptr,
    full_to_swa_ptr,
    commit_lens_ptr,
    swa_loc_ptr,
    positions_ptr,
    rt_stride,
    stride,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = offs // stride
    c = offs % stride

    prefix = tl.load(prefix_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    pos_off = tl.load(block_pos_offsets_ptr + c, mask=mask, other=0).to(tl.int64)
    rp = tl.load(req_pool_ptr + r, mask=mask, other=0).to(tl.int64)
    full_loc = tl.load(
        req_to_token_ptr + rp * rt_stride + prefix + pos_off, mask=mask, other=0
    ).to(tl.int64)
    swa = tl.load(full_to_swa_ptr + full_loc, mask=mask, other=-1).to(tl.int32)

    commit_len = tl.load(commit_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    swa = tl.where(c.to(tl.int64) < commit_len, swa, -1)

    tl.store(swa_loc_ptr + offs, swa, mask=mask)
    tl.store(positions_ptr + offs, prefix + pos_off, mask=mask)


@triton.jit
def _commit_inject_layout_from_window_kernel(
    cache_2d_ptr,
    positions_2d_ptr,
    full_to_swa_ptr,
    commit_lens_ptr,
    swa_loc_ptr,
    positions_ptr,
    stride,
    n,
    cache_stride0: tl.constexpr,
    cache_stride1: tl.constexpr,
    pos_stride0: tl.constexpr,
    pos_stride1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = offs // stride
    c = offs % stride

    cache_idx = r * cache_stride0 + c * cache_stride1
    pos_idx = r * pos_stride0 + c * pos_stride1
    full_loc = tl.load(cache_2d_ptr + cache_idx, mask=mask, other=0).to(tl.int64)
    swa = tl.load(full_to_swa_ptr + full_loc, mask=mask, other=-1).to(tl.int32)
    pos = tl.load(positions_2d_ptr + pos_idx, mask=mask, other=0).to(tl.int64)

    commit_len = tl.load(commit_lens_ptr + r, mask=mask, other=0).to(tl.int64)
    swa = tl.where(c.to(tl.int64) < commit_len, swa, -1)

    tl.store(swa_loc_ptr + offs, swa, mask=mask)
    tl.store(positions_ptr + offs, pos, mask=mask)


def build_commit_inject_layout_triton(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    bs = req_pool_indices.shape[0]
    n = bs * stride
    device = req_pool_indices.device

    swa_loc = torch.empty(n, dtype=torch.int32, device=device)
    positions = torch.empty(n, dtype=torch.int64, device=device)
    BLOCK = 256
    _commit_inject_layout_kernel[(triton.cdiv(n, BLOCK),)](
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa_mapping,
        commit_lens,
        swa_loc,
        positions,
        req_to_token.stride(0),
        stride,
        n,
        BLOCK=BLOCK,
    )
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


def build_commit_inject_layout_from_window_triton(
    *,
    cache_loc_2d: torch.Tensor,
    positions_2d: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    bs = cache_loc_2d.shape[0]
    n = bs * stride
    device = cache_loc_2d.device

    cache_loc_2d = cache_loc_2d.to(dtype=torch.int64)
    positions_2d = positions_2d.to(dtype=torch.int64)
    swa_loc = torch.empty(n, dtype=torch.int32, device=device)
    positions = torch.empty(n, dtype=torch.int64, device=device)
    BLOCK = 256
    _commit_inject_layout_from_window_kernel[(triton.cdiv(n, BLOCK),)](
        cache_loc_2d,
        positions_2d,
        full_to_swa_mapping,
        commit_lens,
        swa_loc,
        positions,
        stride,
        n,
        cache_loc_2d.stride(0),
        cache_loc_2d.stride(1),
        positions_2d.stride(0),
        positions_2d.stride(1),
        BLOCK=BLOCK,
    )
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)
