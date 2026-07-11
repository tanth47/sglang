from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_PADDED_TO_BUCKET.get()


class PaddedToBucket:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
    ) -> torch.Tensor:
        return pad_verify_lens_to_bucket_triton(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
        )


class PadVerifyLensWithinRows:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        verify_lens = kwargs["verify_lens"]
        if _KERNEL_IMPL == "torch" or not verify_lens.is_cuda:
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
        max_verify_len: int,
    ) -> torch.Tensor:
        return pad_verify_lens_within_rows(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
            max_verify_len=max_verify_len,
        )

    @classmethod
    def triton(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        bs: int,
        padded_bs: int,
        max_verify_len: int,
    ) -> torch.Tensor:
        return pad_verify_lens_within_rows_triton(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            bs=bs,
            padded_bs=padded_bs,
            max_verify_len=max_verify_len,
        )


def pad_verify_lens_to_bucket(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
) -> torch.Tensor:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    device = verify_lens.device
    num_pad_reqs = padded_bs - bs
    padded = verify_lens.to(torch.int32)
    leftover = graph_num_tokens - padded.to(torch.int64).sum()
    if num_pad_reqs > 0:
        base = leftover // num_pad_reqs
        rem = leftover - base * num_pad_reqs
        pad_block = base + (
            torch.arange(num_pad_reqs, device=device, dtype=torch.int64) < rem
        )
        padded = torch.cat([padded, pad_block.to(torch.int32)])
    else:
        padded = padded.clone()
        padded[-1] = (padded[-1].to(torch.int64) + leftover).to(torch.int32)
    return padded


def _validate_within_rows_args(
    *,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
    max_verify_len: int,
) -> None:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    assert max_verify_len >= 1, f"max_verify_len must be >= 1, got {max_verify_len}"
    if graph_num_tokens > padded_bs * max_verify_len:
        raise ValueError(
            f"graph_num_tokens={graph_num_tokens} cannot fit into "
            f"{padded_bs} rows of width {max_verify_len}"
        )


def pad_verify_lens_within_rows(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
    max_verify_len: int,
) -> torch.Tensor:
    _validate_within_rows_args(
        graph_num_tokens=graph_num_tokens,
        bs=bs,
        padded_bs=padded_bs,
        max_verify_len=max_verify_len,
    )
    device = verify_lens.device
    verify_lens = verify_lens.to(torch.int32)
    padded = torch.zeros(padded_bs, dtype=torch.int32, device=device)
    padded[:bs].copy_(verify_lens[:bs])

    cap = torch.clamp(int(max_verify_len) - padded.to(torch.int64), min=0)
    row = torch.arange(padded_bs, dtype=torch.int64, device=device)
    is_real = row < int(bs)
    num_dummy_rows = int(padded_bs) - int(bs)

    real_cap = torch.where(is_real, cap, torch.zeros_like(cap))
    real_prior = torch.cumsum(real_cap, dim=0) - real_cap
    dummy_prior = (row - int(bs)) * int(max_verify_len)
    prior = torch.where(
        is_real,
        int(num_dummy_rows) * int(max_verify_len) + real_prior,
        dummy_prior,
    )
    extra = int(graph_num_tokens) - padded.to(torch.int64).sum()
    add = torch.minimum(torch.clamp(extra - prior, min=0), cap)
    return (padded.to(torch.int64) + add).to(torch.int32)


@triton.jit
def _padded_to_bucket_kernel(
    verify_lens_ptr,
    out_ptr,
    bs,
    padded_bs,
    graph_num_tokens,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < padded_bs
    is_real = idx < bs
    vl = tl.load(verify_lens_ptr + idx, mask=is_real, other=0).to(tl.int64)
    leftover = graph_num_tokens - tl.sum(vl)
    num_pad = padded_bs - bs
    num_pad_safe = tl.maximum(num_pad, 1)
    base = leftover // num_pad_safe
    rem = leftover - base * num_pad_safe
    pad_len = base + tl.where((idx - bs) < rem, 1, 0)
    final = tl.where(is_real, vl, pad_len)
    final = final + tl.where((num_pad == 0) & (idx == bs - 1), leftover, 0)
    tl.store(out_ptr + idx, final.to(tl.int32), mask=valid)


@triton.jit
def _pad_within_rows_kernel(
    verify_lens_ptr,
    out_ptr,
    bs,
    padded_bs,
    graph_num_tokens,
    max_verify_len,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    valid = idx < padded_bs
    is_real = idx < bs
    raw = tl.load(verify_lens_ptr + idx, mask=is_real, other=0).to(tl.int64)
    cap = tl.maximum(max_verify_len - raw, 0)
    real_cap = tl.where(is_real, cap, 0)
    current = tl.sum(tl.where(is_real, raw, 0))
    extra = graph_num_tokens - current
    real_prior = tl.cumsum(real_cap, axis=0) - real_cap
    num_dummy_rows = padded_bs - bs
    dummy_prior = (idx - bs) * max_verify_len
    prior = tl.where(is_real, num_dummy_rows * max_verify_len + real_prior, dummy_prior)
    add = tl.minimum(tl.maximum(extra - prior, 0), cap)
    final = raw + add
    tl.store(out_ptr + idx, final.to(tl.int32), mask=valid)


def pad_verify_lens_to_bucket_triton(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
) -> torch.Tensor:
    assert padded_bs >= bs, (
        f"padded_bs {padded_bs} < bs {bs}: the captured tier cannot hold this "
        "batch's requests"
    )
    device = verify_lens.device
    verify_lens = verify_lens.to(torch.int32).contiguous()
    out = torch.empty(padded_bs, dtype=torch.int32, device=device)
    BLOCK = triton.next_power_of_2(max(padded_bs, 1))
    _padded_to_bucket_kernel[(1,)](
        verify_lens,
        out,
        bs,
        padded_bs,
        graph_num_tokens,
        BLOCK=BLOCK,
    )
    return out


def pad_verify_lens_within_rows_triton(
    *,
    verify_lens: torch.Tensor,
    graph_num_tokens: int,
    bs: int,
    padded_bs: int,
    max_verify_len: int,
) -> torch.Tensor:
    _validate_within_rows_args(
        graph_num_tokens=graph_num_tokens,
        bs=bs,
        padded_bs=padded_bs,
        max_verify_len=max_verify_len,
    )
    verify_lens = verify_lens.to(torch.int32).contiguous()
    out = torch.empty(padded_bs, dtype=torch.int32, device=verify_lens.device)
    BLOCK = triton.next_power_of_2(max(padded_bs, 1))
    _pad_within_rows_kernel[(1,)](
        verify_lens,
        out,
        bs,
        padded_bs,
        graph_num_tokens,
        max_verify_len,
        BLOCK=BLOCK,
    )
    return out
