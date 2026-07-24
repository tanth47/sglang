from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dspark_components.kernels.dispatch import (
    inputs_on_cuda,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        DSparkScheduleConfig,
    )


class ScheduleVerifyLensTopk:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
        exact_budget: bool = False,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk(
            confidence=confidence,
            budget=budget,
            cfg=cfg,
            exact_budget=exact_budget,
        )

    @classmethod
    def triton(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
        exact_budget: bool = False,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk_triton(
            confidence=confidence,
            budget=budget,
            cfg=cfg,
            exact_budget=exact_budget,
        )

    @classmethod
    def execute_with_sps_budget(
        cls,
        *,
        confidence: torch.Tensor,
        execution_budget: int,
        sps_budget: int,
        cfg: DSparkScheduleConfig,
        exact_execution: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_on_cuda(confidence):
            return cls.triton_with_sps_budget(
                confidence=confidence,
                execution_budget=execution_budget,
                sps_budget=sps_budget,
                cfg=cfg,
                exact_execution=exact_execution,
            )
        return cls.torch_with_sps_budget(
            confidence=confidence,
            execution_budget=execution_budget,
            sps_budget=sps_budget,
            cfg=cfg,
            exact_execution=exact_execution,
        )

    @classmethod
    def torch_with_sps_budget(
        cls,
        *,
        confidence: torch.Tensor,
        execution_budget: int,
        sps_budget: int,
        cfg: DSparkScheduleConfig,
        exact_execution: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return schedule_verify_lens_topk_with_sps_budget(
            confidence=confidence,
            execution_budget=execution_budget,
            sps_budget=sps_budget,
            cfg=cfg,
            exact_execution=exact_execution,
        )

    @classmethod
    def triton_with_sps_budget(
        cls,
        *,
        confidence: torch.Tensor,
        execution_budget: int,
        sps_budget: int,
        cfg: DSparkScheduleConfig,
        exact_execution: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return schedule_verify_lens_topk_with_sps_budget_triton(
            confidence=confidence,
            execution_budget=execution_budget,
            sps_budget=sps_budget,
            cfg=cfg,
            exact_execution=exact_execution,
        )


def compute_sort_survival(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
    exact_budget: bool = False,
) -> torch.Tensor:
    return schedule_verify_lens_topk_from_survival(
        survival_probs=compute_sort_survival(confidence),
        budget=budget,
        cfg=cfg,
        exact_budget=exact_budget,
    )


def schedule_verify_lens_topk_from_survival(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
    exact_budget: bool = False,
) -> torch.Tensor:
    num_requests, gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device
    candidate_start, candidate_stop, base_verify_len = _schedule_candidate_bounds(
        gamma=gamma, cfg=cfg, exact_budget=exact_budget
    )

    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        candidate_window = survival_probs[:, candidate_start:candidate_stop]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(candidate_window.shape[1], device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = (
                torch.ones_like(candidate_window, dtype=torch.bool)
                if exact_budget
                else candidate_window >= cfg.survival_eps
            )

            flat_prob = candidate_window.reshape(-1).to(torch.float64)
            flat_request = request_index.reshape(-1)
            flat_position = position_index.reshape(-1)
            flat_valid = valid.reshape(-1)

            order = _value_independent_descending_order(
                probs=flat_prob,
                positions=flat_position,
                requests=flat_request,
                valid=flat_valid,
            )

            take = min(int(budget), num_candidates)
            chosen = order[:take]
            chosen_requests = flat_request[chosen]
            chosen_valid = flat_valid[chosen].to(torch.int64)
            selected_extra.scatter_add_(0, chosen_requests, chosen_valid)

    min_len = torch.full(
        (num_requests,), base_verify_len, dtype=torch.int64, device=device
    )
    verify_lens = min_len + selected_extra
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = torch.clamp(verify_lens, min=lower_bound, max=max_len)
    return verify_lens.to(torch.int32)


def schedule_verify_lens_topk_with_sps_budget(
    *,
    confidence: torch.Tensor,
    execution_budget: int,
    sps_budget: int,
    cfg: DSparkScheduleConfig,
    exact_execution: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return schedule_verify_lens_topk_with_sps_budget_from_survival(
        survival_probs=compute_sort_survival(confidence),
        execution_budget=execution_budget,
        sps_budget=sps_budget,
        cfg=cfg,
        exact_execution=exact_execution,
    )


def schedule_verify_lens_topk_with_sps_budget_from_survival(
    *,
    survival_probs: torch.Tensor,
    execution_budget: int,
    sps_budget: int,
    cfg: DSparkScheduleConfig,
    exact_execution: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Schedule physical and logical lengths from one shared candidate order.

    The SPS result always retains legacy filtering and candidate-window semantics.
    The execution result uses exact semantics only when ``exact_execution`` is set.
    """
    num_requests, gamma = survival_probs.shape
    device = survival_probs.device
    execution_start, execution_stop, execution_base = _schedule_candidate_bounds(
        gamma=gamma, cfg=cfg, exact_budget=exact_execution
    )
    sps_start, sps_stop, sps_base = _schedule_candidate_bounds(
        gamma=gamma, cfg=cfg, exact_budget=False
    )
    candidate_start = min(execution_start, sps_start)
    candidate_stop = max(execution_stop, sps_stop)

    execution_selected_extra = torch.zeros(
        num_requests, dtype=torch.int64, device=device
    )
    sps_selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if (execution_budget > 0 or sps_budget > 0) and candidate_stop > candidate_start:
        candidate_window = survival_probs[:, candidate_start:candidate_stop]
        request_index = (
            torch.arange(num_requests, device=device)
            .view(num_requests, 1)
            .expand_as(candidate_window)
        )
        absolute_position = (
            torch.arange(candidate_start, candidate_stop, device=device)
            .view(1, candidate_stop - candidate_start)
            .expand_as(candidate_window)
        )
        execution_valid = (absolute_position >= execution_start) & (
            absolute_position < execution_stop
        )
        if not exact_execution:
            execution_valid &= candidate_window >= cfg.survival_eps
        sps_valid = (
            (absolute_position >= sps_start)
            & (absolute_position < sps_stop)
            & (candidate_window >= cfg.survival_eps)
        )

        flat_prob = candidate_window.reshape(-1).to(torch.float64)
        flat_request = request_index.reshape(-1)
        flat_position = absolute_position.reshape(-1)
        flat_execution_valid = execution_valid.reshape(-1)
        flat_sps_valid = sps_valid.reshape(-1)
        order = _value_independent_descending_order(
            probs=flat_prob,
            positions=flat_position,
            requests=flat_request,
            valid=flat_execution_valid | flat_sps_valid,
        )
        execution_selected_extra = _selected_extra_from_shared_order(
            order=order,
            requests=flat_request,
            valid=flat_execution_valid,
            budget=execution_budget,
            num_requests=num_requests,
        )
        sps_selected_extra = _selected_extra_from_shared_order(
            order=order,
            requests=flat_request,
            valid=flat_sps_valid,
            budget=sps_budget,
            num_requests=num_requests,
        )

    physical_verify_lens = _finalize_scheduled_verify_lens(
        selected_extra=execution_selected_extra,
        base_verify_len=execution_base,
        cfg=cfg,
    )
    sps_verify_lens = _finalize_scheduled_verify_lens(
        selected_extra=sps_selected_extra,
        base_verify_len=sps_base,
        cfg=cfg,
    )
    return physical_verify_lens, sps_verify_lens


def _selected_extra_from_shared_order(
    *,
    order: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
    budget: int,
    num_requests: int,
) -> torch.Tensor:
    selected_extra = torch.zeros(
        num_requests, dtype=torch.int64, device=requests.device
    )
    if budget <= 0:
        return selected_extra
    eligible_order = order[valid[order]]
    chosen = eligible_order[: min(int(budget), eligible_order.numel())]
    selected_extra.scatter_add_(
        0,
        requests[chosen],
        torch.ones_like(chosen, dtype=torch.int64),
    )
    return selected_extra


def _finalize_scheduled_verify_lens(
    *, selected_extra: torch.Tensor, base_verify_len: int, cfg: DSparkScheduleConfig
) -> torch.Tensor:
    verify_lens = base_verify_len + selected_extra
    return torch.clamp(
        verify_lens,
        min=max(cfg.min_verify_len, 1),
        max=cfg.resolved_max_verify_len(),
    ).to(torch.int32)


def _schedule_candidate_bounds(
    *, gamma: int, cfg: DSparkScheduleConfig, exact_budget: bool
) -> tuple[int, int, int]:
    max_len = cfg.resolved_max_verify_len()
    if not exact_budget:
        return 0, min(max_len, gamma), cfg.min_verify_len

    # survival[p] scores the prefix extension from p + 1 to p + 2 tokens.
    floor = max(cfg.min_verify_len, 1)
    candidate_start = floor - 1
    candidate_stop = min(max_len - 1, gamma)
    return candidate_start, max(candidate_start, candidate_stop), floor


def _value_independent_descending_order(
    *,
    probs: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    lowest_prob = torch.full_like(probs, float("-inf"))
    canonical_prob = torch.where(torch.isnan(probs), lowest_prob, probs)
    masked_prob = torch.where(valid, canonical_prob, lowest_prob)
    num_candidates = masked_prob.numel()
    order = torch.arange(num_candidates, device=probs.device)
    order = order[torch.argsort(requests[order], stable=True)]
    order = order[torch.argsort(positions[order], stable=True)]
    order = order[torch.argsort(-masked_prob[order], stable=True)]
    return order


@triton.jit
def _schedule_topk_prep_kernel(
    confidence_ptr,
    survival_ptr,
    selected_extra_ptr,
    gamma,
    cols,
    candidate_start,
    G_P2: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.arange(0, G_P2)
    conf = tl.load(
        confidence_ptr + row.to(tl.int64) * gamma + g, mask=g < gamma, other=1.0
    ).to(tl.float32)
    surv = tl.cumprod(conf, axis=0)
    candidate_col = g - candidate_start
    candidate_mask = (candidate_col >= 0) & (candidate_col < cols)
    tl.store(
        survival_ptr + row.to(tl.int64) * cols + candidate_col,
        surv,
        mask=candidate_mask,
    )
    tl.store(selected_extra_ptr + row, 0)


@triton.jit
def _schedule_topk_dual_prep_kernel(
    confidence_ptr,
    survival_ptr,
    execution_selected_extra_ptr,
    sps_selected_extra_ptr,
    gamma,
    cols,
    candidate_start,
    G_P2: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.arange(0, G_P2)
    conf = tl.load(
        confidence_ptr + row.to(tl.int64) * gamma + g, mask=g < gamma, other=1.0
    ).to(tl.float32)
    surv = tl.cumprod(conf, axis=0)
    candidate_col = g - candidate_start
    candidate_mask = (candidate_col >= 0) & (candidate_col < cols)
    tl.store(
        survival_ptr + row.to(tl.int64) * cols + candidate_col,
        surv,
        mask=candidate_mask,
    )
    tl.store(execution_selected_extra_ptr + row, 0)
    tl.store(sps_selected_extra_ptr + row, 0)


@triton.jit
def _schedule_topk_finalize_kernel(
    selected_extra_ptr,
    out_ptr,
    min_verify_len,
    lower_bound,
    max_len,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    extra = tl.load(selected_extra_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = min_verify_len + extra
    lens = tl.maximum(lens, lower_bound)
    lens = tl.minimum(lens, max_len)
    tl.store(out_ptr + offs, lens, mask=mask)


@triton.jit
def _schedule_topk_dual_finalize_kernel(
    execution_selected_extra_ptr,
    sps_selected_extra_ptr,
    physical_out_ptr,
    sps_out_ptr,
    execution_base_verify_len,
    sps_base_verify_len,
    lower_bound,
    max_len,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    execution_extra = tl.load(
        execution_selected_extra_ptr + offs, mask=mask, other=0
    ).to(tl.int32)
    sps_extra = tl.load(sps_selected_extra_ptr + offs, mask=mask, other=0).to(tl.int32)
    physical_lens = execution_base_verify_len + execution_extra
    physical_lens = tl.maximum(physical_lens, lower_bound)
    physical_lens = tl.minimum(physical_lens, max_len)
    sps_lens = sps_base_verify_len + sps_extra
    sps_lens = tl.maximum(sps_lens, lower_bound)
    sps_lens = tl.minimum(sps_lens, max_len)
    tl.store(physical_out_ptr + offs, physical_lens, mask=mask)
    tl.store(sps_out_ptr + offs, sps_lens, mask=mask)


@triton.jit
def _schedule_topk_selected_extra_kernel(
    survival_ptr,
    selected_extra_ptr,
    budget,
    cols,
    n,
    survival_eps,
    BLOCK_C: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < n
    r = c // cols
    p = c % cols
    sp = tl.load(survival_ptr + c, mask=cmask, other=0.0)
    sp = tl.where(sp == sp, sp, float("-inf"))
    valid_c = sp >= survival_eps
    mp = tl.where(valid_c, sp, float("-inf"))
    rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    for cp0 in range(0, n, BLOCK_CP):
        cp = cp0 + tl.arange(0, BLOCK_CP)
        cpmask = cp < n
        rp = cp // cols
        pp = cp % cols
        spp = tl.load(survival_ptr + cp, mask=cpmask, other=0.0)
        spp = tl.where(spp == spp, spp, float("-inf"))
        validp = spp >= survival_eps
        mpp = tl.where(validp, spp, float("-inf"))
        gt = mpp[None, :] > mp[:, None]
        eq = mpp[None, :] == mp[:, None]
        pos_lt = pp[None, :] < p[:, None]
        pos_eq = pp[None, :] == p[:, None]
        req_lt = rp[None, :] < r[:, None]
        before = gt | (eq & (pos_lt | (pos_eq & req_lt)))
        before = before & cpmask[None, :]
        rank += tl.sum(before.to(tl.int32), axis=1)
    selected = valid_c & (rank < budget)
    tl.atomic_add(selected_extra_ptr + r, selected.to(tl.int32), mask=cmask)


@triton.jit
def _schedule_topk_dual_selected_extra_kernel(
    survival_ptr,
    execution_selected_extra_ptr,
    sps_selected_extra_ptr,
    execution_budget,
    sps_budget,
    cols,
    n,
    candidate_start,
    execution_start,
    execution_stop,
    sps_start,
    sps_stop,
    survival_eps,
    EXACT_EXECUTION: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < n
    r = c // cols
    p = c % cols + candidate_start
    sp = tl.load(survival_ptr + c, mask=cmask, other=0.0)
    sp = tl.where(sp == sp, sp, float("-inf"))
    execution_in_c = cmask & (p >= execution_start) & (p < execution_stop)
    if EXACT_EXECUTION:
        execution_valid_c = execution_in_c
    else:
        execution_valid_c = execution_in_c & (sp >= survival_eps)
    sps_valid_c = cmask & (p >= sps_start) & (p < sps_stop) & (sp >= survival_eps)
    execution_mp = tl.where(execution_valid_c, sp, float("-inf"))
    sps_mp = tl.where(sps_valid_c, sp, float("-inf"))
    execution_rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    sps_rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    for cp0 in range(0, n, BLOCK_CP):
        cp = cp0 + tl.arange(0, BLOCK_CP)
        cpmask = cp < n
        rp = cp // cols
        pp = cp % cols + candidate_start
        spp = tl.load(survival_ptr + cp, mask=cpmask, other=0.0)
        spp = tl.where(spp == spp, spp, float("-inf"))
        execution_in_p = cpmask & (pp >= execution_start) & (pp < execution_stop)
        if EXACT_EXECUTION:
            execution_valid_p = execution_in_p
        else:
            execution_valid_p = execution_in_p & (spp >= survival_eps)
        sps_valid_p = (
            cpmask & (pp >= sps_start) & (pp < sps_stop) & (spp >= survival_eps)
        )
        pos_lt = pp[None, :] < p[:, None]
        pos_eq = pp[None, :] == p[:, None]
        req_lt = rp[None, :] < r[:, None]
        tie_before = pos_lt | (pos_eq & req_lt)

        execution_mpp = tl.where(execution_valid_p, spp, float("-inf"))
        execution_before = (execution_mpp[None, :] > execution_mp[:, None]) | (
            (execution_mpp[None, :] == execution_mp[:, None]) & tie_before
        )
        execution_before &= execution_valid_p[None, :]
        execution_rank += tl.sum(execution_before.to(tl.int32), axis=1)

        sps_mpp = tl.where(sps_valid_p, spp, float("-inf"))
        sps_before = (sps_mpp[None, :] > sps_mp[:, None]) | (
            (sps_mpp[None, :] == sps_mp[:, None]) & tie_before
        )
        sps_before &= sps_valid_p[None, :]
        sps_rank += tl.sum(sps_before.to(tl.int32), axis=1)

    execution_selected = execution_valid_c & (execution_rank < execution_budget)
    sps_selected = sps_valid_c & (sps_rank < sps_budget)
    tl.atomic_add(
        execution_selected_extra_ptr + r,
        execution_selected.to(tl.int32),
        mask=cmask,
    )
    tl.atomic_add(sps_selected_extra_ptr + r, sps_selected.to(tl.int32), mask=cmask)


def schedule_verify_lens_topk_triton(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
    exact_budget: bool = False,
) -> torch.Tensor:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    candidate_start, candidate_stop, base_verify_len = _schedule_candidate_bounds(
        gamma=gamma, cfg=cfg, exact_budget=exact_budget
    )
    cols = candidate_stop - candidate_start
    n = num_requests * cols

    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    _schedule_topk_prep_kernel[(num_requests,)](
        confidence.contiguous(),
        survival,
        selected_extra,
        gamma,
        cols,
        candidate_start,
        G_P2=triton.next_power_of_2(max(gamma, 1)),
    )
    if budget > 0 and n > 0:
        BLOCK_C = 64
        BLOCK_CP = 256
        grid = (triton.cdiv(n, BLOCK_C),)
        _schedule_topk_selected_extra_kernel[grid](
            survival,
            selected_extra,
            int(budget),
            cols,
            n,
            float("-inf") if exact_budget else float(cfg.survival_eps),
            BLOCK_C=BLOCK_C,
            BLOCK_CP=BLOCK_CP,
        )

    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    BLOCK = 256
    _schedule_topk_finalize_kernel[(triton.cdiv(num_requests, BLOCK),)](
        selected_extra,
        verify_lens,
        int(base_verify_len),
        max(cfg.min_verify_len, 1),
        int(max_len),
        num_requests,
        BLOCK=BLOCK,
    )
    return verify_lens


def schedule_verify_lens_topk_with_sps_budget_triton(
    *,
    confidence: torch.Tensor,
    execution_budget: int,
    sps_budget: int,
    cfg: DSparkScheduleConfig,
    exact_execution: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    execution_start, execution_stop, execution_base = _schedule_candidate_bounds(
        gamma=gamma, cfg=cfg, exact_budget=exact_execution
    )
    sps_start, sps_stop, sps_base = _schedule_candidate_bounds(
        gamma=gamma, cfg=cfg, exact_budget=False
    )
    candidate_start = min(execution_start, sps_start)
    candidate_stop = max(execution_stop, sps_stop)
    cols = candidate_stop - candidate_start
    n = num_requests * cols

    execution_selected_extra = torch.empty(
        num_requests, dtype=torch.int32, device=device
    )
    sps_selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    _schedule_topk_dual_prep_kernel[(num_requests,)](
        confidence.contiguous(),
        survival,
        execution_selected_extra,
        sps_selected_extra,
        gamma,
        cols,
        candidate_start,
        G_P2=triton.next_power_of_2(max(gamma, 1)),
    )
    if (execution_budget > 0 or sps_budget > 0) and n > 0:
        BLOCK_C = 64
        BLOCK_CP = 256
        grid = (triton.cdiv(n, BLOCK_C),)
        _schedule_topk_dual_selected_extra_kernel[grid](
            survival,
            execution_selected_extra,
            sps_selected_extra,
            int(execution_budget),
            int(sps_budget),
            cols,
            n,
            candidate_start,
            execution_start,
            execution_stop,
            sps_start,
            sps_stop,
            float(cfg.survival_eps),
            EXACT_EXECUTION=exact_execution,
            BLOCK_C=BLOCK_C,
            BLOCK_CP=BLOCK_CP,
        )

    physical_verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    sps_verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    BLOCK = 256
    _schedule_topk_dual_finalize_kernel[(triton.cdiv(num_requests, BLOCK),)](
        execution_selected_extra,
        sps_selected_extra,
        physical_verify_lens,
        sps_verify_lens,
        int(execution_base),
        int(sps_base),
        max(cfg.min_verify_len, 1),
        int(max_len),
        num_requests,
        BLOCK=BLOCK,
    )
    return physical_verify_lens, sps_verify_lens
