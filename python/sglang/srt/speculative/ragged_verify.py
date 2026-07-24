from __future__ import annotations

import bisect
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.environ import envs


class RaggedVerifyMode(str, Enum):
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"


DSA_TARGET_VERIFY_PRE_TOPK_GRAPH = "dsa_pre_topk"
DSA_TARGET_VERIFY_POST_TOPK_GRAPH = "dsa_post_topk"
DSA_TARGET_VERIFY_MIXED_TRANSITION_REJECT = (
    "rocm_dsa_target_verify_index_topk_mixed_transition"
)
DSA_TARGET_VERIFY_BATCH_MIXED_REGIONS_REJECT = (
    "rocm_dsa_target_verify_index_topk_batch_mixed_regions"
)
DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT = (
    "rocm_dsa_target_verify_index_topk_window_transition"
)
DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT = (
    "rocm_dsa_target_verify_post_topk_no_capture_contract"
)
DSA_TARGET_VERIFY_POST_TOPK_ABOVE_CAPTURE_REJECT = (
    "rocm_dsa_target_verify_post_topk_above_capture_contract"
)
DSA_TARGET_VERIFY_POST_TOPK_CAPTURE_MISMATCH_REJECT = (
    "rocm_dsa_target_verify_post_topk_capture_seq_len_mismatch"
)


def read_ragged_verify_mode() -> RaggedVerifyMode:
    value = envs.SGLANG_RAGGED_VERIFY_MODE.get()
    for mode in RaggedVerifyMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid SGLANG_RAGGED_VERIFY_MODE={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in RaggedVerifyMode)}"
    )


def ragged_verify_compact_enabled() -> bool:
    return read_ragged_verify_mode() == RaggedVerifyMode.COMPACT


def build_ragged_verify_token_buckets(
    *,
    capture_bs: Sequence[int],
    num_tokens_per_req: int,
    fine_grained_min_tokens: Optional[int] = None,
    fine_grained_max_tokens: Optional[int] = None,
) -> list[int]:
    buckets = {int(bs) * num_tokens_per_req for bs in capture_bs}
    fine_grained_max = (
        envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MAX_TOKENS.get()
        if fine_grained_max_tokens is None
        else int(fine_grained_max_tokens)
    )
    if fine_grained_max > 0:
        fine_grained_min = max(
            1,
            (
                envs.SGLANG_RAGGED_VERIFY_FINE_GRAINED_GRAPH_MIN_TOKENS.get()
                if fine_grained_min_tokens is None
                else int(fine_grained_min_tokens)
            ),
        )
        max_bucket = max(buckets)
        upper = min(fine_grained_max, max_bucket)
        if fine_grained_min <= upper:
            buckets.update(range(fine_grained_min, upper + 1))
    buckets = sorted(buckets)
    assert buckets and buckets[0] > 0, f"{buckets=}"
    return buckets


def round_up_grid(total: int, grid: Sequence[int]) -> int:
    if not grid:
        raise ValueError("round_up_grid requires a non-empty grid")
    if total > grid[-1]:
        raise ValueError(
            f"total {total} exceeds max grid tier {grid[-1]}; "
            "the caller must reject this batch before selecting a graph tier"
        )
    index = bisect.bisect_left(grid, total)
    return grid[index]


def classify_dsa_target_verify_graph_regime(
    *,
    seq_lens_cpu: Sequence[int],
    verify_lens_cpu: Sequence[int],
    dsa_index_topk: int,
    post_topk_guard_tokens: int = 0,
    post_topk_capture_seq_len: Optional[int] = None,
) -> Optional[str]:
    """Classify target-verify windows by DSA sparse-index regime.

    DSA changes metadata semantics around ``index_topk`` because attention sees
    clipped DSA sequence lengths while indexer/page-table metadata still maps
    real KV positions. A graph captured below the boundary should not be replayed
    for post-boundary verify. Mixed windows straddle the boundary inside one
    verify block and should stay on the eager path until there is a dedicated
    transition contract.
    """
    if dsa_index_topk <= 0:
        return None

    windows = [
        (int(seq_len), int(verify_len))
        for seq_len, verify_len in zip(seq_lens_cpu, verify_lens_cpu, strict=True)
        if int(verify_len) > 0
    ]
    if not windows:
        return DSA_TARGET_VERIFY_PRE_TOPK_GRAPH

    if all(seq_len + verify_len < dsa_index_topk for seq_len, verify_len in windows):
        return DSA_TARGET_VERIFY_PRE_TOPK_GRAPH

    if post_topk_capture_seq_len is None:
        # TODO(GLM/ROCm DSA): add a verify-specific post-topk graph contract or
        # kernel instead of replaying the generic captured target-verify graph.
        # MI350 validation showed memory faults for post-topk replay without a
        # bounded capture contract, so mixed/post-topk windows stay eager unless
        # an explicit experimental capture contract is supplied by the caller.
        return None

    post_topk_graph_threshold = dsa_index_topk + max(0, int(post_topk_guard_tokens))
    if all(
        seq_len + 1 >= post_topk_graph_threshold
        and seq_len == post_topk_capture_seq_len
        for seq_len, _ in windows
    ):
        return DSA_TARGET_VERIFY_POST_TOPK_GRAPH
    return None


def classify_dsa_target_verify_graph_reject_reason(
    *,
    seq_lens_cpu: Sequence[int],
    verify_lens_cpu: Sequence[int],
    dsa_index_topk: int,
    post_topk_guard_tokens: int = 0,
    post_topk_capture_seq_len: Optional[int] = None,
) -> Optional[str]:
    """Return a precise ROCm DSA target-verify graph reject reason.

    ``classify_dsa_target_verify_graph_regime`` intentionally keeps the public
    admission result compact: either a safe graph label or ``None``. For
    observability and graph-admission fast-fail paths, split the ``None`` case
    into mixed/transition windows versus fully post-topk windows that simply do
    not have a validated capture contract yet.
    """
    regime = classify_dsa_target_verify_graph_regime(
        seq_lens_cpu=seq_lens_cpu,
        verify_lens_cpu=verify_lens_cpu,
        dsa_index_topk=dsa_index_topk,
        post_topk_guard_tokens=post_topk_guard_tokens,
        post_topk_capture_seq_len=post_topk_capture_seq_len,
    )
    if regime is not None or dsa_index_topk <= 0:
        return None

    windows = [
        (int(seq_len), int(verify_len))
        for seq_len, verify_len in zip(seq_lens_cpu, verify_lens_cpu, strict=True)
        if int(verify_len) > 0
    ]
    if not windows:
        return None

    post_topk_graph_threshold = dsa_index_topk + max(0, int(post_topk_guard_tokens))
    fully_post_topk = all(
        seq_len + 1 >= post_topk_graph_threshold for seq_len, _ in windows
    )
    if fully_post_topk:
        if post_topk_capture_seq_len is None:
            return DSA_TARGET_VERIFY_POST_TOPK_NO_CAPTURE_REJECT
        if any(seq_len > post_topk_capture_seq_len for seq_len, _ in windows):
            return DSA_TARGET_VERIFY_POST_TOPK_ABOVE_CAPTURE_REJECT
        return DSA_TARGET_VERIFY_POST_TOPK_CAPTURE_MISMATCH_REJECT

    any_pre_topk = False
    any_post_topk = False
    for seq_len, verify_len in windows:
        if seq_len + verify_len < dsa_index_topk:
            any_pre_topk = True
            continue
        if seq_len + 1 >= post_topk_graph_threshold:
            any_post_topk = True
            continue
        return DSA_TARGET_VERIFY_WINDOW_TRANSITION_REJECT

    if any_pre_topk and any_post_topk:
        return DSA_TARGET_VERIFY_BATCH_MIXED_REGIONS_REJECT
    return DSA_TARGET_VERIFY_MIXED_TRANSITION_REJECT


class RaggedVerifyLayout(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    graph_num_tokens: int
    extend_start_loc: torch.Tensor
    qo_indptr_device: torch.Tensor
    verify_lens_cpu: Optional[list[int]] = None
    total_verify_tokens: Optional[int] = None
    qo_indptr_host: Optional[torch.Tensor] = None
    kv_indptr_host: Optional[torch.Tensor] = None
    kv_lens_host: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None
    max_kv_len: Optional[int] = None
    # Logical pre-alignment SPS decision; physical layout code must not read it.
    sps_verify_lens: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.sps_verify_lens is not None:
            if self.sps_verify_lens.shape != self.verify_lens.shape:
                raise ValueError(
                    "sps_verify_lens must match the physical verify_lens shape, got "
                    f"{tuple(self.sps_verify_lens.shape)} and "
                    f"{tuple(self.verify_lens.shape)}"
                )
            if self.sps_verify_lens.dtype is not torch.int32:
                raise ValueError(
                    "sps_verify_lens must use torch.int32, got "
                    f"{self.sps_verify_lens.dtype}"
                )
            if self.sps_verify_lens.device != self.verify_lens.device:
                raise ValueError(
                    "sps_verify_lens and verify_lens must share a device, got "
                    f"{self.sps_verify_lens.device} and {self.verify_lens.device}"
                )
        if self.verify_lens_cpu is None:
            return
        if not self.verify_lens_cpu:
            raise ValueError("RaggedVerifyLayout requires at least one request")
        if min(self.verify_lens_cpu) < 1:
            raise ValueError(
                f"every request must verify the anchor (verify_len >= 1), got "
                f"{self.verify_lens_cpu}"
            )
        if self.total_verify_tokens != sum(self.verify_lens_cpu):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} != "
                f"sum(verify_lens_cpu) {sum(self.verify_lens_cpu)}"
            )
        if not (self.total_verify_tokens <= self.graph_num_tokens):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} exceeds "
                f"graph_num_tokens {self.graph_num_tokens}"
            )

    @property
    def bs(self) -> int:
        return int(self.verify_lens.shape[0])

    @classmethod
    def _assemble_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        verify_lens_cpu: Optional[list[int]] = None,
        total_verify_tokens: Optional[int] = None,
        sps_verify_lens: Optional[torch.Tensor] = None,
    ) -> RaggedVerifyLayout:
        from sglang.srt.speculative.ragged_verify_kernels import (
            BuildQoIndptr,
        )

        verify_lens = verify_lens.to(torch.int32)
        indptr = BuildQoIndptr.execute(verify_lens=verify_lens)
        return cls(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            extend_start_loc=indptr.extend_start_loc,
            qo_indptr_device=indptr.qo_indptr,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
            sps_verify_lens=sps_verify_lens,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        verify_lens_cpu: list[int],
        total_verify_tokens: int,
        graph_num_tokens: int,
        device: torch.device,
        sps_verify_lens: Optional[torch.Tensor] = None,
    ) -> RaggedVerifyLayout:
        verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32, device=device)
        return cls._assemble_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
            sps_verify_lens=sps_verify_lens,
        )

    @classmethod
    def from_verify_lens_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        sps_verify_lens: Optional[torch.Tensor] = None,
    ) -> RaggedVerifyLayout:
        return cls._assemble_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            sps_verify_lens=sps_verify_lens,
        )

    @classmethod
    def from_verify_lens(
        cls,
        *,
        verify_lens_cpu: Sequence[int],
        device: torch.device,
        grid: Sequence[int],
        graph_num_tokens_floor: int = 0,
        sps_verify_lens: Optional[torch.Tensor] = None,
    ) -> RaggedVerifyLayout:
        verify_lens_list = [int(v) for v in verify_lens_cpu]
        total_verify_tokens = sum(verify_lens_list)
        bucket_input = max(total_verify_tokens, graph_num_tokens_floor)
        graph_num_tokens = round_up_grid(total=bucket_input, grid=grid)

        return cls._assemble(
            verify_lens_cpu=verify_lens_list,
            total_verify_tokens=total_verify_tokens,
            graph_num_tokens=graph_num_tokens,
            device=device,
            sps_verify_lens=sps_verify_lens,
        )

    def padded_to_bucket(self, *, padded_bs: int) -> RaggedVerifyLayout:
        from sglang.srt.speculative.ragged_verify_kernels import (
            PaddedToBucket,
        )

        padded = PaddedToBucket.execute(
            verify_lens=self.verify_lens,
            graph_num_tokens=self.graph_num_tokens,
            bs=self.bs,
            padded_bs=padded_bs,
        )

        return RaggedVerifyLayout._assemble_device(
            verify_lens=padded,
            graph_num_tokens=self.graph_num_tokens,
            total_verify_tokens=self.graph_num_tokens,
        )


def materialize_verify_lens_cpu(layout: RaggedVerifyLayout) -> list[int]:
    """Return host verify lengths, syncing from device only for layouts that were
    intentionally built without a host mirror."""
    if layout.verify_lens_cpu is not None:
        return [int(x) for x in layout.verify_lens_cpu]
    return [int(x) for x in layout.verify_lens.detach().cpu().tolist()]


def materialize_total_verify_tokens(layout: RaggedVerifyLayout) -> int:
    if layout.total_verify_tokens is not None:
        return int(layout.total_verify_tokens)
    return sum(materialize_verify_lens_cpu(layout))


def required_padded_verify_slots(
    layout: RaggedVerifyLayout, *, num_tokens_per_req: int
) -> int:
    """Slots needed to pad a graph tier without enlarging a live request."""
    if num_tokens_per_req < 1:
        raise ValueError("num_tokens_per_req must be positive")
    slack_tokens = layout.graph_num_tokens - materialize_total_verify_tokens(layout)
    if slack_tokens <= 0:
        return layout.bs
    padding_slots = (slack_tokens + num_tokens_per_req - 1) // num_tokens_per_req
    return layout.bs + padding_slots


def is_static_full_verify_layout(
    layout: RaggedVerifyLayout, *, num_tokens_per_req: int
) -> bool:
    """Whether a compact layout is equivalent to static full-block verify.

    DSpark compact mode still attaches a RaggedVerifyLayout when every request
    verifies the full block. Attention backends do not need ragged metadata for
    that case; the static target-verify graph has the same q shape and avoids
    experimental compact-ragged graph metadata paths.
    """
    verify_lens_cpu = materialize_verify_lens_cpu(layout)
    if not verify_lens_cpu:
        return False
    if any(verify_len != num_tokens_per_req for verify_len in verify_lens_cpu):
        return False
    total_verify_tokens = materialize_total_verify_tokens(layout)
    full_width_tokens = len(verify_lens_cpu) * num_tokens_per_req
    return (
        total_verify_tokens == full_width_tokens
        and layout.graph_num_tokens >= total_verify_tokens
        and layout.graph_num_tokens % num_tokens_per_req == 0
    )


def build_capture_verify_lens(
    *,
    num_tokens: int,
    num_slots: int,
    num_draft_tokens: int,
) -> list[int]:
    if num_slots < 1 or num_tokens < num_slots:
        raise ValueError(
            f"capture layout needs 1 <= num_slots <= num_tokens, got "
            f"num_slots={num_slots}, num_tokens={num_tokens}"
        )
    if num_tokens > num_slots * num_draft_tokens:
        raise ValueError(
            f"capture layout cannot pack num_tokens={num_tokens} into "
            f"{num_slots} rows of at most {num_draft_tokens} tokens"
        )
    base = num_tokens // num_slots
    rem = num_tokens - base * num_slots
    return [base + 1] * rem + [base] * (num_slots - rem)


def resolve_ragged_verify_layout(forward_batch) -> Optional[RaggedVerifyLayout]:
    """Layout riding the batch's spec input, or None. Tolerates the runner's
    ad-hoc replay batch views, which may not carry spec_info at all."""
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is None:
        return None
    return spec_info.ragged_verify_layout


class RaggedTargetVerifyGeometry(msgspec.Struct):
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: Optional[int]


def build_ragged_target_verify_geometry(
    *,
    seq_lens: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> RaggedTargetVerifyGeometry:
    cache_seqlens_int32 = (seq_lens + layout.verify_lens).to(torch.int32)
    cu_seqlens_q = layout.qo_indptr_device.to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seq_len_q = max(materialize_verify_lens_cpu(layout))
    return RaggedTargetVerifyGeometry(
        cache_seqlens_int32=cache_seqlens_int32,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
    )


def expand_target_verify_page_table(
    *,
    page_table: torch.Tensor,
    verify_lens: torch.Tensor,
    output_num_tokens: int,
) -> torch.Tensor:
    """Expand request-keyed page rows into query-token order.

    ``output_num_tokens`` is an explicit graph-tier size, so repeat_interleave
    does not need to read the device-side sum of ``verify_lens`` on the host.
    """
    if page_table.shape[0] != verify_lens.shape[0]:
        raise ValueError(
            f"page-table rows ({page_table.shape[0]}) must match verify_lens "
            f"({verify_lens.shape[0]})"
        )
    return torch.repeat_interleave(
        page_table, repeats=verify_lens, dim=0, output_size=output_num_tokens
    )


def compute_target_verify_graph_key(
    *,
    bs: int,
    num_draft_tokens: int,
    ragged_layout: Optional[RaggedVerifyLayout],
) -> Tuple[int, int]:
    num_tokens_full_block = num_draft_tokens * bs
    if ragged_layout is None:
        return bs, num_tokens_full_block
    graph_num_tokens = ragged_layout.graph_num_tokens
    assert graph_num_tokens <= num_tokens_full_block, (
        f"ragged verify graph_num_tokens={graph_num_tokens} exceeds full block "
        f"num_draft*bs={num_tokens_full_block}"
    )
    total_verify_tokens = ragged_layout.total_verify_tokens
    if total_verify_tokens is not None:
        assert total_verify_tokens <= graph_num_tokens, (
            f"ragged verify total_verify_tokens={total_verify_tokens} exceeds the "
            f"round-up bucket graph_num_tokens={graph_num_tokens}"
        )
    return graph_num_tokens, graph_num_tokens


class VerifyExtendLengths(msgspec.Struct, frozen=True):
    seq_lens_extended: torch.Tensor
    seq_lens_cpu_extended: List[int]
    extend_seq_lens_cpu: List[int]
    num_tokens: int
    extend_start_loc: Optional[torch.Tensor]


def compute_uniform_extend_lengths(
    *,
    seq_lens: torch.Tensor,
    seq_lens_cpu: List[int],
    extend_len: int,
) -> VerifyExtendLengths:
    batch_size = len(seq_lens_cpu)
    seq_lens_extended = seq_lens + extend_len
    seq_lens_cpu_extended = [x + extend_len for x in seq_lens_cpu]
    extend_seq_lens_cpu = [extend_len] * batch_size
    num_tokens = extend_len * batch_size
    return VerifyExtendLengths(
        seq_lens_extended=seq_lens_extended,
        seq_lens_cpu_extended=seq_lens_cpu_extended,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        num_tokens=num_tokens,
        extend_start_loc=None,
    )


def compute_ragged_extend_lengths(
    *,
    seq_lens: torch.Tensor,
    seq_lens_cpu: List[int],
    ragged_layout: RaggedVerifyLayout,
) -> VerifyExtendLengths:
    extend_seq_lens_cpu = materialize_verify_lens_cpu(ragged_layout)
    seq_lens_extended = seq_lens + ragged_layout.verify_lens
    seq_lens_cpu_extended = [
        raw + length for raw, length in zip(seq_lens_cpu, extend_seq_lens_cpu)
    ]
    num_tokens = materialize_total_verify_tokens(ragged_layout)
    extend_start_loc = ragged_layout.extend_start_loc
    return VerifyExtendLengths(
        seq_lens_extended=seq_lens_extended,
        seq_lens_cpu_extended=seq_lens_cpu_extended,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        num_tokens=num_tokens,
        extend_start_loc=extend_start_loc,
    )
