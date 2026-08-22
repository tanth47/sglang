from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import accumulate
from typing import Sequence, Tuple


class MixedSpecMode(Enum):
    """Execution mode for a batch that mixes prefill with speculative requests."""

    TARGET_ONLY = auto()


@dataclass(frozen=True)
class MixedSpecBatchInfo:
    """Authoritative host layout for a mixed prefill/speculative batch.

    The first implementation intentionally gives every running speculative
    request one target-only row. A future mixed-verify implementation can add a
    VERIFY mode with a wider query length without changing the partition API.
    """

    mode: MixedSpecMode
    prefill_bs: int
    decode_bs: int
    query_lens: Tuple[int, ...]
    query_start_loc: Tuple[int, ...]

    @classmethod
    def target_only(
        cls, prefill_query_lens: Sequence[int], decode_bs: int
    ) -> "MixedSpecBatchInfo":
        query_lens = tuple(int(x) for x in prefill_query_lens) + (1,) * decode_bs
        return cls(
            mode=MixedSpecMode.TARGET_ONLY,
            prefill_bs=len(prefill_query_lens),
            decode_bs=decode_bs,
            query_lens=query_lens,
            query_start_loc=(0, *accumulate(query_lens)),
        )

    def __post_init__(self) -> None:
        if self.prefill_bs < 0 or self.decode_bs <= 0:
            raise ValueError(
                "A mixed speculative batch requires a non-negative prefill batch "
                "and at least one decode request."
            )
        if len(self.query_lens) != self.prefill_bs + self.decode_bs:
            raise ValueError("query_lens does not match the mixed batch size.")
        if len(self.query_start_loc) != len(self.query_lens) + 1:
            raise ValueError("query_start_loc must be a query indptr.")
        if self.query_start_loc[0] != 0 or any(x <= 0 for x in self.query_lens):
            raise ValueError("Mixed query lengths must be positive and start at zero.")
        expected_start_loc = (0, *accumulate(self.query_lens))
        if self.query_start_loc != expected_start_loc:
            raise ValueError("query_start_loc is inconsistent with query_lens.")
        if self.mode is MixedSpecMode.TARGET_ONLY and any(
            x != 1 for x in self.query_lens[self.prefill_bs :]
        ):
            raise ValueError("Target-only decode requests must have query length one.")

    @property
    def batch_size(self) -> int:
        return self.prefill_bs + self.decode_bs

    @property
    def num_tokens(self) -> int:
        return self.query_start_loc[-1]

    @property
    def decode_start(self) -> int:
        return self.prefill_bs

    @property
    def prefill_indices(self) -> range:
        return range(self.prefill_bs)

    @property
    def decode_indices(self) -> range:
        return range(self.decode_start, self.batch_size)

    def is_decode_index(self, index: int) -> bool:
        return self.decode_start <= index < self.batch_size
