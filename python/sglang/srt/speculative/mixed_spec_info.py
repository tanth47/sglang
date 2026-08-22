from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import accumulate
from typing import Sequence, Tuple


class MixedSpecMode(Enum):
    """Execution mode for a batch that mixes prefill with speculative requests."""

    TARGET_ONLY = auto()
    VERIFY = auto()


EAGLE_VERIFY_WIDTH = 6


@dataclass(frozen=True)
class MixedSpecBatchInfo:
    """Authoritative host layout for a mixed prefill/speculative batch.

    Requests are flattened in partition order: all prefill requests first,
    followed by all running speculative requests. TARGET_ONLY gives each
    running request one row. VERIFY gives each running request the complete
    six-row EAGLE target-verify chain.
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

    @classmethod
    def verify(
        cls, prefill_query_lens: Sequence[int], verify_bs: int
    ) -> "MixedSpecBatchInfo":
        query_lens = tuple(int(x) for x in prefill_query_lens) + (
            EAGLE_VERIFY_WIDTH,
        ) * verify_bs
        return cls(
            mode=MixedSpecMode.VERIFY,
            prefill_bs=len(prefill_query_lens),
            decode_bs=verify_bs,
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
        if self.mode is MixedSpecMode.VERIFY and any(
            x != EAGLE_VERIFY_WIDTH for x in self.query_lens[self.prefill_bs :]
        ):
            raise ValueError(
                f"EAGLE verify requests must have query length {EAGLE_VERIFY_WIDTH}."
            )

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
    def verify_bs(self) -> int:
        return self.decode_bs

    @property
    def verify_start(self) -> int:
        return self.decode_start

    @property
    def prefill_num_tokens(self) -> int:
        return self.query_start_loc[self.prefill_bs]

    @property
    def verify_num_tokens(self) -> int:
        return self.num_tokens - self.prefill_num_tokens

    @property
    def target_output_rows(self) -> int:
        if self.mode is MixedSpecMode.VERIFY:
            return self.prefill_bs + self.verify_num_tokens
        return self.batch_size

    @property
    def target_logit_row_indices(self) -> Tuple[int, ...]:
        """Rows sent to lm_head, in prefill-output then verify-output order."""
        prefill_last_rows = tuple(
            self.query_start_loc[index + 1] - 1 for index in self.prefill_indices
        )
        if self.mode is MixedSpecMode.VERIFY:
            verify_rows = tuple(range(self.prefill_num_tokens, self.num_tokens))
        else:
            verify_rows = tuple(
                self.query_start_loc[index + 1] - 1 for index in self.decode_indices
            )
        return prefill_last_rows + verify_rows

    @property
    def prefill_indices(self) -> range:
        return range(self.prefill_bs)

    @property
    def decode_indices(self) -> range:
        return range(self.decode_start, self.batch_size)

    def is_decode_index(self, index: int) -> bool:
        return self.decode_start <= index < self.batch_size

    def split_target_outputs(self, values):
        """Split pruned target outputs into prefill and speculative partitions."""
        if values.shape[0] != self.target_output_rows:
            raise ValueError(
                f"Expected {self.target_output_rows} mixed target rows, "
                f"got {values.shape[0]}."
            )
        return values[: self.prefill_bs], values[self.prefill_bs :]

    def split_flattened_tokens(self, values):
        """Split unpruned token-aligned values at the partition boundary."""
        if values.shape[0] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} flattened mixed tokens, "
                f"got {values.shape[0]}."
            )
        return values[: self.prefill_num_tokens], values[self.prefill_num_tokens :]

    def causal_context_lens(
        self, prefix_lens: Sequence[int]
    ) -> Tuple[int, ...]:
        """Expand per-request prefixes into causal lengths for every query row."""
        if len(prefix_lens) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} prefix lengths, got {len(prefix_lens)}."
            )
        return tuple(
            int(prefix_len) + row
            for prefix_len, query_len in zip(prefix_lens, self.query_lens)
            for row in range(1, query_len + 1)
        )
