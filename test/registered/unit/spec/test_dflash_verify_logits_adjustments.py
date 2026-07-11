import types

import torch

from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SamplingInfo:
    def __init__(self, *, vocab_mask=None):
        self.has_custom_logit_processor = False
        self.acc_linear_penalties = None
        self.penalizer_orchestrator = types.SimpleNamespace(is_required=False)
        self.vocab_mask = vocab_mask
        self.logit_bias = None

    def __len__(self):
        return 2


def test_noop_verify_logits_adjustment_returns_before_shape_checks():
    apply_dflash_verify_logits_adjustments(
        next_token_logits=torch.empty((3, 5)),
        sampling_info=_SamplingInfo(),
        draft_token_num=4,
    )


def test_non_noop_verify_logits_adjustment_keeps_shape_checks():
    try:
        apply_dflash_verify_logits_adjustments(
            next_token_logits=torch.empty((3, 5)),
            sampling_info=_SamplingInfo(vocab_mask=torch.ones((2, 5), dtype=torch.bool)),
            draft_token_num=4,
        )
    except ValueError as exc:
        assert "row count mismatch" in str(exc)
    else:
        raise AssertionError("expected row-count validation for non-noop adjustments")
