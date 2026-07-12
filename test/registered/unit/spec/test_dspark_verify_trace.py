from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_verify_trace import (
    DsparkVerifyTracer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _build_non_greedy_record(*, accept_assert_enabled: bool) -> dict:
    tracer = DsparkVerifyTracer(gamma=2, verify_num_draft_tokens=3, tp_rank=0)
    with patch(
        "sglang.srt.speculative.dspark_components.dspark_verify_trace."
        "envs.SGLANG_DSPARK_ACCEPT_SAMPLING_TRACE_ASSERT.get",
        return_value=accept_assert_enabled,
    ):
        return tracer._build_record(
            forward_ct=1,
            bs=1,
            mode="compact",
            budget=2,
            layout_graph_num_tokens=3,
            folded_accept=False,
            run_compact=True,
            can_run_cuda_graph=False,
            all_greedy=False,
            sampling_trace_info={
                "backend": "pytorch",
                "seed_present": True,
                "seed_rows": 1,
                "any_greedy": False,
                "need_top_k": False,
                "need_top_p": True,
                "need_min_p": False,
            },
            verify_lens=None,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            rids=["rid"],
            prefix_lens=torch.tensor([10], dtype=torch.int32),
            candidates=torch.tensor([[7, 11, 12]], dtype=torch.int64),
            draft_tokens=torch.tensor([[11, 12]], dtype=torch.int64),
            target_logits=torch.zeros((3, 16), dtype=torch.float32),
            greedy_mask=torch.tensor([False]),
            correct_len=torch.tensor([1], dtype=torch.int32),
            bonus=torch.tensor([9], dtype=torch.int64),
            cap_trim_lens=torch.tensor([0], dtype=torch.int32),
            commit_lens=torch.tensor([2], dtype=torch.int32),
            new_seq_lens=torch.tensor([12], dtype=torch.int32),
            out_tokens=torch.tensor([[11, 9, 0]], dtype=torch.int64),
            simulated_accept=False,
        )


def test_non_greedy_trace_covered_by_accept_sampling_reference():
    record = _build_non_greedy_record(accept_assert_enabled=True)

    assert record["coverage"]["non_greedy_accept"] == "accept_sampling_reference"
    assert record["verdict"]["passed"]
    assert record["verdict"]["skipped"] == []


def test_non_greedy_trace_requires_accept_sampling_reference_for_decision():
    record = _build_non_greedy_record(accept_assert_enabled=False)

    assert record["coverage"]["non_greedy_accept"] == "not_replayed"
    assert record["verdict"]["passed"]
    assert record["verdict"]["skipped"] == [
        "non-greedy accept decision is not replayed by verifier trace; "
        "enable SGLANG_DSPARK_ACCEPT_SAMPLING_TRACE_ASSERT"
    ]


def test_trace_counter_reset_rearms_limit():
    with patch(
        "sglang.srt.speculative.dspark_components.dspark_verify_trace."
        "envs.SGLANG_DSPARK_VERIFY_TRACE_LIMIT.get",
        return_value=1,
    ):
        tracer = DsparkVerifyTracer(gamma=2, verify_num_draft_tokens=3, tp_rank=0)

    tracer._records = 1

    assert not tracer._should_write()

    tracer.reset_records()

    assert tracer._should_write()
