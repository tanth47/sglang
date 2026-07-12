import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.srt.mem_cache.common import (
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import _copy_prefix_seq_lens_cpu
from sglang.srt.speculative.dspark_components.dspark_draft_proposer import (
    DraftBlockProposer,
)
from sglang.srt.speculative.dspark_components.dspark_target_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_verify_epilogue import (
    DsparkVerifyEpilogue,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
    _should_arm_verify_epilogue_commit,
    _should_enable_verify_epilogue,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _spec_args(*, draft_tokens: int = 8, algorithm: str = "DSPARK") -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.speculative_num_draft_tokens = draft_tokens
    args.max_speculative_num_draft_tokens = draft_tokens
    args.speculative_num_steps = None
    args.speculative_eagle_topk = 1
    args.page_size = 1
    return args


class _FakeBatch:
    def __init__(self, *, committed_lens, allocated_lens, row_width: int):
        self.device = torch.device("cpu")
        self.reqs = [
            SimpleNamespace(
                kv_committed_len=committed_len,
                kv_allocated_len=allocated_len,
                sampling_params=SimpleNamespace(top_k=1),
            )
            for committed_len, allocated_len in zip(committed_lens, allocated_lens)
        ]
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.empty((len(self.reqs), row_width), dtype=torch.int32)
        )

    def batch_size(self):
        return len(self.reqs)


class _FakeTargetWorker:
    def __init__(self):
        self.model_runner = SimpleNamespace(attn_backend=SimpleNamespace())

    def forward_batch_generation(self, **kwargs):
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.empty(0)),
            can_run_cuda_graph=False,
        )


class TestDFlashDSparkVerifyLengths(CustomTestCase):
    def test_req_to_token_headroom_covers_spec_v2_double_buffer(self):
        args = _spec_args(draft_tokens=8)
        self.assertEqual(get_alloc_reserve_per_decode(args), 16)
        self.assertGreaterEqual(get_req_to_token_extra_context_len(args), 16)

    def test_req_to_token_headroom_keeps_non_tree_eagle_default(self):
        args = _spec_args(draft_tokens=8, algorithm="EAGLE")
        self.assertEqual(get_alloc_reserve_per_decode(args), 16)
        self.assertEqual(get_req_to_token_extra_context_len(args), 12)

    def test_dspark_forces_future_map_seq_lens_cpu(self):
        args = _spec_args(draft_tokens=8)
        backend = SimpleNamespace(needs_cpu_seq_lens=False)

        self.assertTrue(decide_needs_cpu_seq_lens(args, [backend]))

    def test_prepare_for_decode_keeps_committed_and_reserved_lengths_separate(self):
        args = _spec_args(draft_tokens=4)
        set_global_server_args_for_scheduler(args)
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        batch = _FakeBatch(
            committed_lens=[10, 20],
            allocated_lens=[18, 28],
            row_width=64,
        )

        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(False):
            draft_input.prepare_for_decode(batch)

        self.assertEqual(batch.seq_lens_cpu.tolist(), [10, 20])
        self.assertEqual(batch.seq_lens_sum, 30)
        self.assertEqual(draft_input.reserved_seq_lens_cpu.tolist(), [18, 28])
        self.assertEqual(draft_input.reserved_seq_lens_sum, 46)

    def test_prepare_for_decode_fails_before_req_to_token_oob(self):
        args = _spec_args(draft_tokens=8)
        set_global_server_args_for_scheduler(args)
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        batch = _FakeBatch(committed_lens=[8], allocated_lens=[24], row_width=23)

        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(False):
            with self.assertRaisesRegex(AssertionError, "over-allocation"):
                draft_input.prepare_for_decode(batch)

    def test_dspark_sts_collection_rejects_compact_mode(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker._verify_planner = SimpleNamespace(is_compact_mode=True)

        with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override("/tmp/dspark-sts"):
            with self.assertRaisesRegex(RuntimeError, "cap-accept or static"):
                worker._maybe_record_sts_collect(
                    num_correct_drafts=torch.tensor([0], dtype=torch.int32),
                )

    def test_dspark_scheduled_verify_tokens_use_layout_total_before_graph_key(self):
        worker = object.__new__(DSparkWorkerV2)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[3, 3],
            device=torch.device("cpu"),
            grid=[16],
            num_draft_tokens=8,
        )

        self.assertEqual(
            worker._scheduled_verify_tokens(
                layout=layout,
                fallback=16,
                local_tier=6,
                run_compact=True,
            ),
            6,
        )

    def test_dspark_scheduled_verify_tokens_fall_back_to_local_tier(self):
        worker = object.__new__(DSparkWorkerV2)
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([3, 3], dtype=torch.int32),
            graph_num_tokens=16,
        )

        self.assertEqual(
            worker._scheduled_verify_tokens(
                layout=layout,
                fallback=16,
                local_tier=6,
                run_compact=True,
            ),
            6,
        )

    def test_dspark_scheduled_verify_tokens_use_fallback_without_compact(self):
        worker = object.__new__(DSparkWorkerV2)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[3, 3],
            device=torch.device("cpu"),
            grid=[16],
            num_draft_tokens=8,
        )

        self.assertEqual(
            worker._scheduled_verify_tokens(
                layout=layout,
                fallback=16,
                local_tier=6,
                run_compact=False,
            ),
            16,
        )

    def test_dspark_dsa_multi_request_compact_verify_falls_back(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(attention_backend="dsa")
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 2)

        with envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.override(
            False
        ), envs.SGLANG_DSA_TOPK_BROADCAST.override(False):
            self.assertFalse(
                worker._should_run_compact_target_verify(batch=batch, layout=layout)
            )
        self.assertTrue(worker._warned_dsa_compact_batch_fallback)

    def test_dspark_nsa_multi_request_compact_verify_falls_back(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(attention_backend="nsa")
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 2)

        with envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.override(
            False
        ), envs.SGLANG_DSA_TOPK_BROADCAST.override(False):
            self.assertFalse(
                worker._should_run_compact_target_verify(batch=batch, layout=layout)
            )
        self.assertTrue(worker._warned_dsa_compact_batch_fallback)

    def test_dspark_split_prefill_dsa_compact_verify_falls_back(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(
            attention_backend="flashinfer",
            prefill_attention_backend="dsa",
            decode_attention_backend="flashinfer",
            speculative_attention_mode="prefill",
        )
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 2)

        with envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.override(
            False
        ), envs.SGLANG_DSA_TOPK_BROADCAST.override(False):
            self.assertFalse(
                worker._should_run_compact_target_verify(batch=batch, layout=layout)
            )
        self.assertTrue(worker._warned_dsa_compact_batch_fallback)

    def test_dspark_split_decode_dsa_compact_verify_falls_back(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(
            attention_backend="flashinfer",
            prefill_attention_backend="flashinfer",
            decode_attention_backend="nsa",
            speculative_attention_mode="decode",
        )
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 2)

        with envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.override(
            False
        ), envs.SGLANG_DSA_TOPK_BROADCAST.override(False):
            self.assertFalse(
                worker._should_run_compact_target_verify(batch=batch, layout=layout)
            )
        self.assertTrue(worker._warned_dsa_compact_batch_fallback)

    def test_dspark_dsa_single_request_keeps_compact_verify(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(attention_backend="dsa")
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 1)

        self.assertTrue(
            worker._should_run_compact_target_verify(batch=batch, layout=layout)
        )

    def test_dspark_dsa_multi_request_compact_verify_requires_topk_broadcast(
        self,
    ):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(attention_backend="dsa")
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 2)

        with envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.override(
            True
        ), envs.SGLANG_DSA_TOPK_BROADCAST.override(False):
            self.assertFalse(
                worker._should_run_compact_target_verify(batch=batch, layout=layout)
            )
        self.assertTrue(worker._warned_dsa_compact_batch_fallback)

    def test_dspark_dsa_multi_request_compact_verify_with_topk_broadcast(
        self,
    ):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 0
        worker.server_args = SimpleNamespace(attention_backend="dsa")
        worker._warned_dsa_compact_batch_fallback = False
        worker._verify_planner = SimpleNamespace(
            should_run_compact=lambda *, layout: layout is not None
        )
        layout = object()
        batch = SimpleNamespace(batch_size=lambda: 2)

        with envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.override(
            True
        ), envs.SGLANG_DSA_TOPK_BROADCAST.override(True):
            self.assertTrue(
                worker._should_run_compact_target_verify(batch=batch, layout=layout)
            )
        self.assertTrue(worker._warned_dsa_compact_batch_fallback)

    def test_dspark_verify_epilogue_allows_cuda_alike_platforms(self):
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_cuda_alike",
            return_value=True,
        ):
            self.assertTrue(
                _should_enable_verify_epilogue(
                    is_compact_mode=True,
                    disable_cuda_graph=False,
                )
            )

    def test_dspark_verify_epilogue_requires_compact_graph_enabled(self):
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.is_cuda_alike",
            return_value=True,
        ):
            self.assertFalse(
                _should_enable_verify_epilogue(
                    is_compact_mode=False,
                    disable_cuda_graph=False,
                )
            )
            self.assertFalse(
                _should_enable_verify_epilogue(
                    is_compact_mode=True,
                    disable_cuda_graph=True,
                )
            )

    def test_dspark_verify_epilogue_commit_not_folded_before_tp_sync(self):
        self.assertTrue(
            _should_arm_verify_epilogue_commit(
                fold_eligible=True,
                epilogue_folds_commit=True,
                tp_size=1,
            )
        )
        self.assertFalse(
            _should_arm_verify_epilogue_commit(
                fold_eligible=True,
                epilogue_folds_commit=True,
                tp_size=4,
            )
        )
        self.assertFalse(
            _should_arm_verify_epilogue_commit(
                fold_eligible=True,
                epilogue_folds_commit=False,
                tp_size=1,
            )
        )

    def test_dspark_compact_dsa_defaults_enable_scoped_topk_broadcast(self):
        args = object.__new__(ServerArgs)
        args.speculative_algorithm = "DSPARK"
        args.attention_backend = "dsa"
        args.prefill_attention_backend = None
        args.decode_attention_backend = None
        args.speculative_attention_mode = "prefill"

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
            },
            clear=False,
        ):
            os.environ.pop("SGLANG_DSA_TOPK_BROADCAST", None)
            os.environ.pop("SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH", None)
            args._handle_dspark_dsa_compact_defaults()

            self.assertTrue(envs.SGLANG_DSA_TOPK_BROADCAST.get())
            self.assertTrue(envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.get())

    def test_dspark_compact_dsa_defaults_respect_disabled_topk_broadcast(self):
        args = object.__new__(ServerArgs)
        args.speculative_algorithm = "DSPARK"
        args.attention_backend = "dsa"
        args.prefill_attention_backend = None
        args.decode_attention_backend = None
        args.speculative_attention_mode = "prefill"

        with patch.dict(
            os.environ,
            {
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_DSA_TOPK_BROADCAST": "0",
            },
            clear=False,
        ):
            os.environ.pop("SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH", None)
            args._handle_dspark_dsa_compact_defaults()

            self.assertFalse(envs.SGLANG_DSA_TOPK_BROADCAST.get())
            self.assertFalse(envs.SGLANG_DSPARK_ALLOW_DSA_COMPACT_BATCH.is_set())

    def test_dspark_accept_result_sync_overwrites_local_tp_state(self):
        class FakeBroadcastGroup:
            def __init__(self):
                self.src_values = [
                    torch.tensor([2, 0], dtype=torch.int32),
                    torch.tensor([101, 202], dtype=torch.int64),
                    torch.tensor([0, 1], dtype=torch.int32),
                    torch.tensor([3, 1], dtype=torch.int32),
                    torch.tensor([13, 21], dtype=torch.int64),
                    torch.tensor(
                        [[11, 12, 101, 0], [202, 22, 23, 0]], dtype=torch.int64
                    ),
                ]
                self.calls = []

            def broadcast(self, tensor, src=0):
                self.calls.append((tensor.dtype, tuple(tensor.shape), src))
                tensor.copy_(self.src_values[len(self.calls) - 1])

        worker = object.__new__(DSparkWorkerV2)
        worker.server_args = SimpleNamespace(tp_size=4)
        group = FakeBroadcastGroup()

        correct_len = torch.tensor([0, 3], dtype=torch.int32)
        bonus = torch.tensor([999, 888], dtype=torch.int64)
        cap_trim_lens = torch.tensor([4, 4], dtype=torch.int32)
        commit_lens = torch.tensor([1, 4], dtype=torch.int32)
        new_seq_lens = torch.tensor([10, 24], dtype=torch.int64)
        out_tokens = torch.tensor([[99, 98, 97, 0], [88, 87, 86, 0]], dtype=torch.int64)

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.verify_lens_broadcast_group",
            return_value=(group, 4),
        ):
            worker._sync_accept_result_across_tp(
                correct_len=correct_len,
                bonus=bonus,
                cap_trim_lens=cap_trim_lens,
                commit_lens=commit_lens,
                new_seq_lens=new_seq_lens,
                out_tokens=out_tokens,
            )

        self.assertTrue(torch.equal(correct_len, group.src_values[0]))
        self.assertTrue(torch.equal(bonus, group.src_values[1]))
        self.assertTrue(torch.equal(cap_trim_lens, group.src_values[2]))
        self.assertTrue(torch.equal(commit_lens, group.src_values[3]))
        self.assertTrue(torch.equal(new_seq_lens, group.src_values[4]))
        self.assertTrue(torch.equal(out_tokens, group.src_values[5]))
        self.assertEqual(
            group.calls,
            [
                (torch.int32, (2,), 0),
                (torch.int64, (2,), 0),
                (torch.int32, (2,), 0),
                (torch.int32, (2,), 0),
                (torch.int64, (2,), 0),
                (torch.int64, (2, 4), 0),
            ],
        )

    def test_dspark_draft_proposal_sync_overwrites_local_tp_candidates(self):
        class FakeBroadcastGroup:
            def __init__(self):
                self.src_values = [
                    torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64),
                    torch.tensor([[11, 12], [21, 22]], dtype=torch.int64),
                    torch.tensor([True, False], dtype=torch.bool),
                    torch.tensor([1.0, 0.7], dtype=torch.float32),
                    torch.full((2, 2, 5), 3.0, dtype=torch.float32),
                    torch.full((2, 2, 3), 4.0, dtype=torch.float32),
                    torch.full((4, 3), 5.0, dtype=torch.float32),
                    torch.tensor([[0.9, 0.8], [0.6, 0.4]], dtype=torch.float32),
                    torch.tensor([[1.9, 1.8], [1.6, 1.4]], dtype=torch.float32),
                ]
                self.calls = []

            def broadcast(self, tensor, src=0):
                self.calls.append((tensor.dtype, tuple(tensor.shape), src))
                tensor.copy_(self.src_values[len(self.calls) - 1])

        worker = object.__new__(DSparkWorkerV2)
        worker.server_args = SimpleNamespace(tp_size=4)
        group = FakeBroadcastGroup()
        proposal = SimpleNamespace(
            draft_block_ids=torch.tensor(
                [[10, 99, 98], [20, 88, 87]], dtype=torch.int64
            ),
            draft_block=SimpleNamespace(
                draft_tokens=torch.tensor([[99, 98], [88, 87]], dtype=torch.int64),
                corrected_logits=torch.zeros((2, 2, 5), dtype=torch.float32),
                greedy_mask=torch.tensor([False, False], dtype=torch.bool),
                temperatures=torch.tensor([0.1, 0.2], dtype=torch.float32),
            ),
            draft_hidden=torch.zeros((2, 2, 3), dtype=torch.float32),
            confidence_tap=torch.zeros((4, 3), dtype=torch.float32),
            confidence=torch.zeros((2, 2), dtype=torch.float32),
            confidence_raw=torch.zeros((2, 2), dtype=torch.float32),
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.verify_lens_broadcast_group",
            return_value=(group, 4),
        ):
            worker._sync_draft_proposal_across_tp(proposal)

        self.assertTrue(torch.equal(proposal.draft_block_ids, group.src_values[0]))
        self.assertTrue(
            torch.equal(proposal.draft_block.draft_tokens, group.src_values[1])
        )
        self.assertTrue(
            torch.equal(proposal.draft_block.greedy_mask, group.src_values[2])
        )
        self.assertTrue(
            torch.equal(proposal.draft_block.temperatures, group.src_values[3])
        )
        self.assertTrue(
            torch.equal(proposal.draft_block.corrected_logits, group.src_values[4])
        )
        self.assertTrue(torch.equal(proposal.draft_hidden, group.src_values[5]))
        self.assertTrue(torch.equal(proposal.confidence_tap, group.src_values[6]))
        self.assertTrue(torch.equal(proposal.confidence, group.src_values[7]))
        self.assertTrue(torch.equal(proposal.confidence_raw, group.src_values[8]))
        self.assertEqual(
            group.calls,
            [
                (torch.int64, (2, 3), 0),
                (torch.int64, (2, 2), 0),
                (torch.bool, (2,), 0),
                (torch.float32, (2,), 0),
                (torch.float32, (2, 2, 5), 0),
                (torch.float32, (2, 2, 3), 0),
                (torch.float32, (4, 3), 0),
                (torch.float32, (2, 2), 0),
                (torch.float32, (2, 2), 0),
            ],
        )

    def test_dspark_sts_collection_writes_static_shard(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.gamma = 3
        worker.verify_num_draft_tokens = 4
        worker.tp_rank = 0
        worker._sts_recorder = None
        worker._verify_planner = SimpleNamespace(
            is_compact_mode=False,
            carries_confidence=True,
            last_confidence_raw=torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "sts")
            with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(
                stem
            ), envs.SGLANG_DSPARK_STS_FLUSH_EVERY.override(1):
                worker._maybe_record_sts_collect(
                    num_correct_drafts=torch.tensor([2, 0], dtype=torch.int32),
                )
                worker.flush_sts_records()

            shards = list(Path(tmp).glob("sts.tp0-pid*.0.pt"))
            self.assertEqual(len(shards), 1)
            shard = torch.load(shards[0])

        self.assertTrue(
            torch.equal(shard["logits"], worker._verify_planner.last_confidence_raw)
        )
        self.assertTrue(
            torch.equal(
                shard["prefix_mask"],
                torch.tensor(
                    [[1, 1, 0], [0, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )

    def test_dspark_sts_collection_prefers_explicit_confidence_raw(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.gamma = 3
        worker.verify_num_draft_tokens = 4
        worker.tp_rank = 0
        worker._sts_recorder = None
        worker._verify_planner = SimpleNamespace(
            is_compact_mode=False,
            carries_confidence=True,
            last_confidence_raw=torch.full((1, 3), -99.0, dtype=torch.float32),
        )
        confidence_raw = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "sts")
            with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(
                stem
            ), envs.SGLANG_DSPARK_STS_FLUSH_EVERY.override(1):
                worker._maybe_record_sts_collect(
                    num_correct_drafts=torch.tensor([1], dtype=torch.int32),
                    confidence_raw=confidence_raw,
                )
                worker.flush_sts_records()

            shards = list(Path(tmp).glob("sts.tp0-pid*.0.pt"))
            self.assertEqual(len(shards), 1)
            shard = torch.load(shards[0])

        self.assertTrue(torch.equal(shard["logits"], confidence_raw))

    def test_dspark_sts_collection_skips_non_greedy_batch(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.gamma = 3
        worker.verify_num_draft_tokens = 4
        worker.tp_rank = 0
        worker._sts_recorder = None
        worker._verify_planner = SimpleNamespace(
            is_compact_mode=False,
            carries_confidence=True,
            last_confidence_raw=None,
        )
        confidence_raw = torch.tensor(
            [[0.1, 0.2, 0.3], [9.1, 9.2, 9.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "sts")
            with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(
                stem
            ), envs.SGLANG_DSPARK_STS_FLUSH_EVERY.override(1):
                worker._maybe_record_sts_collect(
                    num_correct_drafts=torch.tensor([2, 3, 1], dtype=torch.int32),
                    confidence_raw=confidence_raw,
                    all_rows_greedy=False,
                )

            self.assertEqual(list(Path(tmp).glob("*.pt")), [])
            self.assertIsNone(worker._sts_recorder)

    def test_dspark_sts_collection_only_records_on_tp0(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.tp_rank = 1
        worker._sts_recorder = None
        worker._verify_planner = SimpleNamespace(is_compact_mode=False)

        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "sts")
            with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(stem):
                worker._maybe_record_sts_collect(
                    num_correct_drafts=torch.tensor([0], dtype=torch.int32),
                )

            self.assertEqual(list(Path(tmp).glob("*.pt")), [])
            self.assertIsNone(worker._sts_recorder)

    def test_dspark_sts_flush_syncs_tp_ranks(self):
        events = []
        worker = object.__new__(DSparkWorkerV2)
        worker.server_args = SimpleNamespace(tp_size=4)
        worker._sts_recorder = SimpleNamespace(flush=lambda: events.append("flush"))
        group = SimpleNamespace(barrier=lambda: events.append("barrier"))

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_attention_tp_group",
            return_value=group,
        ):
            worker.flush_sts_records()

        self.assertEqual(events, ["barrier", "flush", "barrier"])

    def test_dspark_sts_flush_syncs_tp_ranks_on_error(self):
        def fail_flush():
            events.append("flush")
            raise RuntimeError("sts flush failed")

        events = []
        worker = object.__new__(DSparkWorkerV2)
        worker.server_args = SimpleNamespace(tp_size=4)
        worker._sts_recorder = SimpleNamespace(flush=fail_flush)
        group = SimpleNamespace(barrier=lambda: events.append("barrier"))

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_attention_tp_group",
            return_value=group,
        ):
            with self.assertRaisesRegex(RuntimeError, "sts flush failed"):
                worker.flush_sts_records()

        self.assertEqual(events, ["barrier", "flush", "barrier"])

    def test_dspark_sts_collection_skips_health_check_rids(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.gamma = 3
        worker.verify_num_draft_tokens = 4
        worker.tp_rank = 0
        worker._sts_recorder = None
        worker._verify_planner = SimpleNamespace(
            is_compact_mode=False,
            carries_confidence=True,
            last_confidence_raw=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        )

        with tempfile.TemporaryDirectory() as tmp:
            stem = str(Path(tmp) / "sts")
            with envs.SGLANG_DSPARK_STS_COLLECT_PATH.override(stem):
                worker._maybe_record_sts_collect(
                    num_correct_drafts=torch.tensor([0], dtype=torch.int32),
                    rids=["HEALTH_CHECK_WARMUP_test"],
                )

            self.assertEqual(list(Path(tmp).glob("*.pt")), [])
            self.assertIsNone(worker._sts_recorder)

    def test_dflash_draft_block_uses_prefix_seq_lens_cpu(self):
        dst = torch.empty((2,), dtype=torch.int32)
        prefix_lens = torch.tensor([10, 20], dtype=torch.int32)
        host_lens = torch.tensor([10, 20], dtype=torch.int64)

        seq_lens_sum = _copy_prefix_seq_lens_cpu(dst, prefix_lens, host_lens)
        self.assertEqual(dst.tolist(), [10, 20])
        self.assertEqual(seq_lens_sum, 30)

        dst.fill_(-1)
        seq_lens_sum = _copy_prefix_seq_lens_cpu(dst, prefix_lens, None)
        self.assertEqual(dst.tolist(), [10, 20])
        self.assertEqual(seq_lens_sum, 30)

    def test_dspark_target_verify_passes_prefix_seq_lens_cpu_not_reserved(self):
        target_worker = _FakeTargetWorker()
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            verify_num_draft_tokens=4,
            model_runner=target_worker.model_runner,
            kv_injector=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=None,
        )
        draft_input = SimpleNamespace(
            reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
            reserved_seq_lens_sum=46,
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(8, dtype=torch.int64).view(2, 4),
            verify_cache_loc=torch.arange(8, dtype=torch.int64),
        )
        cutoff_layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 3],
            device=torch.device("cpu"),
            grid=[4],
            num_draft_tokens=4,
        )
        seen = {}

        def capture_prepare(_verify_input, verify_batch, _target_worker):
            seen["seq_lens_cpu"] = verify_batch.seq_lens_cpu.clone()
            seen["seq_lens_sum"] = verify_batch.seq_lens_sum
            seen["ragged_verify_layout"] = _verify_input.ragged_verify_layout
            return SimpleNamespace(), False

        with patch.object(DFlashVerifyInput, "prepare_for_verify", new=capture_prepare):
            executor.run_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=torch.ones((2, 4), dtype=torch.int64),
                verify_window=verify_window,
                layout=cutoff_layout,
                sampling_info=None,
            )

        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)
        self.assertIsNone(seen["ragged_verify_layout"])
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)

    def test_dspark_compact_eager_skips_epilogue_when_layout_exceeds_graph_bs(self):
        stride = 8
        bs = 5
        vocab_size = 7
        hidden_size = 3
        epilogue = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=stride,
            device=torch.device("cpu"),
        )
        executor = TargetVerifyExecutor(
            target_worker=SimpleNamespace(),
            verify_num_draft_tokens=stride,
            model_runner=SimpleNamespace(),
            kv_injector=SimpleNamespace(),
            verify_epilogue=epilogue,
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[stride] * bs,
            device=torch.device("cpu"),
            grid=[bs * stride],
            num_draft_tokens=stride,
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * stride, dtype=torch.int64).view(bs, stride),
            verify_cache_loc_2d=torch.arange(bs * stride, dtype=torch.int64).view(
                bs, stride
            ),
        )
        logits_output = SimpleNamespace(
            next_token_logits=torch.arange(
                bs * stride * vocab_size, dtype=torch.float32
            ).view(bs * stride, vocab_size),
            hidden_states=torch.arange(
                bs * stride * hidden_size, dtype=torch.float32
            ).view(bs * stride, hidden_size),
        )
        target_result = SimpleNamespace(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

        with patch.object(executor, "_run_ragged", return_value=target_result):
            target_verify, hidden_strided = executor.run_compact(
                batch=SimpleNamespace(),
                layout=layout,
                verify_ids_2d=torch.arange(bs * stride, dtype=torch.int64).view(
                    bs, stride
                ),
                verify_window=verify_window,
                bs=bs,
                device="cpu",
                sampling_info=None,
            )

        self.assertIs(target_verify, target_result)
        self.assertEqual(
            target_verify.logits_output.next_token_logits.shape,
            (bs * stride, vocab_size),
        )
        self.assertEqual(hidden_strided.shape, (bs * stride, hidden_size))
        self.assertEqual(epilogue.inject_gate_buf.item(), 0)
        self.assertEqual(epilogue.verify_lens_buf.sum().item(), 0)

    def test_dspark_draft_proposer_passes_prefix_seq_lens_cpu_and_crops_anchor_hidden(
        self,
    ):
        seen = {}
        gamma = 4
        draft_width = gamma + 1
        bs = 2

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["seq_lens"] = forward_batch.seq_lens.clone()
                seen["seq_lens_cpu"] = forward_batch.seq_lens_cpu.clone()
                seen["seq_lens_sum"] = forward_batch.seq_lens_sum
                hidden = torch.arange(bs * draft_width * 16, dtype=torch.float32).view(
                    bs * draft_width, 16
                )
                return SimpleNamespace(
                    logits_output=SimpleNamespace(hidden_states=hidden),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_sum=30,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
            verify_cache_loc_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
        )

        out = proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, 16),
        )

        self.assertEqual(seen["seq_lens"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)
        self.assertEqual(tuple(out.draft_block_ids.shape), (bs, draft_width))
        self.assertEqual(tuple(out.draft_hidden_3d.shape), (bs, gamma, 16))
        self.assertEqual(out.raw_hidden[0].tolist(), list(range(16, 32)))

    def test_dspark_draft_proposer_derives_cpu_lens_from_gpu_only_batch(self):
        seen = {}
        gamma = 4
        draft_width = gamma + 1
        bs = 2

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["seq_lens"] = forward_batch.seq_lens.clone()
                seen["seq_lens_cpu"] = forward_batch.seq_lens_cpu.clone()
                seen["seq_lens_sum"] = forward_batch.seq_lens_sum
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=torch.empty((bs * draft_width, 16))
                    ),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
            reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
            reserved_seq_lens_sum=46,
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
            verify_cache_loc_2d=torch.arange(bs * draft_width, dtype=torch.int64).view(
                bs, draft_width
            ),
        )

        proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, 16),
        )

        self.assertEqual(seen["seq_lens"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)

    def test_dspark_draft_dummy_verify_input_uses_verify_window(self):
        args = _spec_args(draft_tokens=8)
        spec_algorithm = SpeculativeAlgorithm.DSPARK

        draft_spec = create_dummy_verify_input(
            spec_algorithm=spec_algorithm,
            server_args=args,
            custom_mask=torch.empty(0, dtype=torch.bool),
            num_tokens_per_bs=8,
            is_draft_worker=True,
        )
        target_spec = create_dummy_verify_input(
            spec_algorithm=spec_algorithm,
            server_args=args,
            custom_mask=torch.empty(0, dtype=torch.bool),
            num_tokens_per_bs=8,
            is_draft_worker=False,
        )

        self.assertEqual(draft_spec.draft_token_num, 8)
        self.assertEqual(target_spec.draft_token_num, 8)

    def test_target_verify_width_adjustment_keeps_dspark_window(self):
        self.assertEqual(
            SpeculativeAlgorithm.DSPARK.get_num_tokens_per_bs_for_target_verify(
                8, is_draft_worker=True
            ),
            8,
        )
        self.assertEqual(
            SpeculativeAlgorithm.DSPARK.get_num_tokens_per_bs_for_target_verify(
                8, is_draft_worker=False
            ),
            8,
        )
        self.assertEqual(
            SpeculativeAlgorithm.EAGLE.get_num_tokens_per_bs_for_target_verify(
                8, is_draft_worker=True
            ),
            8,
        )
        self.assertEqual(
            SpeculativeAlgorithm.NGRAM.get_num_tokens_per_bs_for_target_verify(
                8, is_draft_worker=True
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
