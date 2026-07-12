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
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
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
        seen = {}

        def capture_prepare(_verify_input, verify_batch, _target_worker):
            seen["seq_lens_cpu"] = verify_batch.seq_lens_cpu.clone()
            seen["seq_lens_sum"] = verify_batch.seq_lens_sum
            return SimpleNamespace(), False

        with patch.object(DFlashVerifyInput, "prepare_for_verify", new=capture_prepare):
            executor.run_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=torch.ones((2, 4), dtype=torch.int64),
                verify_window=verify_window,
                sampling_info=None,
            )

        self.assertEqual(seen["seq_lens_cpu"].tolist(), [10, 20])
        self.assertEqual(seen["seq_lens_sum"], 30)
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)

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
