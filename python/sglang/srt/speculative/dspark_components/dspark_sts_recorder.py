from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import torch


class StsDataRecorder:
    def __init__(
        self,
        *,
        path_stem: str,
        gamma: int,
        flush_every: int,
        shard_tag: Optional[str] = None,
    ) -> None:
        self.path_stem = path_stem
        self.gamma = int(gamma)
        self.flush_every = int(flush_every)
        self.shard_tag = shard_tag
        self._logits_buffer: list[torch.Tensor] = []
        self._prefix_mask_buffer: list[torch.Tensor] = []
        self._shard_ct = 0
        self._writer_queue: Optional[
            Queue[Optional[tuple[Path, dict[str, torch.Tensor]]]]
        ] = None
        self._writer_thread: Optional[Thread] = None
        self._writer_error: Optional[BaseException] = None

    def record(
        self, *, confidence_raw: torch.Tensor, num_correct_drafts: torch.Tensor
    ) -> None:
        self._check_writer_error()
        logits = confidence_raw.detach().to(device="cpu", dtype=torch.float32)
        positions = torch.arange(self.gamma).view(1, -1)
        counts = (
            num_correct_drafts.detach().to(device="cpu", dtype=torch.int64).view(-1, 1)
        )
        prefix_mask = (positions < counts).to(torch.float32)
        self._logits_buffer.append(logits)
        self._prefix_mask_buffer.append(prefix_mask)
        if len(self._logits_buffer) >= self.flush_every:
            self.flush(wait=False)

    def flush(self, *, wait: bool = True) -> None:
        self._check_writer_error()
        if not self._logits_buffer:
            if wait and self._writer_queue is not None:
                self._writer_queue.join()
                self._check_writer_error()
            return
        suffix = (
            f".{self._shard_ct}.pt"
            if self.shard_tag is None
            else f".{self.shard_tag}.{self._shard_ct}.pt"
        )
        shard_path = Path(f"{self.path_stem}{suffix}")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard = {
            "logits": torch.cat(self._logits_buffer, dim=0),
            "prefix_mask": torch.cat(self._prefix_mask_buffer, dim=0),
        }
        self._logits_buffer.clear()
        self._prefix_mask_buffer.clear()
        self._shard_ct += 1
        self._enqueue_shard(shard_path, shard)
        if wait and self._writer_queue is not None:
            self._writer_queue.join()
            self._check_writer_error()

    def _enqueue_shard(self, shard_path: Path, shard: dict[str, torch.Tensor]) -> None:
        if self._writer_queue is None:
            self._writer_queue = Queue()
            self._writer_thread = Thread(
                target=self._writer_loop,
                name="dspark-sts-writer",
                daemon=True,
            )
            self._writer_thread.start()
        self._writer_queue.put((shard_path, shard))

    def _writer_loop(self) -> None:
        assert self._writer_queue is not None
        while True:
            item = self._writer_queue.get()
            try:
                if item is None:
                    return
                shard_path, shard = item
                torch.save(shard, shard_path)
            except BaseException as exc:
                self._writer_error = exc
            finally:
                self._writer_queue.task_done()

    def _check_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("DSpark STS writer failed") from self._writer_error
