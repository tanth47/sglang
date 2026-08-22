"""External-server lifecycle smoke for mixed speculative scheduling.

The server is launched separately so the operator can constrain it to an exact
GPU device cgroup.  This script keeps a long speculative decode alive, submits
a chunked cold prefill so mixed scheduling is exercised, aborts the decode, and
then verifies that allocator accounting returns to an idle state.

Example:
    python test/manual/spec/test_mixed_spec_lifecycle_runtime.py \
        --base-url http://127.0.0.1:31280
"""

import argparse
import json
import threading
import time
from typing import Any

import requests


METRIC_NAMES = (
    "sglang:num_used_tokens",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
)


def read_allocator_metrics(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=10)
    response.raise_for_status()
    metrics = {}
    for line in response.text.splitlines():
        for name in METRIC_NAMES:
            if line.startswith(f"{name}{{"):
                metrics[name] = float(line.rsplit(" ", 1)[1])
    missing = set(METRIC_NAMES) - metrics.keys()
    if missing:
        raise AssertionError(f"Missing allocator metrics: {sorted(missing)}")
    return metrics


def generate(
    base_url: str,
    *,
    rid: str,
    text: str,
    max_new_tokens: int,
    output: dict[str, Any],
) -> None:
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": text,
                "rid": rid,
                "stream": False,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=240,
        )
        output["status_code"] = response.status_code
        output["body"] = response.json()
    except Exception as exc:  # pragma: no cover - manual diagnostic path
        output["error"] = repr(exc)


def finish_reason(output: dict[str, Any]) -> Any:
    return output.get("body", {}).get("meta_info", {}).get("finish_reason")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:31280")
    parser.add_argument("--decode-settle-seconds", type=float, default=2.0)
    parser.add_argument("--mixed-settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--control-mode",
        choices=("abort", "retract"),
        default="abort",
        help="Optionally retract and resume both requests before aborting decode.",
    )
    args = parser.parse_args()

    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    before = read_allocator_metrics(args.base_url)

    run_tag = str(time.time_ns())
    decode_rid = f"mixed-spec-lifecycle-decode-{run_tag}"
    prefill_rid = f"mixed-spec-lifecycle-prefill-{run_tag}"
    decode_output: dict[str, Any] = {}
    prefill_output: dict[str, Any] = {}

    decode_thread = threading.Thread(
        target=generate,
        kwargs={
            "base_url": args.base_url,
            "rid": decode_rid,
            "text": (
                f"Unique run {run_tag}. "
                "List the positive integers from 1 through 2000 in order, "
                "one integer per line. Do not stop early."
            ),
            "max_new_tokens": 2048,
            "output": decode_output,
        },
    )
    decode_thread.start()
    time.sleep(args.decode_settle_seconds)

    # Roughly 1K+ tokenizer tokens, forcing multiple 128-token prefill chunks.
    cold_prompt = (
        f"Unique run {run_tag}. Read the following repeated record and "
        "summarize it in one sentence.\n"
        + "alpha beta gamma delta " * 320
    )
    prefill_thread = threading.Thread(
        target=generate,
        kwargs={
            "base_url": args.base_url,
            "rid": prefill_rid,
            "text": cold_prompt,
            "max_new_tokens": 8,
            "output": prefill_output,
        },
    )
    prefill_thread.start()
    time.sleep(args.mixed_settle_seconds)

    pause_response = None
    continue_response = None
    paused_metrics = None
    if args.control_mode == "retract":
        pause_response = requests.post(
            f"{args.base_url}/pause_generation",
            json={"mode": "retract"},
            timeout=30,
        )
        pause_response.raise_for_status()
        # Metrics are exported through a separate process; allow the scheduler's
        # forced paused-state snapshot to propagate before sampling it.
        time.sleep(2)
        paused_metrics = read_allocator_metrics(args.base_url)
        continue_response = requests.post(
            f"{args.base_url}/continue_generation",
            json={"torch_empty_cache": False},
            timeout=30,
        )
        continue_response.raise_for_status()
        time.sleep(1)

    abort_response = requests.post(
        f"{args.base_url}/abort_request",
        json={"rid": decode_rid},
        timeout=10,
    )
    abort_response.raise_for_status()

    decode_thread.join(timeout=120)
    prefill_thread.join(timeout=240)
    if decode_thread.is_alive() or prefill_thread.is_alive():
        raise AssertionError("Lifecycle requests did not drain before timeout")

    # Let the scheduler publish the post-release metrics snapshot.
    time.sleep(2)
    after = read_allocator_metrics(args.base_url)
    final_health = requests.get(f"{args.base_url}/health", timeout=10)
    final_health.raise_for_status()

    decode_reason = finish_reason(decode_output)
    if decode_output.get("status_code") != 200:
        raise AssertionError(f"Decode request failed: {decode_output}")
    if not decode_reason or decode_reason.get("type") != "abort":
        raise AssertionError(f"Decode did not finish by abort: {decode_output}")
    if prefill_output.get("status_code") != 200:
        raise AssertionError(f"Cold prefill request failed: {prefill_output}")
    num_retractions = decode_output.get("body", {}).get("meta_info", {}).get(
        "num_retractions", 0
    )
    if args.control_mode == "retract" and num_retractions < 1:
        raise AssertionError(f"Decode was not retracted: {decode_output}")
    if (
        args.control_mode == "retract"
        and paused_metrics["sglang:num_used_tokens"] != 0
    ):
        raise AssertionError(
            f"Retraction left live allocator tokens while paused: {paused_metrics}"
        )

    before_total = (
        before["sglang:kv_available_tokens"]
        + before["sglang:kv_evictable_tokens"]
    )
    after_total = (
        after["sglang:kv_available_tokens"]
        + after["sglang:kv_evictable_tokens"]
    )
    if after["sglang:num_used_tokens"] != 0:
        raise AssertionError(f"Allocator still has live tokens: {after}")
    if after_total != before_total:
        raise AssertionError(
            f"Allocator capacity did not recover: before={before}, after={after}"
        )

    print(
        json.dumps(
            {
                "abort_status": abort_response.status_code,
                "control_mode": args.control_mode,
                "decode_finish_reason": decode_reason,
                "decode_num_retractions": num_retractions,
                "prefill_finish_reason": finish_reason(prefill_output),
                "allocator_before": before,
                "allocator_while_paused": paused_metrics,
                "allocator_after": after,
                "health_after": final_health.status_code,
                "pause_status": (
                    pause_response.status_code if pause_response is not None else None
                ),
                "continue_status": (
                    continue_response.status_code
                    if continue_response is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
