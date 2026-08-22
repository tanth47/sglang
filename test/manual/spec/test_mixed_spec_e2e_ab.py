"""Real-weight end-to-end A/B gate for mixed speculative scheduling.

Run ``capture`` once against each externally managed server mode, then run
``compare`` over the three JSON artifacts.  The workload deliberately keeps
two decodes at different sequence lengths alive while three different cold
prefills are admitted together.

The server process is managed separately so a shared-node operator can enforce
an exact GPU device cgroup around model loading and execution.
"""

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import requests


SPEC_DRAFTS_PER_VERIFY = 5


def _request_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "decode_alpha",
            "text": (
                "Write the positive integers from 1 through 400 in order, "
                "separated by one comma and one space."
            ),
            "max_new_tokens": 256,
        },
        {
            "name": "decode_beta",
            "text": (
                "Write the lowercase English alphabet repeatedly, separating "
                "each letter with one space."
            ),
            "max_new_tokens": 192,
        },
        {
            "name": "prefill_short",
            "text": (
                "Read these records, then state the color named in the final "
                "instruction. "
                + "amber cedar delta quartz " * 80
                + " Final instruction: state the color violet."
            ),
            "max_new_tokens": 32,
        },
        {
            "name": "prefill_medium",
            "text": (
                "Read these records, then state the animal named in the final "
                "instruction. "
                + "falcon harbor maple river " * 160
                + " Final instruction: state the animal otter."
            ),
            "max_new_tokens": 32,
        },
        {
            "name": "prefill_long",
            "text": (
                "Read these records, then state the number named in the final "
                "instruction. "
                + "indigo lantern meadow silver " * 240
                + " Final instruction: state the number 731."
            ),
            "max_new_tokens": 32,
        },
    ]


def _token_ids(meta_info: dict[str, Any]) -> list[int]:
    logprobs = meta_info.get("output_token_logprobs")
    if logprobs is None:
        raise AssertionError("Server response omitted output_token_logprobs")
    return [int(item[1]) for item in logprobs]


def _generate(
    base_url: str,
    mode: str,
    run_id: str,
    spec: dict[str, Any],
    output: dict[str, Any],
) -> None:
    started_at = time.monotonic()
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "rid": f"g4-{mode}-{run_id}-{spec['name']}",
                "text": spec["text"],
                "stream": False,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": spec["max_new_tokens"],
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 0,
                "return_text_in_logprobs": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
        meta_info = body["meta_info"]
        token_ids = _token_ids(meta_info)
        completion_tokens = int(meta_info["completion_tokens"])
        if len(token_ids) != completion_tokens:
            raise AssertionError(
                f"{spec['name']}: {len(token_ids)=} != {completion_tokens=}"
            )
        output.update(
            {
                "status_code": response.status_code,
                "text": body["text"],
                "token_ids": token_ids,
                "finish_reason": meta_info.get("finish_reason"),
                "prompt_tokens": int(meta_info["prompt_tokens"]),
                "completion_tokens": completion_tokens,
                "spec_accept_rate": meta_info.get("spec_accept_rate"),
                "spec_accept_length": meta_info.get("spec_accept_length"),
                "spec_num_correct_drafts": meta_info.get(
                    "spec_num_correct_drafts"
                ),
                "spec_num_proposed_drafts": meta_info.get(
                    "spec_num_proposed_drafts"
                ),
                "spec_verify_ct": meta_info.get("spec_verify_ct"),
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
    except Exception as exc:  # pragma: no cover - manual diagnostic path
        output["error"] = repr(exc)


def capture(args: argparse.Namespace) -> None:
    if args.max_new_tokens_cap is not None and args.max_new_tokens_cap <= 0:
        raise ValueError("--max-new-tokens-cap must be positive")
    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    cache_flush_response = None
    if args.flush_cache_before:
        flush = requests.post(f"{args.base_url}/flush_cache", timeout=30)
        flush.raise_for_status()
        cache_flush_response = flush.text.strip()

    specs = _request_specs()
    if args.max_new_tokens_cap is not None:
        for spec in specs:
            spec["max_new_tokens"] = min(
                spec["max_new_tokens"], args.max_new_tokens_cap
            )
    run_id = args.run_id or str(time.time_ns())
    outputs = {spec["name"]: {} for spec in specs}
    threads: dict[str, threading.Thread] = {}

    def start(spec: dict[str, Any]) -> None:
        thread = threading.Thread(
            target=_generate,
            args=(
                args.base_url,
                args.mode,
                run_id,
                spec,
                outputs[spec["name"]],
            ),
        )
        threads[spec["name"]] = thread
        thread.start()

    if args.serial:
        for spec in specs:
            start(spec)
            threads[spec["name"]].join(timeout=360)
    else:
        # Staggering gives the two running requests different decode lengths
        # before the three heterogeneous prefills arrive together.
        start(specs[0])
        time.sleep(args.decode_stagger_seconds)
        start(specs[1])
        time.sleep(args.prefill_delay_seconds)
        prefill_specs = specs[2:]
        for index, spec in enumerate(prefill_specs):
            start(spec)
            if index + 1 < len(prefill_specs):
                time.sleep(args.prefill_request_stagger_seconds)

    for thread in threads.values():
        thread.join(timeout=360)
    alive = [name for name, thread in threads.items() if thread.is_alive()]
    if alive:
        raise AssertionError(f"Requests did not drain: {alive}")
    failures = {name: value for name, value in outputs.items() if "error" in value}
    if failures:
        raise AssertionError(f"Generation failures: {failures}")

    final_health = requests.get(f"{args.base_url}/health", timeout=10)
    final_health.raise_for_status()
    artifact = {
        "mode": args.mode,
        "run_id": run_id,
        "cache_flushed_before": args.flush_cache_before,
        "cache_flush_response": cache_flush_response,
        "scheduling": "serial" if args.serial else "concurrent",
        "max_new_tokens_cap": args.max_new_tokens_cap,
        "health_before": health.status_code,
        "health_after": final_health.status_code,
        "requests": outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


def _load(path: str, expected_mode: str) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("mode") != expected_mode:
        raise AssertionError(
            f"{path}: expected mode {expected_mode!r}, got {artifact.get('mode')!r}"
        )
    if artifact.get("health_before") != 200 or artifact.get("health_after") != 200:
        raise AssertionError(f"{path}: server health gate failed")
    return artifact


def _validate_spec_accounting(
    mode: str, requests_by_name: dict[str, dict[str, Any]]
) -> dict[str, float]:
    total_completion = 0
    total_verify = 0
    total_correct = 0
    total_proposed = 0
    for name, result in requests_by_name.items():
        verify_ct = result.get("spec_verify_ct")
        correct = result.get("spec_num_correct_drafts")
        proposed = result.get("spec_num_proposed_drafts")
        accept_rate = result.get("spec_accept_rate")
        accept_length = result.get("spec_accept_length")
        if None in (verify_ct, correct, proposed, accept_rate, accept_length):
            raise AssertionError(f"{mode}/{name}: missing speculative accounting")

        verify_ct = int(verify_ct)
        correct = int(correct)
        proposed = int(proposed)
        completion = int(result["completion_tokens"])
        expected_proposed = verify_ct * SPEC_DRAFTS_PER_VERIFY
        if verify_ct <= 0 or proposed != expected_proposed:
            raise AssertionError(
                f"{mode}/{name}: bad proposal accounting: "
                f"{verify_ct=}, {proposed=}, {expected_proposed=}"
            )
        if not 0 <= correct <= proposed:
            raise AssertionError(
                f"{mode}/{name}: correct drafts outside proposal range"
            )
        if not math.isclose(accept_rate, correct / proposed, abs_tol=1e-12):
            raise AssertionError(f"{mode}/{name}: accept-rate formula mismatch")
        if not math.isclose(
            accept_length, completion / verify_ct, abs_tol=1e-12
        ):
            raise AssertionError(f"{mode}/{name}: accept-length formula mismatch")

        total_completion += completion
        total_verify += verify_ct
        total_correct += correct
        total_proposed += proposed

    return {
        "completion_tokens": total_completion,
        "verify_ct": total_verify,
        "correct_drafts": total_correct,
        "proposed_drafts": total_proposed,
        "accept_length": total_completion / total_verify,
        "accept_rate": total_correct / total_proposed,
    }


def _first_token_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _finish_type(result: dict[str, Any]) -> Any:
    finish_reason = result.get("finish_reason")
    return finish_reason.get("type") if isinstance(finish_reason, dict) else None


def _compare_result_pair(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    mismatch = _first_token_mismatch(left["token_ids"], right["token_ids"])
    return {
        "token_ids_equal": mismatch is None,
        "first_token_mismatch": mismatch,
        "text_equal": left["text"] == right["text"],
        "finish_type_equal": _finish_type(left) == _finish_type(right),
    }


def _comparison_passed(comparison: dict[str, Any]) -> bool:
    return (
        comparison["token_ids_equal"]
        and comparison["text_equal"]
        and comparison["finish_type_equal"]
    )


def compare(args: argparse.Namespace) -> None:
    target = _load(args.target, "target")
    unmixed = _load(args.unmixed, "unmixed")
    mixed = _load(args.mixed, "mixed")
    mixed_repeat = (
        _load(args.mixed_repeat, "mixed") if args.mixed_repeat else None
    )
    modes = {"target": target, "unmixed": unmixed, "mixed": mixed}
    if mixed_repeat is not None:
        modes["mixed_repeat"] = mixed_repeat
    request_names = set(target["requests"])
    if any(set(artifact["requests"]) != request_names for artifact in modes.values()):
        raise AssertionError("Mode artifacts do not contain the same requests")

    comparisons = {}
    target_mismatches = []
    mixed_scheduler_mismatches = []
    mixed_repeat_mismatches = []
    for name in sorted(request_names):
        target_result = target["requests"][name]
        unmixed_result = unmixed["requests"][name]
        mixed_result = mixed["requests"][name]
        per_request = {
            "target_vs_unmixed": _compare_result_pair(
                target_result, unmixed_result
            ),
            "target_vs_mixed": _compare_result_pair(target_result, mixed_result),
            "unmixed_vs_mixed": _compare_result_pair(
                unmixed_result, mixed_result
            ),
        }
        for pair in ("target_vs_unmixed", "target_vs_mixed"):
            if not _comparison_passed(per_request[pair]):
                target_mismatches.append(
                    {"request": name, "pair": pair, **per_request[pair]}
                )
        if not _comparison_passed(per_request["unmixed_vs_mixed"]):
            mixed_scheduler_mismatches.append(
                {
                    "request": name,
                    "pair": "unmixed_vs_mixed",
                    **per_request["unmixed_vs_mixed"],
                }
            )
        if mixed_repeat is not None:
            per_request["mixed_vs_mixed_repeat"] = _compare_result_pair(
                mixed_result, mixed_repeat["requests"][name]
            )
            if not _comparison_passed(per_request["mixed_vs_mixed_repeat"]):
                mixed_repeat_mismatches.append(
                    {
                        "request": name,
                        "pair": "mixed_vs_mixed_repeat",
                        **per_request["mixed_vs_mixed_repeat"],
                    }
                )
        comparisons[name] = per_request

    unmixed_acceptance = _validate_spec_accounting(
        "unmixed", unmixed["requests"]
    )
    mixed_acceptance = _validate_spec_accounting("mixed", mixed["requests"])
    mixed_repeat_acceptance = (
        _validate_spec_accounting("mixed_repeat", mixed_repeat["requests"])
        if mixed_repeat is not None
        else None
    )
    acceptance_exact = all(
        unmixed["requests"][name].get(field)
        == mixed["requests"][name].get(field)
        for name in request_names
        for field in (
            "spec_verify_ct",
            "spec_num_correct_drafts",
            "spec_num_proposed_drafts",
        )
    )

    mixed_scheduler_passed = not mixed_scheduler_mismatches
    target_oracle_passed = not target_mismatches
    full_g4_passed = mixed_scheduler_passed and target_oracle_passed
    mixed_repeat_deterministic = (
        not mixed_repeat_mismatches if mixed_repeat is not None else None
    )
    if args.require_repeat_determinism and mixed_repeat is None:
        raise AssertionError(
            "--require-repeat-determinism requires --mixed-repeat"
        )
    passed = (
        full_g4_passed
        if args.require_target_equivalence
        else mixed_scheduler_passed
    )
    if args.require_repeat_determinism:
        passed = passed and bool(mixed_repeat_deterministic)
    strict_gate = (
        "full_g4" if args.require_target_equivalence else "mixed_scheduler"
    )
    if args.require_repeat_determinism:
        strict_gate += "+repeat_determinism"
    summary = {
        "passed": passed,
        "strict_gate": strict_gate,
        "mixed_scheduler_passed": mixed_scheduler_passed,
        "target_oracle_passed": target_oracle_passed,
        "full_g4_passed": full_g4_passed,
        "comparisons": comparisons,
        "mixed_scheduler_mismatches": mixed_scheduler_mismatches,
        "mixed_repeat_mismatches": mixed_repeat_mismatches,
        "mixed_repeat_deterministic": mixed_repeat_deterministic,
        "target_mismatches": target_mismatches,
        "acceptance_accounting_valid": True,
        "acceptance_decisions_exact": acceptance_exact,
        "unmixed_acceptance": unmixed_acceptance,
        "mixed_acceptance": mixed_acceptance,
        "mixed_repeat_acceptance": mixed_repeat_acceptance,
        "mixed_accept_length_relative_delta": (
            mixed_acceptance["accept_length"]
            / unmixed_acceptance["accept_length"]
            - 1
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if mixed_scheduler_mismatches:
        raise AssertionError(
            "Mixed scheduling changed end-to-end output: "
            f"{mixed_scheduler_mismatches}"
        )
    if args.require_target_equivalence and target_mismatches:
        raise AssertionError(
            f"Full target-oracle G4 equivalence failed: {target_mismatches}"
        )
    if args.require_repeat_determinism and mixed_repeat_mismatches:
        raise AssertionError(
            "Repeated mixed capture changed end-to-end output: "
            f"{mixed_repeat_mismatches}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--base-url", default="http://127.0.0.1:31280")
    capture_parser.add_argument(
        "--mode", choices=("target", "unmixed", "mixed"), required=True
    )
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--run-id")
    capture_parser.add_argument("--flush-cache-before", action="store_true")
    capture_parser.add_argument("--serial", action="store_true")
    capture_parser.add_argument("--max-new-tokens-cap", type=int)
    capture_parser.add_argument("--decode-stagger-seconds", type=float, default=0.6)
    capture_parser.add_argument("--prefill-delay-seconds", type=float, default=0.6)
    capture_parser.add_argument(
        "--prefill-request-stagger-seconds", type=float, default=0.02
    )
    capture_parser.set_defaults(func=capture)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--target", required=True)
    compare_parser.add_argument("--unmixed", required=True)
    compare_parser.add_argument("--mixed", required=True)
    compare_parser.add_argument("--mixed-repeat")
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument(
        "--require-target-equivalence",
        action="store_true",
        help="Fail unless both speculative modes also match target-only output",
    )
    compare_parser.add_argument(
        "--require-repeat-determinism",
        action="store_true",
        help="Fail unless --mixed and --mixed-repeat have identical outputs",
    )
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
