"""Aggregate DSpark collect/server_info perf evidence.

This helper is intentionally offline-only. It consumes artifacts produced by
benchmark/dspark_accuracy_harness.py collect, preferably with both
--server-info-output and --manifest-output, and prints one compact row per run.

Examples:
  python3 benchmark/dspark_perf_report.py \
    --manifest-run dspark /tmp/dspark_manifest.json

  python3 benchmark/dspark_perf_report.py \
    --run target /tmp/target.jsonl \
    --run compact /tmp/dspark_compact.jsonl /tmp/dspark_compact_server_info.json \
    --elapsed-s target 82.4 \
    --elapsed-s compact 51.7 \
    --json-output /tmp/dspark_perf_report.json \
    --jsonl-output /tmp/dspark_perf_records.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from benchmark.dspark_profile_artifacts import (
        add_provenance_args,
        provenance_from_args,
    )
except ModuleNotFoundError:
    from dspark_profile_artifacts import add_provenance_args, provenance_from_args

try:
    from dspark_accuracy_harness import (
        iter_dspark_info_records,
        read_json_or_jsonl,
        read_jsonl,
        summarize_run,
    )
except ModuleNotFoundError:
    from benchmark.dspark_accuracy_harness import (
        iter_dspark_info_records,
        read_json_or_jsonl,
        read_jsonl,
        summarize_run,
    )


SCHEMA = "sglang-dspark-perf-report-v3"


@dataclass(frozen=True)
class RunInput:
    label: str
    collect_path: Path | None
    server_info_path: Path | None = None
    manifest_path: Path | None = None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_path_maps(values: list[str] | None) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--path-map expects FROM=TO, got {value!r}")
        src, dst = value.split("=", 1)
        if not src or not dst:
            raise SystemExit(f"--path-map expects non-empty FROM=TO, got {value!r}")
        mappings.append((src.rstrip("/"), dst.rstrip("/")))
    return mappings


def apply_path_maps(path: Path, mappings: list[tuple[str, str]]) -> Path:
    raw = str(path)
    for src, dst in mappings:
        if raw == src:
            return Path(dst)
        prefix = src + "/"
        if raw.startswith(prefix):
            return Path(dst) / raw[len(prefix) :]
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path | None, *, path_maps: list[tuple[str, str]]
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "resolved_path": None, "exists": False}
    resolved = apply_path_maps(path.expanduser(), path_maps)
    exists = resolved.exists()
    record: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": exists,
    }
    if exists and resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = sha256_file(resolved)
    return record


def resolve_artifact_path(raw_path: Any, *, manifest_path: Path | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute() or manifest_path is None:
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def manifest_artifact_path(
    manifest: dict[str, Any], key: str, *, manifest_path: Path
) -> Path | None:
    artifact = (manifest.get("artifacts") or {}).get(key) or {}
    return resolve_artifact_path(artifact.get("path"), manifest_path=manifest_path)


def run_input_from_manifest(label: str, manifest_path: Path) -> RunInput:
    manifest = read_json(manifest_path)
    summary = manifest.get("summary") or {}
    args = manifest.get("args") or {}
    resolved_label = (
        label or summary.get("run_label") or args.get("run_label") or manifest_path.stem
    )
    collect_path = manifest_artifact_path(
        manifest, "collect_output", manifest_path=manifest_path
    ) or resolve_artifact_path(summary.get("output"), manifest_path=manifest_path)
    server_info_path = manifest_artifact_path(
        manifest, "server_info", manifest_path=manifest_path
    ) or resolve_artifact_path(
        summary.get("server_info_output") or args.get("server_info_output"),
        manifest_path=manifest_path,
    )
    return RunInput(
        label=str(resolved_label),
        collect_path=collect_path,
        server_info_path=server_info_path,
        manifest_path=manifest_path,
    )


def parse_run_inputs(args: argparse.Namespace) -> list[RunInput]:
    runs: list[RunInput] = []
    for values in args.run or []:
        if not 2 <= len(values) <= 4:
            raise SystemExit(
                "--run expects LABEL COLLECT [SERVER_INFO] [MANIFEST]; "
                f"got {len(values)} values: {values!r}"
            )
        label = values[0]
        collect_path = None if values[1] == "-" else Path(values[1]).expanduser()
        server_info_path = (
            None
            if len(values) < 3 or values[2] == "-"
            else Path(values[2]).expanduser()
        )
        manifest_path = (
            None
            if len(values) < 4 or values[3] == "-"
            else Path(values[3]).expanduser()
        )
        if manifest_path is not None and (
            collect_path is None or server_info_path is None
        ):
            manifest_run = run_input_from_manifest(label, manifest_path)
            collect_path = collect_path or manifest_run.collect_path
            server_info_path = server_info_path or manifest_run.server_info_path
        runs.append(
            RunInput(
                label=label,
                collect_path=collect_path,
                server_info_path=server_info_path,
                manifest_path=manifest_path,
            )
        )

    for label, manifest in args.manifest_run or []:
        runs.append(run_input_from_manifest(label, Path(manifest).expanduser()))

    if not runs:
        raise SystemExit("Provide at least one --run or --manifest-run.")
    return runs


def parse_elapsed_overrides(args: argparse.Namespace) -> dict[str, float]:
    elapsed: dict[str, float] = {}
    for label, value in args.elapsed_s or []:
        seconds = float(value)
        if seconds <= 0:
            raise SystemExit(f"--elapsed-s for {label!r} must be positive.")
        elapsed[label] = seconds
    return elapsed


def load_optional_json(path: Path | None, *, path_maps: list[tuple[str, str]]) -> Any:
    if path is None:
        return None
    resolved = apply_path_maps(path.expanduser(), path_maps)
    return read_json(resolved)


def manifest_elapsed_s(manifest_path: Path | None) -> float | None:
    if manifest_path is None:
        return None
    manifest = read_json(manifest_path)
    summary = manifest.get("summary") or {}
    value = summary.get("elapsed_s")
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def manifest_throughput_window(
    manifest_path: Path | None,
) -> tuple[float, int | None, str] | None:
    if manifest_path is None:
        return None
    manifest = read_json(manifest_path)
    summary = manifest.get("summary") or {}
    value = summary.get("elapsed_s")
    if value is not None:
        value = float(value)
        if value > 0:
            return value, None, "manifest_summary_elapsed_s"

    value = summary.get("elapsed_s_new_requests")
    tokens = summary.get("completion_tokens_new_requests")
    if value is None or tokens is None:
        return None
    value = float(value)
    tokens = int(tokens)
    if value <= 0 or tokens <= 0:
        return None
    return value, tokens, "manifest_summary_elapsed_s_new_requests"


def summarize_collect(
    path: Path, *, path_maps: list[tuple[str, str]]
) -> dict[str, Any]:
    rows = read_jsonl(apply_path_maps(path, path_maps))
    summary = summarize_run(rows)
    fallback_completion_tokens = sum(
        int(row.get("completion_tokens") or 0) for row in rows if row.get("ok")
    )
    if not summary.get("completion_tokens") and fallback_completion_tokens:
        summary["completion_tokens"] = fallback_completion_tokens
    return summary


def nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def quantile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_timing(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": quantile(values, 50),
        "p90": quantile(values, 90),
        "max": max(values),
    }


def summarize_server_info(
    path: Path | None, *, path_maps: list[tuple[str, str]]
) -> dict[str, Any]:
    if path is None:
        return {}

    dumps = list(
        iter_dspark_info_records(read_json_or_jsonl(apply_path_maps(path, path_maps)))
    )
    records_with_cfg = [
        (dump, record)
        for dump in dumps
        for record in dump.get("records", [])
        if isinstance(record, dict)
    ]
    compact_records = 0
    non_uniform_records = 0
    padded_graph_records = 0
    full_verify_token_sum = 0
    scheduled_verify_token_sum = 0
    graph_token_sum = 0
    request_rows = 0
    max_bs = 0

    for dump, record in records_with_cfg:
        mode = record.get("mode") or dump.get("mode")
        if mode == "compact":
            compact_records += 1

        verify_width = int(dump.get("verify_num_draft_tokens") or 0)
        reqs = record.get("reqs") or []
        bs = int(record.get("bs") or len(reqs) or 0)
        max_bs = max(max_bs, bs)
        request_rows += len(reqs)

        verify_lens = [
            int(req["verify_len"])
            for req in reqs
            if isinstance(req, dict) and req.get("verify_len") is not None
        ]
        if verify_lens and len(set(verify_lens)) > 1:
            non_uniform_records += 1

        num_verify_tokens = nonnegative_int(record.get("num_verify_tokens"))
        if verify_lens:
            scheduled_tokens = sum(verify_lens)
        else:
            scheduled_tokens = num_verify_tokens or 0

        graph_tokens = nonnegative_int(record.get("verify_tokens_graph_key"))
        if graph_tokens is None:
            graph_tokens = max(scheduled_tokens, num_verify_tokens or 0)
        if graph_tokens > scheduled_tokens:
            padded_graph_records += 1

        if verify_width and bs:
            full_verify_token_sum += bs * verify_width
        scheduled_verify_token_sum += scheduled_tokens
        graph_token_sum += graph_tokens

    saved_verify_tokens = full_verify_token_sum - scheduled_verify_token_sum
    return {
        "dumps": len(dumps),
        "records": len(records_with_cfg),
        "request_rows": request_rows,
        "compact_records": compact_records,
        "non_uniform_verify_lens_records": non_uniform_records,
        "padded_graph_records": padded_graph_records,
        "full_verify_token_sum": full_verify_token_sum,
        "scheduled_verify_token_sum": scheduled_verify_token_sum,
        "saved_verify_tokens": saved_verify_tokens,
        "scheduled_verify_token_ratio": (
            scheduled_verify_token_sum / full_verify_token_sum
            if full_verify_token_sum
            else None
        ),
        "graph_token_sum": graph_token_sum,
        "graph_padding_tokens": graph_token_sum - scheduled_verify_token_sum,
        "max_bs": max_bs,
        "timing_ms": {
            "step_cpu": summarize_timing(
                [record for _dump, record in records_with_cfg], "step_cpu_ms"
            ),
            "step_gpu": summarize_timing(
                [record for _dump, record in records_with_cfg], "step_gpu_ms"
            ),
            "draft_gpu": summarize_timing(
                [record for _dump, record in records_with_cfg], "draft_gpu_ms"
            ),
            "target_verify_gpu": summarize_timing(
                [record for _dump, record in records_with_cfg],
                "target_verify_gpu_ms",
            ),
        },
    }


def build_record(
    run: RunInput,
    *,
    elapsed_overrides: dict[str, float],
    path_maps: list[tuple[str, str]],
) -> dict[str, Any]:
    if run.collect_path is None:
        raise SystemExit(f"Run {run.label!r} has no collect JSONL path.")
    collect = summarize_collect(run.collect_path, path_maps=path_maps)
    info = summarize_server_info(run.server_info_path, path_maps=path_maps)
    completion_tokens = int(collect.get("completion_tokens") or 0)

    elapsed_s = elapsed_overrides.get(run.label)
    throughput_basis = None
    throughput_completion_tokens = completion_tokens
    if elapsed_s is not None:
        throughput_basis = "elapsed_override"
    else:
        throughput_window = manifest_throughput_window(run.manifest_path)
        if throughput_window is not None:
            elapsed_s, tokens, throughput_basis = throughput_window
            if tokens is not None:
                throughput_completion_tokens = tokens

    throughput = (
        throughput_completion_tokens / elapsed_s
        if elapsed_s is not None and elapsed_s > 0 and throughput_completion_tokens
        else None
    )
    accept_length = (
        collect.get("aggregate_accept_length_including_bonus")
        or collect.get("aggregate_accept_length")
        or collect.get("mean_accept_length")
    )
    timing = info.get("timing_ms") or {}
    step_gpu = timing.get("step_gpu") or {}
    draft_gpu = timing.get("draft_gpu") or {}
    target_verify_gpu = timing.get("target_verify_gpu") or {}

    return {
        "label": run.label,
        "collect_path": str(run.collect_path),
        "server_info_path": str(run.server_info_path) if run.server_info_path else None,
        "manifest_path": str(run.manifest_path) if run.manifest_path else None,
        "artifacts": {
            "collect": artifact_record(run.collect_path, path_maps=path_maps),
            "server_info": artifact_record(run.server_info_path, path_maps=path_maps),
            "manifest": artifact_record(run.manifest_path, path_maps=path_maps),
        },
        "requests": collect.get("requests"),
        "ok_requests": collect.get("ok_requests"),
        "error_requests": collect.get("error_requests"),
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed_s,
        "throughput_tokens_s": throughput,
        "throughput_completion_tokens": throughput_completion_tokens,
        "throughput_basis": throughput_basis,
        "ar": collect.get("aggregate_accept_rate"),
        "al": accept_length,
        "draft_al": collect.get("aggregate_draft_accept_length"),
        "mean_ar": collect.get("mean_accept_rate"),
        "mean_al": collect.get("mean_accept_length"),
        "spec_metric_rows": collect.get("spec_metric_rows"),
        "spec_completion_tokens": collect.get("spec_completion_tokens"),
        "server_info_records": info.get("records"),
        "server_info_request_rows": info.get("request_rows"),
        "compact_records": info.get("compact_records"),
        "saved_verify_tokens": info.get("saved_verify_tokens"),
        "scheduled_verify_token_ratio": info.get("scheduled_verify_token_ratio"),
        "full_verify_token_sum": info.get("full_verify_token_sum"),
        "scheduled_verify_token_sum": info.get("scheduled_verify_token_sum"),
        "non_uniform_verify_lens_records": info.get("non_uniform_verify_lens_records"),
        "padded_graph_records": info.get("padded_graph_records"),
        "graph_padding_tokens": info.get("graph_padding_tokens"),
        "max_bs": info.get("max_bs"),
        "step_gpu_mean_ms": step_gpu.get("mean"),
        "draft_gpu_mean_ms": draft_gpu.get("mean"),
        "target_verify_gpu_mean_ms": target_verify_gpu.get("mean"),
        "timing_ms": timing,
    }


def build_evidence(
    args: argparse.Namespace, *, path_maps: list[tuple[str, str]]
) -> dict[str, Any]:
    trace_summary = load_optional_json(args.trace_summary, path_maps=path_maps)
    return {
        "trace_summary": trace_summary,
        "artifacts": {
            "trace_summary": artifact_record(args.trace_summary, path_maps=path_maps),
            "sps_table": artifact_record(args.sps_table, path_maps=path_maps),
            "sps_manifest": artifact_record(args.sps_manifest, path_maps=path_maps),
            "sts_calibration": artifact_record(
                args.sts_calibration, path_maps=path_maps
            ),
        },
    }


def annotate_speedups(
    records: list[dict[str, Any]], *, baseline_label: str | None
) -> None:
    if not baseline_label:
        return
    baseline = next(
        (record for record in records if record["label"] == baseline_label), None
    )
    if baseline is None:
        for record in records:
            record["baseline_label"] = baseline_label
            record["speedup_vs_baseline"] = None
        return
    baseline_tps = baseline.get("throughput_tokens_s")
    for record in records:
        record["baseline_label"] = baseline_label
        record_tps = record.get("throughput_tokens_s")
        if baseline_tps is None or record_tps is None or float(baseline_tps) <= 0:
            record["speedup_vs_baseline"] = None
            record["throughput_delta_tokens_s"] = None
            continue
        record["speedup_vs_baseline"] = float(record_tps) / float(baseline_tps)
        record["throughput_delta_tokens_s"] = float(record_tps) - float(baseline_tps)


def aggregate_evidence_metric(
    records: list[dict[str, Any]],
    evidence: dict[str, Any],
    key: str,
) -> tuple[int, str]:
    trace_summary = evidence.get("trace_summary")
    if isinstance(trace_summary, dict) and trace_summary.get(key) is not None:
        return int(trace_summary.get(key) or 0), "trace_summary"
    return sum(int(record.get(key) or 0) for record in records), "runs"


def build_verdict(
    records: list[dict[str, Any]],
    evidence: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    failures: list[str] = []
    labels = {record["label"] for record in records}
    baseline_label = getattr(args, "baseline_label", None)
    if (
        getattr(args, "min_speedup_vs_baseline", None) is not None
        and not baseline_label
    ):
        failures.append("--min-speedup-vs-baseline requires --baseline-label")
    if baseline_label and baseline_label not in labels:
        failures.append(f"baseline label {baseline_label!r} was not found")

    for record in records:
        label = record["label"]
        has_spec_metrics = int(record.get("spec_metric_rows") or 0) > 0
        if (
            args.min_ok_requests is not None
            and int(record.get("ok_requests") or 0) < args.min_ok_requests
        ):
            failures.append(
                f"{label}: ok_requests {record.get('ok_requests')} is below {args.min_ok_requests}"
            )
        has_ar = has_spec_metrics or record.get("ar") is not None
        has_al = has_spec_metrics or record.get("al") is not None
        if (
            args.min_ar is not None
            and has_ar
            and (record.get("ar") is None or float(record["ar"]) < args.min_ar)
        ):
            failures.append(f"{label}: AR {record.get('ar')} is below {args.min_ar}")
        if (
            args.min_al is not None
            and has_al
            and (record.get("al") is None or float(record["al"]) < args.min_al)
        ):
            failures.append(f"{label}: AL {record.get('al')} is below {args.min_al}")
        if (
            getattr(args, "min_speedup_vs_baseline", None) is not None
            and label != baseline_label
        ):
            speedup = record.get("speedup_vs_baseline")
            if speedup is None or float(speedup) < args.min_speedup_vs_baseline:
                failures.append(
                    f"{label}: speedup_vs_baseline {speedup} is below "
                    f"{args.min_speedup_vs_baseline}"
                )

    compact_records, compact_source = aggregate_evidence_metric(
        records, evidence, "compact_records"
    )
    if args.require_compact and compact_records <= 0:
        failures.append(
            "no compact records were observed across runs/evidence "
            f"(source={compact_source})"
        )

    non_uniform_records, non_uniform_source = aggregate_evidence_metric(
        records, evidence, "non_uniform_verify_lens_records"
    )
    if args.require_non_uniform_verify_lens and non_uniform_records <= 0:
        failures.append(
            "no non-uniform verify-lens records were observed across "
            f"runs/evidence (source={non_uniform_source})"
        )

    non_greedy_records, non_greedy_source = aggregate_evidence_metric(
        records, evidence, "non_greedy_records"
    )
    if getattr(args, "require_non_greedy", False) and non_greedy_records <= 0:
        failures.append(
            "no non-greedy records were observed across runs/evidence "
            f"(source={non_greedy_source})"
        )

    seeded_records, seeded_source = aggregate_evidence_metric(
        records, evidence, "seeded_sampling_records"
    )
    if getattr(args, "require_seeded_sampling", False) and seeded_records <= 0:
        failures.append(
            "no seeded sampling records were observed across runs/evidence "
            f"(source={seeded_source})"
        )

    skipped_records, skipped_source = aggregate_evidence_metric(
        records, evidence, "skipped_records"
    )
    if getattr(args, "require_no_skipped", False) and skipped_records > 0:
        failures.append(
            f"{skipped_records} skipped records were observed across "
            f"runs/evidence (source={skipped_source})"
        )

    if getattr(args, "require_non_greedy_accept_coverage", False):
        covered_records, covered_source = aggregate_evidence_metric(
            records, evidence, "non_greedy_accept_covered_records"
        )
        uncovered_records, uncovered_source = aggregate_evidence_metric(
            records, evidence, "non_greedy_accept_uncovered_records"
        )
        if non_greedy_records <= 0:
            failures.append(
                "non-greedy accept coverage was required, but no non-greedy "
                f"records were observed (source={non_greedy_source})"
            )
        elif uncovered_records > 0:
            failures.append(
                f"{uncovered_records} non-greedy records lacked accept coverage "
                f"(source={uncovered_source})"
            )
        elif covered_records < non_greedy_records:
            failures.append(
                "non-greedy accept coverage is incomplete: "
                f"covered={covered_records} from {covered_source}, "
                f"non_greedy={non_greedy_records} from {non_greedy_source}"
            )

    saved_verify_tokens, saved_source = aggregate_evidence_metric(
        records, evidence, "saved_verify_tokens"
    )
    if (
        args.min_saved_verify_tokens is not None
        and saved_verify_tokens < args.min_saved_verify_tokens
    ):
        failures.append(
            f"saved_verify_tokens {saved_verify_tokens} from {saved_source} "
            f"is below {args.min_saved_verify_tokens}"
        )

    artifacts = evidence.get("artifacts") or {}
    if args.require_sps_table and not (artifacts.get("sps_table") or {}).get("exists"):
        failures.append("required SPS table artifact is missing")
    if args.require_sts_calibration and not (
        artifacts.get("sts_calibration") or {}
    ).get("exists"):
        failures.append("required STS calibration artifact is missing")

    trace_summary = evidence.get("trace_summary")
    if getattr(args, "require_trace_verdict", False):
        if not isinstance(trace_summary, dict):
            failures.append("trace summary is required for --require-trace-verdict")
        else:
            trace_verdict = trace_summary.get("verdict")
            if (
                not isinstance(trace_verdict, dict)
                or trace_verdict.get("passed") is not True
            ):
                failures.append("trace summary verdict did not pass")

    if args.expect_target_verify_eager or args.expect_target_verify_graph:
        if not isinstance(trace_summary, dict):
            failures.append(
                "trace summary is required for target-verify graph/eager gates"
            )
        else:
            cuda_graph_records = int(trace_summary.get("cuda_graph_records") or 0)
            eager_records = int(trace_summary.get("eager_records") or 0)
            if args.expect_target_verify_eager and eager_records <= 0:
                failures.append("expected eager target-verify records, observed none")
            if args.expect_target_verify_graph and cuda_graph_records <= 0:
                failures.append("expected graph target-verify records, observed none")

    return {"passed": not failures, "failures": failures}


def sort_records(records: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "input":
        return records
    reverse = sort_by in {
        "ar",
        "al",
        "throughput_tokens_s",
        "saved_verify_tokens",
        "compact_records",
    }

    def key(record: dict[str, Any]):
        value = record.get(sort_by)
        if value is None:
            return float("-inf") if reverse else float("inf")
        return value

    return sorted(records, key=key, reverse=reverse)


def fmt_int(value: Any) -> str:
    return "n/a" if value is None else str(int(value))


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(
    records: list[dict[str, Any]], evidence: dict[str, Any] | None = None
) -> str:
    include_speedup = any(
        record.get("speedup_vs_baseline") is not None for record in records
    )
    include_timing = any(
        record.get("target_verify_gpu_mean_ms") is not None
        or record.get("draft_gpu_mean_ms") is not None
        for record in records
    )
    headers = [
        "run",
        "ok/req",
        "AR",
        "AL",
        "tokens/s",
        "compact",
        "saved verify",
        "scheduled ratio",
    ]
    if include_speedup:
        headers.insert(5, "speedup")
    if include_timing:
        headers.extend(["draft ms", "target verify ms"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in records:
        ok_req = (
            f"{fmt_int(record.get('ok_requests'))}/{fmt_int(record.get('requests'))}"
        )
        row = [
            str(record["label"]),
            ok_req,
            fmt_float(record.get("ar")),
            fmt_float(record.get("al")),
            fmt_float(record.get("throughput_tokens_s"), digits=1),
            fmt_int(record.get("compact_records")),
            fmt_int(record.get("saved_verify_tokens")),
            fmt_float(record.get("scheduled_verify_token_ratio")),
        ]
        if include_speedup:
            row.insert(5, fmt_float(record.get("speedup_vs_baseline"), digits=3) + "x")
        if include_timing:
            row.extend(
                [
                    fmt_float(record.get("draft_gpu_mean_ms"), digits=2),
                    fmt_float(record.get("target_verify_gpu_mean_ms"), digits=2),
                ]
            )
        lines.append("| " + " | ".join(row) + " |")
    trace_summary = (evidence or {}).get("trace_summary")
    if isinstance(trace_summary, dict):
        trace_verdict = trace_summary.get("verdict") or {}
        lines.extend(
            [
                "",
                "Trace evidence: "
                f"verdict={trace_verdict.get('passed')}, "
                f"records={fmt_int(trace_summary.get('records'))}, "
                f"compact={fmt_int(trace_summary.get('compact_records'))}, "
                "non_uniform="
                f"{fmt_int(trace_summary.get('non_uniform_verify_lens_records'))}, "
                f"saved_verify={fmt_int(trace_summary.get('saved_verify_tokens'))}, "
                "scheduled_ratio="
                f"{fmt_float(trace_summary.get('scheduled_verify_token_ratio'))}, "
                f"eager={fmt_int(trace_summary.get('eager_records'))}, "
                f"cuda_graph={fmt_int(trace_summary.get('cuda_graph_records'))}",
            ]
        )
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize multiple DSpark collect/server_info artifacts into "
            "AR, AL, throughput tokens/s, compact records, saved verify tokens, "
            "and scheduled verify-token ratio."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs="+",
        metavar="VALUE",
        help=(
            "Run spec: LABEL COLLECT [SERVER_INFO] [MANIFEST]. Repeat for multiple "
            "runs. Use '-' for COLLECT or SERVER_INFO to resolve that path from "
            "the manifest."
        ),
    )
    parser.add_argument(
        "--manifest-run",
        action="append",
        nargs=2,
        metavar=("LABEL", "MANIFEST"),
        help=(
            "Run spec from a collect manifest produced by "
            "dspark_accuracy_harness.py collect --manifest-output."
        ),
    )
    parser.add_argument(
        "--elapsed-s",
        action="append",
        nargs=2,
        metavar=("LABEL", "SECONDS"),
        help=(
            "Wall-clock elapsed seconds override for throughput. Needed when a "
            "collect manifest with summary.elapsed_s is not available."
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=[
            "input",
            "label",
            "ar",
            "al",
            "throughput_tokens_s",
            "saved_verify_tokens",
            "compact_records",
        ],
        default="input",
    )
    parser.add_argument(
        "--baseline-label",
        help=(
            "Optional run label used as the throughput baseline. When set, each "
            "run gets speedup_vs_baseline and throughput_delta_tokens_s."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--jsonl-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument(
        "--require-throughput",
        action="store_true",
        help="Exit nonzero if any run lacks manifest/override elapsed seconds.",
    )
    parser.add_argument(
        "--trace-summary",
        type=Path,
        help="Optional trace-summary JSON from dspark_accuracy_harness.py.",
    )
    parser.add_argument("--sps-table", type=Path)
    parser.add_argument("--sps-manifest", type=Path)
    parser.add_argument("--sts-calibration", type=Path)
    parser.add_argument(
        "--path-map",
        action="append",
        help="Map artifact path prefixes, e.g. /artifacts=/tmp/run-artifacts.",
    )
    parser.add_argument("--min-ok-requests", type=int)
    parser.add_argument("--min-ar", type=float)
    parser.add_argument("--min-al", type=float)
    parser.add_argument("--min-speedup-vs-baseline", type=float)
    parser.add_argument("--require-compact", action="store_true")
    parser.add_argument("--require-non-uniform-verify-lens", action="store_true")
    parser.add_argument("--require-non-greedy", action="store_true")
    parser.add_argument("--require-seeded-sampling", action="store_true")
    parser.add_argument("--require-non-greedy-accept-coverage", action="store_true")
    parser.add_argument("--require-no-skipped", action="store_true")
    parser.add_argument("--min-saved-verify-tokens", type=int)
    parser.add_argument("--require-sps-table", action="store_true")
    parser.add_argument("--require-sts-calibration", action="store_true")
    parser.add_argument("--require-trace-verdict", action="store_true")
    parser.add_argument("--expect-target-verify-eager", action="store_true")
    parser.add_argument("--expect-target-verify-graph", action="store_true")
    parser.add_argument("--fail-on-verdict", action="store_true")
    add_provenance_args(parser)
    args = parser.parse_args()

    path_maps = parse_path_maps(args.path_map)
    elapsed_overrides = parse_elapsed_overrides(args)
    records = [
        build_record(run, elapsed_overrides=elapsed_overrides, path_maps=path_maps)
        for run in parse_run_inputs(args)
    ]
    records = sort_records(records, args.sort_by)
    annotate_speedups(records, baseline_label=args.baseline_label)
    evidence = build_evidence(args, path_maps=path_maps)
    verdict = build_verdict(records, evidence, args)

    missing_throughput = [
        record["label"] for record in records if record["throughput_tokens_s"] is None
    ]
    if args.require_throughput and missing_throughput:
        raise SystemExit(
            "Missing throughput for run(s): "
            + ", ".join(missing_throughput)
            + ". Provide --manifest-run or --elapsed-s."
        )

    report = {
        "schema": SCHEMA,
        "created_unix_s": time.time(),
        "provenance": provenance_from_args(args),
        "path_maps": path_maps,
        "runs": records,
        "evidence": evidence,
        "verdict": verdict,
    }
    if args.json_output:
        write_text(args.json_output, json.dumps(report, indent=2, sort_keys=True))
    if args.jsonl_output:
        write_jsonl(args.jsonl_output, records)

    markdown = render_markdown(records, evidence=evidence)
    if args.markdown_output:
        write_text(args.markdown_output, markdown)
    if not args.no_markdown:
        print(markdown)
    if args.fail_on_verdict and not verdict["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
