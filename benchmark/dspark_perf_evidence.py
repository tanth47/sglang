"""Build a PR-ready DSpark performance evidence bundle.

This helper intentionally does not launch a server or run traffic. Existing
tools already own that:

- ``benchmark/dspark_accuracy_harness.py`` for accuracy and AR/AL.
- ``python -m sglang.benchmark.serving`` for throughput and latency.
- ``python -m sglang.profiler`` plus DSpark debug dumps for profiling.

This script ties those artifacts together into one manifest and one short
Markdown report so DSpark performance PRs can show accuracy, throughput, GPU
timing, and launch/profile artifacts in a consistent format.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

SERVING_KEYS = [
    "backend",
    "dataset_name",
    "request_rate",
    "max_concurrency",
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "concurrency",
    "accept_length",
]

DSPARK_TIMING_FIELDS = [
    "step_cpu_ms",
    "step_gpu_ms",
    "draft_gpu_ms",
    "target_verify_gpu_ms",
]


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    obj = read_json(path)
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"Unsupported JSON payload in {path}: {type(obj).__name__}")


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def quantile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    upper_weight = pos - lower
    return ordered[lower] * (1 - upper_weight) + ordered[upper] * upper_weight


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def numeric_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        value = numeric(record.get(field))
        if value is not None:
            values.append(value)
    return values


def summarize_numeric(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = numeric_values(records, field)
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": quantile(values, 50),
        "p90": quantile(values, 90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_unattributed_gpu_ms(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        step = numeric(record.get("step_gpu_ms"))
        draft = numeric(record.get("draft_gpu_ms"))
        verify = numeric(record.get("target_verify_gpu_ms"))
        if step is None or draft is None or verify is None:
            continue
        rows.append({"unattributed_gpu_ms": step - draft - verify})
    return summarize_numeric(rows, "unattributed_gpu_ms")


def summarize_accuracy(path: str | Path) -> dict[str, Any]:
    obj = read_json(path)
    if "spec_run" in obj and isinstance(obj["spec_run"], dict):
        run = obj["spec_run"]
        compare = obj
    else:
        run = obj
        compare = None

    verdict = obj.get("verdict") if isinstance(obj.get("verdict"), dict) else {}
    return {
        "source": str(path),
        "requests": run.get("requests"),
        "ok_requests": run.get("ok_requests"),
        "error_requests": run.get("error_requests"),
        "completion_tokens": run.get("completion_tokens"),
        "spec_metric_rows": run.get("spec_metric_rows"),
        "aggregate_accept_rate": run.get("aggregate_accept_rate"),
        "aggregate_accept_length": run.get("aggregate_accept_length"),
        "mean_accept_rate": run.get("mean_accept_rate"),
        "mean_accept_length": run.get("mean_accept_length"),
        "token_exact_match_rate": (
            compare.get("token_exact_match_rate") if compare else None
        ),
        "verdict_passed": verdict.get("passed"),
        "verdict_failures": verdict.get("failures") or [],
        "verdict_warnings": verdict.get("warnings") or [],
    }


def summarize_serving_output(path: str | Path) -> dict[str, Any]:
    records = load_json_records(path)
    summary_records: list[dict[str, Any]] = []
    for record in records:
        summary_records.append(
            {
                key: record.get(key)
                for key in SERVING_KEYS
                if key in record and record.get(key) is not None
            }
        )
    latest = summary_records[-1] if summary_records else {}
    return {
        "source": str(path),
        "records": len(summary_records),
        "latest": latest,
    }


def extract_dspark_info(obj: dict[str, Any]) -> dict[str, Any] | None:
    if "records" in obj and isinstance(obj["records"], list):
        return obj
    if "dspark_info_record" in obj and isinstance(obj["dspark_info_record"], dict):
        return obj["dspark_info_record"]
    internal_states = obj.get("internal_states")
    if isinstance(internal_states, list):
        for state in internal_states:
            if not isinstance(state, dict):
                continue
            info = state.get("dspark_info_record")
            if isinstance(info, dict):
                return info
    return None


def summarize_dspark_info(path: str | Path) -> dict[str, Any]:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected DSpark info object in {path}")
    info = extract_dspark_info(obj)
    if info is None:
        raise ValueError(f"No DSpark info record found in {path}")

    records = [r for r in info.get("records", []) if isinstance(r, dict)]
    timing = {
        field: summarize_numeric(records, field) for field in DSPARK_TIMING_FIELDS
    }
    unattributed = summarize_unattributed_gpu_ms(records)

    return {
        "source": str(path),
        "mode": info.get("mode"),
        "gamma": info.get("gamma"),
        "verify_num_draft_tokens": info.get("verify_num_draft_tokens"),
        "components": info.get("components") or [],
        "records": len(records),
        "timing_ms": timing,
        "unattributed_gpu_ms": unattributed,
        "mean_unattributed_gpu_ms": unattributed["mean"],
    }


def first_numeric_arg(args: dict[str, Any]) -> float | None:
    for value in args.values():
        number = numeric(value)
        if number is not None:
            return number
    return None


def summarize_trace_idle(events: list[dict[str, Any]]) -> dict[str, Any]:
    idle_points: list[tuple[float, float]] = []
    queue_points: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "C":
            continue
        name = str(event.get("name", "")).lower()
        timestamp = numeric(event.get("ts"))
        value = first_numeric_arg(event.get("args") or {})
        if timestamp is None or value is None:
            continue
        if name == "idle":
            idle_points.append((timestamp, value))
        elif name in ("queuedepth", "queue_depth", "queue depth"):
            queue_points.append((timestamp, value))

    result: dict[str, Any] = {
        "idle_counter_events": len(idle_points),
        "queue_depth_events": len(queue_points),
    }
    if len(idle_points) >= 2:
        idle_points.sort()
        total_duration = 0.0
        idle_duration = 0.0
        for idx, (timestamp, value) in enumerate(idle_points[:-1]):
            next_timestamp = idle_points[idx + 1][0]
            duration = max(0.0, next_timestamp - timestamp)
            total_duration += duration
            if value > 0:
                idle_duration += duration
        result.update(
            {
                "idle_time_us": idle_duration,
                "observed_time_us": total_duration,
                "idle_fraction": (
                    idle_duration / total_duration if total_duration else None
                ),
            }
        )
    if queue_points:
        queue_values = [value for _, value in queue_points]
        result.update(
            {
                "mean_queue_depth": mean(queue_values),
                "max_queue_depth": max(queue_values),
            }
        )
    return result


def summarize_profile_trace(path: Path, max_bytes: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    if summary["size_bytes"] > max_bytes:
        summary["skipped"] = f"larger than max_profile_json_bytes={max_bytes}"
        return summary
    obj = read_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("traceEvents"), list):
        summary.update(summarize_trace_idle(obj["traceEvents"]))
    else:
        summary["skipped"] = "not a Chrome trace JSON object"
    return summary


def summarize_profile_dir(
    path: str | Path, max_profile_json_bytes: int
) -> dict[str, Any]:
    root = Path(path)
    files = [p for p in root.rglob("*") if p.is_file()]
    server_args_path = root / "server_args.json"
    trace_summaries = []
    for candidate in files:
        if candidate.name == "server_args.json" or candidate.suffix != ".json":
            continue
        trace_summaries.append(
            summarize_profile_trace(candidate, max_profile_json_bytes)
        )
    return {
        "source": str(path),
        "files": len(files),
        "server_args_json": (
            str(server_args_path) if server_args_path.exists() else None
        ),
        "trace_summaries": trace_summaries,
    }


def load_optional_json(path: str | None) -> Any:
    return read_json(path) if path else None


def build_manifest(args) -> dict[str, Any]:
    accuracy = [
        summarize_accuracy(path) for path in getattr(args, "accuracy_summary", []) or []
    ]
    serving = [
        summarize_serving_output(path)
        for path in getattr(args, "serving_output", []) or []
    ]
    dspark = [
        summarize_dspark_info(path)
        for path in getattr(args, "dspark_info_json", []) or []
    ]
    profiles = [
        summarize_profile_dir(path, args.max_profile_json_bytes)
        for path in getattr(args, "profile_dir", []) or []
    ]

    return {
        "schema_version": 1,
        "label": args.label,
        "created_at_unix": time.time(),
        "node": args.node,
        "gpu_ids": args.gpu_ids,
        "image": args.image,
        "server_command": (
            Path(args.server_command_file).read_text(encoding="utf-8")
            if args.server_command_file
            else None
        ),
        "server_env": load_optional_json(args.server_env_json),
        "accuracy": accuracy,
        "serving": serving,
        "dspark": dspark,
        "profiles": profiles,
        "notes": args.notes or [],
    }


def fmt(value: Any, digits: int = 2) -> str:
    number = numeric(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def fmt_pct(value: Any) -> str:
    number = numeric(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        f"# DSpark Perf Evidence: {manifest.get('label') or 'run'}",
        "",
        "## Run Context",
        "",
        f"- Node: `{manifest.get('node') or '-'}`",
        f"- GPU IDs: `{manifest.get('gpu_ids') or '-'}`",
        f"- Image: `{manifest.get('image') or '-'}`",
        "",
    ]

    accuracy_rows = []
    for item in manifest["accuracy"]:
        accuracy_rows.append(
            [
                item["source"],
                str(item.get("ok_requests") or "-"),
                str(item.get("error_requests") or "-"),
                fmt_pct(item.get("aggregate_accept_rate")),
                fmt(item.get("aggregate_accept_length")),
                str(item.get("verdict_passed")),
            ]
        )
    lines.extend(["## Accuracy", ""])
    lines.extend(
        markdown_table(
            ["source", "ok", "errors", "aggregate AR", "aggregate AL", "pass"],
            accuracy_rows,
        )
        or ["No accuracy summary provided."]
    )
    lines.append("")

    serving_rows = []
    for item in manifest["serving"]:
        latest = item.get("latest") or {}
        serving_rows.append(
            [
                item["source"],
                str(latest.get("completed") or "-"),
                fmt(latest.get("request_throughput")),
                fmt(latest.get("output_throughput")),
                fmt(latest.get("mean_ttft_ms")),
                fmt(latest.get("mean_tpot_ms")),
                fmt(latest.get("p99_itl_ms")),
                fmt(latest.get("accept_length")),
            ]
        )
    lines.extend(["## Serving Throughput", ""])
    lines.extend(
        markdown_table(
            [
                "source",
                "completed",
                "req/s",
                "out tok/s",
                "mean TTFT ms",
                "mean TPOT ms",
                "p99 ITL ms",
                "accept len",
            ],
            serving_rows,
        )
        or ["No serving benchmark output provided."]
    )
    lines.append("")

    dspark_rows = []
    for item in manifest["dspark"]:
        timing = item.get("timing_ms") or {}
        dspark_rows.append(
            [
                item["source"],
                str(item.get("records") or "-"),
                fmt((timing.get("step_gpu_ms") or {}).get("mean")),
                fmt((timing.get("draft_gpu_ms") or {}).get("mean")),
                fmt((timing.get("target_verify_gpu_ms") or {}).get("mean")),
                fmt(item.get("mean_unattributed_gpu_ms")),
            ]
        )
    lines.extend(["## DSpark Timing", ""])
    lines.extend(
        markdown_table(
            [
                "source",
                "records",
                "mean step GPU ms",
                "mean draft GPU ms",
                "mean verify GPU ms",
                "mean unattributed GPU ms",
            ],
            dspark_rows,
        )
        or ["No DSpark timing dump provided."]
    )
    lines.append("")

    profile_rows = []
    for profile in manifest["profiles"]:
        idle_fractions = [
            (trace.get("idle_fraction"), trace.get("path"))
            for trace in profile.get("trace_summaries", [])
            if trace.get("idle_fraction") is not None
        ]
        best_idle = idle_fractions[0] if idle_fractions else (None, None)
        profile_rows.append(
            [
                profile["source"],
                str(profile.get("files") or 0),
                profile.get("server_args_json") or "-",
                fmt_pct(best_idle[0]),
                best_idle[1] or "-",
            ]
        )
    lines.extend(["## Profile Artifacts", ""])
    lines.extend(
        markdown_table(
            ["source", "files", "server_args", "idle fraction", "idle source"],
            profile_rows,
        )
        or ["No profile directory provided."]
    )
    lines.append("")

    if manifest.get("notes"):
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in manifest["notes"])
        lines.append("")

    lines.extend(
        [
            "## Required Follow-up For Perf Claims",
            "",
            "- Include the manifest JSON and this report in the PR body or artifacts.",
            "- A speedup claim needs both serving throughput/latency and profile "
            "attribution.",
            "- If no RPD/trace idle counter is present, do not claim GPU bubble "
            "reduction.",
            "",
        ]
    )
    return "\n".join(lines)


def command_summarize(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = out_dir / "dspark_perf_manifest.json"
    report_path = out_dir / "dspark_perf_report.md"
    write_json(manifest_path, manifest)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


def add_summarize(subparsers) -> None:
    parser = subparsers.add_parser("summarize")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="dspark-perf")
    parser.add_argument("--node")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--image")
    parser.add_argument("--server-command-file")
    parser.add_argument("--server-env-json")
    parser.add_argument("--accuracy-summary", action="append", default=[])
    parser.add_argument("--serving-output", action="append", default=[])
    parser.add_argument("--dspark-info-json", action="append", default=[])
    parser.add_argument("--profile-dir", action="append", default=[])
    parser.add_argument("--notes", action="append", default=[])
    parser.add_argument("--max-profile-json-bytes", type=int, default=256 * 1024 * 1024)
    parser.set_defaults(func=command_summarize)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_summarize(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
