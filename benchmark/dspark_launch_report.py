#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from benchmark.dspark_profile_artifacts import (
        add_provenance_args,
        provenance_from_args,
    )
except ModuleNotFoundError:
    from dspark_profile_artifacts import add_provenance_args, provenance_from_args

SCHEMA = "sglang-dspark-launch-report-v6"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
START_RE = re.compile(
    r"^[A-Z][a-z]{2} [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}:\d{2} UTC \d{4}$"
)
TIMESTAMP_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: TP(?P<rank>\d+))?\]"
)
LOAD_RE = re.compile(
    r"Load weight end\. elapsed=(?P<elapsed>[0-9.]+) s, type=(?P<type>[^,\n]+)"
)
LOAD_BEGIN_RE = re.compile(r"Load weight begin\.")
GRAPH_BEGIN_RE = re.compile(r"Capture draft verify CUDA graph begin\.")
GRAPH_RE = re.compile(
    r"Capture draft verify CUDA graph end\. elapsed=(?P<elapsed>[0-9.]+) s"
)
BUILD_START_RE = re.compile(r"start build \[(?P<module>[^\]]+)\] under (?P<path>\S+)")
BUILD_FINISH_RE = re.compile(
    r"finish build \[(?P<module>[^\]]+)\], cost (?P<cost>[0-9.]+)s"
)
IMPORT_RE = re.compile(r"import \[(?P<module>[^\]]+)\] under (?P<path>\S+)")
TUNED_MISS_RE = re.compile(
    r"shape is M:(?P<m>\d+), N:(?P<n>\d+), K:(?P<k>\d+).*"
    r"not found tuned config in (?P<config>\S+)"
)
SHARD_PROGRESS_RE = re.compile(
    r"Multi-thread loading shards:\s+"
    r"(?P<pct>\d+)% Completed \| (?P<done>\d+)/(?P<total>\d+) "
    r"\[(?P<elapsed>[0-9:]+)(?:<|,)"
)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    return entry


def parse_log_timestamp(line: str) -> tuple[datetime | None, int | None]:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None, None
    ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    rank = match.group("rank")
    return ts, int(rank) if rank is not None else None


def parse_start_line(line: str) -> datetime | None:
    line = line.strip()
    if not START_RE.match(line):
        return None
    return datetime.strptime(line, "%a %b %d %H:%M:%S UTC %Y").replace(
        tzinfo=timezone.utc
    )


def parse_elapsed_to_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 1:
        return float(parts[0])
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


def summarize_rank_seconds(values: dict[int, float]) -> dict[str, Any]:
    if not values:
        return {
            "by_rank": {},
            "rank_count": 0,
            "ranks": [],
            "min_s": None,
            "max_s": None,
            "skew_s": None,
            "slowest_rank": None,
        }
    min_s = min(values.values())
    max_s = max(values.values())
    slowest_rank = max(values, key=lambda rank: values[rank])
    return {
        "by_rank": {str(rank): values[rank] for rank in sorted(values)},
        "rank_count": len(values),
        "ranks": sorted(values),
        "min_s": min_s,
        "max_s": max_s,
        "skew_s": max_s - min_s,
        "slowest_rank": slowest_rank,
    }


def isoformat_timestamp(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def summarize_rank_timestamps(values: dict[int, datetime]) -> dict[str, Any]:
    if not values:
        return {
            "by_rank": {},
            "rank_count": 0,
            "ranks": [],
            "first_time": None,
            "last_time": None,
            "first_rank": None,
            "last_rank": None,
            "skew_s": None,
        }
    first_rank = min(values, key=lambda rank: values[rank])
    last_rank = max(values, key=lambda rank: values[rank])
    first_time = values[first_rank]
    last_time = values[last_rank]
    return {
        "by_rank": {str(rank): values[rank].isoformat() for rank in sorted(values)},
        "rank_count": len(values),
        "ranks": sorted(values),
        "first_time": first_time.isoformat(),
        "last_time": last_time.isoformat(),
        "first_rank": first_rank,
        "last_rank": last_rank,
        "skew_s": (last_time - first_time).total_seconds(),
    }


def timestamp_bounds(
    values: dict[int, datetime],
) -> tuple[datetime | None, datetime | None]:
    if not values:
        return None, None
    return min(values.values()), max(values.values())


def infer_load_post_shard_seconds(
    load_summary: dict[str, Any], shard_summary: dict[str, Any]
) -> float | None:
    load_s = load_summary.get("max_s")
    shard_s = shard_summary.get("elapsed_s_max")
    if load_s is None or shard_s is None:
        return None
    return max(0.0, float(load_s) - float(shard_s))


def build_phase_summary(
    *,
    start_ts: datetime | None,
    server_args_ts: datetime | None,
    target_begin: dict[int, datetime],
    target_end: dict[int, datetime],
    draft_begin: dict[int, datetime],
    draft_end: dict[int, datetime],
    graph_begin: dict[int, datetime],
    graph_end: dict[int, datetime],
    app_startup_ts: datetime | None,
    uvicorn_running_ts: datetime | None,
    ready_ts: datetime | None,
    target_load: dict[str, Any],
    draft_load: dict[str, Any],
    draft_graph: dict[str, Any],
    target_shard_progress: dict[str, Any],
    draft_shard_progress: dict[str, Any],
) -> dict[str, Any]:
    first_target_begin, last_target_begin = timestamp_bounds(target_begin)
    first_target_end, last_target_end = timestamp_bounds(target_end)
    first_draft_begin, last_draft_begin = timestamp_bounds(draft_begin)
    first_draft_end, last_draft_end = timestamp_bounds(draft_end)
    first_graph_begin, last_graph_begin = timestamp_bounds(graph_begin)
    first_graph_end, last_graph_end = timestamp_bounds(graph_end)

    return {
        "timestamps": {
            "start": isoformat_timestamp(start_ts),
            "server_args": isoformat_timestamp(server_args_ts),
            "first_target_load_begin": isoformat_timestamp(first_target_begin),
            "last_target_load_begin": isoformat_timestamp(last_target_begin),
            "first_target_load_end": isoformat_timestamp(first_target_end),
            "last_target_load_end": isoformat_timestamp(last_target_end),
            "first_draft_load_begin": isoformat_timestamp(first_draft_begin),
            "last_draft_load_begin": isoformat_timestamp(last_draft_begin),
            "first_draft_load_end": isoformat_timestamp(first_draft_end),
            "last_draft_load_end": isoformat_timestamp(last_draft_end),
            "first_draft_graph_begin": isoformat_timestamp(first_graph_begin),
            "last_draft_graph_begin": isoformat_timestamp(last_graph_begin),
            "first_draft_graph_end": isoformat_timestamp(first_graph_end),
            "last_draft_graph_end": isoformat_timestamp(last_graph_end),
            "app_startup_complete": isoformat_timestamp(app_startup_ts),
            "uvicorn_running": isoformat_timestamp(uvicorn_running_ts),
            "ready": isoformat_timestamp(ready_ts),
        },
        "target_load_begin": summarize_rank_timestamps(target_begin),
        "target_load_end": summarize_rank_timestamps(target_end),
        "draft_load_begin": summarize_rank_timestamps(draft_begin),
        "draft_load_end": summarize_rank_timestamps(draft_end),
        "draft_graph_begin": summarize_rank_timestamps(graph_begin),
        "draft_graph_end": summarize_rank_timestamps(graph_end),
        "durations_s": {
            "start_to_server_args": seconds_between(start_ts, server_args_ts),
            "start_to_first_target_load_begin": seconds_between(
                start_ts, first_target_begin
            ),
            "server_args_to_first_target_load_begin": seconds_between(
                server_args_ts, first_target_begin
            ),
            "target_load_wall": seconds_between(first_target_begin, last_target_end),
            "target_load_max": target_load.get("max_s"),
            "target_shard_progress": target_shard_progress.get("elapsed_s_max"),
            "target_load_post_shard_inferred": infer_load_post_shard_seconds(
                target_load, target_shard_progress
            ),
            "target_to_draft_load_begin": seconds_between(
                last_target_end, first_draft_begin
            ),
            "draft_load_wall": seconds_between(first_draft_begin, last_draft_end),
            "draft_load_max": draft_load.get("max_s"),
            "draft_shard_progress": draft_shard_progress.get("elapsed_s_max"),
            "draft_load_post_shard_inferred": infer_load_post_shard_seconds(
                draft_load, draft_shard_progress
            ),
            "draft_to_graph_begin": seconds_between(last_draft_end, first_graph_begin),
            "draft_graph_wall": seconds_between(first_graph_begin, last_graph_end),
            "draft_graph_max": draft_graph.get("max_s"),
            "draft_graph_to_app_startup": seconds_between(
                last_graph_end, app_startup_ts
            ),
            "app_startup_to_ready": seconds_between(app_startup_ts, ready_ts),
            "uvicorn_running_to_ready": seconds_between(uvicorn_running_ts, ready_ts),
        },
    }


def parse_shard_progress_line(line: str) -> dict[str, Any] | None:
    matches = list(SHARD_PROGRESS_RE.finditer(line))
    if not matches:
        return None

    updates: list[dict[str, Any]] = []
    for match in matches:
        updates.append(
            {
                "percent": int(match.group("pct")),
                "completed": int(match.group("done")),
                "total": int(match.group("total")),
                "elapsed_s": parse_elapsed_to_seconds(match.group("elapsed")),
            }
        )

    final_updates = [
        update for update in updates if update["completed"] == update["total"]
    ]
    best = max(
        final_updates or updates,
        key=lambda update: (update["completed"], update["elapsed_s"]),
    )
    return {
        "update_count": len(updates),
        "completed": best["completed"],
        "total": best["total"],
        "elapsed_s": best["elapsed_s"],
        "final_seen": bool(final_updates),
    }


def summarize_shard_progress(
    records: list[dict[str, Any]], *, kind: str | None = None
) -> dict[str, Any]:
    filtered = [
        record for record in records if kind is None or record.get("kind") == kind
    ]
    if not filtered:
        return {
            "event_count": 0,
            "elapsed_s_max": None,
            "completed_shards_max": None,
            "total_shards_max": None,
            "update_count_total": 0,
        }
    return {
        "event_count": len(filtered),
        "elapsed_s_max": max(float(record["elapsed_s"]) for record in filtered),
        "completed_shards_max": max(int(record["completed"]) for record in filtered),
        "total_shards_max": max(int(record["total"]) for record in filtered),
        "update_count_total": sum(int(record["update_count"]) for record in filtered),
    }


def parse_launch_log(label: str, path: Path) -> dict[str, Any]:
    start_ts: datetime | None = None
    ready_ts: datetime | None = None
    first_log_ts: datetime | None = None
    target_load: dict[int, float] = {}
    draft_load: dict[int, float] = {}
    draft_graph: dict[int, float] = {}
    pending_load_begin: dict[int, datetime] = {}
    target_load_begin: dict[int, datetime] = {}
    target_load_end: dict[int, datetime] = {}
    draft_load_begin: dict[int, datetime] = {}
    draft_load_end: dict[int, datetime] = {}
    draft_graph_begin: dict[int, datetime] = {}
    draft_graph_end: dict[int, datetime] = {}
    server_args_ts: datetime | None = None
    app_startup_ts: datetime | None = None
    uvicorn_running_ts: datetime | None = None
    aiter_imports: Counter[tuple[str, str]] = Counter()
    aiter_build_starts: Counter[tuple[str, str]] = Counter()
    aiter_build_events: list[dict[str, Any]] = []
    tuned_misses: Counter[tuple[str, str, str, str]] = Counter()
    shard_progress_records: list[dict[str, Any]] = []
    pending_shard_progress: dict[str, Any] | None = None
    ready_seen = False

    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for raw_line in fin:
            line = ANSI_RE.sub("", raw_line.rstrip("\n"))
            if start_ts is None:
                start_ts = parse_start_line(line)
            ts, rank = parse_log_timestamp(line)
            if ts is not None and first_log_ts is None:
                first_log_ts = ts
            if "server_args=" in line and ts is not None:
                server_args_ts = server_args_ts or ts
            if "Application startup complete." in line and ts is not None:
                app_startup_ts = app_startup_ts or ts
            if "Uvicorn running on" in line and ts is not None:
                uvicorn_running_ts = uvicorn_running_ts or ts
            if "The server is fired up and ready to roll!" in line and ts is not None:
                ready_ts = ts

            load_begin_match = LOAD_BEGIN_RE.search(line)
            if load_begin_match and rank is not None and ts is not None:
                pending_load_begin[rank] = ts

            load_match = LOAD_RE.search(line)
            if load_match and rank is not None:
                load_type = load_match.group("type")
                elapsed = float(load_match.group("elapsed"))
                if load_type == "DSparkDraftModel":
                    draft_load[rank] = elapsed
                    if ts is not None:
                        draft_load_end[rank] = ts
                    begin_ts = pending_load_begin.pop(rank, None)
                    if begin_ts is not None:
                        draft_load_begin[rank] = begin_ts
                    load_kind = "draft"
                else:
                    target_load[rank] = elapsed
                    if ts is not None:
                        target_load_end[rank] = ts
                    begin_ts = pending_load_begin.pop(rank, None)
                    if begin_ts is not None:
                        target_load_begin[rank] = begin_ts
                    load_kind = "target"
                if pending_shard_progress is not None:
                    shard_progress_records.append(
                        {
                            **pending_shard_progress,
                            "kind": load_kind,
                            "model_type": load_type,
                            "first_end_rank": rank,
                        }
                    )
                    pending_shard_progress = None

            graph_begin_match = GRAPH_BEGIN_RE.search(line)
            if graph_begin_match and rank is not None and ts is not None:
                draft_graph_begin[rank] = ts

            graph_match = GRAPH_RE.search(line)
            if graph_match and rank is not None:
                draft_graph[rank] = float(graph_match.group("elapsed"))
                if ts is not None:
                    draft_graph_end[rank] = ts

            shard_progress = parse_shard_progress_line(line)
            if shard_progress is not None:
                pending_shard_progress = shard_progress

            import_match = IMPORT_RE.search(line)
            if import_match:
                aiter_imports[
                    (import_match.group("module"), import_match.group("path"))
                ] += 1

            build_start_match = BUILD_START_RE.search(line)
            if build_start_match:
                aiter_build_starts[
                    (
                        build_start_match.group("module"),
                        build_start_match.group("path"),
                    )
                ] += 1

            build_finish_match = BUILD_FINISH_RE.search(line)
            if build_finish_match:
                aiter_build_events.append(
                    {
                        "module": build_finish_match.group("module"),
                        "cost_s": float(build_finish_match.group("cost")),
                        "phase": "post_ready" if ready_seen else "pre_ready",
                    }
                )

            tuned_miss_match = TUNED_MISS_RE.search(line)
            if tuned_miss_match:
                config_path = tuned_miss_match.group("config").rstrip(",")
                tuned_misses[
                    (
                        config_path,
                        tuned_miss_match.group("m"),
                        tuned_miss_match.group("n"),
                        tuned_miss_match.group("k"),
                    )
                ] += 1

            if ready_ts == ts and "The server is fired up and ready to roll!" in line:
                ready_seen = True

    effective_start = start_ts or first_log_ts
    start_to_ready_s = (
        (ready_ts - effective_start).total_seconds()
        if effective_start is not None and ready_ts is not None
        else None
    )
    target_load_summary = summarize_rank_seconds(target_load)
    draft_load_summary = summarize_rank_seconds(draft_load)
    draft_graph_summary = summarize_rank_seconds(draft_graph)
    target_shard_progress_summary = summarize_shard_progress(
        shard_progress_records, kind="target"
    )
    draft_shard_progress_summary = summarize_shard_progress(
        shard_progress_records, kind="draft"
    )
    phase_summary = build_phase_summary(
        start_ts=effective_start,
        server_args_ts=server_args_ts,
        target_begin=target_load_begin,
        target_end=target_load_end,
        draft_begin=draft_load_begin,
        draft_end=draft_load_end,
        graph_begin=draft_graph_begin,
        graph_end=draft_graph_end,
        app_startup_ts=app_startup_ts,
        uvicorn_running_ts=uvicorn_running_ts,
        ready_ts=ready_ts,
        target_load=target_load_summary,
        draft_load=draft_load_summary,
        draft_graph=draft_graph_summary,
        target_shard_progress=target_shard_progress_summary,
        draft_shard_progress=draft_shard_progress_summary,
    )

    return {
        "label": label,
        "log": artifact_entry(path),
        "start_time": effective_start.isoformat() if effective_start else None,
        "ready_time": ready_ts.isoformat() if ready_ts else None,
        "start_to_ready_s": start_to_ready_s,
        "target_weight_load": target_load_summary,
        "draft_weight_load": draft_load_summary,
        "draft_verify_graph_capture": draft_graph_summary,
        "phase_summary": phase_summary,
        "aiter_builds": summarize_aiter_builds(aiter_build_events),
        "aiter_build_events": aiter_build_events,
        "shard_loading_progress": shard_progress_records,
        "target_shard_loading_progress": target_shard_progress_summary,
        "draft_shard_loading_progress": draft_shard_progress_summary,
        "aiter_build_starts": [
            {"module": module, "path": build_path, "count": count}
            for (module, build_path), count in sorted(aiter_build_starts.items())
        ],
        "aiter_imports": [
            {"module": module, "path": import_path, "count": count}
            for (module, import_path), count in sorted(aiter_imports.items())
        ],
        "tuned_config_misses": [
            {
                "config": config,
                "m": int(m),
                "n": int(n),
                "k": int(k),
                "count": count,
            }
            for (config, m, n, k), count in sorted(tuned_misses.items())
        ],
        "tuned_config_miss_summary": summarize_tuned_misses(tuned_misses),
    }


def summarize_aiter_builds(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_module: dict[str, dict[str, Any]] = {}
    for event in events:
        module = event["module"]
        summary = by_module.setdefault(
            module,
            {
                "module": module,
                "count": 0,
                "costs_s": [],
                "total_cost_s": 0.0,
                "pre_ready_cost_s": 0.0,
                "post_ready_cost_s": 0.0,
            },
        )
        cost = float(event["cost_s"])
        summary["count"] += 1
        summary["costs_s"].append(cost)
        summary["total_cost_s"] += cost
        if event["phase"] == "pre_ready":
            summary["pre_ready_cost_s"] += cost
        else:
            summary["post_ready_cost_s"] += cost
    return [by_module[module] for module in sorted(by_module)]


def summarize_tuned_misses(
    tuned_misses: Counter[tuple[str, str, str, str]],
) -> dict[str, Any]:
    by_config: Counter[str] = Counter()
    for (config, _m, _n, _k), count in tuned_misses.items():
        by_config[config] += count
    return {
        "event_count": sum(tuned_misses.values()),
        "unique_shape_count": len(tuned_misses),
        "by_config": [
            {"config": config, "count": count}
            for config, count in sorted(by_config.items())
        ],
    }


def sum_aiter_build_seconds(run: dict[str, Any], field: str) -> float:
    return sum(float(build.get(field, 0.0)) for build in run.get("aiter_builds", []))


def observed_launch_stages(run: dict[str, Any]) -> list[dict[str, Any]]:
    durations = (run.get("phase_summary") or {}).get("durations_s") or {}
    stage_specs = [
        (
            "start_to_first_target_load_begin",
            durations.get("start_to_first_target_load_begin"),
        ),
        ("target_weight_load", run["target_weight_load"].get("max_s")),
        (
            "target_shard_loading_progress",
            run["target_shard_loading_progress"].get("elapsed_s_max"),
        ),
        (
            "target_load_post_shard_inferred",
            durations.get("target_load_post_shard_inferred"),
        ),
        ("draft_weight_load", run["draft_weight_load"].get("max_s")),
        (
            "draft_shard_loading_progress",
            run["draft_shard_loading_progress"].get("elapsed_s_max"),
        ),
        (
            "draft_verify_graph_capture",
            run["draft_verify_graph_capture"].get("max_s"),
        ),
        ("aiter_build_pre_ready", sum_aiter_build_seconds(run, "pre_ready_cost_s")),
    ]
    return [
        {"stage": name, "seconds": seconds}
        for name, seconds in stage_specs
        if seconds is not None and float(seconds) > 0.0
    ]


def summarize_launch_insight(run: dict[str, Any]) -> dict[str, Any]:
    stages = observed_launch_stages(run)
    dominant = max(stages, key=lambda item: item["seconds"]) if stages else None
    build_pre_ready = sum_aiter_build_seconds(run, "pre_ready_cost_s")
    build_total = sum_aiter_build_seconds(run, "total_cost_s")
    miss_summary = run.get("tuned_config_miss_summary") or {}
    miss_events = int(miss_summary.get("event_count", 0))
    miss_shapes = int(miss_summary.get("unique_shape_count", 0))

    if build_pre_ready > 0.0:
        aiter_cache_action = "prewarm_or_reuse_aiter_jit_cache"
    elif run.get("aiter_imports"):
        aiter_cache_action = "preserve_warm_aiter_jit_cache"
    else:
        aiter_cache_action = "no_aiter_cache_signal"

    return {
        "observed_stages": stages,
        "dominant_observed_stage": dominant,
        "aiter_build_pre_ready_s": build_pre_ready,
        "aiter_build_total_s": build_total,
        "aiter_cache_action": aiter_cache_action,
        "tuned_miss_events": miss_events,
        "tuned_miss_shapes": miss_shapes,
        "target_shard_progress_elapsed_s": run["target_shard_loading_progress"].get(
            "elapsed_s_max"
        ),
        "target_load_post_shard_inferred_s": (
            (run.get("phase_summary") or {})
            .get("durations_s", {})
            .get("target_load_post_shard_inferred")
        ),
        "draft_shard_progress_elapsed_s": run["draft_shard_loading_progress"].get(
            "elapsed_s_max"
        ),
        "draft_load_post_shard_inferred_s": (
            (run.get("phase_summary") or {})
            .get("durations_s", {})
            .get("draft_load_post_shard_inferred")
        ),
        "tuned_miss_action": (
            "generate_tuned_miss_inputs" if miss_events else "no_tuned_miss_action"
        ),
    }


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise SystemExit(f"expected non-empty LABEL=PATH, got {value!r}")
    return label, Path(raw_path).expanduser()


def inventory_cache_dir(label: str, path: Path) -> dict[str, Any]:
    entry = artifact_entry(path)
    entry["label"] = label
    if not path.exists() or not path.is_dir():
        entry["file_count"] = 0
        entry["total_size_bytes"] = 0
        return entry

    file_count = 0
    total_size = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            total_size += child.stat().st_size
    entry["file_count"] = file_count
    entry["total_size_bytes"] = total_size
    return entry


def inventory_resource_snapshot(label: str, path: Path) -> dict[str, Any]:
    entry = artifact_entry(path)
    entry["label"] = label
    return entry


def build_report(
    *,
    runs: list[tuple[str, Path]],
    cache_dirs: list[tuple[str, Path]],
    resource_snapshots: list[tuple[str, Path]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_records = [parse_launch_log(label, path) for label, path in runs]
    for run in run_records:
        run["launch_insight"] = summarize_launch_insight(run)
    return {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "provenance": provenance or {},
        "runs": run_records,
        "cache_dirs": [inventory_cache_dir(label, path) for label, path in cache_dirs],
        "resource_snapshots": [
            inventory_resource_snapshot(label, path)
            for label, path in resource_snapshots or []
        ],
    }


def format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DSpark Launch Report",
        "",
        "| run | ready_s | target_load_ranks | target_load_max_s | target_load_skew_s | draft_graph_ranks | draft_graph_max_s | aiter_build_pre_ready_s | aiter_build_total_s | tuned_miss_events | tuned_miss_shapes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        build_total = sum_aiter_build_seconds(run, "total_cost_s")
        build_pre_ready = sum_aiter_build_seconds(run, "pre_ready_cost_s")
        miss_summary = run.get("tuned_config_miss_summary") or {}
        lines.append(
            "| {label} | {ready} | {target_ranks} | {target_max} | {target_skew} | {graph_ranks} | {graph_max} | {build_pre_ready:.2f} | {build_total:.2f} | {miss_events} | {miss_shapes} |".format(
                label=run["label"],
                ready=format_seconds(run.get("start_to_ready_s")),
                target_ranks=run["target_weight_load"].get("rank_count", 0),
                target_max=format_seconds(run["target_weight_load"].get("max_s")),
                target_skew=format_seconds(run["target_weight_load"].get("skew_s")),
                graph_ranks=run["draft_verify_graph_capture"].get("rank_count", 0),
                graph_max=format_seconds(
                    run["draft_verify_graph_capture"].get("max_s")
                ),
                build_pre_ready=build_pre_ready,
                build_total=build_total,
                miss_events=miss_summary.get(
                    "event_count",
                    sum(
                        int(item.get("count") or 0)
                        for item in run.get("tuned_config_misses", [])
                    ),
                ),
                miss_shapes=miss_summary.get(
                    "unique_shape_count", len(run.get("tuned_config_misses", []))
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Launch Insights",
            "",
            "| run | dominant_stage | dominant_s | aiter_cache_action | tuned_miss_action |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for run in report["runs"]:
        insight = run.get("launch_insight") or {}
        dominant = insight.get("dominant_observed_stage") or {}
        lines.append(
            "| {label} | {stage} | {seconds} | {aiter_action} | {miss_action} |".format(
                label=run["label"],
                stage=dominant.get("stage", "-"),
                seconds=format_seconds(dominant.get("seconds")),
                aiter_action=insight.get("aiter_cache_action", "-"),
                miss_action=insight.get("tuned_miss_action", "-"),
            )
        )

    phase_rows = [
        (run, (run.get("phase_summary") or {}).get("durations_s") or {})
        for run in report["runs"]
        if (run.get("phase_summary") or {}).get("durations_s")
    ]
    if phase_rows:
        lines.extend(
            [
                "",
                "## Launch Phase Breakdown",
                "",
                "| run | start_to_target_begin_s | target_wall_s | target_shard_s | target_post_shard_inferred_s | target_to_draft_s | draft_wall_s | draft_to_graph_s | graph_wall_s | app_to_ready_s | uvicorn_to_ready_s |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run, durations in phase_rows:
            lines.append(
                "| {label} | {start_to_target} | {target_wall} | {target_shard} | {target_post_shard} | {target_to_draft} | {draft_wall} | {draft_to_graph} | {graph_wall} | {app_to_ready} | {uvicorn_to_ready} |".format(
                    label=run["label"],
                    start_to_target=format_seconds(
                        durations.get("start_to_first_target_load_begin")
                    ),
                    target_wall=format_seconds(durations.get("target_load_wall")),
                    target_shard=format_seconds(durations.get("target_shard_progress")),
                    target_post_shard=format_seconds(
                        durations.get("target_load_post_shard_inferred")
                    ),
                    target_to_draft=format_seconds(
                        durations.get("target_to_draft_load_begin")
                    ),
                    draft_wall=format_seconds(durations.get("draft_load_wall")),
                    draft_to_graph=format_seconds(
                        durations.get("draft_to_graph_begin")
                    ),
                    graph_wall=format_seconds(durations.get("draft_graph_wall")),
                    app_to_ready=format_seconds(durations.get("app_startup_to_ready")),
                    uvicorn_to_ready=format_seconds(
                        durations.get("uvicorn_running_to_ready")
                    ),
                )
            )

    shard_rows = [
        (run, record)
        for run in report["runs"]
        for record in run.get("shard_loading_progress", [])
    ]
    if shard_rows:
        lines.extend(
            [
                "",
                "## Shard Loading Progress",
                "",
                "| run | kind | model_type | completed | total | progress_elapsed_s | updates | first_end_rank |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run, record in shard_rows:
            lines.append(
                "| {label} | {kind} | {model_type} | {completed} | {total} | {elapsed} | {updates} | {rank} |".format(
                    label=run["label"],
                    kind=record.get("kind", "-"),
                    model_type=record.get("model_type", "-"),
                    completed=record.get("completed", "-"),
                    total=record.get("total", "-"),
                    elapsed=format_seconds(record.get("elapsed_s")),
                    updates=record.get("update_count", "-"),
                    rank=record.get("first_end_rank", "-"),
                )
            )

    if report.get("cache_dirs"):
        lines.extend(
            [
                "",
                "## Cache Directories",
                "",
                "| label | path | exists | files | size_bytes |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for cache in report["cache_dirs"]:
            lines.append(
                f"| {cache['label']} | `{cache['path']}` | {cache['exists']} | "
                f"{cache['file_count']} | {cache['total_size_bytes']} |"
            )

    if report.get("resource_snapshots"):
        lines.extend(
            [
                "",
                "## Resource Snapshots",
                "",
                "| label | path | exists | size_bytes | sha256 |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for snapshot in report["resource_snapshots"]:
            lines.append(
                f"| {snapshot['label']} | `{snapshot['path']}` | "
                f"{snapshot['exists']} | {snapshot.get('size_bytes', 0)} | "
                f"{snapshot.get('sha256', '-')} |"
            )

    lines.append("")
    return "\n".join(lines)


def iter_tuned_miss_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in report.get("runs", []):
        for item in run.get("tuned_config_misses", []):
            rows.append(
                {
                    "run": run.get("label"),
                    "config": item.get("config"),
                    "m": item.get("m"),
                    "n": item.get("n"),
                    "k": item.get("k"),
                    "count": item.get("count"),
                }
            )
    return rows


def write_tuned_misses_csv(path: Path, report: dict[str, Any]) -> None:
    rows = iter_tuned_miss_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=["run", "config", "m", "n", "k", "count"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_tuned_misses_jsonl(path: Path, report: dict[str, Any]) -> None:
    rows = iter_tuned_miss_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize DSpark server launch/cache evidence from saved logs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "LOG"),
        required=True,
        help="Saved launch log to parse. Can be repeated.",
    )
    parser.add_argument(
        "--cache-dir",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional cache directory to inventory. Can be repeated.",
    )
    parser.add_argument(
        "--resource-snapshot",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Optional resource/profiling snapshot to hash into the report, e.g. "
            "rocm_smi=/artifacts/preflight_rocm_smi.txt. Can be repeated."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--tuned-misses-csv-output",
        type=Path,
        help="Optional CSV with one row per run/config/M/N/K tuned-config miss.",
    )
    parser.add_argument(
        "--tuned-misses-jsonl-output",
        type=Path,
        help="Optional JSONL with one row per run/config/M/N/K tuned-config miss.",
    )
    add_provenance_args(parser)
    args = parser.parse_args()

    report = build_report(
        runs=[(label, Path(path).expanduser()) for label, path in args.run],
        cache_dirs=[parse_label_path(value) for value in args.cache_dir],
        resource_snapshots=[
            parse_label_path(value) for value in args.resource_snapshot
        ],
        provenance=provenance_from_args(args),
    )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    if args.tuned_misses_csv_output:
        write_tuned_misses_csv(args.tuned_misses_csv_output, report)
    if args.tuned_misses_jsonl_output:
        write_tuned_misses_jsonl(args.tuned_misses_jsonl_output, report)
    if not any(
        [
            args.json_output,
            args.markdown_output,
            args.tuned_misses_csv_output,
            args.tuned_misses_jsonl_output,
        ]
    ):
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
