#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SweepRun:
    label: str
    frac: Optional[float]
    output: Path
    server_info_output: Path
    manifest_output: Path
    info_summary_output: Path


@dataclass(frozen=True)
class CommandSpec:
    label: str
    command: list[str]


@dataclass(frozen=True)
class CommandResult:
    label: str
    command: list[str]
    log_path: str
    started_unix_s: float
    ended_unix_s: float
    elapsed_s: float
    returncode: int


def parse_frac_list(value: str) -> list[float]:
    fracs: list[float] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        frac = float(token)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"budget frac must be in (0, 1], got {frac}")
        fracs.append(frac)
    if not fracs:
        raise ValueError("at least one budget frac is required")
    return fracs


def frac_slug(frac: Optional[float]) -> str:
    if frac is None:
        return "auto"
    return f"frac_{frac:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_sweep_runs(
    *,
    output_dir: Path,
    run_prefix: str,
    fracs: list[float],
    include_auto: bool,
) -> list[SweepRun]:
    labels: list[tuple[str, Optional[float]]] = []
    if include_auto:
        labels.append((f"{run_prefix}_auto", None))
    labels.extend((f"{run_prefix}_{frac_slug(frac)}", frac) for frac in fracs)
    return [
        SweepRun(
            label=label,
            frac=frac,
            output=output_dir / f"{label}.jsonl",
            server_info_output=output_dir / f"{label}_server_info.json",
            manifest_output=output_dir / f"{label}_manifest.json",
            info_summary_output=output_dir / f"{label}_info_summary.json",
        )
        for label, frac in labels
    ]


def build_collect_command(args, run: SweepRun) -> list[str]:
    cmd = [
        sys.executable,
        str(args.harness),
        "collect",
        "--base-url",
        args.base_url,
        "--prompts",
        str(args.prompts),
        "--output",
        str(run.output),
        "--run-label",
        run.label,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--min-p",
        str(args.min_p),
        "--concurrency",
        str(args.concurrency),
        "--timeout-s",
        str(args.timeout_s),
        "--retries",
        str(args.retries),
        "--retry-sleep-s",
        str(args.retry_sleep_s),
        "--server-info-output",
        str(run.server_info_output),
        "--manifest-output",
        str(run.manifest_output),
        "--dspark-clear-info-records",
        "--no-print-records",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.start_idx is not None:
        cmd.extend(["--start-idx", str(args.start_idx)])
    if args.end_idx is not None:
        cmd.extend(["--end-idx", str(args.end_idx)])
    if args.sampling_seed is not None:
        cmd.extend(["--sampling-seed", str(args.sampling_seed)])
    if args.allow_nondeterministic_sampling:
        cmd.append("--allow-nondeterministic-sampling")
    if not args.ignore_eos:
        cmd.append("--no-ignore-eos")
    if run.frac is not None:
        cmd.extend(["--dspark-force-budget-frac", str(run.frac)])
    cmd.extend(args.extra_collect_arg)
    return cmd


def build_info_summary_command(args, run: SweepRun) -> list[str]:
    cmd = [
        sys.executable,
        str(args.harness),
        "info-summary",
        "--input",
        str(run.server_info_output),
        "--summary-output",
        str(run.info_summary_output),
    ]
    cmd.extend(args.extra_info_summary_arg)
    return cmd


def build_report_command(args, runs: list[SweepRun]) -> list[str]:
    cmd = [sys.executable, str(args.report)]
    for run in runs:
        cmd.extend(["--manifest-run", run.label, str(run.manifest_output)])
    cmd.extend(
        [
            "--json-output",
            str(args.output_dir / f"{args.run_prefix}_budget_sweep_report.json"),
            "--jsonl-output",
            str(args.output_dir / f"{args.run_prefix}_budget_sweep_report.jsonl"),
            "--markdown-output",
            str(args.output_dir / f"{args.run_prefix}_budget_sweep_report.md"),
        ]
    )
    cmd.extend(args.extra_report_arg)
    return cmd


def build_command_specs(
    runs: list[SweepRun],
    collect_commands: list[list[str]],
    info_summary_commands: list[list[str]],
    report_command: list[str],
) -> list[CommandSpec]:
    specs: list[CommandSpec] = []
    for run, collect_cmd, info_summary_cmd in zip(
        runs, collect_commands, info_summary_commands
    ):
        specs.append(CommandSpec(label=f"{run.label}_collect", command=collect_cmd))
        specs.append(
            CommandSpec(label=f"{run.label}_info_summary", command=info_summary_cmd)
        )
    specs.append(CommandSpec(label="perf_report", command=report_command))
    return specs


def command_manifest(args, runs: list[SweepRun], specs: list[CommandSpec]) -> dict:
    return {
        "base_url": args.base_url,
        "prompts": str(args.prompts),
        "run_prefix": args.run_prefix,
        "fracs": [run.frac for run in runs],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "command_log_dir": str(args.output_dir / f"{args.run_prefix}_command_logs"),
        "runs": [
            {
                "label": run.label,
                "frac": run.frac,
                "output": str(run.output),
                "server_info_output": str(run.server_info_output),
                "manifest_output": str(run.manifest_output),
                "info_summary_output": str(run.info_summary_output),
            }
            for run in runs
        ],
        "commands": [spec.command for spec in specs],
        "command_specs": [
            {"label": spec.label, "command": spec.command} for spec in specs
        ],
        "command_results": [],
    }


def command_log_path(log_dir: Path, index: int, label: str) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    return log_dir / f"{index:02d}_{safe_label}.log"


def run_command(
    spec: CommandSpec, *, dry_run: bool, log_path: Path | None = None
) -> CommandResult | None:
    cmd = spec.command
    printable = " ".join(cmd)
    print(printable, flush=True)
    if dry_run:
        return None

    if log_path is None:
        raise ValueError("log_path is required when dry_run=False")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        returncode = proc.wait()
    ended = time.time()
    return CommandResult(
        label=spec.label,
        command=cmd,
        log_path=str(log_path),
        started_unix_s=started,
        ended_unix_s=ended,
        elapsed_s=ended - started,
        returncode=returncode,
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DSpark forced-budget sweep and emit comparable collect manifests "
            "plus a perf report."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="dspark_budget_sweep")
    parser.add_argument("--fracs", default="0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--include-auto", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--allow-nondeterministic-sampling", action="store_true")
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument(
        "--harness", type=Path, default=SCRIPT_DIR / "dspark_accuracy_harness.py"
    )
    parser.add_argument(
        "--report", type=Path, default=SCRIPT_DIR / "dspark_perf_report.py"
    )
    parser.add_argument("--extra-collect-arg", action="append", default=[])
    parser.add_argument(
        "--extra-info-summary-arg",
        action="append",
        default=[],
        help=(
            "Additional argument passed verbatim to the per-run "
            "dspark_accuracy_harness.py info-summary command."
        ),
    )
    parser.add_argument(
        "--extra-report-arg",
        action="append",
        default=[],
        help=(
            "Additional argument passed verbatim to dspark_perf_report.py. Repeat "
            "for flags with values, for example: --extra-report-arg --min-ar "
            "--extra-report-arg 0.5."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fracs = parse_frac_list(args.fracs)
    runs = build_sweep_runs(
        output_dir=args.output_dir,
        run_prefix=args.run_prefix,
        fracs=fracs,
        include_auto=args.include_auto,
    )
    collect_commands = [build_collect_command(args, run) for run in runs]
    info_summary_commands = [build_info_summary_command(args, run) for run in runs]
    report_command = build_report_command(args, runs)
    command_specs = build_command_specs(
        runs, collect_commands, info_summary_commands, report_command
    )
    manifest = command_manifest(args, runs, command_specs)
    sweep_manifest = args.output_dir / f"{args.run_prefix}_budget_sweep_manifest.json"

    if args.dry_run:
        for spec in command_specs:
            run_command(spec, dry_run=True)
        print(str(sweep_manifest), flush=True)
        return

    log_dir = Path(manifest["command_log_dir"])
    write_manifest(sweep_manifest, manifest)
    for index, spec in enumerate(command_specs):
        result = run_command(
            spec, dry_run=False, log_path=command_log_path(log_dir, index, spec.label)
        )
        assert result is not None
        manifest["command_results"].append(asdict(result))
        write_manifest(sweep_manifest, manifest)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, spec.command)
    print(str(sweep_manifest), flush=True)


if __name__ == "__main__":
    main()
