from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key_values(values: list[str] | None, *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"{option} expects KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        if not key:
            raise SystemExit(f"{option} expects non-empty KEY, got {value!r}")
        parsed[key] = val
    return parsed


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


def artifact_record(
    path: Path | None, *, path_maps: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "resolved_path": None, "exists": False}

    mappings = path_maps or []
    resolved = apply_path_maps(path.expanduser(), mappings)
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


def artifact_records(
    paths: dict[str, Path | None], *, path_maps: list[tuple[str, str]] | None = None
) -> dict[str, dict[str, Any]]:
    return {
        label: artifact_record(path, path_maps=path_maps)
        for label, path in paths.items()
    }


def add_provenance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Optional provenance metadata, e.g. --metadata git_commit=abc "
            "--metadata image=lmsysorg/sglang:tag."
        ),
    )
    parser.add_argument(
        "--env-key",
        action="append",
        default=[],
        metavar="NAME",
        help="Environment variable name to copy into the provenance block.",
    )


def provenance(
    *,
    argv: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    env_keys: list[str] | None = None,
) -> dict[str, Any]:
    env = {
        key: os.environ[key] for key in sorted(set(env_keys or [])) if key in os.environ
    }
    return {
        "created_unix_s": time.time(),
        "argv": list(sys.argv if argv is None else argv),
        "metadata": metadata or {},
        "env": env,
    }


def provenance_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return provenance(
        metadata=parse_key_values(args.metadata, option="--metadata"),
        env_keys=args.env_key,
    )
