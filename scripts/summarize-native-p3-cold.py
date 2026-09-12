#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAGES = (
    ("real_cold", "busybox-real-cold.log", "BUSYBOX_REAL_COLD"),
    ("cached_reset", "busybox-real-cached-cold.log", "CACHED_COLD_RESET"),
    ("ror32", "busybox-real-ror32-cold.log", "CACHED_COLD_ROR32"),
    ("i128", "busybox-real-i128-cold.log", "CACHED_COLD_I128"),
    ("scalar", "busybox-real-scalar-cold.log", "CACHED_COLD_SCALAR"),
    ("i128_value", "busybox-real-i128-value-cold.log", "CACHED_COLD_I128_VALUE"),
    ("memory", "busybox-real-memory-cold.log", "CACHED_COLD_MEMORY"),
    ("fast_memory", "busybox-real-fast-memory-cold.log", "CACHED_COLD_FAST_MEMORY"),
)

TIME_RE_TEMPLATE = (
    r"{prefix}_WALL=(?P<wall>[0-9.]+)\s+"
    r"USER=(?P<user>[0-9.]+)\s+"
    r"SYS=(?P<sys>[0-9.]+)\s+"
    r"MAXRSS_KB=(?P<maxrss>[0-9]+)"
)
CHECKPOINT_RE = re.compile(
    r"BOOT_EXEC_CHECKPOINT_SAVED .*?steps=(?P<steps>[0-9]+).*?"
    r"function=kernel_execve"
)


def _last_match(pattern: re.Pattern[str], text: str):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _parse_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def summarize(build_dir: Path, *, allow_missing: bool = False) -> dict[str, object]:
    stages: dict[str, object] = {}
    missing: list[str] = []

    for stage_name, log_name, prefix in STAGES:
        path = build_dir / log_name
        if not path.exists():
            missing.append(log_name)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        timing_re = re.compile(TIME_RE_TEMPLATE.format(prefix=re.escape(prefix)))
        timing_match = _last_match(timing_re, text)
        status_match = _last_match(
            re.compile(rf"^{re.escape(prefix)}_STATUS=(?P<status>[0-9]+)$", re.MULTILINE),
            text,
        )
        checkpoint_match = _last_match(CHECKPOINT_RE, text)

        if timing_match is None:
            missing.append(f"{log_name}:{prefix}_WALL")
            continue
        if status_match is None:
            missing.append(f"{log_name}:{prefix}_STATUS")
            continue
        if checkpoint_match is None:
            missing.append(f"{log_name}:kernel_execve_steps")
            continue

        stages[stage_name] = {
            "wall_seconds": float(timing_match.group("wall")),
            "user_seconds": float(timing_match.group("user")),
            "sys_seconds": float(timing_match.group("sys")),
            "maxrss_kb": int(timing_match.group("maxrss")),
            "status": int(status_match.group("status")),
            "kernel_execve_steps": int(checkpoint_match.group("steps")),
        }

    if missing and not allow_missing:
        raise SystemExit("missing benchmark evidence: " + ", ".join(missing))

    env = _parse_env(build_dir / "input-runs.env")
    summary: dict[str, object] = {
        "schema_version": 1,
        "generation": {
            key.lower(): value
            for key, value in env.items()
            if key
            in {
                "LOWERING_SHA",
                "LINUX_SHA",
                "NATIVE_PACK_SHA",
                "P3_REFRESH_RUN_ID",
                "LINUX_RUN_ID",
                "BUSYBOX_SHA",
                "BUSYBOX_RUN_ID",
            }
        },
        "stages": stages,
        "missing": missing,
    }

    ordered = [name for name, _log, _prefix in STAGES if name in stages]
    comparisons: dict[str, object] = {}
    for control_name, candidate_name in zip(ordered, ordered[1:]):
        control = stages[control_name]
        candidate = stages[candidate_name]
        control_wall = control["wall_seconds"]
        candidate_wall = candidate["wall_seconds"]
        delta = candidate_wall - control_wall
        comparisons[f"{control_name}_to_{candidate_name}"] = {
            "control_seconds": control_wall,
            "candidate_seconds": candidate_wall,
            "delta_seconds": round(delta, 6),
            "delta_percent": round(delta * 100.0 / control_wall, 6),
            "steps_equal": (
                control["kernel_execve_steps"]
                == candidate["kernel_execve_steps"]
            ),
        }
    summary["comparisons"] = comparisons
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize MiniMachine Native P3 formal cold benchmark logs."
    )
    parser.add_argument(
        "build_dir",
        nargs="?",
        type=Path,
        default=Path("build"),
        help="directory containing busybox-real-*-cold.log files",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    summary = summarize(args.build_dir, allow_missing=args.allow_missing)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
