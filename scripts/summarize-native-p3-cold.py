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

GENERATION_KEYS = (
    "LOWERING_SHA",
    "LINUX_SHA",
    "NATIVE_PACK_SHA",
    "P3_REFRESH_RUN_ID",
    "LINUX_RUN_ID",
    "BUSYBOX_SHA",
    "BUSYBOX_RUN_ID",
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
REAL_SOFTWARE_TIME_RE = re.compile(
    r"MINIMACHINE_REAL_SOFTWARE_WALL=(?P<wall>[0-9.]+)\s+"
    r"USER=(?P<user>[0-9.]+)\s+SYS=(?P<sys>[0-9.]+)\s+"
    r"MAXRSS_KB=(?P<maxrss>[0-9]+)"
)
REAL_SOFTWARE_STATUS_RE = re.compile(
    r"MINIMACHINE_REAL_SOFTWARE_STATUS=(?P<status>[0-9]+)"
)
REAL_SOFTWARE_FRONTIER_RE = re.compile(
    r"MINIMACHINE_REAL_FRONTIER=(?P<frontier>[A-Za-z0-9_.-]+)"
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


def _parse_generation_from_text(text: str) -> dict[str, str]:
    generation: dict[str, str] = {}
    for key in GENERATION_KEYS:
        if key.endswith("_RUN_ID"):
            value_re = r"[0-9]+"
        else:
            value_re = r"[0-9a-fA-F]{7,64}"
        matches = list(re.finditer(rf"{key}=(?P<value>{value_re})\b", text))
        if matches:
            generation[key] = matches[-1].group("value")
    return generation


def _parse_stage_text(
    text: str,
    *,
    prefix: str,
    source_name: str,
    missing: list[str],
) -> dict[str, object] | None:
    timing_re = re.compile(TIME_RE_TEMPLATE.format(prefix=re.escape(prefix)))
    timing_match = _last_match(timing_re, text)
    status_match = _last_match(
        re.compile(rf"{re.escape(prefix)}_STATUS=(?P<status>[0-9]+)\b"),
        text,
    )
    checkpoint_match = _last_match(CHECKPOINT_RE, text)

    if timing_match is None:
        missing.append(f"{source_name}:{prefix}_WALL")
        return None
    if status_match is None:
        missing.append(f"{source_name}:{prefix}_STATUS")
        return None
    if checkpoint_match is None:
        missing.append(f"{source_name}:kernel_execve_steps")
        return None

    return {
        "wall_seconds": float(timing_match.group("wall")),
        "user_seconds": float(timing_match.group("user")),
        "sys_seconds": float(timing_match.group("sys")),
        "maxrss_kb": int(timing_match.group("maxrss")),
        "status": int(status_match.group("status")),
        "kernel_execve_steps": int(checkpoint_match.group("steps")),
    }


def _add_comparisons(summary: dict[str, object]) -> None:
    stages = summary["stages"]
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


def summarize(build_dir: Path, *, allow_missing: bool = False) -> dict[str, object]:
    stages: dict[str, object] = {}
    missing: list[str] = []

    for stage_name, log_name, prefix in STAGES:
        path = build_dir / log_name
        if not path.exists():
            missing.append(log_name)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_stage_text(
            text,
            prefix=prefix,
            source_name=log_name,
            missing=missing,
        )
        if parsed is not None:
            stages[stage_name] = parsed

    if missing and not allow_missing:
        raise SystemExit("missing benchmark evidence: " + ", ".join(missing))

    env = _parse_env(build_dir / "input-runs.env")
    summary: dict[str, object] = {
        "schema_version": 1,
        "source": "stage_logs",
        "generation": {
            key.lower(): value for key, value in env.items() if key in GENERATION_KEYS
        },
        "stages": stages,
        "missing": missing,
    }
    _add_comparisons(summary)
    return summary


def summarize_combined_log(
    path: Path, *, allow_missing: bool = False
) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stages: dict[str, object] = {}
    missing: list[str] = []

    # `gh run view --log` prefixes each payload line with job/step/timestamp text.
    # Marker regexes deliberately search anywhere in the line, so the summary
    # remains independent of GitHub's presentation prefix.
    for stage_name, _log_name, prefix in STAGES:
        parsed = _parse_stage_text(
            text,
            prefix=prefix,
            source_name=path.name,
            missing=missing,
        )
        if parsed is not None:
            stages[stage_name] = parsed

    if missing and not allow_missing:
        raise SystemExit("missing benchmark evidence: " + ", ".join(missing))

    summary: dict[str, object] = {
        "schema_version": 1,
        "source": "combined_github_log",
        "generation": {
            key.lower(): value
            for key, value in _parse_generation_from_text(text).items()
        },
        "stages": stages,
        "missing": missing,
    }

    real_time = _last_match(REAL_SOFTWARE_TIME_RE, text)
    real_status = _last_match(REAL_SOFTWARE_STATUS_RE, text)
    real_frontier = _last_match(REAL_SOFTWARE_FRONTIER_RE, text)
    if real_time is not None or real_status is not None or real_frontier is not None:
        summary["real_software"] = {
            "wall_seconds": (
                float(real_time.group("wall")) if real_time is not None else None
            ),
            "user_seconds": (
                float(real_time.group("user")) if real_time is not None else None
            ),
            "sys_seconds": (
                float(real_time.group("sys")) if real_time is not None else None
            ),
            "maxrss_kb": (
                int(real_time.group("maxrss")) if real_time is not None else None
            ),
            "status": (
                int(real_status.group("status")) if real_status is not None else None
            ),
            "frontier": (
                real_frontier.group("frontier")
                if real_frontier is not None
                else None
            ),
        }

    _add_comparisons(summary)
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
    parser.add_argument(
        "--combined-log",
        type=Path,
        help="combined output from `gh run view RUN_ID --log`",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if args.combined_log is not None:
        summary = summarize_combined_log(
            args.combined_log, allow_missing=args.allow_missing
        )
    else:
        summary = summarize(args.build_dir, allow_missing=args.allow_missing)

    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
