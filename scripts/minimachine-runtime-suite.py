#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeCase:
    level: str
    case_id: str
    markers: tuple[str, ...]
    command: str
    coverage: str


CASES = (
    RuntimeCase(
        "L0-shell",
        "builtin",
        ("MMRT_BUILTIN_PASS",),
        r"""printf 'MMRT_BUILTIN_%s\n' PASS""",
        "BusyBox multicall dispatch, ash builtin execution, console write",
    ),
    RuntimeCase(
        "L0-shell",
        "shell-state",
        ("MMRT_VAR_42", "MMRT_FALSE_1"),
        r"""value=42; printf 'MMRT_VAR_%s\n' "$value"; false; rc=$?; printf 'MMRT_FALSE_%s\n' "$rc" """,
        "shell variables, builtin false, status propagation",
    ),
    RuntimeCase(
        "L1-process",
        "external-status",
        ("MMRT_WAIT_STATUS_PASS",),
        r"""/bin/false; rc=$?; case "$rc" in 1) printf 'MMRT_WAIT_STATUS_%s\n' PASS;; *) exit 61;; esac""",
        "external BusyBox applet dispatch, exec, wait, child exit status",
    ),
    RuntimeCase(
        "L1-process",
        "subshell",
        ("MMRT_SUBSHELL_PASS",),
        r"""/bin/sh -c 'printf "MMRT_SUBSHELL_%s\n" PASS'""",
        "fork/exec/wait and a second BusyBox shell userspace entry",
    ),
    RuntimeCase(
        "L1-process",
        "uname",
        ("MMRT_UNAME_PASS",),
        r"""/bin/uname -s; rc=$?; case "$rc" in 0) printf 'MMRT_UNAME_%s\n' PASS;; *) exit 62;; esac""",
        "external applet plus Linux uname syscall and parent resume",
    ),
    RuntimeCase(
        "L2-vfs",
        "vfs-file",
        ("MMRT_VFS_PASS",),
        r"""printf 'alpha\nbeta\n' > /tmp/mmrt-file || exit 71; out=$(/bin/cat /tmp/mmrt-file) || exit 72; case "$out" in "$(printf 'alpha\nbeta')") printf 'MMRT_VFS_%s\n' PASS;; *) exit 73;; esac; /bin/rm -f /tmp/mmrt-file""",
        "open/create/truncate/write/read/close, redirection, cat, unlink",
    ),
    RuntimeCase(
        "L2-vfs",
        "directory",
        ("MMRT_DIR_PASS",),
        r"""/bin/rm -rf /tmp/mmrt-dir; /bin/mkdir /tmp/mmrt-dir || exit 81; printf 'item\n' > /tmp/mmrt-dir/item || exit 82; out=$(/bin/ls /tmp/mmrt-dir) || exit 83; case "$out" in item) printf 'MMRT_DIR_%s\n' PASS;; *) exit 84;; esac; /bin/rm -f /tmp/mmrt-dir/item; /bin/rmdir /tmp/mmrt-dir""",
        "mkdir, path lookup, directory iteration, ls, unlink, rmdir",
    ),
    RuntimeCase(
        "L3-pipe",
        "pipeline",
        ("MMRT_PIPE_CHILD_PASS", "MMRT_PIPE_PARENT_PASS"),
        r"""printf 'MMRT_PIPE_CHILD_%s\n' PASS | /bin/cat && printf 'MMRT_PIPE_PARENT_%s\n' PASS""",
        "pipe, fd redirection, fork/exec/wait, parent resume",
    ),
)

PROFILES = {
    "smoke": CASES[:2],
    "core": CASES,
}

BAD_MARKERS = (
    "BOOT_EXEC_BLOCKED ",
    "Kernel panic",
    "Oops:",
    "BUG:",
    "Unable to handle kernel",
    "Attempted to kill init",
    "soft lockup",
    "workqueue lockup",
)


def _line_present(text: str, marker: str) -> bool:
    return marker in text.replace("\r", "").splitlines()


def write_init(path: Path, profile: str) -> None:
    cases = PROFILES[profile]
    lines = [
        "#!/bin/sh",
        "PATH=/bin",
        "export PATH",
        f"printf 'MMRT_SUITE_START profile=%s cases=%s\\n' {profile} {len(cases)}",
    ]
    for case in cases:
        for marker in case.markers:
            if marker in case.command:
                raise ValueError(
                    f"runtime case {case.case_id!r} embeds exact marker {marker!r}"
                )
        lines.extend(
            [
                (
                    "printf 'MMRT_CASE_START id=%s level=%s\\n' "
                    f"{case.case_id} {case.level}"
                ),
                f"if ( {case.command} ); then",
                f"  printf 'MMRT_CASE_PASS id=%s\\n' {case.case_id}",
                "else",
                "  rc=$?",
                (
                    "  printf 'MMRT_CASE_FAIL id=%s status=%s\\n' "
                    f'{case.case_id} "$rc"'
                ),
                '  exit "$rc"',
                "fi",
            ]
        )
    lines.extend(
        [
            f"printf 'MMRT_SUITE_PASS profile=%s cases=%s\\n' {profile} {len(cases)}",
            "exit 0",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"MMRT_INIT_READY profile={profile} cases={len(cases)} path={path}")
    for case in cases:
        print(
            "MMRT_CASE "
            f"level={case.level} id={case.case_id} "
            f"coverage={case.coverage}"
        )


def verify_log(path: Path, profile: str) -> None:
    cases = PROFILES[profile]
    text = path.read_text(errors="replace").replace("\r", "")
    for bad in BAD_MARKERS:
        if bad in text:
            raise RuntimeError(f"bad runtime marker observed: {bad.strip()}")
    if "BOOT_EXEC_USER_HANDOFF " not in text or "function=__mm_user_main" not in text:
        raise RuntimeError("BusyBox multicall userspace handoff was not reached")
    for case in cases:
        if f"MMRT_CASE_START id={case.case_id} level={case.level}" not in text:
            raise RuntimeError(f"runtime case did not start: {case.case_id}")
        if f"MMRT_CASE_PASS id={case.case_id}" not in text:
            raise RuntimeError(f"runtime case did not pass: {case.case_id}")
        for marker in case.markers:
            if not _line_present(text, marker):
                raise RuntimeError(
                    f"runtime output marker missing: {case.case_id}: {marker}"
                )
    endpoint = f"MMRT_SUITE_PASS profile={profile} cases={len(cases)}"
    if not _line_present(text, endpoint):
        raise RuntimeError(f"runtime suite endpoint missing: {endpoint}")
    if "BOOT_EXEC_USER_EXIT status=0" not in text:
        raise RuntimeError("BusyBox runtime suite did not exit cleanly")
    print(f"MMRT_SUITE_VERIFIED profile={profile} cases={len(cases)} log={path}")


def summarize_log(path: Path, profile: str) -> None:
    cases = PROFILES[profile]
    text = path.read_text(errors="replace").replace("\r", "")
    passed = 0
    for case in cases:
        if f"MMRT_CASE_PASS id={case.case_id}" in text:
            passed += 1
        else:
            break
    next_case = cases[passed].case_id if passed < len(cases) else "-"
    started = [
        case.case_id
        for case in cases
        if f"MMRT_CASE_START id={case.case_id} level={case.level}" in text
    ]
    blocked = [
        line
        for line in text.splitlines()
        if line.startswith("BOOT_EXEC_BLOCKED ")
    ]
    failed = [
        line
        for line in text.splitlines()
        if line.startswith("MMRT_CASE_FAIL ")
    ]
    print(
        "MMRT_FRONTIER "
        f"profile={profile} passed={passed}/{len(cases)} "
        f"next={next_case} started={','.join(started) if started else '-'}"
    )
    if failed:
        print(f"MMRT_FRONTIER_CASE_FAIL {failed[-1]}")
    if blocked:
        print(f"MMRT_FRONTIER_BLOCKED {blocked[-1]}")


def list_cases(profile: str) -> None:
    for case in PROFILES[profile]:
        print(
            f"{case.level}\t{case.case_id}\t"
            f"{','.join(case.markers)}\t{case.coverage}"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="MiniMachine cumulative BusyBox/Linux runtime suite."
    )
    p.add_argument(
        "action",
        choices=("write-init", "verify-log", "frontier-log", "list"),
    )
    p.add_argument("path", nargs="?", type=Path)
    p.add_argument("--profile", choices=tuple(PROFILES), default="core")
    args = p.parse_args()

    if args.action == "list":
        list_cases(args.profile)
        return 0
    if args.path is None:
        p.error(f"{args.action} requires PATH")
    if args.action == "write-init":
        write_init(args.path, args.profile)
    elif args.action == "verify-log":
        verify_log(args.path, args.profile)
    else:
        summarize_log(args.path, args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
