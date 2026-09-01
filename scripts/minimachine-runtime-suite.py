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


CORE_CASES = (
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
        "L2-vfs",
        "vfs-file",
        ("MMRT_VFS_PASS",),
        r"""printf 'alpha\nbeta\n' > /tmp/mmrt-file || exit 71; IFS= read -r first < /tmp/mmrt-file || exit 72; case "$first" in alpha) printf 'MMRT_VFS_%s\n' PASS;; *) exit 73;; esac""",
        "open/create/truncate/write/read/close through ash redirection and read",
    ),
    RuntimeCase(
        "L2-vfs",
        "cwd",
        ("MMRT_CWD_PASS",),
        r"""cd /tmp || exit 81; pwd > /tmp/mmrt-pwd || exit 82; IFS= read -r here < /tmp/mmrt-pwd || exit 83; case "$here" in /tmp) printf 'MMRT_CWD_%s\n' PASS;; *) exit 84;; esac; cd / || exit 85""",
        "chdir/getcwd, path lookup, shell cwd state, VFS readback",
    ),
    RuntimeCase(
        "L2-fd",
        "fd-redirection",
        ("MMRT_FD_PASS",),
        r"""printf 'one\n' > /tmp/mmrt-fd || exit 86; exec 3>>/tmp/mmrt-fd || exit 87; printf 'two\n' >&3 || exit 88; exec 3>&-; { IFS= read -r a; IFS= read -r b; } < /tmp/mmrt-fd; case "$a:$b" in one:two) printf 'MMRT_FD_%s\n' PASS;; *) exit 89;; esac""",
        "append open, fd duplication/redirection, close, sequential read",
    ),
    RuntimeCase(
        "L2-vfs",
        "path-error",
        ("MMRT_PATH_ERROR_PASS",),
        r"""cd /__mmrt_path_that_does_not_exist__; rc=$?; case "$rc" in 0) exit 90;; *) printf 'MMRT_PATH_ERROR_%s\n' PASS;; esac""",
        "negative path lookup and shell propagation of Linux VFS error",
    ),
    RuntimeCase(
        "L2-vfs",
        "test-file",
        ("MMRT_TEST_FILE_PASS",),
        r"""printf 'x\n' > /tmp/mmrt-test || exit 101; if test -f /tmp/mmrt-test && test -s /tmp/mmrt-test && test -r /tmp/mmrt-test; then printf 'MMRT_TEST_FILE_%s\n' PASS; else exit 102; fi""",
        "ash test builtin with file type, size, readability and stat-family VFS metadata",
    ),
    RuntimeCase(
        "L2-vfs",
        "noclobber",
        ("MMRT_NOCLOBBER_PASS",),
        r"""printf 'first\n' > /tmp/mmrt-noclobber || exit 103; set -C; printf 'second\n' > /tmp/mmrt-noclobber; rc=$?; set +C; case "$rc" in 0) exit 104;; *) printf 'MMRT_NOCLOBBER_%s\n' PASS;; esac""",
        "ash noclobber, exclusive create/open failure and errno propagation",
    ),
)

PROCESS_CASES = (
    RuntimeCase(
        "L1-process",
        "external-status",
        ("MMRT_WAIT_STATUS_PASS",),
        r"""/bin/false; rc=$?; case "$rc" in 1) printf 'MMRT_WAIT_STATUS_%s\n' PASS;; *) exit 61;; esac""",
        "external BusyBox applet dispatch, fork/exec/wait, child exit status",
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
        "directory",
        ("MMRT_DIR_PASS",),
        r"""/bin/rm -rf /tmp/mmrt-dir; /bin/mkdir /tmp/mmrt-dir || exit 91; printf 'item\n' > /tmp/mmrt-dir/item || exit 92; /bin/ls /tmp/mmrt-dir; rc=$?; case "$rc" in 0) printf 'MMRT_DIR_%s\n' PASS;; *) exit 93;; esac; /bin/rm -f /tmp/mmrt-dir/item; /bin/rmdir /tmp/mmrt-dir""",
        "forked applets plus mkdir/path lookup/directory iteration/unlink/rmdir",
    ),
    RuntimeCase(
        "L3-pipe",
        "pipeline",
        ("MMRT_PIPE_CHILD_PASS", "MMRT_PIPE_PARENT_PASS"),
        r"""printf 'MMRT_PIPE_CHILD_%s\n' PASS | /bin/cat && printf 'MMRT_PIPE_PARENT_%s\n' PASS""",
        "pipe, fd redirection, fork/exec/wait, parent resume",
    ),
)

CASES = CORE_CASES + PROCESS_CASES

PROFILES = {
    "smoke": CORE_CASES[:2],
    "core": CORE_CASES,
    "process": PROCESS_CASES,
    "full": CASES,
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


def case_script_name(index: int) -> str:
    return f"mmrt-c{index:02d}"


def write_init(path: Path, profile: str) -> None:
    cases = PROFILES[profile]
    path.parent.mkdir(parents=True, exist_ok=True)

    case_paths: list[Path] = []
    for index, case in enumerate(cases):
        for marker in case.markers:
            if marker in case.command:
                raise ValueError(
                    f"runtime case {case.case_id!r} embeds exact marker {marker!r}"
                )
        case_path = path.parent / case_script_name(index)
        case_path.write_text(
            "#!/bin/sh\n"
            "PATH=/bin\n"
            "export PATH\n"
            f"{case.command}\n"
        )
        case_paths.append(case_path)

    lines = [
        "#!/bin/sh",
        "PATH=/bin",
        "export PATH",
        "run_case() {",
        "  id=$1; level=$2; script=$3",
        "  printf 'MMRT_CASE_START id=%s level=%s\\n' \"$id\" \"$level\"",
        "  if . \"$script\"; then",
        "    printf 'MMRT_CASE_PASS id=%s\\n' \"$id\"",
        "  else",
        "    rc=$?",
        "    printf 'MMRT_CASE_FAIL id=%s status=%s\\n' \"$id\" \"$rc\"",
        "    exit \"$rc\"",
        "  fi",
        "}",
        f"printf 'MMRT_SUITE_START profile=%s cases=%s\\n' {profile} {len(cases)}",
    ]
    for index, case in enumerate(cases):
        lines.append(
            f"run_case {case.case_id} {case.level} /{case_script_name(index)}"
        )
    lines.extend(
        [
            f"printf 'MMRT_SUITE_PASS profile=%s cases=%s\\n' {profile} {len(cases)}",
            "exit 0",
            "",
        ]
    )
    text = "\n".join(lines)
    if len(text.encode()) >= 1024:
        raise ValueError(
            f"runtime dispatcher exceeds one ash input buffer: {len(text.encode())}"
        )
    for case_path in case_paths:
        size = case_path.stat().st_size
        if size >= 1024:
            raise ValueError(
                f"runtime case script exceeds one ash input buffer: "
                f"{case_path.name}={size}"
            )
    path.write_text(text)
    print(
        f"MMRT_INIT_READY profile={profile} cases={len(cases)} path={path} "
        f"dispatcher_bytes={len(text.encode())} sidecars={len(case_paths)}"
    )
    for index, case in enumerate(cases):
        print(
            "MMRT_CASE "
            f"level={case.level} id={case.case_id} "
            f"script={case_script_name(index)} "
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
    cursor = 0
    for case in cases:
        pass_marker = f"MMRT_CASE_PASS id={case.case_id}"
        pass_at = text.find(pass_marker, cursor)
        if pass_at < 0:
            raise RuntimeError(f"runtime case did not pass: {case.case_id}")
        for marker in case.markers:
            marker_at = text.find("\n" + marker + "\n", cursor)
            if marker_at < 0 or marker_at > pass_at:
                raise RuntimeError(
                    f"runtime output marker missing or out of order: "
                    f"{case.case_id}: {marker}"
                )
        cursor = pass_at + len(pass_marker)
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
