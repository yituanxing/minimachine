#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import stat


def _pad4(data: bytearray) -> None:
    while len(data) & 3:
        data.append(0)


def _newc_entry(
    out: bytearray,
    name: str,
    *,
    mode: int,
    data: bytes = b"",
    ino: int,
    nlink: int = 1,
    rdevmajor: int = 0,
    rdevminor: int = 0,
) -> None:
    encoded = name.encode("utf-8") + b"\0"
    fields = (
        ino, mode, 0, 0, nlink, 0, len(data),
        0, 0, rdevmajor, rdevminor, len(encoded), 0,
    )
    out.extend(b"070701")
    out.extend("".join(f"{value:08x}" for value in fields).encode("ascii"))
    out.extend(encoded)
    _pad4(out)
    out.extend(data)
    _pad4(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build a minimal initramfs that launches a MiniMachine user "
            "program through Linux binfmt_script."
        )
    )
    p.add_argument("output", type=Path)
    p.add_argument("--program-image", type=Path, required=True)
    p.add_argument(
        "--program-path",
        default="/bin/program",
        help="absolute guest path of the interpreter/program image",
    )
    p.add_argument(
        "--init-script-file",
        type=Path,
        required=True,
        help="script body installed as /init",
    )
    args = p.parse_args()

    if not args.program_path.startswith("/"):
        p.error("--program-path must be absolute")
    if args.program_path == "/init":
        p.error("--program-path must differ from /init")

    program = args.program_image.read_bytes()
    script = args.init_script_file.read_bytes()
    if not program:
        raise SystemExit("empty --program-image")
    if not script:
        raise SystemExit("empty --init-script-file")

    expected_shebang = f"#!{args.program_path}".encode()
    first_line = script.splitlines()[0] if script.splitlines() else b""
    if first_line != expected_shebang:
        raise SystemExit(
            f"/init must start with exact shebang {expected_shebang!r}; "
            f"got {first_line!r}"
        )

    out = bytearray()
    ino = 1
    entries = 0

    def add(name: str, **kwargs) -> None:
        nonlocal ino, entries
        _newc_entry(out, name, ino=ino, **kwargs)
        ino += 1
        entries += 1

    components = [part for part in args.program_path.split("/") if part]
    parent_parts = components[:-1]

    directories = {".", "dev", "tmp"}
    prefix = ""
    for part in parent_parts:
        prefix = f"{prefix}/{part}" if prefix else part
        directories.add(prefix)
    for name in sorted(directories, key=lambda x: (x.count("/"), x)):
        add(
            name,
            mode=stat.S_IFDIR | (0o1777 if name == "tmp" else 0o755),
            nlink=2,
        )

    add(
        "dev/console",
        mode=stat.S_IFCHR | 0o600,
        rdevmajor=240,
        rdevminor=0,
    )
    add(
        args.program_path.lstrip("/"),
        mode=stat.S_IFREG | 0o755,
        data=program,
    )
    add("init", mode=stat.S_IFREG | 0o755, data=script)
    add("TRAILER!!!", mode=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    print(
        "PROGRAM_ROOTFS_READY "
        f"path={args.output} bytes={len(out)} entries={entries} "
        f"program={args.program_path} program_bytes={len(program)} "
        f"init_bytes={len(script)} mode=binfmt-script",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
