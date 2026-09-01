#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import stat


def _pad4(data: bytearray) -> None:
    while len(data) & 3:
        data.append(0)


def _entry(out: bytearray, name: str, *, mode: int, data: bytes, ino: int) -> None:
    encoded = name.encode("utf-8") + b"\0"
    fields = (
        ino,
        mode,
        0,
        0,
        1,
        0,
        len(data),
        0,
        0,
        0,
        0,
        len(encoded),
        0,
    )
    out.extend(b"070701")
    out.extend("".join(f"{value:08x}" for value in fields).encode("ascii"))
    out.extend(encoded)
    _pad4(out)
    out.extend(data)
    _pad4(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a tiny newc overlay replacing /init in a live MiniMachine rootfs."
    )
    p.add_argument("output", type=Path)
    p.add_argument("--init-script", type=Path, required=True)
    p.add_argument(
        "--path",
        default="init",
        help="archive path for the injected script; defaults to init",
    )
    p.add_argument(
        "--extra-script",
        action="append",
        type=Path,
        default=[],
        help=(
            "additional executable script to add at archive root using its "
            "basename; may be repeated"
        ),
    )
    args = p.parse_args()

    archive_path = args.path.lstrip("/")
    if not archive_path or ".." in archive_path.split("/"):
        raise SystemExit("invalid --path")
    script = args.init_script.read_bytes()
    if not script.startswith(b"#!"):
        raise SystemExit("runtime init script must start with a shebang")

    scripts = [(archive_path, script)]
    seen = {archive_path}
    for extra in args.extra_script:
        name = extra.name
        if not name or name in seen or "/" in name:
            raise SystemExit(f"invalid or duplicate --extra-script name: {name!r}")
        data = extra.read_bytes()
        if not data.startswith(b"#!"):
            raise SystemExit(f"runtime extra script must start with a shebang: {extra}")
        seen.add(name)
        scripts.append((name, data))

    out = bytearray()
    for ino, (name, data) in enumerate(scripts, start=1):
        _entry(
            out,
            name,
            mode=stat.S_IFREG | 0o755,
            data=data,
            ino=ino,
        )
    _entry(out, "TRAILER!!!", mode=0, data=b"", ino=len(scripts) + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    print(
        "MMRT_OVERLAY_READY "
        f"path={args.output} bytes={len(out)} script={archive_path} "
        f"init_bytes={len(script)} entries={len(scripts) + 1} "
        f"extras={len(scripts) - 1}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
