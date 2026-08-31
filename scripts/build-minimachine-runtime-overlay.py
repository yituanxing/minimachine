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
    args = p.parse_args()

    script = args.init_script.read_bytes()
    if not script.startswith(b"#!"):
        raise SystemExit("runtime init script must start with a shebang")

    out = bytearray()
    _entry(out, "init", mode=stat.S_IFREG | 0o755, data=script, ino=1)
    _entry(out, "TRAILER!!!", mode=0, data=b"", ino=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    print(
        "MMRT_OVERLAY_READY "
        f"path={args.output} bytes={len(out)} init_bytes={len(script)} entries=2",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
