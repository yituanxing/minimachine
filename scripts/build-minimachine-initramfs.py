#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir, p3
from src.minimachine.user_image import build_bflt


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
        ino,
        mode,
        0,  # uid
        0,  # gid
        nlink,
        0,  # mtime
        len(data),
        0,  # devmajor
        0,  # devminor
        rdevmajor,
        rdevminor,
        len(encoded),
        0,  # check
    )
    out.extend(b"070701")
    out.extend("".join(f"{value:08x}" for value in fields).encode("ascii"))
    out.extend(encoded)
    _pad4(out)
    out.extend(data)
    _pad4(out)


def build_init_image() -> bytes:
    # This is an init process, not a shell substitute.  Its only job is to
    # prove the real Linux path VFS -> binfmt_flat -> start_thread ->
    # MiniMachine return-to-user.  A real shell image replaces it next
    # without changing the rootfs/exec route.
    init_fn = p3.Function(
        "__mm_init",
        [
            p3.Block(
                "entry",
                [
                    p3.Br(
                        muir.Width.I8,
                        muir.Cond.EQ,
                        muir.Imm(0),
                        muir.Imm(0),
                        muir.Target(label="entry"),
                        muir.Target(label="entry"),
                    )
                ],
            )
        ],
        set(),
    )
    return build_bflt(init_fn, stack_size=256 * 1024)


def build_cpio() -> tuple[bytes, int]:
    init_image = build_init_image()
    out = bytearray()
    ino = 1

    def add(name: str, **kwargs) -> None:
        nonlocal ino
        _newc_entry(out, name, ino=ino, **kwargs)
        ino += 1

    add(".", mode=stat.S_IFDIR | 0o755, nlink=2)
    add("bin", mode=stat.S_IFDIR | 0o755, nlink=2)
    add("dev", mode=stat.S_IFDIR | 0o755, nlink=2)
    add("proc", mode=stat.S_IFDIR | 0o755, nlink=2)
    add("sys", mode=stat.S_IFDIR | 0o755, nlink=2)
    add(
        "dev/console",
        mode=stat.S_IFCHR | 0o600,
        rdevmajor=5,
        rdevminor=1,
    )
    add("init", mode=stat.S_IFREG | 0o755, data=init_image)
    add("TRAILER!!!", mode=0)
    return bytes(out), len(init_image)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the standard MiniMachine Linux initramfs cpio."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    archive, init_size = build_cpio()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(archive)
    print(
        f"INITRAMFS_READY path={args.output} bytes={len(archive)} "
        f"init_bytes={init_size} format=newc entries=8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
