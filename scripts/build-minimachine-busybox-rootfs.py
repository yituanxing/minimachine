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
        ino,
        mode,
        0,
        0,
        nlink,
        0,
        len(data),
        0,
        0,
        rdevmajor,
        rdevminor,
        len(encoded),
        0,
    )
    out.extend(b"070701")
    out.extend("".join(f"{value:08x}" for value in fields).encode("ascii"))
    out.extend(encoded)
    _pad4(out)
    out.extend(data)
    _pad4(out)


def build_rootfs(
    busybox: bytes,
    *,
    applets: tuple[str, ...],
) -> tuple[bytes, int]:
    out = bytearray()
    ino = 1
    entries = 0

    def add(name: str, **kwargs) -> None:
        nonlocal ino, entries
        _newc_entry(out, name, ino=ino, **kwargs)
        ino += 1
        entries += 1

    for name in (".", "bin", "dev", "proc", "sys", "etc", "tmp", "root"):
        add(name, mode=stat.S_IFDIR | (0o1777 if name == "tmp" else 0o755), nlink=2)

    add(
        "dev/console",
        mode=stat.S_IFCHR | 0o600,
        rdevmajor=240,
        rdevminor=0,
    )

    # Let Linux binfmt_script resolve the first userspace image through
    # /bin/sh. This keeps /init tiny and proves the normal BusyBox multicall
    # dispatch instead of hard-wiring ash_main as the executable entry.
    init_script = b"#!/bin/sh\nexec /bin/sh\n"
    add("init", mode=stat.S_IFREG | 0o755, data=init_script)
    add("bin/busybox", mode=stat.S_IFREG | 0o755, data=busybox)

    installed: set[str] = set()
    for applet in ("sh", *applets):
        applet = applet.strip()
        if not applet or applet in installed or applet == "busybox":
            continue
        installed.add(applet)
        add(
            f"bin/{applet}",
            mode=stat.S_IFLNK | 0o777,
            data=b"busybox",
        )

    add("TRAILER!!!", mode=0)
    return bytes(out), entries


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a BusyBox multicall MiniMachine initramfs."
    )
    p.add_argument("output", type=Path)
    p.add_argument("--busybox-image", type=Path, required=True)
    p.add_argument(
        "--applets",
        default="ls,cat,echo,uname,pwd,mkdir,rm,rmdir,touch,head,tail,wc,true,false",
        help="comma-separated /bin applet symlinks",
    )
    args = p.parse_args()

    busybox = args.busybox_image.read_bytes()
    if not busybox:
        raise SystemExit("empty --busybox-image")
    applets = tuple(x.strip() for x in args.applets.split(",") if x.strip())
    archive, entries = build_rootfs(busybox, applets=applets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(archive)
    print(
        "BUSYBOX_ROOTFS_READY "
        f"path={args.output} bytes={len(archive)} "
        f"busybox_bytes={len(busybox)} entries={entries} "
        f"applets={','.join(('sh', *applets))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
