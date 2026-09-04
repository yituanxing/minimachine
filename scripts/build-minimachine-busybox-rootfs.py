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
    init_script: bytes | None,
    busybox_init: bool = False,
    rc_script: bytes | None = None,
) -> tuple[bytes, int]:
    out = bytearray()
    ino = 1
    entries = 0

    def add(name: str, **kwargs) -> None:
        nonlocal ino, entries
        _newc_entry(out, name, ino=ino, **kwargs)
        ino += 1
        entries += 1

    directories = [".", "bin", "dev", "proc", "sys", "etc", "tmp", "root"]
    if busybox_init:
        directories.append("etc/init.d")
    for name in directories:
        add(name, mode=stat.S_IFDIR | (0o1777 if name == "tmp" else 0o755), nlink=2)

    add(
        "dev/console",
        mode=stat.S_IFCHR | 0o600,
        rdevmajor=240,
        rdevminor=0,
    )

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

    if busybox_init:
        if init_script is not None:
            raise ValueError("busybox init mode cannot also install an init script")
        if "init" not in installed:
            installed.add("init")
            add("bin/init", mode=stat.S_IFLNK | 0o777, data=b"busybox")

        # Exercise the real BusyBox multicall init path.  Linux executes
        # /init, BusyBox dispatches the init applet from argv[0], init reads
        # /etc/inittab, then launches the conventional rcS shell script.
        add("init", mode=stat.S_IFLNK | 0o777, data=b"bin/busybox")
        add(
            "etc/inittab",
            mode=stat.S_IFREG | 0o644,
            data=b"::sysinit:/etc/init.d/rcS\n",
        )
        if rc_script is None:
            rc_script = (
                b"#!/bin/sh\n"
                b"PATH=/bin\n"
                b"export PATH\n"
                b"echo MINIMACHINE_REAL_BUSYBOX_RCS\n"
                b"exec /bin/sh\n"
            )
        if not rc_script.startswith(b"#!"):
            raise ValueError("rcS script must start with a shebang")
        add("etc/init.d/rcS", mode=stat.S_IFREG | 0o755, data=rc_script)
    else:
        # Script mode remains useful for narrow regression tests.  The main
        # software-driven gate uses --busybox-init instead.
        if init_script is None:
            init_script = b"#!/bin/sh\nexit 0\n"
        if not init_script.startswith(b"#!"):
            raise ValueError("init script must start with a shebang")
        add("init", mode=stat.S_IFREG | 0o755, data=init_script)

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
    init_group = p.add_mutually_exclusive_group()
    init_group.add_argument(
        "--init-script-file",
        type=Path,
        help="script installed as /init; defaults to a minimal /bin/sh exit script",
    )
    init_group.add_argument(
        "--busybox-init",
        action="store_true",
        help=(
            "install /init -> /bin/busybox plus /etc/inittab and rcS so the "
            "real BusyBox init applet drives userspace"
        ),
    )
    p.add_argument(
        "--rc-script-file",
        type=Path,
        help="rcS used with --busybox-init; defaults to exec /bin/sh",
    )
    args = p.parse_args()

    if args.rc_script_file is not None and not args.busybox_init:
        p.error("--rc-script-file requires --busybox-init")

    busybox = args.busybox_image.read_bytes()
    if not busybox:
        raise SystemExit("empty --busybox-image")
    applets = tuple(x.strip() for x in args.applets.split(",") if x.strip())
    init_script = (
        args.init_script_file.read_bytes()
        if args.init_script_file is not None
        else None
    )
    rc_script = (
        args.rc_script_file.read_bytes()
        if args.rc_script_file is not None
        else None
    )
    archive, entries = build_rootfs(
        busybox,
        applets=applets,
        init_script=init_script,
        busybox_init=args.busybox_init,
        rc_script=rc_script,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(archive)
    print(
        "BUSYBOX_ROOTFS_READY "
        f"path={args.output} bytes={len(archive)} "
        f"busybox_bytes={len(busybox)} entries={entries} "
        f"mode={'busybox-init' if args.busybox_init else 'script'} "
        f"applets={','.join(('sh', *applets))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
