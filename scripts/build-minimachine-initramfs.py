#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir, p3
from src.minimachine.user_image import build_bflt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the first standard Linux initramfs payload for MiniMachine."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = args.output
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "dev").mkdir(parents=True, exist_ok=True)
    (root / "proc").mkdir(parents=True, exist_ok=True)
    (root / "sys").mkdir(parents=True, exist_ok=True)

    # This is deliberately an init process, not a fake shell.  It provides the
    # smallest executable proof that Linux VFS + binfmt_flat + start_thread +
    # MiniMachine return-to-user all work end to end.  The rootfs layout is
    # already conventional so the next userspace image can replace /init
    # without changing the kernel boot route.
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
    image = build_bflt(init_fn, stack_size=256 * 1024)
    init_path = root / "init"
    init_path.write_bytes(image)
    init_path.chmod(0o755)

    print(
        f"INITRAMFS_READY root={root} init={init_path} bytes={len(image)} "
        f"dirs=bin,dev,proc,sys"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
