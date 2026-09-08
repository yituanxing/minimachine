from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Execute a linked MiniMachine Linux LLVM image in the P3 VM."
    )
    p.add_argument("input", type=Path)
    p.add_argument(
        "--linker-contract",
        type=Path,
        required=True,
        help="MiniMachine linker contract used to install the Linux data image",
    )
    p.add_argument("--entry", default="start_kernel")
    p.add_argument(
        "--native-report-every",
        type=int,
        default=0,
        help=(
            "native VM only: return to Python every N P3 steps for a "
            "low-overhead throughput/location sample; 0 disables"
        ),
    )
    p.add_argument(
        "--native-report-slot",
        action="append",
        default=[],
        help=(
            "native VM only: include the named current-function P3 frame slot "
            "in each --native-report-every sample; may be repeated"
        ),
    )
    p.add_argument(
        "--native-boot-phases-only",
        action="store_true",
        help=(
            "for native full boot, trace only low-frequency boot phase "
            "milestones so scheduler/initcall tracing does not dominate runtime"
        ),
    )
    p.add_argument(
        "--stop-on-panic",
        action="store_true",
        help="stop immediately after recording the first kernel panic entry",
    )
    p.add_argument(
        "--stop-after-user-handoff",
        action="store_true",
        help=(
            "stop after Linux has successfully exec'd and transferred to "
            "the first MiniMachine userspace P3 function"
        ),
    )
    p.add_argument(
        "--native-vm",
        action="store_true",
        help="execute strict P3 with the C native VM backend",
    )
    p.add_argument(
        "--native-hot-cache-in",
        type=Path,
        help="load slim Python metadata for mmap-backed native hot replay",
    )
    p.add_argument(
        "--native-pack-cache-in",
        type=Path,
        help="mmap a generation-matched prepacked native P3 image",
    )
    p.add_argument(
        "--native-pack-cache-out",
        type=Path,
        help="save the fully resolved native P3 packed image for reuse",
    )
    p.add_argument(
        "--native-append-pack-cache-in-dir",
        type=Path,
        help="load mmap-backed cached native P3 append segments from this directory",
    )
    p.add_argument(
        "--native-append-pack-cache-out-dir",
        type=Path,
        help="save dynamically appended native P3 segments into this directory",
    )
    p.add_argument("--max-steps", type=int, default=10_000_000)
    p.add_argument("--progress-every", type=int, default=250_000)
    p.add_argument(
        "--probe-initcall-table",
        action="store_true",
        help="dump installed initcall table/boundaries and exit before execution",
    )
    p.add_argument(
        "--probe-kallsyms",
        metavar="SYMBOL",
        help="probe Linux kallsyms against one P3 function and exit",
    )
    p.add_argument(
        "--initramfs",
        type=Path,
        help=(
            "raw newc/cpio image backing Linux __initramfs_start in the "
            "LLVM/P3 execution image"
        ),
    )
    p.add_argument(
        "--checkpoint-in",
        type=Path,
        help="resume VM state from a checkpoint for the exact linked image",
    )
    p.add_argument(
        "--inject-init",
        type=Path,
        help=(
            "after restoring a checkpoint, create /init in the live Linux "
            "rootfs through filp_open/kernel_write/fput before replay"
        ),
    )
    p.add_argument(
        "--inject-init-path",
        default="/init",
        help=(
            "guest path written by --inject-init; defaults to /init"
        ),
    )
    p.add_argument(
        "--inject-file",
        action="append",
        nargs=2,
        metavar=("HOST", "GUEST"),
        default=[],
        help=(
            "after restoring a checkpoint, write an additional host file to "
            "the given guest path through filp_open/kernel_write/fput; may be repeated"
        ),
    )
    p.add_argument(
        "--inject-initramfs-cpio",
        type=Path,
        help=(
            "after restoring a checkpoint, unpack a newc/cpio archive into "
            "the live Linux rootfs through Linux unpack_to_rootfs()"
        ),
    )
    p.add_argument(
        "--trace-hot-filp-open",
        action="store_true",
        help=(
            "after checkpoint restore, trace the Linux filp_open O_CREAT "
            "control path during hot /init injection"
        ),
    )
    p.add_argument(
        "--probe-rootfs-after-checkpoint",
        action="store_true",
        help="probe live Linux rootfs paths and current task after checkpoint restore",
    )
    p.add_argument(
        "--probe-run-init-after-checkpoint",
        action="store_true",
        help=(
            "after checkpoint restore/injection, call Linux "
            "run_init_process('/init'), report the exact return code, and exit"
        ),
    )
    p.add_argument(
        "--restart-run-init-after-checkpoint",
        action="store_true",
        help=(
            "after checkpoint restore/injection, discard the in-flight exec "
            "frame and restart Linux run_init_process('/init') as the active "
            "control path so a successful userspace handoff is preserved"
        ),
    )
    p.add_argument(
        "--restart-init-path",
        default="/init",
        help=(
            "path passed to run_init_process when "
            "--restart-run-init-after-checkpoint is used; defaults to /init"
        ),
    )
    p.add_argument(
        "--skip-prepare-namespace-after-inject",
        action="store_true",
        help=(
            "with --inject-init at a prepare_namespace entry checkpoint, "
            "restore ramdisk_execute_command and resume its caller as if "
            "/init had existed at the preceding init_eaccess check"
        ),
    )
    p.add_argument(
        "--checkpoint-out",
        type=Path,
        help="write a resumable VM checkpoint during replay",
    )
    p.add_argument(
        "--checkpoint-initcall",
        help="write --checkpoint-out on entry to this Linux initcall",
    )
    p.add_argument(
        "--checkpoint-after-initcall",
        help="write --checkpoint-out when the initcall after this one begins",
    )
    p.add_argument(
        "--checkpoint-function",
        help="write --checkpoint-out on an occurrence of this function entry",
    )
    p.add_argument(
        "--checkpoint-function-hit",
        type=int,
        default=1,
        help="1-based occurrence of --checkpoint-function to capture",
    )
    p.add_argument(
        "--stop-after-checkpoint",
        action="store_true",
        help="stop replay immediately after writing a checkpoint",
    )
    p.add_argument(
        "--program-cache-in",
        type=Path,
        help="load a lowered P3 program cache for the exact linked image",
    )
    p.add_argument(
        "--program-cache-out",
        type=Path,
        help="write the lowered P3 program before runtime callbacks are bound",
    )
    p.add_argument(
        "--checkpoint-at-limit",
        action="store_true",
        help="write --checkpoint-out when execution stops at the step limit",
    )
    p.add_argument(
        "--checkpoint-on-error",
        action="store_true",
        help="write --checkpoint-out before reporting any VM execution error",
    )
    return p.parse_args()
