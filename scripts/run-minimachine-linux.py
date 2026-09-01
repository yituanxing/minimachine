#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir
from src.minimachine.abi import CALLER_SP, RESULT_COUNT, RESULT_PTR, RET_PC, expand_function
from src.minimachine.checkpoint import (
    CheckpointError,
    image_fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from src.minimachine.image import ImageError, install_module_image, parse_module_image
from src.minimachine.legalize import legalize_module
from src.minimachine.layout import DataLayout
from src.minimachine.kallsyms import install_p3_kallsyms
from src.minimachine.linker import LinkerContract
from src.minimachine.lower_p3 import lower_function
from src.minimachine.program_cache import (
    ProgramCache,
    ProgramCacheError,
    load_program_cache,
    save_program_cache,
)
from src.minimachine.runtime import (
    accelerate_direct_runtime,
    collect_runtime_surface,
    direct_runtime_callback,
    helper_callback,
    install_runtime,
)
from src.minimachine.user_image import UserImageError, unpack_user_image
from src.minimachine.verify import verify_muir, verify_p3
from src.minimachine.vm import HOST_CONTROL_TRANSFER, Program, VMError


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


def trap_symbol(reason: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", reason).strip("_")
    return "__mm_trap_" + (tag or "unknown")


def arch_escapes(function: muir.Function):
    return [
        inst
        for block in function.blocks
        for inst in block.instructions
        if isinstance(inst, muir.ArchEscape)
    ]


def trap_reasons(function: muir.Function):
    return {
        inst.reason
        for block in function.blocks
        for inst in block.instructions
        if isinstance(inst, muir.Trap)
    }


def register_traps(program: Program, reasons: set[str]) -> None:
    def callback(reason: str):
        def trap(_vm, _args):
            raise VMError(f"MiniMachine trap reached: {reason}")
        return trap

    for reason in sorted(reasons):
        symbol = trap_symbol(reason)
        if symbol not in program.symbol_addresses:
            program.register_service(symbol, callback(reason))



def refresh_host_service_descriptors(vm) -> None:
    """Reapply immutable host-service descriptors after checkpoint restore.

    Native sparse-page checkpoints capture guest memory exactly, so a
    checkpoint created before a newly registered host service cannot contain
    that service's descriptor.  Program descriptors are immutable linker
    metadata; restoring them does not alter live Linux data.
    """
    refreshed = 0
    accelerated = 0
    for symbol in vm.program.host_services:
        descriptor_symbol = symbol
        fast_prefix = "__mm_fast_"
        if symbol.startswith(fast_prefix):
            descriptor_symbol = symbol[len(fast_prefix):]
            accelerated += 1
        descriptor = vm.program.symbol_addresses.get(descriptor_symbol)
        if descriptor is None:
            continue
        for offset in range(16):
            byte = vm.program.initial_memory.read(descriptor + offset, 8)
            vm.memory.write(descriptor + offset, 8, byte)
        refreshed += 1
    print(
        "BOOT_EXEC_HOST_DESCRIPTORS_REFRESHED "
        f"count={refreshed} accelerated={accelerated}",
        flush=True,
    )


def _guest_function_name_from_descriptor(vm, descriptor: int) -> str:
    matches = [
        name
        for name, address in vm.program.symbol_addresses.items()
        if address == descriptor and name in vm.program.functions
    ]
    if not matches:
        raise VMError(
            f"guest callback descriptor 0x{descriptor:x} is not a P3 function"
        )
    return sorted(matches)[0]


def _call_guest_descriptor_preserving_control(
    vm,
    descriptor: int,
    args: tuple[int, ...],
    *,
    result_count: int = 1,
    max_extra_steps: int = 1_000_000,
) -> tuple[int, ...]:
    name = _guest_function_name_from_descriptor(vm, descriptor)
    return _call_linux_function_preserving_control(
        vm,
        name,
        args,
        result_count=result_count,
        max_extra_steps=max_extra_steps,
    )


def _user_external_original(symbol: str) -> str:
    prefix = "__mm_user_ext_"
    return symbol[len(prefix):] if symbol.startswith(prefix) else symbol


def _user_libc_callback(symbol: str, errno_address: int | None):
    original = _user_external_original(symbol)

    direct = direct_runtime_callback(original)
    if direct is not None:
        return direct

    if original == "bcmp":
        return direct_runtime_callback("memcmp")

    if original == "__errno_location" and errno_address is not None:
        def errno_location(_vm, args):
            if args:
                raise VMError("__errno_location expects no arguments")
            return errno_address
        return errno_location

    if original == "_setjmp":
        def user_setjmp(vm, args):
            if len(args) != 1:
                raise VMError("_setjmp expects jmp_buf")
            env = int(args[0])
            expected = vm.memory.read(vm.sp + RESULT_COUNT, 64)
            if expected != 1:
                raise VMError(
                    f"_setjmp caller expects {expected} results, expected 1"
                )
            state = (
                vm.memory.read(vm.sp + CALLER_SP, 64),
                vm.memory.read(vm.sp + RET_PC, 64),
                vm.memory.read(vm.sp + RESULT_PTR, 64),
                int(vm.heap_next),
            )
            table = getattr(vm, "user_setjmp_states", None)
            if table is None:
                table = {}
                vm.user_setjmp_states = table
            table[env] = state
            print(
                "BOOT_EXEC_USER_SETJMP "
                f"env=0x{env:x} resume_sp=0x{state[0]:x} "
                f"resume_pc=0x{state[1]:x} result_ptr=0x{state[2]:x}",
                flush=True,
            )
            return 0
        return user_setjmp

    if original == "longjmp":
        def user_longjmp(vm, args):
            if len(args) != 2:
                raise VMError("longjmp expects jmp_buf,value")
            env, value = map(int, args)
            table = getattr(vm, "user_setjmp_states", None)
            state = table.get(env) if table is not None else None
            if state is None:
                raise VMError(
                    f"longjmp received unknown jmp_buf 0x{env:x}"
                )
            resume_sp, resume_pc, result_ptr, heap_next = state
            result = value if value != 0 else 1
            vm.memory.write(result_ptr, 64, result)
            vm.heap_next = heap_next
            vm.sp = resume_sp
            vm.halted = False
            vm._set_code(resume_pc)
            print(
                "BOOT_EXEC_USER_LONGJMP "
                f"env=0x{env:x} value={result} "
                f"resume_sp=0x{resume_sp:x} resume_pc=0x{resume_pc:x}",
                flush=True,
            )
            return HOST_CONTROL_TRANSFER
        return user_longjmp

    def set_errno(vm, value: int) -> None:
        if errno_address is not None:
            vm.memory.write(errno_address, 32, value & 0xFFFFFFFF)

    def libc_linux_result(vm, raw: int) -> int:
        signed = raw - (1 << 64) if raw & (1 << 63) else raw
        if -4095 <= signed < 0:
            set_errno(vm, -signed)
            return (1 << 64) - 1
        return raw

    def stdio_fd(stream: int) -> int:
        # Userspace external FILE* values are opaque MiniMachine handles.
        # Keep stdio unbuffered and route visible I/O through Linux fds.
        if stream == 2:
            return 2
        if stream == 1:
            return 1
        return 0

    if original == "bsearch":
        def user_bsearch(vm, args):
            if len(args) != 5:
                raise VMError("bsearch expects key,base,nmemb,size,compar")
            key, base, nmemb, size, compar = map(int, args)
            if size <= 0:
                return 0

            def preview_cstring(ptr: int, limit: int = 48) -> str:
                if not ptr:
                    return "<null>"
                data = bytearray()
                for i in range(limit):
                    byte = vm.memory.read(ptr + i, 8)
                    if byte == 0:
                        break
                    data.append(byte)
                return data.decode("utf-8", errors="backslashreplace")

            comparator_name = _guest_function_name_from_descriptor(vm, compar)
            key_text = preview_cstring(key)
            lo = 0
            hi = nmemb
            calls = 0
            while lo < hi:
                mid = lo + (hi - lo) // 2
                element = base + mid * size
                element_name_ptr = vm.memory.read(element, 64)
                raw_name = preview_cstring(element_name_ptr)
                first = (
                    vm.memory.read(element_name_ptr, 8)
                    if element_name_ptr else 0
                )
                logical_name = (
                    preview_cstring(element_name_ptr + 1)
                    if 0 < first < 8
                    else raw_name
                )
                raw, = _call_guest_descriptor_preserving_control(
                    vm,
                    compar,
                    (key, element),
                    result_count=1,
                    max_extra_steps=500_000,
                )
                calls += 1
                raw32 = raw & 0xFFFFFFFF
                cmp_value = (
                    raw32 - (1 << 32)
                    if raw32 & (1 << 31)
                    else raw32
                )
                print(
                    "BOOT_EXEC_USER_BSEARCH_CMP "
                    f"compar={comparator_name} key={key_text!r} "
                    f"nmemb={nmemb} size={size} lo={lo} hi={hi} mid={mid} "
                    f"element=0x{element:x} name_ptr=0x{element_name_ptr:x} "
                    f"first=0x{first:x} name={logical_name!r} "
                    f"raw=0x{raw:x} cmp={cmp_value}",
                    flush=True,
                )
                if cmp_value < 0:
                    hi = mid
                elif cmp_value > 0:
                    lo = mid + 1
                else:
                    print(
                        "BOOT_EXEC_USER_BSEARCH "
                        f"compar={comparator_name} key={key_text!r} "
                        f"nmemb={nmemb} size={size} calls={calls} "
                        f"index={mid} result=0x{element:x}",
                        flush=True,
                    )
                    return element

            print(
                "BOOT_EXEC_USER_BSEARCH "
                f"compar={comparator_name} key={key_text!r} "
                f"nmemb={nmemb} size={size} calls={calls} result=0x0",
                flush=True,
            )
            return 0

        return user_bsearch

    if original == "fflush":
        def user_fflush(_vm, args):
            if len(args) != 1:
                raise VMError("fflush expects FILE*")
            return 0
        return user_fflush

    if original == "clearerr":
        def user_clearerr(_vm, args):
            if len(args) != 1:
                raise VMError("clearerr expects FILE*")
            return None
        return user_clearerr

    if original == "ferror_unlocked":
        def user_ferror(_vm, args):
            if len(args) != 1:
                raise VMError("ferror_unlocked expects FILE*")
            return 0
        return user_ferror

    if original == "fputs_unlocked":
        def user_fputs(vm, args):
            if len(args) != 2:
                raise VMError("fputs_unlocked expects string,FILE*")
            ptr, stream = map(int, args)
            length = 0
            bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
            if bulk_strlen is not None:
                length = bulk_strlen(ptr)
            else:
                while vm.memory.read(ptr + length, 8) != 0:
                    length += 1
            raw = user_syscall(
                vm,
                (64, stdio_fd(stream), ptr, length, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return 0
        return user_fputs

    if original == "putc_unlocked":
        def user_putc(vm, args):
            if len(args) != 2:
                raise VMError("putc_unlocked expects char,FILE*")
            ch, stream = map(int, args)
            ptr = vm.alloc_bytes(1, align=1)
            vm.memory.write(ptr, 8, ch & 0xFF)
            raw = user_syscall(
                vm,
                (64, stdio_fd(stream), ptr, 1, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return ch & 0xFF
        return user_putc

    if original == "puts":
        def user_puts(vm, args):
            if len(args) != 1:
                raise VMError("puts expects string")
            ptr = int(args[0])
            length = 0
            bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
            if bulk_strlen is not None:
                length = bulk_strlen(ptr)
            else:
                while vm.memory.read(ptr + length, 8) != 0:
                    length += 1
            raw = user_syscall(vm, (64, 1, ptr, length, 0, 0, 0))
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            nl = vm.alloc_bytes(1, align=1)
            vm.memory.write(nl, 8, 10)
            raw = user_syscall(vm, (64, 1, nl, 1, 0, 0, 0))
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return 0
        return user_puts

    def encode_dev(major: int, minor: int) -> int:
        return (
            (minor & 0xFF)
            | ((major & 0xFFF) << 8)
            | ((minor & ~0xFF) << 12)
            | ((major & ~0xFFF) << 32)
        ) & ((1 << 64) - 1)

    def write_user_stat_from_statx(vm, stat_ptr: int, statx_ptr: int) -> None:
        # RISC-V glibc struct stat (128 bytes), probed with the exact
        # cross libc used by the BusyBox carrier.
        for i in range(128):
            vm.memory.write(stat_ptr + i, 8, 0)

        stx_blksize = vm.memory.read(statx_ptr + 4, 32)
        stx_nlink = vm.memory.read(statx_ptr + 16, 32)
        stx_uid = vm.memory.read(statx_ptr + 20, 32)
        stx_gid = vm.memory.read(statx_ptr + 24, 32)
        stx_mode = vm.memory.read(statx_ptr + 28, 16)
        stx_ino = vm.memory.read(statx_ptr + 32, 64)
        stx_size = vm.memory.read(statx_ptr + 40, 64)
        stx_blocks = vm.memory.read(statx_ptr + 48, 64)
        stx_rdev_major = vm.memory.read(statx_ptr + 128, 32)
        stx_rdev_minor = vm.memory.read(statx_ptr + 132, 32)
        stx_dev_major = vm.memory.read(statx_ptr + 136, 32)
        stx_dev_minor = vm.memory.read(statx_ptr + 140, 32)

        vm.memory.write(
            stat_ptr + 0,
            64,
            encode_dev(stx_dev_major, stx_dev_minor),
        )
        vm.memory.write(stat_ptr + 8, 64, stx_ino)
        vm.memory.write(stat_ptr + 16, 32, stx_mode)
        vm.memory.write(stat_ptr + 20, 32, stx_nlink)
        vm.memory.write(stat_ptr + 24, 32, stx_uid)
        vm.memory.write(stat_ptr + 28, 32, stx_gid)
        vm.memory.write(
            stat_ptr + 32,
            64,
            encode_dev(stx_rdev_major, stx_rdev_minor),
        )
        vm.memory.write(stat_ptr + 48, 64, stx_size)
        vm.memory.write(stat_ptr + 56, 64, stx_blksize)
        vm.memory.write(stat_ptr + 64, 64, stx_blocks)

        # struct statx timestamps: atime@64, btime@80, ctime@96, mtime@112.
        # struct stat timestamps: atim@72, mtim@88, ctim@104.
        for dst_off, src_off in ((72, 64), (88, 112), (104, 96)):
            sec = vm.memory.read(statx_ptr + src_off, 64)
            nsec = vm.memory.read(statx_ptr + src_off + 8, 32)
            vm.memory.write(stat_ptr + dst_off, 64, sec)
            vm.memory.write(stat_ptr + dst_off + 8, 64, nsec)

    if original in {"stat", "lstat"}:
        def user_stat(vm, args):
            if len(args) != 2:
                raise VMError(f"{original} expects path,stat")
            path_ptr, stat_ptr = map(int, args)
            if "__se_sys_statx" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_statx")

            statx_ptr = vm.alloc_bytes(256, align=8)
            bulk_fill = getattr(vm.memory, "bulk_fill", None)
            if bulk_fill is not None:
                bulk_fill(statx_ptr, 0, 256)
            else:
                for i in range(256):
                    vm.memory.write(statx_ptr + i, 8, 0)

            at_fdcwd = (-100) & ((1 << 64) - 1)
            flags = 0x100 if original == "lstat" else 0
            statx_basic_stats = 0x000007FF
            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_statx",
                (at_fdcwd, path_ptr, flags, statx_basic_stats, statx_ptr),
                result_count=1,
                max_extra_steps=4_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1

            write_user_stat_from_statx(vm, stat_ptr, statx_ptr)
            print(
                "BOOT_EXEC_USER_STAT "
                f"kind={original} path_ptr=0x{path_ptr:x} "
                f"stat_ptr=0x{stat_ptr:x} "
                f"mode=0{vm.memory.read(stat_ptr + 16, 32):o} "
                f"size={vm.memory.read(stat_ptr + 48, 64)}",
                flush=True,
            )
            return 0

        return user_stat

    if original == "isatty":
        def user_isatty(vm, args):
            if len(args) != 1:
                raise VMError("isatty expects fd")
            if "__se_sys_ioctl" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_ioctl")
            fd = int(args[0])
            termios = vm.alloc_bytes(64, align=8)
            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_ioctl",
                (fd, 0x5401, termios),
                result_count=1,
                max_extra_steps=2_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return 0
            return 1

        return user_isatty

    if original == "strerror":
        messages = {
            0: b"Success",
            2: b"No such file or directory",
            5: b"Input/output error",
            9: b"Bad file descriptor",
            12: b"Cannot allocate memory",
            13: b"Permission denied",
            22: b"Invalid argument",
            38: b"Function not implemented",
            95: b"Operation not supported",
        }

        def user_strerror(vm, args):
            if len(args) != 1:
                raise VMError("strerror expects errno")
            err = int(args[0]) & 0xFFFFFFFF
            payload = messages.get(
                err,
                f"Unknown error {err}".encode("ascii"),
            ) + b"\0"
            ptr = vm.alloc_bytes(len(payload), align=1)
            bulk_write = getattr(vm.memory, "bulk_write", None)
            if bulk_write is not None:
                bulk_write(ptr, payload)
            else:
                for i, byte in enumerate(payload):
                    vm.memory.write(ptr + i, 8, byte)
            return ptr

        return user_strerror

    if original in {"fcntl", "fcntl64"}:
        def user_fcntl(vm, args):
            if len(args) not in {2, 3}:
                raise VMError(f"{original} expects fd,cmd[,arg]")
            fd, cmd = map(int, args[:2])
            arg = int(args[2]) if len(args) == 3 else 0
            raw = user_syscall(
                vm,
                (25, fd, cmd, arg, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)
        return user_fcntl

    if original in {"open", "open64"}:
        def user_open(vm, args):
            if len(args) not in {2, 3}:
                raise VMError(f"{original} expects path,flags[,mode]")
            path, flags = map(int, args[:2])
            mode = int(args[2]) if len(args) == 3 else 0
            raw = user_syscall(
                vm,
                (
                    56,  # openat
                    (-100) & ((1 << 64) - 1),  # AT_FDCWD
                    path,
                    flags,
                    mode,
                    0,
                    0,
                ),
            )
            return libc_linux_result(vm, raw)
        return user_open

    if original == "getcwd":
        def user_getcwd(vm, args):
            if len(args) != 2:
                raise VMError("getcwd expects buffer,size")
            buf, size = map(int, args)
            if "__se_sys_getcwd" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_getcwd")

            if buf == 0:
                alloc_size = size if size else 4096
                if alloc_size <= 0:
                    set_errno(vm, 22)
                    return 0
                buf = vm.alloc_bytes(alloc_size, align=1)
                size = alloc_size
            elif size == 0:
                set_errno(vm, 22)
                return 0

            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_getcwd",
                (buf, size),
                result_count=1,
                max_extra_steps=2_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return 0
            return buf

        return user_getcwd

    def check_signal_number(vm, signum: int) -> bool:
        if 1 <= signum <= 64:
            return True
        set_errno(vm, 22)
        return False

    if original in {
        "sigfillset",
        "sigemptyset",
        "sigaddset",
        "sigdelset",
        "sigismember",
    }:
        def sigset_op(vm, args):
            if original in {"sigfillset", "sigemptyset"}:
                if len(args) != 1:
                    raise VMError(f"{original} expects sigset_t*")
                ptr = int(args[0])
                fill = 0xFF if original == "sigfillset" else 0x00
                bulk_fill = getattr(vm.memory, "bulk_fill", None)
                if bulk_fill is not None:
                    bulk_fill(ptr, fill, 128)
                else:
                    for i in range(128):
                        vm.memory.write(ptr + i, 8, fill)
                return 0

            if len(args) != 2:
                raise VMError(f"{original} expects sigset_t*,signum")
            ptr, signum = map(int, args)
            if not check_signal_number(vm, signum):
                return -1 & ((1 << 64) - 1)
            bit = signum - 1
            byte_addr = ptr + (bit // 8)
            mask = 1 << (bit % 8)
            old = vm.memory.read(byte_addr, 8)
            if original == "sigaddset":
                vm.memory.write(byte_addr, 8, old | mask)
                return 0
            if original == "sigdelset":
                vm.memory.write(byte_addr, 8, old & ~mask)
                return 0
            return 1 if old & mask else 0

        return sigset_op

    def call_rt_sigaction(vm, signum: int, act_ptr: int, old_ptr: int) -> int:
        if "__se_sys_rt_sigaction" not in vm.program.functions:
            raise VMError("Linux image is missing __se_sys_rt_sigaction")
        kact = 0
        kold = 0
        if act_ptr:
            kact = vm.alloc_bytes(24, align=8)
            # glibc/RISC-V struct sigaction:
            #   handler @0, 128-byte mask @8, flags @136, restorer @144.
            # MiniMachine kernel struct sigaction:
            #   handler @0, flags @8, kernel_sigset_t(low64) @16.
            handler = vm.memory.read(act_ptr + 0, 64)
            mask = vm.memory.read(act_ptr + 8, 64)
            flags = vm.memory.read(act_ptr + 136, 32)
            vm.memory.write(kact + 0, 64, handler)
            vm.memory.write(kact + 8, 64, flags)
            vm.memory.write(kact + 16, 64, mask)
        if old_ptr:
            kold = vm.alloc_bytes(24, align=8)

        result, = _call_linux_function_preserving_control(
            vm,
            "__se_sys_rt_sigaction",
            (signum, kact, kold, 8),
            result_count=1,
            max_extra_steps=2_000_000,
        )
        signed = result - (1 << 64) if result & (1 << 63) else result
        if signed < 0:
            set_errno(vm, -signed)
            return result

        if old_ptr:
            handler = vm.memory.read(kold + 0, 64)
            flags = vm.memory.read(kold + 8, 64)
            mask = vm.memory.read(kold + 16, 64)
            vm.memory.write(old_ptr + 0, 64, handler)
            for i in range(128):
                vm.memory.write(old_ptr + 8 + i, 8, 0)
            vm.memory.write(old_ptr + 8, 64, mask)
            vm.memory.write(old_ptr + 136, 32, flags)
            vm.memory.write(old_ptr + 144, 64, 0)
        return result

    if original == "sigaction":
        def user_sigaction(vm, args):
            if len(args) != 3:
                raise VMError("sigaction expects signum,act,oldact")
            signum, act_ptr, old_ptr = map(int, args)
            if not check_signal_number(vm, signum):
                return -1 & ((1 << 64) - 1)
            return call_rt_sigaction(vm, signum, act_ptr, old_ptr)
        return user_sigaction

    if original == "sigprocmask":
        def user_sigprocmask(vm, args):
            if len(args) != 3:
                raise VMError("sigprocmask expects how,set,oldset")
            how, set_ptr, old_ptr = map(int, args)
            if "__se_sys_rt_sigprocmask" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_rt_sigprocmask")
            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_rt_sigprocmask",
                (how, set_ptr, old_ptr, 8),
                result_count=1,
                max_extra_steps=2_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return -1 & ((1 << 64) - 1)
            return 0
        return user_sigprocmask

    if original == "signal":
        def user_signal(vm, args):
            if len(args) != 2:
                raise VMError("signal expects signum,handler")
            signum, handler = map(int, args)
            if not check_signal_number(vm, signum):
                return ((1 << 64) - 1)
            act = vm.alloc_bytes(152, align=8)
            old = vm.alloc_bytes(152, align=8)
            bulk_fill = getattr(vm.memory, "bulk_fill", None)
            if bulk_fill is not None:
                bulk_fill(act, 0, 152)
                bulk_fill(old, 0, 152)
            else:
                for i in range(152):
                    vm.memory.write(act + i, 8, 0)
                    vm.memory.write(old + i, 8, 0)
            vm.memory.write(act + 0, 64, handler)
            # SA_RESTART, matching the usual libc signal() semantics.
            vm.memory.write(act + 136, 32, 0x10000000)
            result = call_rt_sigaction(vm, signum, act, old)
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                return ((1 << 64) - 1)
            return vm.memory.read(old + 0, 64)
        return user_signal

    if original == "mallopt":
        def user_mallopt(_vm, args):
            if len(args) != 2:
                raise VMError("mallopt expects parameter,value")
            parameter, value = map(int, args)
            print(
                "BOOT_EXEC_USER_MALLOPT "
                f"parameter={parameter if parameter < (1 << 63) else parameter - (1 << 64)} "
                f"value={value if value < (1 << 63) else value - (1 << 64)}",
                flush=True,
            )
            # BusyBox uses mallopt only to tune the host libc allocator.
            # MiniMachine provides its own userspace allocation arena, so
            # accepting the hint is the correct semantic equivalent.
            return 1
        return user_mallopt

    if original == "getcwd":
        def user_getcwd(vm, args):
            if len(args) != 2:
                raise VMError("getcwd expects buffer,size")
            buf, size = map(int, args)
            allocated = False
            capacity = size

            if buf == 0:
                # GNU/POSIX extension used by BusyBox ash: getcwd(NULL, 0)
                # asks libc to allocate a sufficiently large result buffer.
                capacity = size if size else 4096
                buf = vm.alloc_bytes(capacity, align=16)
                allocations = getattr(vm, "user_allocations", None)
                if allocations is None:
                    allocations = {}
                    vm.user_allocations = allocations
                allocations[buf] = capacity
                allocated = True
            elif size == 0:
                set_errno(vm, 22)
                return 0

            raw = user_syscall(
                vm,
                (17, buf, capacity, 0, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                if allocated:
                    allocations = getattr(vm, "user_allocations", None)
                    if allocations is not None:
                        allocations.pop(buf, None)
                return 0

            preview = bytearray()
            for index in range(min(capacity, 512)):
                byte = vm.memory.read(buf + index, 8)
                if byte == 0:
                    break
                preview.append(byte)
            print(
                "BOOT_EXEC_USER_GETCWD "
                f"ptr=0x{buf:x} size={capacity} "
                f"result={preview.decode('utf-8', errors='replace')!r}",
                flush=True,
            )
            return buf

        return user_getcwd

    if original == "malloc":
        def malloc(vm, args):
            if len(args) != 1:
                raise VMError("malloc expects size")
            size = int(args[0])
            ptr = vm.alloc_bytes(size, align=16)
            allocations = getattr(vm, "user_allocations", None)
            if allocations is None:
                allocations = {}
                vm.user_allocations = allocations
            allocations[ptr] = size
            return ptr
        return malloc

    if original == "free":
        def free(vm, args):
            if len(args) != 1:
                raise VMError("free expects pointer")
            ptr = int(args[0])
            allocations = getattr(vm, "user_allocations", None)
            if allocations is not None:
                allocations.pop(ptr, None)
            return None
        return free

    if original == "realloc":
        def realloc(vm, args):
            if len(args) != 2:
                raise VMError("realloc expects pointer,size")
            ptr, size = map(int, args)
            allocations = getattr(vm, "user_allocations", None)
            if allocations is None:
                allocations = {}
                vm.user_allocations = allocations
            if ptr == 0:
                out = vm.alloc_bytes(size, align=16)
                allocations[out] = size
                return out
            if size == 0:
                allocations.pop(ptr, None)
                return 0
            old_size = allocations.get(ptr)
            if old_size is None:
                raise VMError(
                    f"realloc received unknown userspace allocation 0x{ptr:x}"
                )
            out = vm.alloc_bytes(size, align=16)
            copy_size = min(old_size, size)
            bulk = getattr(vm.memory, "bulk_copy", None)
            if bulk is not None:
                bulk(out, ptr, copy_size)
            else:
                for i in range(copy_size):
                    vm.memory.write(
                        out + i,
                        8,
                        vm.memory.read(ptr + i, 8),
                    )
            allocations.pop(ptr, None)
            allocations[out] = size
            return out
        return realloc

    if original == "atoi":
        def user_atoi(vm, args):
            if len(args) != 1:
                raise VMError("atoi expects string")
            ptr = int(args[0])
            data = bytearray()
            for index in range(256):
                byte = vm.memory.read(ptr + index, 8)
                if byte == 0:
                    break
                data.append(byte)
            text = data.decode("ascii", errors="ignore")
            pos = 0
            while pos < len(text) and text[pos].isspace():
                pos += 1
            sign = 1
            if pos < len(text) and text[pos] in "+-":
                if text[pos] == "-":
                    sign = -1
                pos += 1
            value = 0
            while pos < len(text) and text[pos].isdigit():
                value = value * 10 + (ord(text[pos]) - ord("0"))
                pos += 1
            value *= sign
            print(
                "BOOT_EXEC_USER_ATOI "
                f"text={text!r} result={value}",
                flush=True,
            )
            return value & ((1 << 64) - 1)

        return user_atoi

    if original == "vsnprintf":
        def user_vsnprintf(vm, args):
            if len(args) != 4:
                raise VMError("vsnprintf expects buffer,size,format,va_list")
            outbuf, size, fmt_ptr, ap = map(int, args)

            def read_cstring(ptr: int, limit: int = 1 << 20) -> bytes:
                if not ptr:
                    return b""
                data = bytearray()
                bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
                if bulk_strlen is not None:
                    length = min(int(bulk_strlen(ptr)), limit)
                    bulk_read = getattr(vm.memory, "bulk_read", None)
                    if bulk_read is not None:
                        return bytes(bulk_read(ptr, length))
                for index in range(limit):
                    byte = vm.memory.read(ptr + index, 8)
                    if byte == 0:
                        break
                    data.append(byte)
                return bytes(data)

            fmt = read_cstring(fmt_ptr, 4096).decode("latin1")
            cursor = ap

            def next_arg() -> int:
                nonlocal cursor
                value = vm.memory.read(cursor, 64)
                cursor += 8
                return int(value)

            def signed_value(value: int, bits: int) -> int:
                mask = (1 << bits) - 1
                value &= mask
                sign = 1 << (bits - 1)
                return value - (1 << bits) if value & sign else value

            def unsigned_value(value: int, bits: int) -> int:
                return value & ((1 << bits) - 1)

            out = bytearray()
            index = 0
            while index < len(fmt):
                if fmt[index] != "%":
                    out.append(ord(fmt[index]))
                    index += 1
                    continue

                index += 1
                if index < len(fmt) and fmt[index] == "%":
                    out.append(ord("%"))
                    index += 1
                    continue

                flags = ""
                while index < len(fmt) and fmt[index] in "-+ #0":
                    flags += fmt[index]
                    index += 1

                if index < len(fmt) and fmt[index] == "*":
                    width = signed_value(next_arg(), 32)
                    if width < 0:
                        flags += "-"
                        width = -width
                    index += 1
                else:
                    width_start = index
                    while index < len(fmt) and fmt[index].isdigit():
                        index += 1
                    width = int(fmt[width_start:index] or "0")

                precision = None
                if index < len(fmt) and fmt[index] == ".":
                    index += 1
                    if index < len(fmt) and fmt[index] == "*":
                        raw_precision = signed_value(next_arg(), 32)
                        precision = None if raw_precision < 0 else raw_precision
                        index += 1
                    else:
                        precision_start = index
                        while index < len(fmt) and fmt[index].isdigit():
                            index += 1
                        precision = int(fmt[precision_start:index] or "0")

                length = ""
                if index < len(fmt) and fmt[index] in "hljzt":
                    length = fmt[index]
                    index += 1
                    if (
                        length in {"h", "l"}
                        and index < len(fmt)
                        and fmt[index] == length
                    ):
                        length += fmt[index]
                        index += 1

                if index >= len(fmt):
                    raise VMError("vsnprintf format ends after %")
                spec = fmt[index]
                index += 1
                bits = 64 if length in {"l", "ll", "j", "z", "t"} else 32
                prefix = ""
                numeric = False

                if spec in "di":
                    value = signed_value(next_arg(), bits)
                    sign = "-" if value < 0 else ("+" if "+" in flags else (" " if " " in flags else ""))
                    digits = str(abs(value))
                    if precision is not None:
                        digits = digits.rjust(precision, "0")
                    piece = sign + digits
                    numeric = True
                elif spec in "uoxX":
                    value = unsigned_value(next_arg(), bits)
                    if spec == "u":
                        digits = str(value)
                    elif spec == "o":
                        digits = format(value, "o")
                        if "#" in flags and (not digits.startswith("0")):
                            prefix = "0"
                    else:
                        digits = format(value, "x" if spec == "x" else "X")
                        if "#" in flags and value:
                            prefix = "0x" if spec == "x" else "0X"
                    if precision is not None:
                        digits = digits.rjust(precision, "0")
                    piece = prefix + digits
                    numeric = True
                elif spec == "p":
                    value = next_arg()
                    piece = "(nil)" if value == 0 else f"0x{value:x}"
                elif spec == "c":
                    piece = chr(next_arg() & 0xFF)
                elif spec == "s":
                    raw = read_cstring(next_arg())
                    if precision is not None:
                        raw = raw[:precision]
                    piece_bytes = raw
                    if width > len(piece_bytes):
                        padding = b" " * (width - len(piece_bytes))
                        piece_bytes = (
                            piece_bytes + padding
                            if "-" in flags
                            else padding + piece_bytes
                        )
                    out.extend(piece_bytes)
                    continue
                else:
                    raise VMError(
                        f"unsupported vsnprintf conversion %{length}{spec}"
                    )

                piece_bytes = piece.encode("latin1")
                if width > len(piece_bytes):
                    pad_len = width - len(piece_bytes)
                    if "-" in flags:
                        piece_bytes += b" " * pad_len
                    elif "0" in flags and precision is None and numeric:
                        lead = 0
                        if piece_bytes[:1] in {b"+", b"-", b" "}:
                            lead = 1
                        elif piece_bytes[:2] in {b"0x", b"0X"}:
                            lead = 2
                        piece_bytes = (
                            piece_bytes[:lead]
                            + b"0" * pad_len
                            + piece_bytes[lead:]
                        )
                    else:
                        piece_bytes = b" " * pad_len + piece_bytes
                out.extend(piece_bytes)

            payload = bytes(out)
            if size > 0:
                written = payload[: max(0, size - 1)]
                bulk_write = getattr(vm.memory, "bulk_write", None)
                if bulk_write is not None:
                    if written:
                        bulk_write(outbuf, written)
                else:
                    for offset, byte in enumerate(written):
                        vm.memory.write(outbuf + offset, 8, byte)
                vm.memory.write(outbuf + len(written), 8, 0)
            else:
                written = b""

            print(
                "BOOT_EXEC_USER_VSNPRINTF "
                f"fmt={fmt!r} result={len(payload)} written={len(written)} "
                f"ap=0x{ap:x}",
                flush=True,
            )
            return len(payload)

        return user_vsnprintf

    if original == "_exit":
        def user_exit(vm, args):
            if len(args) != 1:
                raise VMError("_exit expects status")
            status = int(args[0]) & 0xFF
            vm.user_exit_status = status
            parsefile_symbol = vm.program.symbol_addresses.get(
                "__mm_user_g_parsefile"
            )
            if parsefile_symbol is not None:
                parsefile = vm.memory.read(parsefile_symbol, 64)
                if parsefile:
                    next_ptr = vm.memory.read(parsefile + 24, 64)
                    buf_ptr = vm.memory.read(parsefile + 32, 64)
                    preview = bytearray()
                    if next_ptr:
                        for index in range(48):
                            byte = vm.memory.read(next_ptr + index, 8)
                            if byte == 0:
                                break
                            preview.append(byte)
                    print(
                        "BOOT_EXEC_USER_PARSEFILE_EXIT "
                        f"parsefile=0x{parsefile:x} "
                        f"linno={vm.memory.read(parsefile + 8, 32)} "
                        f"fd={vm.memory.read(parsefile + 12, 32)} "
                        f"left_line={vm.memory.read(parsefile + 16, 32)} "
                        f"left_buffer={vm.memory.read(parsefile + 20, 32)} "
                        f"next=0x{next_ptr:x} buf=0x{buf_ptr:x} "
                        f"consumed={next_ptr - buf_ptr if next_ptr and buf_ptr else -1} "
                        f"lastc0={vm.memory.read(parsefile + 120, 32)} "
                        f"lastc1={vm.memory.read(parsefile + 124, 32)} "
                        f"unget={vm.memory.read(parsefile + 128, 32)} "
                        f"preview={bytes(preview)!r}",
                        flush=True,
                    )
            evalskip_symbol = vm.program.symbol_addresses.get(
                "__mm_user_evalskip"
            )
            misc_symbol = vm.program.symbol_addresses.get(
                "__mm_user_ash_ptr_to_globals_misc"
            )
            evalskip = (
                vm.memory.read(evalskip_symbol, 8)
                if evalskip_symbol is not None else -1
            )
            misc = (
                vm.memory.read(misc_symbol, 64)
                if misc_symbol is not None else 0
            )
            nflag = vm.memory.read(misc + 98, 8) if misc else -1
            sflag = vm.memory.read(misc + 99, 8) if misc else -1
            print(
                "BOOT_EXEC_USER_ASH_CONTROL_EXIT "
                f"evalskip={evalskip} nflag={nflag} sflag={sflag} "
                f"misc=0x{misc:x}",
                flush=True,
            )
            vm.halted = True
            print(
                "BOOT_EXEC_USER_EXIT "
                f"status={status}",
                flush=True,
            )
            return HOST_CONTROL_TRANSFER
        return user_exit

    syscall_map = {
        "read": (63, 3),
        "write": (64, 3),
        "chdir": (49, 1),
        "close": (57, 1),
        "gettimeofday": (169, 2),
        "uname": (160, 1),
        "umask": (166, 1),
        "times": (153, 1),
        "getpid": (172, 0),
        "getppid": (173, 0),
        "getuid": (174, 0),
        "geteuid": (175, 0),
        "getgid": (176, 0),
        "getegid": (177, 0),
        "execve": (221, 3),
    }
    syscall_spec = syscall_map.get(original)
    if syscall_spec is not None:
        nr, argc = syscall_spec

        def libc_syscall(vm, args):
            if len(args) != argc:
                raise VMError(
                    f"{original} expects {argc} arguments, got {len(args)}"
                )
            padded = tuple(args) + (0,) * (6 - len(args))
            raw = user_syscall(vm, (nr, *padded))
            return libc_linux_result(vm, raw)
        return libc_syscall

    def unimplemented(_vm, args):
        preview = ",".join(f"0x{x:x}" for x in args[:8])
        print(
            "BOOT_EXEC_USER_EXTERNAL "
            f"name={original} argc={len(args)} args={preview}",
            flush=True,
        )
        raise VMError(
            f"unimplemented BusyBox userspace external {original}"
        )

    return unimplemented


def trace_user_external_descriptor(vm, original: str, stage: str) -> None:
    symbol = f"__mm_user_ext_{original}"
    descriptor = vm.program.symbol_addresses.get(symbol)
    if descriptor is None:
        print(
            "BOOT_EXEC_USER_DESCRIPTOR "
            f"stage={stage} name={original} descriptor=missing",
            flush=True,
        )
        return
    print(
        "BOOT_EXEC_USER_DESCRIPTOR "
        f"stage={stage} name={original} descriptor=0x{descriptor:x} "
        f"initial_entry=0x{vm.program.initial_memory.read(descriptor, 64):x} "
        f"initial_frame={vm.program.initial_memory.read(descriptor + 8, 64)} "
        f"live_entry=0x{vm.memory.read(descriptor, 64):x} "
        f"live_frame={vm.memory.read(descriptor + 8, 64)}",
        flush=True,
    )


def install_user_external_surface(vm, user_image, envp: int) -> None:
    image = user_image.image
    if image is None:
        return

    data_defaults = {
        "environ": envp,
        "optarg": 0,
        "opterr": 1,
        "optind": 1,
        "optopt": 0,
        "stdin": 0,
        "stdout": 1,
        "stderr": 2,
    }

    installed_data = 0
    for symbol in image.external_data:
        if symbol in vm.program.symbol_addresses:
            continue
        original = _user_external_original(symbol)
        value = data_defaults.get(original, 0)
        address = vm.program.define_data_symbol(
            symbol,
            int(value & ((1 << 64) - 1)).to_bytes(8, "little"),
            align=8,
        )
        for offset in range(8):
            vm.memory.write(
                address + offset,
                8,
                vm.program.initial_memory.read(address + offset, 8),
            )
        installed_data += 1
        print(
            "BOOT_EXEC_USER_EXTERNAL_DATA "
            f"name={original} address=0x{address:x} value=0x{value:x}",
            flush=True,
        )

    errno_symbol = "__mm_user_ext___errno_cell"
    errno_address = vm.program.symbol_addresses.get(errno_symbol)
    if errno_address is None:
        errno_address = vm.program.define_data_symbol(
            errno_symbol,
            b"\0" * 8,
            align=8,
        )
        for offset in range(8):
            vm.memory.write(
                errno_address + offset,
                8,
                vm.program.initial_memory.read(errno_address + offset, 8),
            )

    installed_functions = 0
    accelerated = 0
    for symbol in image.external_functions:
        if symbol in vm.program.symbol_addresses:
            continue
        original = _user_external_original(symbol)
        callback = _user_libc_callback(symbol, errno_address)
        if direct_runtime_callback(original) is not None or original == "bcmp":
            accelerated += 1
        vm.program.register_service(symbol, callback)
        descriptor = vm.program.symbol_addresses[symbol]
        for offset in range(16):
            vm.memory.write(
                descriptor + offset,
                8,
                vm.program.initial_memory.read(descriptor + offset, 8),
            )
        installed_functions += 1

    print(
        "BOOT_EXEC_USER_EXTERNAL_SURFACE "
        f"functions={installed_functions} data={installed_data} "
        f"portable_accel={accelerated} live_descriptors={installed_functions}",
        flush=True,
    )


def linux_ecall(vm, args: tuple[int, ...]):
    # Boot-first host ABI:
    #   service 1: write(ptr, len) to the host boot console.
    #   service 2: context_switch(prev,next,fresh_sp,start_fn,start_arg)
    #   service 3: enter_userspace(pt_regs) after a successful Linux exec.
    #   service 4: read host input into a MiniMachine console-device buffer.
    if not args:
        raise VMError("Linux ecall requires a service number")

    service = args[0]
    if service == 1:
        if len(args) != 3:
            raise VMError(
                f"Linux console ecall expects service,ptr,len; got {len(args)} args"
            )
        _, ptr, size = args
        if size > 1 << 20:
            raise VMError(f"MiniMachine Linux console write too large: {size}")
        data = bytes(vm.memory.read(ptr + i, 8) for i in range(size))
        text = data.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        return None

    if service == 4:
        if len(args) != 3:
            raise VMError(
                "Linux input ecall expects service,ptr,len; "
                f"got {len(args)} args"
            )
        _, ptr, size = args
        if size > 1 << 20:
            raise VMError(f"MiniMachine Linux input read too large: {size}")
        data = sys.stdin.buffer.read(size)
        for index, byte in enumerate(data):
            vm.memory.write(ptr + index, 8, byte)
        print(
            "BOOT_EXEC_HOST_INPUT "
            f"requested={size} returned={len(data)} "
            f"ptr=0x{ptr:x} data={data[:32].hex()}",
            flush=True,
        )
        return len(data)

    if service == 2:
        if len(args) != 6:
            raise VMError(
                "Linux context-switch ecall expects "
                "service,prev,next,fresh_sp,start_fn,start_arg"
            )
        _, prev, next_task, fresh_sp, start_fn, start_arg = args

        result_count = vm.memory.read(vm.sp + RESULT_COUNT, 64)
        if result_count != 1:
            raise VMError(
                f"Linux context switch expects one result, got {result_count}"
            )

        contexts = getattr(vm, "linux_task_contexts", None)
        if contexts is None:
            contexts = {}
            vm.linux_task_contexts = contexts

        # Save where __switch_to() must continue when this task is resumed,
        # together with the system-call result slot carrying the last task.
        contexts[prev] = (
            vm.memory.read(vm.sp + CALLER_SP, 64),
            vm.memory.read(vm.sp + RET_PC, 64),
            vm.memory.read(vm.sp + RESULT_PTR, 64),
        )

        saved = contexts.get(next_task)
        if saved is not None:
            resume_sp, resume_pc, result_ptr = saved
            vm.memory.write(result_ptr, 64, prev)
            vm.sp = resume_sp
            vm.halted = False
            vm._set_code(resume_pc)
            return HOST_CONTROL_TRANSFER

        if not fresh_sp or not start_fn:
            raise VMError(
                "first MiniMachine task switch lacks fresh stack/function: "
                f"next=0x{next_task:x} sp=0x{fresh_sp:x} fn=0x{start_fn:x}"
            )
        if "minimachine_ret_from_fork" not in vm.program.functions:
            raise VMError("MiniMachine ret-from-fork trampoline is missing")

        shadow_stacks = getattr(vm, "linux_task_shadow_stacks", None)
        if shadow_stacks is None:
            shadow_stacks = {}
            vm.linux_task_shadow_stacks = shadow_stacks

        shadow_top = shadow_stacks.get(next_task)
        if shadow_top is None:
            shadow_top = vm.linux_shadow_stack_next
            vm.linux_shadow_stack_next -= 0x01000000
            if vm.linux_shadow_stack_next <= vm.heap_next:
                raise VMError("MiniMachine Linux shadow task stacks exhausted")
            shadow_stacks[next_task] = shadow_top
            print(
                "BOOT_EXEC_TASK_SHADOW_STACK "
                f"task=0x{next_task:x} guest_sp=0x{fresh_sp:x} "
                f"p3_stack_top=0x{shadow_top:x}",
                flush=True,
            )

        vm.enter_function(
            "minimachine_ret_from_fork",
            (prev, start_fn, start_arg),
            stack_top=shadow_top,
            result_count=0,
        )
        return HOST_CONTROL_TRANSFER

    if service == 3:
        if len(args) != 2:
            raise VMError(
                "Linux user-mode handoff ecall expects service,pt_regs; "
                f"got {len(args)} args"
            )
        _, regs = args
        pc = vm.memory.read(regs + 0, 64)
        user_sp = vm.memory.read(regs + 8, 64)
        status = vm.memory.read(regs + 80, 64)
        if not (status & 1):
            raise VMError(
                "Linux user-mode handoff received non-user pt_regs: "
                f"regs=0x{regs:x} status=0x{status:x}"
            )

        header = bytes(vm.memory.read(pc + i, 8) for i in range(12))
        if header[:4] != b"MMP3":
            raise VMError(
                "MiniMachine user entry is not an MMP3 payload: "
                f"pc=0x{pc:x} magic={header[:4]!r}"
            )
        size = int.from_bytes(header[8:12], "big")
        if size > 16 * 1024 * 1024:
            raise VMError(f"MiniMachine user payload too large: {size}")
        payload = bytes(vm.memory.read(pc + i, 8) for i in range(12 + size))
        reference_path = os.environ.get("MINIMACHINE_USER_IMAGE_REFERENCE")
        if reference_path:
            reference_blob = Path(reference_path).read_bytes()
            if len(reference_blob) < 64:
                raise VMError(
                    "MiniMachine userspace reference image is truncated: "
                    f"path={reference_path} bytes={len(reference_blob)}"
                )
            reference_data_start = int.from_bytes(
                reference_blob[12:16], "big"
            )
            reference_payload = reference_blob[64:reference_data_start]
            if len(reference_payload) < 12 or reference_payload[:4] != b"MMP3":
                raise VMError(
                    "MiniMachine userspace reference payload is invalid: "
                    f"path={reference_path}"
                )
            reference_logical_size = (
                12 + int.from_bytes(reference_payload[8:12], "big")
            )
            reference_payload = reference_payload[:reference_logical_size]
            guest_hash = hashlib.sha256(payload).hexdigest()
            reference_hash = hashlib.sha256(reference_payload).hexdigest()
            mismatch = next(
                (
                    index
                    for index, (guest_byte, reference_byte) in enumerate(
                        zip(payload, reference_payload)
                    )
                    if guest_byte != reference_byte
                ),
                None,
            )
            if mismatch is None and len(payload) != len(reference_payload):
                mismatch = min(len(payload), len(reference_payload))
            if mismatch is None:
                guest_window = reference_window = "-"
            else:
                start = max(0, mismatch - 16)
                end = mismatch + 17
                guest_window = payload[start:end].hex()
                reference_window = reference_payload[start:end].hex()
            print(
                "BOOT_EXEC_USER_PAYLOAD_REFERENCE "
                f"guest_bytes={len(payload)} reference_bytes={len(reference_payload)} "
                f"guest_sha256={guest_hash} reference_sha256={reference_hash} "
                f"mismatch={mismatch if mismatch is not None else -1} "
                f"guest_window={guest_window} reference_window={reference_window}",
                flush=True,
            )
            if mismatch is not None:
                raise VMError(
                    "MiniMachine userspace payload differs from reference image: "
                    f"path={reference_path} offset={mismatch}"
                )
        try:
            user_image = unpack_user_image(payload)
        except UserImageError as exc:
            raise VMError(f"invalid MiniMachine user payload: {exc}") from exc

        functions = list(user_image.functions)
        entry_name = user_image.entry

        decoded_xxreadtoken = next(
            (
                function
                for function in functions
                if function.name == "__mm_user_xxreadtoken"
            ),
            None,
        )
        if decoded_xxreadtoken is not None:
            decoded_block19 = next(
                (
                    block
                    for block in decoded_xxreadtoken.blocks
                    if block.label == "19"
                ),
                None,
            )
            if (
                decoded_block19 is not None
                and len(decoded_block19.instructions) > 3
            ):
                decoded_inst = decoded_block19.instructions[3]
                decoded_width = getattr(
                    getattr(decoded_inst, "width", None), "value", None
                )
                print(
                    "BOOT_EXEC_USER_P3_DECODE "
                    f"function=__mm_user_xxreadtoken block=19 ip=3 "
                    f"type={type(decoded_inst).__name__} width={decoded_width}",
                    flush=True,
                )

        entry_argv: tuple[int, ...] = ()
        user_envp = 0
        if user_image.entry_args == "linux-main":
            argc = vm.memory.read(user_sp, 64)
            if argc > 4096:
                raise VMError(
                    f"MiniMachine userspace argc is unreasonable: {argc}"
                )
            argv = (user_sp + 8) & ((1 << 64) - 1)
            user_envp = (argv + (argc + 1) * 8) & ((1 << 64) - 1)
            entry_argv = (argc, argv, user_envp)
            vm.user_envp = user_envp
            argv0 = vm.memory.read(argv, 64) if argc else 0
            print(
                "BOOT_EXEC_USER_ARGS "
                f"argc={argc} argv=0x{argv:x} envp=0x{user_envp:x} "
                f"argv0=0x{argv0:x}",
                flush=True,
            )

        # Dynamic P3 userspace descriptors/data are machine representation
        # for this exec image.  Do not append them after the kernel Program data:
        # Linux owns the RAM immediately following the linked image and may
        # already have slab/page state there.  Place the dynamic P3 segment in
        # the task's own post-payload/pre-stack gap instead.
        user_payload_end = (pc + 12 + size + 15) & ~15
        kernel_program_data_end = vm.program._next_data
        if user_payload_end < kernel_program_data_end:
            raise VMError(
                "MiniMachine userspace P3 data arena overlaps kernel Program "
                f"data: user_base=0x{user_payload_end:x} "
                f"kernel_end=0x{kernel_program_data_end:x}"
            )
        if user_payload_end >= user_sp:
            raise VMError(
                "MiniMachine userspace P3 data arena has no stack gap: "
                f"user_base=0x{user_payload_end:x} user_sp=0x{user_sp:x}"
            )
        vm.program._next_data = user_payload_end
        print(
            "BOOT_EXEC_USER_DATA_ARENA "
            f"kernel_end=0x{kernel_program_data_end:x} "
            f"base=0x{user_payload_end:x} limit=0x{user_sp:x} "
            f"capacity={user_sp - user_payload_end}",
            flush=True,
        )

        install_user_external_surface(vm, user_image, user_envp)
        trace_user_external_descriptor(vm, "getcwd", "external-surface")

        registered_user_helpers = 0
        for symbol in user_image.runtime_helpers:
            if symbol in vm.program.symbol_addresses:
                continue
            callback = helper_callback(symbol)
            if callback is None:
                raise VMError(
                    "MiniMachine userspace image requires unsupported "
                    f"runtime helper {symbol}"
                )
            vm.program.register_service(symbol, callback)
            registered_user_helpers += 1
        if user_image.runtime_helpers:
            print(
                "BOOT_EXEC_USER_RUNTIME "
                f"required={len(user_image.runtime_helpers)} "
                f"registered={registered_user_helpers}",
                flush=True,
            )

        # Preserve the collision-safe legacy behavior for single-function
        # probes. Multi-function programs currently require a clean user
        # symbol namespace so internal function descriptors stay coherent.
        if len(functions) == 1:
            function = functions[0]
            base_name = function.name
            suffix = 0
            while (
                function.name in vm.program.functions
                or function.name in vm.program.symbol_addresses
            ):
                suffix += 1
                function.name = f"__mm_user_{base_name}_{suffix}"
            entry_name = function.name
        else:
            collisions = sorted(
                function.name
                for function in functions
                if (
                    function.name in vm.program.functions
                    or function.name in vm.program.symbol_addresses
                )
            )
            if collisions:
                raise VMError(
                    "MiniMachine multi-function userspace symbol collision: "
                    + ",".join(collisions[:8])
                )

        for function in functions:
            verify_p3(function)
        for function in functions:
            vm.program.add_function(function)
        trace_user_external_descriptor(vm, "getcwd", "functions-added")

        parser_needles = (
            "cmdloop",
            "parsecmd",
            "readtoken",
            "xxreadtoken",
            "pgetc",
            "preadbuffer",
            "pungetc",
        )
        parser_codes = {}
        for parser_name, linked_parser in vm.program.functions.items():
            lowered_name = parser_name.lower()
            if not any(needle in lowered_name for needle in parser_needles):
                continue
            if not linked_parser.function.blocks:
                continue
            entry_block = linked_parser.function.blocks[0].label
            entry_code = vm.program.block_code.get((parser_name, entry_block))
            if entry_code is not None:
                parser_codes[entry_code] = parser_name
        if parser_codes:
            vm.trace_user_parser_codes = parser_codes
            existing_watches = tuple(getattr(vm, "_watch_codes", ()))
            vm.set_watch_codes(existing_watches + tuple(parser_codes))
            parser_items = ",".join(
                f"{name}@0x{code:x}"
                for code, name in sorted(parser_codes.items(), key=lambda item: item[1])
            )
            print(
                "BOOT_EXEC_USER_PARSER_WATCHES "
                f"count={len(parser_codes)} functions={parser_items}",
                flush=True,
            )
            pgetc_watch = next(
                (
                    code
                    for code, name in parser_codes.items()
                    if name == "__mm_user_pgetc"
                ),
                None,
            )
            if pgetc_watch is not None:
                vm.trace_user_pgetc_watch_code = pgetc_watch
                vm.trace_user_pgetc_unget_armed = False
                print(
                    "BOOT_EXEC_USER_PGETC_UNGET_WATCH "
                    f"code=0x{pgetc_watch:x}",
                    flush=True,
                )

        setpwd_watch = vm.program.block_code.get(
            ("__mm_user_setpwd", "25")
        )
        if setpwd_watch is not None and hasattr(vm, "set_watch_codes"):
            if (
                not getattr(vm, "trace_user_parser_codes", None)
                and not getattr(vm, "trace_single_step_watch_code", 0)
            ):
                vm.trace_single_step_watch_code = setpwd_watch
                vm.trace_single_step_count = 400
            existing_watches = tuple(getattr(vm, "_watch_codes", ()))
            vm.set_watch_codes(existing_watches + (setpwd_watch,))
            print(
                "BOOT_EXEC_USER_SETPWD_WATCH "
                f"code=0x{setpwd_watch:x} count=400",
                flush=True,
            )

        if user_image.image is not None:
            try:
                install_module_image(vm.program, user_image.image)
            except (ImageError, VMError) as exc:
                raise VMError(
                    f"cannot install MiniMachine userspace data image: {exc}"
                ) from exc
            print(
                "BOOT_EXEC_USER_IMAGE_DATA "
                f"objects={len(user_image.image.objects)} "
                f"bytes={user_image.image.byte_size} "
                f"relocs={user_image.image.relocation_count}",
                flush=True,
            )
        trace_user_external_descriptor(vm, "getcwd", "module-image")
        if vm.program._next_data >= user_sp:
            raise VMError(
                "MiniMachine userspace P3 data arena exhausted: "
                f"end=0x{vm.program._next_data:x} user_sp=0x{user_sp:x}"
            )
        print(
            "BOOT_EXEC_USER_DATA_ARENA_USED "
            f"base=0x{user_payload_end:x} end=0x{vm.program._next_data:x} "
            f"bytes={vm.program._next_data - user_payload_end} "
            f"remaining={user_sp - vm.program._next_data}",
            flush=True,
        )

        print(
            "BOOT_EXEC_USER_HANDOFF "
            f"regs=0x{regs:x} pc=0x{pc:x} user_sp=0x{user_sp:x} "
            f"function={entry_name} functions={len(functions)} "
            f"entry_args={user_image.entry_args} "
            f"payload_bytes={12 + size}",
            flush=True,
        )
        vm.enter_function(
            entry_name,
            entry_argv,
            stack_top=user_sp,
            result_count=0,
        )
        if getattr(vm, "stop_after_user_handoff", False):
            vm.halted = True
            print(
                "BOOT_EXEC_USER_HANDOFF_STOP "
                f"steps={vm.steps} function={entry_name}",
                flush=True,
            )
        return HOST_CONTROL_TRANSFER

    raise VMError(f"unsupported MiniMachine Linux ecall service: {service}")



def user_syscall(vm, args: tuple[int, ...]):
    """Semantic userspace trap into the MiniMachine Linux syscall entry."""
    if len(args) != 7:
        raise VMError(
            "MiniMachine user syscall expects nr,arg0..arg5; "
            f"got {len(args)} arguments"
        )

    nr, *argv = args
    fallback = {
        17: ("__se_sys_getcwd", 2),
        25: ("__se_sys_fcntl", 3),
        49: ("__se_sys_chdir", 1),
        56: ("__se_sys_openat", 4),
        57: ("__se_sys_close", 1),
        63: ("__se_sys_read", 3),
        64: ("__se_sys_write", 3),
        93: ("__se_sys_exit", 1),
        94: ("__se_sys_exit_group", 1),
        153: ("__se_sys_times", 1),
        160: ("__se_sys_newuname", 1),
        166: ("__se_sys_umask", 1),
        169: ("__se_sys_gettimeofday", 2),
        172: ("sys_getpid", 0),
        173: ("sys_getppid", 0),
        174: ("sys_getuid", 0),
        175: ("sys_geteuid", 0),
        176: ("sys_getgid", 0),
        177: ("sys_getegid", 0),
        221: ("__se_sys_execve", 3),
    }

    result = None
    read_watch_previous = None
    if nr == 63:
        descriptor = vm.program.symbol_addresses.get("memcpy")
        linked_memcpy = vm.program.functions.get("memcpy")
        p3_entry = 0
        if linked_memcpy is not None and linked_memcpy.function.blocks:
            p3_entry = vm.program.block_code.get(
                ("memcpy", linked_memcpy.function.blocks[0].label),
                0,
            )
        if descriptor is not None:
            live_entry = vm.memory.read(descriptor, 64)
            initial_entry = vm.program.initial_memory.read(descriptor, 64)
            host_symbol = vm.program.host_code.get(live_entry, "<none>")
            print(
                "BOOT_EXEC_USER_READ_MEMCPY_DESCRIPTOR "
                f"descriptor=0x{descriptor:x} "
                f"live_entry=0x{live_entry:x} "
                f"live_frame={vm.memory.read(descriptor + 8, 64)} "
                f"initial_entry=0x{initial_entry:x} "
                f"initial_frame={vm.program.initial_memory.read(descriptor + 8, 64)} "
                f"p3_entry=0x{p3_entry:x} "
                f"host={host_symbol}",
                flush=True,
            )
        if (
            p3_entry
            and hasattr(vm, "set_watch_codes")
        ):
            read_watch_previous = tuple(
                getattr(vm, "_watch_codes", ())
            )
            vm.trace_user_read_memcpy_code = p3_entry
            vm.set_watch_codes(read_watch_previous + (p3_entry,))

    try:
        if "minimachine_user_syscall" in vm.program.functions:
            result, = _call_linux_function_preserving_control(
                vm,
                "minimachine_user_syscall",
                tuple(args),
                result_count=1,
                max_extra_steps=8_000_000,
            )
    finally:
        if read_watch_previous is not None:
            vm.set_watch_codes(read_watch_previous)
            vm.trace_user_read_memcpy_code = None

    signed_result = (
        result - (1 << 64)
        if result is not None and result & (1 << 63)
        else result
    )
    if result is None or signed_result == -38:
        spec = fallback.get(nr)
        if spec is None:
            if result is None:
                raise VMError(
                    f"MiniMachine userspace syscall {nr} has no semantic dispatch"
                )
        else:
            target, argc = spec
            if target in vm.program.functions:
                result, = _call_linux_function_preserving_control(
                    vm,
                    target,
                    tuple(argv[:argc]),
                    result_count=1,
                    max_extra_steps=8_000_000,
                )
                print(
                    "BOOT_EXEC_USER_SYSCALL_FALLBACK "
                    f"nr={nr} target={target}",
                    flush=True,
                )
            elif result is None:
                raise VMError(
                    f"MiniMachine Linux image is missing syscall wrapper {target}"
                )

    if result is None:
        result = ((1 << 64) - 38) & ((1 << 64) - 1)
    if nr == 63 and argv:
        user_ptr = argv[1]
        preview_len = min(32, max(0, int(result)))
        preview = bytes(
            vm.memory.read(user_ptr + i, 8)
            for i in range(preview_len)
        )
        print(
            "BOOT_EXEC_USER_READ_BUFFER "
            f"ptr=0x{user_ptr:x} result={int(result)} "
            f"data={preview.hex()}",
            flush=True,
        )
        evalskip_symbol = vm.program.symbol_addresses.get(
            "__mm_user_evalskip"
        )
        misc_symbol = vm.program.symbol_addresses.get(
            "__mm_user_ash_ptr_to_globals_misc"
        )
        evalskip = (
            vm.memory.read(evalskip_symbol, 32)
            if evalskip_symbol is not None else -1
        )
        misc = (
            vm.memory.read(misc_symbol, 64)
            if misc_symbol is not None else 0
        )
        print(
            "BOOT_EXEC_USER_ASH_CONTROL_READ "
            f"evalskip={evalskip} "
            f"nflag={vm.memory.read(misc + 98, 8) if misc else -1} "
            f"sflag={vm.memory.read(misc + 99, 8) if misc else -1} "
            f"misc=0x{misc:x}",
            flush=True,
        )

    count = int(getattr(vm, "user_syscall_count", 0)) + 1
    vm.user_syscall_count = count
    if count <= 64:
        signed = result - (1 << 64) if result & (1 << 63) else result
        print(
            "BOOT_EXEC_USER_SYSCALL "
            f"seq={count} nr={nr} "
            f"args={','.join(f'0x{x:x}' for x in argv)} "
            f"result={signed}",
            flush=True,
        )
    return result


def referenced_slots(value):
    seen: set[int] = set()

    def walk(obj):
        identity = id(obj)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(obj, muir.Slot):
            yield obj.name
            return
        if is_dataclass(obj):
            for field in fields(obj):
                yield from walk(getattr(obj, field.name))
            return
        if isinstance(obj, (tuple, list, set, frozenset)):
            for item in obj:
                yield from walk(item)
            return
        if isinstance(obj, dict):
            for key, item in obj.items():
                yield from walk(key)
                yield from walk(item)

    yield from walk(value)


def dump_instruction_slots(vm, inst) -> None:
    if inst is None or vm.current_function is None:
        return
    linked = vm.program.functions.get(vm.current_function)
    if linked is None:
        return

    for name in sorted(set(referenced_slots(inst))):
        offset = linked.slot_offsets.get(name)
        if offset is None:
            continue
        cell = (vm.sp + offset) & ((1 << 64) - 1)
        value = vm.memory.read(cell, 64)
        preview = bytes(vm.memory.read(value + i, 8) for i in range(16))
        print(
            "BOOT_EXEC_SLOT "
            f"name={name} cell=0x{cell:x} value=0x{value:x} "
            f"mem16={preview.hex()}",
            flush=True,
        )


def dump_current_call_frame(vm) -> None:
    sp = vm.sp
    header = (
        ("caller_sp", CALLER_SP),
        ("ret_pc", RET_PC),
        ("entry", 16),
        ("frame_size", 24),
        ("result_ptr", RESULT_PTR),
        ("result_count", RESULT_COUNT),
        ("resume_pc", 48),
        ("arg_count", 56),
    )
    print(
        "BOOT_EXEC_CALL_FRAME "
        + " ".join(
            f"{name}=0x{vm.memory.read(sp + off, 64):x}"
            for name, off in header
        ),
        flush=True,
    )

    caller_sp = vm.memory.read(sp + CALLER_SP, 64)
    if not caller_sp or vm.current_function is None:
        return
    linked = vm.program.functions.get(vm.current_function)
    if linked is None:
        return

    interesting = []
    for name, off in sorted(linked.slot_offsets.items(), key=lambda item: item[1]):
        value = vm.memory.read(caller_sp + off, 64)
        if (
            name.startswith("__abi_")
            or value in vm.program.symbol_addresses.values()
            or value >= 0x10000
        ):
            interesting.append((name, off, value))
    for name, off, value in interesting[-80:]:
        print(
            "BOOT_EXEC_CALLER_SLOT "
            f"name={name} offset={off} address=0x{caller_sp + off:x} "
            f"value=0x{value:x}",
            flush=True,
        )

    entry = vm.memory.read(sp + 16, 64)
    descriptor_candidates = [
        (name, value)
        for name, _off, value in interesting
        if value and vm.memory.read(value, 64) == entry
    ]
    for name, descriptor in descriptor_candidates[-20:]:
        print(
            "BOOT_EXEC_DESCRIPTOR "
            f"slot={name} address=0x{descriptor:x} "
            f"entry=0x{vm.memory.read(descriptor, 64):x} "
            f"frame=0x{vm.memory.read(descriptor + 8, 64):x}",
            flush=True,
        )


def dump_scheduler_indirect_call(vm) -> None:
    if vm.current_function != "dequeue_task":
        return

    frame_size = vm.memory.read(vm.sp + 24, 64)
    argc = vm.memory.read(vm.sp + 56, 64)
    if argc < 2:
        return

    arg_base = vm.sp + frame_size
    rq = vm.memory.read(arg_base, 64)
    task = vm.memory.read(arg_base + 8, 64)
    flags = vm.memory.read(arg_base + 16, 64) if argc >= 3 else 0

    sched_off = getattr(vm, "linux_task_sched_class_offset", None)
    print(
        "BOOT_EXEC_SCHED_CALL "
        f"rq=0x{rq:x} task=0x{task:x} flags=0x{flags:x} "
        f"sched_class_offset={sched_off}",
        flush=True,
    )
    if sched_off is None or not task:
        return

    sched_class = vm.memory.read(task + sched_off, 64)
    dequeue_desc = vm.memory.read(sched_class + 8, 64) if sched_class else 0
    entry = vm.memory.read(dequeue_desc, 64) if dequeue_desc else 0
    desc_frame = vm.memory.read(dequeue_desc + 8, 64) if dequeue_desc else 0
    print(
        "BOOT_EXEC_SCHED_DESCRIPTOR "
        f"sched_class=0x{sched_class:x} dequeue_desc=0x{dequeue_desc:x} "
        f"entry=0x{entry:x} frame_size=0x{desc_frame:x}",
        flush=True,
    )

    for symbol in (
        "dequeue_task_idle",
        "dequeue_task_fair",
        "dequeue_task_rt",
        "dequeue_task_dl",
    ):
        desc = vm.program.symbol_addresses.get(symbol)
        if desc is None:
            continue
        print(
            "BOOT_EXEC_EXPECTED_DESCRIPTOR "
            f"symbol={symbol} address=0x{desc:x} "
            f"entry=0x{vm.memory.read(desc, 64):x} "
            f"frame_size=0x{vm.memory.read(desc + 8, 64):x}",
            flush=True,
        )


def dump_linux_memory_state(vm) -> None:
    def word(name: str, index: int = 0):
        address = vm.program.symbol_addresses.get(name)
        if address is None:
            print(f"BOOT_EXEC_MEM symbol={name} missing=1", flush=True)
            return None
        value = vm.memory.read(address + index * 8, 64)
        print(
            f"BOOT_EXEC_MEM symbol={name} index={index} "
            f"address=0x{address + index * 8:x} value={value} hex=0x{value:x}",
            flush=True,
        )
        return value

    for symbol in (
        "memory_start",
        "memory_end",
        "min_low_pfn",
        "max_low_pfn",
        "max_pfn",
        "nr_kernel_pages",
        "nr_all_pages",
    ):
        word(symbol)

    for index in range(2):
        word("arch_zone_lowest_possible_pfn", index)
        word("arch_zone_highest_possible_pfn", index)

    memblock = vm.program.symbol_addresses.get("memblock")
    if memblock is None:
        print("BOOT_EXEC_MEM symbol=memblock missing=1", flush=True)
        return

    # Linux 6.6 struct memblock on this target:
    #   u8 bottom_up; u64 current_limit;
    #   memblock_type memory; memblock_type reserved;
    # and memblock_type is {cnt,max,total_size,regions,name}.
    memory_cnt = vm.memory.read(memblock + 16, 64)
    memory_max = vm.memory.read(memblock + 24, 64)
    memory_total = vm.memory.read(memblock + 32, 64)
    memory_regions = vm.memory.read(memblock + 40, 64)
    reserved_cnt = vm.memory.read(memblock + 56, 64)
    reserved_total = vm.memory.read(memblock + 72, 64)
    reserved_regions = vm.memory.read(memblock + 80, 64)
    print(
        "BOOT_EXEC_MEMBLOCK "
        f"address=0x{memblock:x} memory_cnt={memory_cnt} "
        f"memory_max={memory_max} memory_total={memory_total} "
        f"memory_regions=0x{memory_regions:x} reserved_cnt={reserved_cnt} "
        f"reserved_total={reserved_total} reserved_regions=0x{reserved_regions:x}",
        flush=True,
    )

    for kind, count, regions in (
        ("memory", memory_cnt, memory_regions),
        ("reserved", reserved_cnt, reserved_regions),
    ):
        for index in range(min(count, 4)):
            region = regions + index * 24
            base = vm.memory.read(region + 0, 64)
            size = vm.memory.read(region + 8, 64)
            flags = vm.memory.read(region + 16, 32)
            print(
                "BOOT_EXEC_REGION "
                f"kind={kind} index={index} address=0x{region:x} "
                f"base=0x{base:x} size=0x{size:x} flags=0x{flags:x}",
                flush=True,
            )


def probe_linux_memory_helpers(vm) -> None:
    def run(name: str, args: tuple[int, ...], result_count: int = 1):
        if name not in vm.program.functions:
            print(f"BOOT_EXEC_PROBE function={name} missing=1", flush=True)
            return None
        try:
            result = vm.run_function(
                name,
                args,
                result_count=result_count,
                max_steps=500_000,
            )
        except VMError as exc:
            print(
                f"BOOT_EXEC_PROBE function={name} error={exc}",
                flush=True,
            )
            return None
        print(
            f"BOOT_EXEC_PROBE function={name} result={result}",
            flush=True,
        )
        return result

    memblock = vm.program.symbol_addresses.get("memblock")
    if memblock is not None:
        regions = vm.memory.read(memblock + 40, 64)
        run("memblock_get_region_node", (regions,))
        run("numa_valid_node", (0,))

        index = vm.alloc_bytes(8, align=8)
        direct_start = vm.alloc_bytes(8, align=8)
        direct_end = vm.alloc_bytes(8, align=8)
        direct_nid = vm.alloc_bytes(8, align=8)
        vm.memory.write(index, 32, 0xFFFFFFFF)
        vm.memory.write(direct_start, 64, 0)
        vm.memory.write(direct_end, 64, 0)
        vm.memory.write(direct_nid, 32, 0xFFFFFFFF)
        run(
            "__next_mem_pfn_range",
            (index, 0, direct_start, direct_end, direct_nid),
            result_count=0,
        )
        print(
            "BOOT_EXEC_PROBE_NEXT_PFN "
            f"index={vm.memory.read(index, 32)} "
            f"start={vm.memory.read(direct_start, 64)} "
            f"end={vm.memory.read(direct_end, 64)} "
            f"nid={vm.memory.read(direct_nid, 32)}",
            flush=True,
        )

    start = vm.alloc_bytes(8, align=8)
    end = vm.alloc_bytes(8, align=8)
    run("get_pfn_range_for_nid", (0, start, end), result_count=0)
    start_pfn = vm.memory.read(start, 64)
    end_pfn = vm.memory.read(end, 64)
    print(
        "BOOT_EXEC_PROBE_PFN "
        f"start={start_pfn} end={end_pfn}",
        flush=True,
    )

    if end_pfn >= start_pfn:
        run(
            "__absent_pages_in_range",
            (0, start_pfn, end_pfn),
        )

        zone_start = vm.alloc_bytes(8, align=8)
        zone_end = vm.alloc_bytes(8, align=8)
        run(
            "zone_spanned_pages_in_node",
            (0, 0, start_pfn, end_pfn, zone_start, zone_end),
        )
        print(
            "BOOT_EXEC_PROBE_ZONE "
            f"start={vm.memory.read(zone_start, 64)} "
            f"end={vm.memory.read(zone_end, 64)}",
            flush=True,
        )


def _call_linux_function_preserving_control(
    vm,
    name: str,
    args: tuple[int, ...],
    *,
    result_count: int,
    max_extra_steps: int = 2_000_000,
) -> tuple[int, ...]:
    if name not in vm.program.functions:
        raise VMError(f"rootfs injection missing Linux function: {name}")

    current_addr = vm.program.symbol_addresses.get(
        "minimachine_current_task"
    )
    saved_current = (
        vm.memory.read(current_addr, 64)
        if current_addr is not None
        else None
    )
    saved_contexts = dict(getattr(vm, "linux_task_contexts", {}))
    saved_shadow_stacks = dict(
        getattr(vm, "linux_task_shadow_stacks", {})
    )
    saved_shadow_next = getattr(vm, "linux_shadow_stack_next", None)
    saved = (
        vm.sp,
        vm.current_function,
        vm.current_block,
        vm.ip,
        vm.halted,
        vm.steps,
    )
    linked = vm.program.functions[name]
    temp_stack_top = 0x0F00_0000
    result_words = max(1, result_count)
    total = linked.frame_size + len(args) * 8 + result_words * 8
    callee_sp = temp_stack_top - total
    result_base = callee_sp + linked.frame_size + len(args) * 8

    try:
        vm.enter_function(
            name,
            args,
            stack_top=temp_stack_top,
            result_count=result_count,
        )
        vm.run(max_steps=saved[5] + max_extra_steps)
        return tuple(
            vm.memory.read(result_base + i * 8, 64)
            for i in range(result_count)
        )
    except VMError as exc:
        current_inst = None
        linked_now = vm.program.functions.get(vm.current_function)
        if (
            linked_now is not None
            and vm.current_block is not None
            and 0 <= vm.ip
        ):
            block_now = next(
                (
                    block
                    for block in linked_now.function.blocks
                    if block.label == vm.current_block
                ),
                None,
            )
            if block_now is not None and vm.ip < len(block_now.instructions):
                current_inst = block_now.instructions[vm.ip]
        print(
            "BOOT_EXEC_PRESERVED_CALL_ERROR "
            f"target={name} function={vm.current_function} "
            f"block={vm.current_block} ip={vm.ip} sp=0x{vm.sp:x} "
            f"inst={current_inst!r} error={exc}",
            flush=True,
        )
        if vm.current_function == "setup_object" and linked_now is not None:
            interesting = (
                "0", "1", "3", "4", "10", "12", "13", "14", "15",
                "17", "18", "21", "22", "23", "25", "26",
            )
            values = {}
            for slot_name in interesting:
                off = linked_now.slot_offsets.get(slot_name)
                if off is not None:
                    values[slot_name] = vm.memory.read(vm.sp + off, 64)
            cache_ptr = values.get("23") or values.get("10") or 0
            ctor_now = (
                vm.memory.read(cache_ptr + 64, 64)
                if cache_ptr else 0
            )
            ctor_block = vm.program.code_block.get(ctor_now)
            ctor_name = ctor_block[0] if ctor_block is not None else None
            ctor_descriptor = (
                vm.program.symbol_addresses.get(ctor_name, 0)
                if ctor_name is not None else 0
            )
            print(
                "BOOT_EXEC_SETUP_OBJECT_CTOR "
                f"value=0x{ctor_now:x} "
                f"code_block={ctor_block!r} "
                f"code_mem_entry=0x{vm.memory.read(ctor_now, 64):x} "
                f"code_mem_frame=0x{vm.memory.read(ctor_now + 8, 64):x} "
                f"descriptor=0x{ctor_descriptor:x} "
                f"descriptor_entry=0x{vm.memory.read(ctor_descriptor, 64):x} "
                f"descriptor_frame=0x{vm.memory.read(ctor_descriptor + 8, 64):x}",
                flush=True,
            )
            print(
                "BOOT_EXEC_SETUP_OBJECT_STATE "
                + " ".join(
                    f"s{slot}=0x{value:x}"
                    for slot, value in sorted(values.items(), key=lambda item: int(item[0]))
                )
                + f" cache=0x{cache_ptr:x} cache_ctor=0x{ctor_now:x} "
                + f"ctor_code={vm.program.code_block.get(ctor_now)!r} "
                + f"ctor_descriptor={next((hex(address) for symbol, address in vm.program.symbol_addresses.items() if symbol in vm.program.functions and vm.program.block_code.get((symbol, vm.program.functions[symbol].function.blocks[0].label)) == ctor_now), None)}",
                flush=True,
            )
        raise
    finally:
        if current_addr is not None and saved_current is not None:
            observed_current = vm.memory.read(current_addr, 64)
            if observed_current != saved_current:
                print(
                    "BOOT_EXEC_PRESERVE_CURRENT_RESTORE "
                    f"function={name} "
                    f"before=0x{saved_current:x} "
                    f"after=0x{observed_current:x}",
                    flush=True,
                )
                vm.memory.write(current_addr, 64, saved_current)
        vm.linux_task_contexts = saved_contexts
        vm.linux_task_shadow_stacks = saved_shadow_stacks
        if saved_shadow_next is not None:
            vm.linux_shadow_stack_next = saved_shadow_next
        (
            vm.sp,
            vm.current_function,
            vm.current_block,
            vm.ip,
            vm.halted,
            vm.steps,
        ) = saved


def skip_prepare_namespace_after_hot_init(vm) -> None:
    if vm.current_function != "prepare_namespace" or vm.ip != 0:
        raise VMError(
            "prepare_namespace hot skip requires an entry checkpoint; "
            f"got function={vm.current_function} block={vm.current_block} ip={vm.ip}"
        )

    command = vm.program.symbol_addresses.get("ramdisk_execute_command")
    if command is None:
        raise VMError("ramdisk_execute_command symbol is missing")
    initial_command = vm.program.initial_memory.read(command, 64)
    if not initial_command:
        raise VMError("ramdisk_execute_command has no initial /init pointer")
    vm.memory.write(command, 64, initial_command)

    result_count = vm.memory.read(vm.sp + RESULT_COUNT, 64)
    if result_count != 0:
        raise VMError(
            f"prepare_namespace caller expects {result_count} results"
        )
    caller_sp = vm.memory.read(vm.sp + CALLER_SP, 64)
    ret_pc = vm.memory.read(vm.sp + RET_PC, 64)
    vm.sp = caller_sp
    vm.halted = False
    vm._set_code(ret_pc)
    print(
        "BOOT_EXEC_PREPARE_NAMESPACE_SKIPPED "
        f"ramdisk_execute_command=0x{initial_command:x} ret_pc=0x{ret_pc:x}",
        flush=True,
    )


def probe_live_rootfs(vm) -> None:
    def guest_cstring(text: str) -> int:
        data = text.encode("utf-8") + b"\0"
        ptr = vm.alloc_bytes(len(data), align=8)
        for i, byte in enumerate(data):
            vm.memory.write(ptr + i, 8, byte)
        return ptr

    current_addr = vm.program.symbol_addresses.get("minimachine_current_task")
    current = vm.memory.read(current_addr, 64) if current_addr is not None else 0
    print(
        "BOOT_EXEC_ROOTFS_PROBE_CURRENT "
        f"symbol=0x{(current_addr or 0):x} task=0x{current:x}",
        flush=True,
    )

    for path in ("/", "/dev", "/dev/console", "/root", "/init"):
        ptr = guest_cstring(path)
        try:
            result, = _call_linux_function_preserving_control(
                vm,
                "init_eaccess",
                (ptr,),
                result_count=1,
            )
        except VMError as exc:
            print(
                f"BOOT_EXEC_ROOTFS_PROBE path={path} error={exc}",
                flush=True,
            )
            continue
        signed = result - (1 << 64) if result & (1 << 63) else result
        print(
            f"BOOT_EXEC_ROOTFS_PROBE path={path} result={signed}",
            flush=True,
        )

    tmp = guest_cstring("/mm-probe")
    try:
        result, = _call_linux_function_preserving_control(
            vm,
            "init_mkdir",
            (tmp, 0o755),
            result_count=1,
        )
        signed = result - (1 << 64) if result & (1 << 63) else result
        print(
            f"BOOT_EXEC_ROOTFS_PROBE mkdir=/mm-probe result={signed}",
            flush=True,
        )
    except VMError as exc:
        print(
            f"BOOT_EXEC_ROOTFS_PROBE mkdir=/mm-probe error={exc}",
            flush=True,
        )


def install_hot_filp_trace(vm) -> None:
    targets = {
        "filp_open",
        "file_open_name",
        "do_filp_open",
        "path_openat",
        "path_init",
        "link_path_walk",
        "open_last_lookups",
        "lookup_open",
        "lookup_fast",
        "d_lookup",
        "d_alloc_parallel",
        "mnt_want_write",
        "may_o_create",
        "do_open",
        "vfs_open",
    }
    traced_codes: dict[int, tuple[str, str]] = {}
    for code, pair in vm.program.code_block.items():
        if pair[0] in targets:
            traced_codes[code] = pair

    original_set_code = vm._set_code
    original_enter_function = vm.enter_function
    trace_count = 0
    trace_limit = 1200

    def emit(function: str, block: str) -> None:
        nonlocal trace_count
        if trace_count >= trace_limit:
            return
        trace_count += 1
        linked = vm.program.functions.get(function)
        frame_size = vm.memory.read(vm.sp + 24, 64) if linked is not None else 0
        argc = vm.memory.read(vm.sp + 56, 64) if linked is not None else 0
        args = []
        if linked is not None and argc <= 8:
            arg_base = vm.sp + frame_size
            args = [vm.memory.read(arg_base + i * 8, 64) for i in range(argc)]
        if function == "open_last_lookups" and block == "entry" and len(args) >= 3:
            op_ptr = args[2]
            fields = [vm.memory.read(op_ptr + i * 4, 32) for i in range(5)]
            print(
                "BOOT_EXEC_HOT_OPEN_FLAGS "
                f"ptr=0x{op_ptr:x} "
                f"f0=0x{fields[0]:x} f1=0x{fields[1]:x} "
                f"f2=0x{fields[2]:x} f3=0x{fields[3]:x} f4=0x{fields[4]:x}",
                flush=True,
            )
        print(
            "BOOT_EXEC_HOT_FILP_TRACE "
            f"seq={trace_count} steps={vm.steps} "
            f"function={function} block={block} sp=0x{vm.sp:x} "
            f"argc={argc} args={','.join(f'0x{x:x}' for x in args)}",
            flush=True,
        )

    def traced_set_code(code: int) -> None:
        original_set_code(code)
        pair = traced_codes.get(code)
        if pair is not None:
            emit(pair[0], pair[1])

    def traced_enter_function(name, args=(), *, stack_top, result_count=0):
        original_enter_function(
            name,
            args,
            stack_top=stack_top,
            result_count=result_count,
        )
        if name in targets:
            linked = vm.program.functions[name]
            emit(name, linked.function.blocks[0].label)

    vm._set_code = traced_set_code
    vm.enter_function = traced_enter_function
    and32 = vm.program.host_services.get("__mm_and_32")
    if and32 is not None:
        def traced_and32(inner_vm, args):
            result = and32(inner_vm, args)
            if len(args) == 2 and (args[1] & 0xFFFFFFFF) == 579:
                print(
                    "BOOT_EXEC_HOT_AND32 "
                    f"steps={inner_vm.steps} "
                    f"a=0x{args[0] & 0xffffffff:x} "
                    f"b=0x{args[1] & 0xffffffff:x} "
                    f"result=0x{int(result) & 0xffffffff:x}",
                    flush=True,
                )
            return result
        vm.program.host_services["__mm_and_32"] = traced_and32

    print(
        "BOOT_EXEC_HOT_FILP_TRACE_ARMED "
        f"functions={len(targets)} codes={len(traced_codes)} limit={trace_limit}",
        flush=True,
    )


def inject_live_root_init(
    vm,
    path: Path,
    *,
    guest_path_text: str = "/init",
) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VMError(
            f"cannot read injected rootfs image for {guest_path_text}: {exc}"
        ) from exc
    if not data:
        raise VMError(f"injected rootfs image for {guest_path_text} is empty")
    if not guest_path_text.startswith("/"):
        raise VMError("--inject-init-path must be absolute")

    guest_path = guest_path_text.encode("utf-8") + b"\0"
    path_ptr = vm.alloc_bytes(len(guest_path), align=8)
    for i, byte in enumerate(guest_path):
        vm.memory.write(path_ptr + i, 8, byte)

    data_ptr = vm.alloc_bytes(len(data), align=8)
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        started = time.perf_counter()
        bulk_write(data_ptr, data)
        print(
            "BOOT_EXEC_HOST_BULK_WRITE "
            f"bytes={len(data)} seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    else:
        for i, byte in enumerate(data):
            vm.memory.write(data_ptr + i, 8, byte)

    pos_ptr = vm.alloc_bytes(8, align=8)
    vm.memory.write(pos_ptr, 64, 0)

    # Linux asm-generic flags: O_WRONLY | O_CREAT | O_TRUNC.
    open_flags = 0x1 | 0x40 | 0x200
    file_ptr, = _call_linux_function_preserving_control(
        vm,
        "filp_open",
        (path_ptr, open_flags, 0o755),
        result_count=1,
        max_extra_steps=max(
            16_000_000,
            len(data) * 2,
        ),
    )
    print(
        "BOOT_EXEC_ROOTFS_OPEN "
        f"path={guest_path_text} file=0x{file_ptr:x} bytes={len(data)}",
        flush=True,
    )
    if file_ptr >= (1 << 64) - 4095:
        signed = file_ptr - (1 << 64)
        raise VMError(f"filp_open({guest_path_text}) failed: {signed}")

    try:
        written, = _call_linux_function_preserving_control(
            vm,
            "kernel_write",
            (file_ptr, data_ptr, len(data), pos_ptr),
            result_count=1,
            max_extra_steps=max(
                8_000_000,
                len(data) * 32,
            ),
        )
        if written != len(data):
            signed = written - (1 << 64) if written & (1 << 63) else written
            raise VMError(
                f"kernel_write({guest_path_text}) wrote {signed}, expected {len(data)}"
            )
    finally:
        if "__fput_sync" in vm.program.functions:
            _call_linux_function_preserving_control(
                vm,
                "__fput_sync",
                (file_ptr,),
                result_count=0,
            )
            print(
                "BOOT_EXEC_ROOTFS_CLOSE method=__fput_sync",
                flush=True,
            )
        else:
            _call_linux_function_preserving_control(
                vm,
                "fput",
                (file_ptr,),
                result_count=0,
            )
            if "flush_delayed_fput" in vm.program.functions:
                _call_linux_function_preserving_control(
                    vm,
                    "flush_delayed_fput",
                    (),
                    result_count=0,
                )
                print(
                    "BOOT_EXEC_ROOTFS_CLOSE "
                    "method=fput+flush_delayed_fput",
                    flush=True,
                )
            else:
                print(
                    "BOOT_EXEC_ROOTFS_CLOSE method=fput-only",
                    flush=True,
                )

    access, = _call_linux_function_preserving_control(
        vm,
        "init_eaccess",
        (path_ptr,),
        result_count=1,
    )
    if access != 0:
        signed = access - (1 << 64) if access & (1 << 63) else access
        raise VMError(
            f"init_eaccess({guest_path_text}) after injection failed: {signed}"
        )

    print(
        "BOOT_EXEC_ROOTFS_INJECTED "
        f"path={guest_path_text} bytes={len(data)} mode=0755",
        flush=True,
    )



def inject_live_root_initramfs(vm, path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VMError(f"cannot read injected initramfs: {exc}") from exc
    if not data:
        raise VMError("injected initramfs is empty")
    if "unpack_to_rootfs" not in vm.program.functions:
        raise VMError("Linux unpack_to_rootfs is missing from P3 program")

    data_ptr = vm.alloc_bytes(len(data), align=8)
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        started = time.perf_counter()
        bulk_write(data_ptr, data)
        print(
            "BOOT_EXEC_HOST_BULK_WRITE "
            f"bytes={len(data)} seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    else:
        for i, byte in enumerate(data):
            vm.memory.write(data_ptr + i, 8, byte)

    error_ptr, = _call_linux_function_preserving_control(
        vm,
        "unpack_to_rootfs",
        (data_ptr, len(data)),
        result_count=1,
        max_extra_steps=25_000_000,
    )
    if error_ptr:
        raw = bytearray()
        for i in range(256):
            byte = vm.memory.read(error_ptr + i, 8)
            if byte == 0:
                break
            raw.append(byte)
        message = raw.decode("utf-8", errors="replace")
        raise VMError(
            f"unpack_to_rootfs(initramfs) failed: ptr=0x{error_ptr:x} "
            f"message={message!r}"
        )

    guest_path = b"/init\0"
    path_ptr = vm.alloc_bytes(len(guest_path), align=8)
    for i, byte in enumerate(guest_path):
        vm.memory.write(path_ptr + i, 8, byte)
    access, = _call_linux_function_preserving_control(
        vm,
        "init_eaccess",
        (path_ptr,),
        result_count=1,
    )
    if access != 0:
        signed = access - (1 << 64) if access & (1 << 63) else access
        raise VMError(f"init_eaccess(/init) after initramfs unpack failed: {signed}")

    print(
        "BOOT_EXEC_ROOTFS_UNPACKED "
        f"path=/init archive_bytes={len(data)} via=unpack_to_rootfs",
        flush=True,
    )

def current_instruction(vm):
    if vm.current_function is None or vm.current_block is None:
        return None
    linked = vm.program.functions.get(vm.current_function)
    if linked is None:
        return None
    block = linked.block_map.get(vm.current_block)
    if block is None or vm.ip >= len(block.instructions):
        return None
    return block.instructions[vm.ip]


def main() -> int:
    runner_started = time.perf_counter()
    args = parse_args()
    llvm_text = args.input.read_text()
    linked_image_sha256 = image_fingerprint(llvm_text)
    linker_contract = LinkerContract.load(args.linker_contract)

    if args.program_cache_in is not None:
        try:
            cached = load_program_cache(
                args.program_cache_in,
                image_sha256=linked_image_sha256,
            )
        except ProgramCacheError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=program-cache error={exc}")
            return 1
        program = cached.program
        surface = cached.surface
        reasons = set(cached.reasons)
        blocked_functions = list(cached.blocked_functions)
        image = cached.image
        task_sched_class_offset = cached.task_sched_class_offset
        p3_function_count = cached.function_count
        print(
            "BOOT_EXEC_PROGRAM_CACHE_LOADED "
            f"path={args.program_cache_in} functions={p3_function_count}",
            flush=True,
        )
        print(
            "BOOT_EXEC_STAGE "
            f"stage=program-cache elapsed_s={time.perf_counter() - runner_started:.3f}",
            flush=True,
        )
    else:
        functions, _stats = legalize_module(llvm_text)
        image = parse_module_image(llvm_text)

        strict_source = []
        p3_functions = []
        blocked_functions: list[tuple[str, int]] = []
        reasons: set[str] = set()

        for function in functions:
            verify_muir(function)
            reasons.update(trap_reasons(function))
            expanded, _abi_stats = expand_function(function)
            escapes = arch_escapes(expanded)
            if escapes:
                blocked_functions.append((function.name, len(escapes)))
                continue
            lowered = lower_function(expanded)
            verify_p3(lowered)
            strict_source.append(function)
            p3_functions.append(lowered)

        program = Program(p3_functions)
        surface = collect_runtime_surface(strict_source)
        layout = DataLayout.from_module(llvm_text)
        task_info = layout.info("%struct.task_struct")
        task_sched_class_offset = (
            task_info.field_offsets[15]
            if task_info.field_offsets is not None
            and len(task_info.field_offsets) > 15
            else None
        )
        p3_function_count = len(p3_functions)

        if args.program_cache_out is not None:
            save_program_cache(
                ProgramCache(
                    image_sha256=linked_image_sha256,
                    program=program,
                    surface=surface,
                    reasons=frozenset(reasons),
                    blocked_functions=tuple(blocked_functions),
                    image=image,
                    task_sched_class_offset=task_sched_class_offset,
                ),
                args.program_cache_out,
            )
            print(
                "BOOT_EXEC_PROGRAM_CACHE_SAVED "
                f"path={args.program_cache_out} functions={p3_function_count}",
                flush=True,
            )

    image_sections = {
        obj.section for obj in image.objects if obj.section is not None
    }

    install_runtime(program, surface)
    accelerated_runtime = accelerate_direct_runtime(program)
    if accelerated_runtime:
        print(
            "BOOT_EXEC_FAST_RUNTIME symbols=" + ",".join(accelerated_runtime),
            flush=True,
        )
    register_traps(program, reasons)

    missing_helpers = tuple(getattr(program, "runtime_missing_helpers", ()))
    missing_systems = tuple(getattr(program, "runtime_missing_systems", ()))
    if missing_helpers or missing_systems:
        print(
            "BOOT_EXEC_BLOCKED "
            f"stage=runtime helpers={len(missing_helpers)} systems={len(missing_systems)}"
        )
        return 1

    # Native Linux adds kallsyms during final ELF linking. P3 has its own
    # final code-address domain, so synthesize the same generated globals
    # before installing the module image. This ensures semantic _end includes
    # the tables and Linux reserves their storage during setup_arch.
    try:
        installed_kallsyms = install_p3_kallsyms(
            program,
            external_data=image.external_data,
        )
    except VMError as exc:
        print(f"BOOT_EXEC_BLOCKED stage=kallsyms error={exc}")
        return 1
    if installed_kallsyms:
        print(
            "BOOT_EXEC_KALLSYMS "
            f"generated={len(installed_kallsyms)} "
            f"symbols={program.initial_memory.read(program.symbol_addresses['kallsyms_num_syms'], 32)}",
            flush=True,
        )

    initramfs_sha256 = None
    if args.initramfs is not None:
        try:
            initramfs_data = args.initramfs.read_bytes()
        except OSError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=initramfs error={exc}")
            return 1
        if not initramfs_data:
            print("BOOT_EXEC_BLOCKED stage=initramfs error=empty image")
            return 1
        initramfs_sha256 = hashlib.sha256(initramfs_data).hexdigest()
        try:
            initramfs_start = program.define_data_symbol(
                "__initramfs_start",
                initramfs_data,
                align=4,
            )
            initramfs_size_addr = program.define_data_symbol(
                "__initramfs_size",
                len(initramfs_data).to_bytes(8, "little"),
                align=8,
            )
        except VMError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=initramfs error={exc}")
            return 1
        print(
            "BOOT_EXEC_INITRAMFS "
            f"path={args.initramfs} bytes={len(initramfs_data)} "
            f"start=0x{initramfs_start:x} size_symbol=0x{initramfs_size_addr:x}",
            flush=True,
        )

    try:
        install_module_image(
            program,
            image,
            symbol_aliases=dict(linker_contract.aliases),
            linker_contract=linker_contract,
        )
    except (ImageError, VMError, ValueError) as exc:
        print(f"BOOT_EXEC_BLOCKED stage=image error={exc}")
        return 1

    if args.probe_initcall_table:
        reverse_symbols: dict[int, list[str]] = {}
        for name, address in program.symbol_addresses.items():
            reverse_symbols.setdefault(address, []).append(name)

        boundary_names = (
            "__initcall5_start",
            "__initcallrootfs_start",
            "__initcall6_start",
            "__initcall7_start",
            "__initcall_end",
        )
        for name in boundary_names:
            address = program.symbol_addresses.get(name)
            print(
                "BOOT_EXEC_INITCALL_BOUNDARY "
                f"name={name} "
                f"address={f'0x{address:x}' if address is not None else 'missing'}",
                flush=True,
            )

        start = program.symbol_addresses.get("__initcall5_start")
        end = program.symbol_addresses.get("__initcall6_start")
        if start is not None and end is not None:
            print(
                "BOOT_EXEC_INITCALL_RANGE "
                f"start=0x{start:x} end=0x{end:x} "
                f"bytes={end-start} entries={(end-start)//8}",
                flush=True,
            )
            for address in range(start, end, 8):
                value = program.initial_memory.read(address, 64)
                names = reverse_symbols.get(value, ())
                print(
                    "BOOT_EXEC_INITCALL_ENTRY "
                    f"address=0x{address:x} value=0x{value:x} "
                    f"symbols={'|'.join(sorted(names)[:6]) if names else '<unknown>'}",
                    flush=True,
                )

        levels = program.symbol_addresses.get("initcall_levels")
        if levels is not None:
            print(f"BOOT_EXEC_INITCALL_LEVELS address=0x{levels:x}", flush=True)
            for index in range(9):
                value = program.initial_memory.read(levels + index * 8, 64)
                names = reverse_symbols.get(value, ())
                print(
                    "BOOT_EXEC_INITCALL_LEVEL "
                    f"index={index} value=0x{value:x} "
                    f"symbols={'|'.join(sorted(names)[:6]) if names else '<unknown>'}",
                    flush=True,
                )
        else:
            print("BOOT_EXEC_INITCALL_LEVELS address=missing", flush=True)
        return 0

    if args.entry not in program.functions:
        print(f"BOOT_EXEC_BLOCKED stage=entry missing={args.entry}")
        return 1

    print(
        "BOOT_EXEC_START "
        f"entry={args.entry} functions={p3_function_count} "
        f"blocked_functions={len(blocked_functions)} "
        f"arch_escape_sites={sum(sites for _, sites in blocked_functions)} "
        f"image_objects={len(image.objects)} image_bytes={image.byte_size} "
        f"linker_boundaries={len(linker_contract.active_boundary_symbols(image_sections))}"
    )

    if "__mm_user_syscall" not in program.symbol_addresses:
        program.register_service("__mm_user_syscall", user_syscall)

    if args.native_vm:
        from src.minimachine.native_vm import NativeVM
        native_init_started = time.perf_counter()
        vm = NativeVM(program)
        vm.native_report_every = max(0, args.native_report_every)
        print("BOOT_EXEC_BACKEND backend=native-c", flush=True)
        print(
            "BOOT_EXEC_STAGE "
            f"stage=native-vm-init seconds={time.perf_counter() - native_init_started:.3f} "
            f"elapsed_s={time.perf_counter() - runner_started:.3f}",
            flush=True,
        )
    else:
        vm = program.new_vm()
        print("BOOT_EXEC_BACKEND backend=python", flush=True)
    vm.ecall_handler = linux_ecall
    vm.stop_after_user_handoff = args.stop_after_user_handoff
    vm.linux_task_contexts = {}
    vm.linux_task_shadow_stacks = {}
    # Reserve the top 16 MiB for the boot task's existing P3 stack. New
    # Linux tasks receive independent P3 continuation stacks below it.
    vm.linux_shadow_stack_next = vm.stack_top - 0x01000000
    if task_sched_class_offset is not None:
        vm.linux_task_sched_class_offset = task_sched_class_offset

    resumed_from_checkpoint = False
    if args.checkpoint_in is not None:
        try:
            load_checkpoint(
                vm,
                args.checkpoint_in,
                image_sha256=linked_image_sha256,
                initramfs_sha256=initramfs_sha256,
            )
        except CheckpointError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=checkpoint error={exc}")
            return 1
        vm.ecall_handler = linux_ecall
        refresh_host_service_descriptors(vm)
        resumed_from_checkpoint = True
        if args.trace_hot_filp_open:
            install_hot_filp_trace(vm)
        if args.probe_rootfs_after_checkpoint:
            try:
                probe_live_rootfs(vm)
            except VMError as exc:
                print(f"BOOT_EXEC_BLOCKED stage=rootfs-probe error={exc}")
                return 1
        if args.inject_init is not None and args.inject_initramfs_cpio is not None:
            print(
                "BOOT_EXEC_BLOCKED stage=rootfs-inject "
                "error=choose only one of --inject-init or --inject-initramfs-cpio"
            )
            return 1
        if args.inject_initramfs_cpio is not None:
            try:
                inject_live_root_initramfs(vm, args.inject_initramfs_cpio)
                if args.skip_prepare_namespace_after_inject:
                    skip_prepare_namespace_after_hot_init(vm)
            except VMError as exc:
                print(f"BOOT_EXEC_BLOCKED stage=rootfs-inject error={exc}")
                return 1
        elif args.inject_init is not None:
            try:
                inject_live_root_init(
                    vm,
                    args.inject_init,
                    guest_path_text=args.inject_init_path,
                )
                if args.skip_prepare_namespace_after_inject:
                    skip_prepare_namespace_after_hot_init(vm)
            except VMError as exc:
                print(f"BOOT_EXEC_BLOCKED stage=rootfs-inject error={exc}")
                return 1
        elif args.skip_prepare_namespace_after_inject:
            print(
                "BOOT_EXEC_BLOCKED stage=rootfs-inject "
                "error=--skip-prepare-namespace-after-inject requires an injection source"
            )
            return 1
        print(
            "BOOT_EXEC_CHECKPOINT_RESTORED "
            f"path={args.checkpoint_in} steps={vm.steps} "
            f"function={vm.current_function} block={vm.current_block} ip={vm.ip}",
            flush=True,
        )

        if args.probe_run_init_after_checkpoint:
            if "run_init_process" not in vm.program.functions:
                print(
                    "BOOT_EXEC_BLOCKED stage=init-exec-probe "
                    "error=missing-run_init_process",
                    flush=True,
                )
                return 1
            guest_path = b"/init\0"
            path_ptr = vm.alloc_bytes(len(guest_path), align=8)
            bulk_write = getattr(vm.memory, "bulk_write", None)
            if bulk_write is not None:
                bulk_write(path_ptr, guest_path)
            else:
                for i, byte in enumerate(guest_path):
                    vm.memory.write(path_ptr + i, 8, byte)
            try:
                result, = _call_linux_function_preserving_control(
                    vm,
                    "run_init_process",
                    (path_ptr,),
                    result_count=1,
                    max_extra_steps=20_000_000,
                )
            except VMError as exc:
                print(
                    "BOOT_EXEC_BLOCKED stage=init-exec-probe "
                    f"error={exc}",
                    flush=True,
                )
                return 1
            raw32 = result & 0xFFFFFFFF
            signed = raw32 - (1 << 32) if raw32 & (1 << 31) else raw32
            print(
                "BOOT_EXEC_INIT_EXEC_PROBE "
                f"path=/init result={signed} raw=0x{result:x} "
                f"steps={vm.steps}",
                flush=True,
            )
            return 0

        if args.restart_run_init_after_checkpoint:
            if "run_init_process" not in vm.program.functions:
                print(
                    "BOOT_EXEC_BLOCKED stage=init-exec-restart "
                    "error=missing-run_init_process",
                    flush=True,
                )
                return 1
            restart_path = args.restart_init_path
            if not restart_path.startswith("/"):
                print(
                    "BOOT_EXEC_BLOCKED stage=init-exec-restart "
                    "error=--restart-init-path-must-be-absolute",
                    flush=True,
                )
                return 1
            guest_path = restart_path.encode("utf-8") + b"\0"
            path_ptr = vm.alloc_bytes(len(guest_path), align=8)
            bulk_write = getattr(vm.memory, "bulk_write", None)
            if bulk_write is not None:
                bulk_write(path_ptr, guest_path)
            else:
                for i, byte in enumerate(guest_path):
                    vm.memory.write(path_ptr + i, 8, byte)

            linked = vm.program.functions["run_init_process"]
            restart_stack_top = 0x0F00_0000
            result_words = 1
            total = linked.frame_size + 8 + result_words * 8
            restart_sp = restart_stack_top - total
            result_base = restart_sp + linked.frame_size + 8
            vm.enter_function(
                "run_init_process",
                (path_ptr,),
                stack_top=restart_stack_top,
                result_count=1,
            )
            vm.init_exec_restart_result_base = result_base
            print(
                "BOOT_EXEC_INIT_EXEC_RESTART "
                f"path={restart_path} path_ptr=0x{path_ptr:x} "
                f"sp=0x{vm.sp:x} steps={vm.steps}",
                flush=True,
            )

    if args.probe_kallsyms is not None:
        symbol = args.probe_kallsyms
        linked = program.functions.get(symbol)
        if linked is None or not linked.function.blocks:
            print(f"BOOT_EXEC_BLOCKED stage=kallsyms-probe missing={symbol}")
            return 1
        pc = program.block_code[(symbol, linked.function.blocks[0].label)]
        size_ptr = vm.alloc_bytes(8, align=8)
        offset_ptr = vm.alloc_bytes(8, align=8)
        name_ptr = vm.alloc_bytes(512, align=8)
        try:
            found = vm.run_function(
                "kallsyms_lookup_size_offset",
                (pc, size_ptr, offset_ptr),
                result_count=1,
                max_steps=args.max_steps,
            )
            named = vm.run_function(
                "lookup_symbol_name",
                (pc, name_ptr),
                result_count=1,
                max_steps=args.max_steps,
            )
        except VMError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=kallsyms-probe error={exc}")
            return 1
        raw_name = bytearray()
        for i in range(512):
            byte = vm.memory.read(name_ptr + i, 8)
            if byte == 0:
                break
            raw_name.append(byte)
        decoded = raw_name.decode("utf-8", errors="replace")
        size = vm.memory.read(size_ptr, 64)
        offset = vm.memory.read(offset_ptr, 64)
        print(
            "BOOT_EXEC_KALLSYMS_PROBE "
            f"symbol={symbol} pc=0x{pc:x} found={found[0]} "
            f"name_rc={named[0]} name={decoded} size={size} offset={offset}",
            flush=True,
        )
        if found[0] != 1 or named[0] != 0 or decoded != symbol:
            print("BOOT_EXEC_BLOCKED stage=kallsyms-probe mismatch=1")
            return 1
        return 0

    last_milestone_function = None
    last_milestone_time = None
    execution_started = None
    milestone_functions = {
        "sched_init",
        "sched_fork",
        "early_irq_init",
        "init_IRQ",
        "tick_init",
        "init_timers",
        "srcu_init",
        "hrtimers_init",
        "softirq_init",
        "timekeeping_init",
        "time_init",
        "console_init",
        "arch_call_rest_init",
        "rest_init",
        "minimachine_ret_from_fork",
        "kernel_init",
        "kernel_init_freeable",
        "init_post",
        "console_on_rootfs",
        "do_mounts_initrd",
        "prepare_namespace",
        "do_basic_setup",
        "do_pre_smp_initcalls",
        "do_initcalls",
        "do_initcall_level",
        "do_one_initcall",
        "wait_for_initramfs",
        "async_synchronize_full",
        "async_synchronize_full_domain",
        "async_synchronize_cookie_domain",
        "async_run_entry_fn",
        "worker_thread",
        "process_one_work",
        "schedule",
        "__schedule",
        "schedule_timeout",
        "schedule_timeout_uninterruptible",
        "msleep",
        "__const_udelay",
        "free_initmem",
        "mark_readonly",
        "run_init_process",
        "try_to_run_init_process",
        "kernel_execve",
        "kthreadd",
    }

    if args.native_vm and args.native_boot_phases_only:
        milestone_functions = {
            "console_init",
            "arch_call_rest_init",
            "rest_init",
            "kernel_init",
            "kernel_init_freeable",
            "do_pre_smp_initcalls",
            "do_basic_setup",
            "do_initcalls",
            "free_initmem",
            "mark_readonly",
            "console_on_rootfs",
            "do_mounts_initrd",
            "prepare_namespace",
            "init_post",
            "run_init_process",
            "try_to_run_init_process",
            "kernel_execve",
            "wait_for_initramfs",
            "populate_rootfs",
            "do_populate_rootfs",
            "unpack_to_rootfs",
            "do_name",
            "do_copy",
            "do_reset",
            "panic",
        }
        print(
            "BOOT_EXEC_NATIVE_BOOT_PHASE_TRACE "
            f"functions={len(milestone_functions)}",
            flush=True,
        )

    sched_watch_installed = False

    # Track initcall identity only on real function-entry transitions.  This
    # avoids wrapping every P3 instruction just to discover coarse milestones.
    symbols_by_address: dict[int, list[str]] = {}
    for symbol, address in program.symbol_addresses.items():
        symbols_by_address.setdefault(address, []).append(symbol)
    last_initcall_enter_step: int | None = (
        vm.steps if resumed_from_checkpoint and args.checkpoint_initcall else None
    )
    last_initcall_symbol = (
        args.checkpoint_initcall
        if resumed_from_checkpoint and args.checkpoint_initcall
        else "<none>"
    )
    checkpoint_written = False
    checkpoint_after_armed = bool(
        resumed_from_checkpoint
        and args.checkpoint_after_initcall is not None
        and args.checkpoint_initcall == args.checkpoint_after_initcall
    )
    if checkpoint_after_armed:
        print(
            "BOOT_EXEC_CHECKPOINT_ARMED "
            f"after_initcall={args.checkpoint_after_initcall} "
            f"steps={vm.steps} source=resume",
            flush=True,
        )
    checkpoint_function_hits = 0
    initramfs_last_name = None
    initramfs_copy_seen = False

    def install_sched_watch() -> None:
        nonlocal sched_watch_installed
        if sched_watch_installed:
            return
        linked = vm.program.functions.get("sched_fork")
        sched_off = getattr(vm, "linux_task_sched_class_offset", None)
        if linked is None or sched_off is None:
            return
        frame_size = vm.memory.read(vm.sp + 24, 64)
        argc = vm.memory.read(vm.sp + 56, 64)
        if argc < 2:
            return
        arg_base = vm.sp + frame_size
        task = vm.memory.read(arg_base + 8, 64)
        watched = task + sched_off
        print(
            "BOOT_EXEC_SCHED_WATCH_START "
            f"steps={vm.steps} task=0x{task:x} address=0x{watched:x} "
            f"value=0x{vm.memory.read(watched, 64):x}",
            flush=True,
        )
        original_memory_write = vm.memory.write

        def watched_write(address, bits, value):
            byte_count = bits // 8
            if address <= watched < address + byte_count:
                before = vm.memory.read(watched, 64)
                original_memory_write(address, bits, value)
                after = vm.memory.read(watched, 64)
                print(
                    "BOOT_EXEC_SCHED_WATCH_WRITE "
                    f"steps={vm.steps} function={vm.current_function} "
                    f"block={vm.current_block} ip={vm.ip} "
                    f"address=0x{address:x} bits={bits} "
                    f"before=0x{before:x} after=0x{after:x}",
                    flush=True,
                )
                return
            original_memory_write(address, bits, value)

        vm.memory.write = watched_write
        sched_watch_installed = True

    def observe_function_entry(function: str) -> None:
        nonlocal last_milestone_function, last_milestone_time
        nonlocal last_initcall_enter_step, last_initcall_symbol
        nonlocal checkpoint_written, checkpoint_after_armed
        nonlocal checkpoint_function_hits
        nonlocal initramfs_last_name, initramfs_copy_seen

        if function == "do_name":
            collected_addr = vm.program.symbol_addresses.get("collected")
            name_len_addr = vm.program.symbol_addresses.get("name_len")
            body_len_addr = vm.program.symbol_addresses.get("body_len")
            mode_addr = vm.program.symbol_addresses.get("mode")
            wfile_addr = vm.program.symbol_addresses.get("wfile")
            collected_ptr = (
                vm.memory.read(collected_addr, 64)
                if collected_addr is not None else 0
            )
            name_len = (
                vm.memory.read(name_len_addr, 64)
                if name_len_addr is not None else 0
            )
            body_len = (
                vm.memory.read(body_len_addr, 64)
                if body_len_addr is not None else 0
            )
            mode = (
                vm.memory.read(mode_addr, 16)
                if mode_addr is not None else 0
            )
            raw = bytearray()
            if collected_ptr:
                for i in range(min(int(name_len or 0), 256)):
                    b = vm.memory.read(collected_ptr + i, 8)
                    if b == 0:
                        break
                    raw.append(b)
            initramfs_last_name = raw.decode("utf-8", errors="replace")
            initramfs_copy_seen = False
            old_wfile = (
                vm.memory.read(wfile_addr, 64)
                if wfile_addr is not None else 0
            )
            print(
                "BOOT_EXEC_INITRAMFS_NAME "
                f"steps={vm.steps} name={initramfs_last_name!r} "
                f"mode=0{mode:o} body_len={body_len} "
                f"wfile_before=0x{old_wfile:x}",
                flush=True,
            )

        if function == "do_copy":
            initramfs_copy_seen = True
            wfile_addr = vm.program.symbol_addresses.get("wfile")
            body_len_addr = vm.program.symbol_addresses.get("body_len")
            wfile = vm.memory.read(wfile_addr, 64) if wfile_addr is not None else 0
            body_len = (
                vm.memory.read(body_len_addr, 64)
                if body_len_addr is not None else 0
            )
            print(
                "BOOT_EXEC_INITRAMFS_COPY "
                f"steps={vm.steps} name={initramfs_last_name!r} "
                f"wfile=0x{wfile:x} body_len={body_len}",
                flush=True,
            )

        if function == "do_reset" and initramfs_last_name is not None:
            wfile_addr = vm.program.symbol_addresses.get("wfile")
            wfile = vm.memory.read(wfile_addr, 64) if wfile_addr is not None else 0
            signed_wfile = wfile - (1 << 64) if wfile & (1 << 63) else wfile
            print(
                "BOOT_EXEC_INITRAMFS_RESET "
                f"steps={vm.steps} name={initramfs_last_name!r} "
                f"copy_seen={int(initramfs_copy_seen)} "
                f"wfile=0x{wfile:x} signed_wfile={signed_wfile}",
                flush=True,
            )
            initramfs_last_name = None
            initramfs_copy_seen = False

        if function == "unpack_to_rootfs":
            frame_size = vm.memory.read(vm.sp + 24, 64)
            argc = vm.memory.read(vm.sp + 56, 64)
            arg_base = vm.sp + frame_size
            raw_args = [
                vm.memory.read(arg_base + 8 * i, 64)
                for i in range(min(argc, 4))
            ]
            ptr = raw_args[0] if raw_args else 0
            length = raw_args[1] if len(raw_args) > 1 else 0
            expected_ptr = vm.program.symbol_addresses.get("__initramfs_start", 0)
            expected_size_addr = vm.program.symbol_addresses.get("__initramfs_size")
            expected_size = (
                vm.memory.read(expected_size_addr, 64)
                if expected_size_addr is not None
                else 0
            )
            print(
                "BOOT_EXEC_INITRAMFS_UNPACK_ENTER "
                f"steps={vm.steps} ptr=0x{ptr:x} len={length} "
                f"expected_ptr=0x{expected_ptr:x} expected_len={expected_size}",
                flush=True,
            )

        if function == "panic":
            frame_size = vm.memory.read(vm.sp + 24, 64)
            argc = vm.memory.read(vm.sp + 56, 64)
            arg_base = vm.sp + frame_size
            raw_args = [
                vm.memory.read(arg_base + 8 * i, 64)
                for i in range(min(argc, 8))
            ]
            fmt_ptr = raw_args[0] if raw_args else 0
            raw_fmt = bytearray()
            if fmt_ptr:
                for i in range(512):
                    byte = vm.memory.read(fmt_ptr + i, 8)
                    if byte == 0:
                        break
                    raw_fmt.append(byte)
            print(
                "BOOT_EXEC_PANIC_ENTER "
                f"steps={vm.steps} argc={argc} "
                f"fmt_ptr=0x{fmt_ptr:x} "
                f"fmt={raw_fmt.decode('utf-8', errors='replace')!r} "
                f"args={','.join(f'0x{x:x}' for x in raw_args)}",
                flush=True,
            )
            if args.stop_on_panic:
                raise VMError("native-stop-on-panic: kernel panic entered")

        if function == "do_one_initcall":
            frame_size = vm.memory.read(vm.sp + 24, 64)
            argc = vm.memory.read(vm.sp + 56, 64)
            initcall_ptr = vm.memory.read(vm.sp + frame_size, 64) if argc else 0
            names = symbols_by_address.get(initcall_ptr, ())
            initcall_symbol = "|".join(sorted(names)[:4]) if names else "<unknown>"
            since_previous = (
                vm.steps - last_initcall_enter_step
                if last_initcall_enter_step is not None
                else 0
            )
            print(
                "BOOT_EXEC_INITCALL_ENTER "
                f"steps={vm.steps} since_previous={since_previous} "
                f"ptr=0x{initcall_ptr:x} symbol={initcall_symbol}",
                flush=True,
            )
            previous_initcall_symbol = last_initcall_symbol
            last_initcall_enter_step = vm.steps
            last_initcall_symbol = initcall_symbol

            if (
                not checkpoint_written
                and checkpoint_after_armed
                and args.checkpoint_out is not None
            ):
                save_checkpoint(
                    vm,
                    args.checkpoint_out,
                    image_sha256=linked_image_sha256,
                    initramfs_sha256=initramfs_sha256,
                )
                checkpoint_written = True
                checkpoint_after_armed = False
                print(
                    "BOOT_EXEC_CHECKPOINT_SAVED "
                    f"path={args.checkpoint_out} steps={vm.steps} "
                    f"after_initcall={previous_initcall_symbol} "
                    f"next_initcall={initcall_symbol}",
                    flush=True,
                )
                if args.stop_after_checkpoint:
                    vm.halted = True

            if (
                not checkpoint_written
                and args.checkpoint_out is not None
                and args.checkpoint_initcall == initcall_symbol
            ):
                save_checkpoint(
                    vm,
                    args.checkpoint_out,
                    image_sha256=linked_image_sha256,
                    initramfs_sha256=initramfs_sha256,
                )
                checkpoint_written = True
                print(
                    "BOOT_EXEC_CHECKPOINT_SAVED "
                    f"path={args.checkpoint_out} steps={vm.steps} "
                    f"initcall={initcall_symbol}",
                    flush=True,
                )
                if args.stop_after_checkpoint:
                    vm.halted = True

            if (
                not checkpoint_written
                and args.checkpoint_out is not None
                and args.checkpoint_after_initcall == initcall_symbol
            ):
                checkpoint_after_armed = True
                print(
                    "BOOT_EXEC_CHECKPOINT_ARMED "
                    f"after_initcall={initcall_symbol} steps={vm.steps}",
                    flush=True,
                )

        if (
            not checkpoint_written
            and args.checkpoint_function is not None
            and function == args.checkpoint_function
        ):
            checkpoint_function_hits += 1
            if checkpoint_function_hits == args.checkpoint_function_hit:
                if args.checkpoint_out is None:
                    raise VMError(
                        "--checkpoint-function requires --checkpoint-out"
                    )
                save_checkpoint(
                    vm,
                    args.checkpoint_out,
                    image_sha256=linked_image_sha256,
                    initramfs_sha256=initramfs_sha256,
                )
                checkpoint_written = True
                print(
                    "BOOT_EXEC_CHECKPOINT_SAVED "
                    f"path={args.checkpoint_out} steps={vm.steps} "
                    f"function={function} hit={checkpoint_function_hits}",
                    flush=True,
                )
                if args.stop_after_checkpoint:
                    vm.halted = True

        if function == "sched_fork":
            install_sched_watch()

        if function in milestone_functions:
            now = time.perf_counter()
            elapsed = now - execution_started if execution_started is not None else 0.0
            since_last = (
                now - last_milestone_time
                if last_milestone_time is not None
                else elapsed
            )
            print(
                "BOOT_EXEC_MILESTONE "
                f"steps={vm.steps} function={function} "
                f"elapsed_s={elapsed:.3f} since_last_s={since_last:.3f}",
                flush=True,
            )
            last_milestone_function = function
            last_milestone_time = now

    traced_entry_codes: dict[int, str] = {}
    for function in milestone_functions:
        linked = vm.program.functions.get(function)
        if linked is None or not linked.function.blocks:
            continue
        entry_block = linked.function.blocks[0].label
        traced_entry_codes[
            vm.program.block_code[(function, entry_block)]
        ] = function

    if args.checkpoint_function is not None:
        linked = vm.program.functions.get(args.checkpoint_function)
        if linked is None or not linked.function.blocks:
            print(
                "BOOT_EXEC_BLOCKED stage=checkpoint "
                f"missing_function={args.checkpoint_function}"
            )
            return 1
        entry_block = linked.function.blocks[0].label
        traced_entry_codes[
            vm.program.block_code[(args.checkpoint_function, entry_block)]
        ] = args.checkpoint_function

    original_set_code = vm._set_code

    def traced_set_code(code: int) -> None:
        function = traced_entry_codes.get(code)
        parser_function = getattr(
            vm, "trace_user_parser_codes", {}
        ).get(code)
        trace_memcpy = (
            getattr(vm, "trace_user_read_memcpy_code", None) == code
        )
        original_set_code(code)
        if parser_function is not None:
            parsefile_symbol = vm.program.symbol_addresses.get(
                "__mm_user_g_parsefile"
            )
            parsefile = (
                vm.memory.read(parsefile_symbol, 64)
                if parsefile_symbol is not None
                else 0
            )
            if parsefile:
                next_ptr = vm.memory.read(parsefile + 24, 64)
                buf_ptr = vm.memory.read(parsefile + 32, 64)
                left_line = vm.memory.read(parsefile + 16, 32)
                left_buffer = vm.memory.read(parsefile + 20, 32)
                lastc0 = vm.memory.read(parsefile + 120, 32)
                lastc1 = vm.memory.read(parsefile + 124, 32)
                unget = vm.memory.read(parsefile + 128, 32)
            else:
                next_ptr = buf_ptr = 0
                left_line = left_buffer = lastc0 = lastc1 = unget = -1
            try:
                frame_size = vm.memory.read(vm.sp + FRAME_SIZE, 64)
                argc = vm.memory.read(vm.sp + ARG_COUNT, 64)
                arg_base = vm.sp + frame_size
                raw_args = tuple(
                    vm.memory.read(arg_base + i * 8, 64)
                    for i in range(min(int(argc), 6))
                )
            except Exception:
                frame_size = argc = -1
                raw_args = ()
            print(
                "BOOT_EXEC_USER_PARSER_ENTRY "
                f"function={parser_function} code=0x{code:x} "
                f"sp=0x{vm.sp:x} frame_size={frame_size} argc={argc} "
                f"args={','.join(f'0x{x:x}' for x in raw_args)} "
                f"parsefile=0x{parsefile:x} left_line={left_line} "
                f"left_buffer={left_buffer} next=0x{next_ptr:x} "
                f"buf=0x{buf_ptr:x} "
                f"consumed={next_ptr - buf_ptr if next_ptr and buf_ptr else -1} "
                f"lastc0={lastc0} lastc1={lastc1} unget={unget}",
                flush=True,
            )
            if (
                parser_function == "__mm_user_pgetc"
                and unget > 0
                and left_buffer > 0
                and not getattr(vm, "trace_user_pgetc_unget_armed", False)
            ):
                vm.trace_user_pgetc_unget_armed = True
                vm.trace_single_step_watch_code = code
                vm.trace_single_step_count = 240
                print(
                    "BOOT_EXEC_USER_PGETC_UNGET_SINGLE_STEP "
                    f"code=0x{code:x} count=240 "
                    f"left_buffer={left_buffer} lastc0={lastc0} "
                    f"lastc1={lastc1} unget={unget}",
                    flush=True,
                )
        if trace_memcpy:
            linked = vm.program.functions.get("memcpy")
            if linked is not None:
                frame_size = vm.memory.read(vm.sp + FRAME_SIZE, 64)
                argc = vm.memory.read(vm.sp + ARG_COUNT, 64)
                arg_base = vm.sp + frame_size
                raw_args = tuple(
                    vm.memory.read(arg_base + i * 8, 64)
                    for i in range(min(int(argc), 6))
                )
                print(
                    "BOOT_EXEC_USER_READ_MEMCPY_ENTRY "
                    f"sp=0x{vm.sp:x} frame_size={frame_size} argc={argc} "
                    f"args={','.join(f'0x{x:x}' for x in raw_args)}",
                    flush=True,
                )
        if function is not None:
            observe_function_entry(function)

    vm._set_code = traced_set_code
    if hasattr(vm, "set_watch_codes"):
        vm.set_watch_codes(traced_entry_codes.keys())

    original_enter_function = vm.enter_function

    def traced_enter_function(name, args=(), *, stack_top, result_count=0):
        original_enter_function(
            name,
            args,
            stack_top=stack_top,
            result_count=result_count,
        )
        if name in milestone_functions:
            observe_function_entry(name)

    vm.enter_function = traced_enter_function

    # Progress sampling is optional.  Production replay uses zero so the hot
    # interpreter path is not wrapped once per P3 instruction.
    if args.progress_every > 0:
        original_step = vm.step
        next_progress = args.progress_every

        def progress_step():
            nonlocal next_progress
            if vm.steps >= next_progress:
                print(
                    "BOOT_EXEC_PROGRESS "
                    f"steps={vm.steps} function={vm.current_function} "
                    f"block={vm.current_block} ip={vm.ip} "
                    f"last_initcall={last_initcall_symbol}",
                    flush=True,
                )
                next_progress += args.progress_every
            original_step()

        vm.step = progress_step

    execution_started = time.perf_counter()
    print(
        "BOOT_EXEC_STAGE "
        f"stage=execute-start elapsed_s={execution_started - runner_started:.3f}",
        flush=True,
    )
    try:
        if resumed_from_checkpoint:
            if args.max_steps <= 0 and args.native_vm:
                resume_limit = 0
                absolute_text = "unlimited"
            else:
                resume_limit = vm.steps + args.max_steps
                absolute_text = str(resume_limit)
            print(
                "BOOT_EXEC_CHECKPOINT_BUDGET "
                f"saved_steps={vm.steps} additional_steps={args.max_steps} "
                f"absolute_limit={absolute_text}",
                flush=True,
            )
            vm.run(max_steps=resume_limit)
        else:
            vm.run_function(
                args.entry,
                (),
                result_count=0,
                max_steps=args.max_steps,
            )
    except VMError as exc:
        if args.native_vm and "step limit exceeded" in str(exc):
            print(
                "BOOT_EXEC_NATIVE_LIMIT "
                f"steps={vm.steps} function={vm.current_function} "
                f"block={vm.current_block} ip={vm.ip}",
                flush=True,
            )
            return 0
        checkpoint_reason = None
        if args.checkpoint_on_error and args.checkpoint_out is not None:
            checkpoint_reason = "error"
        elif (
            args.checkpoint_at_limit
            and args.checkpoint_out is not None
            and "step limit exceeded" in str(exc)
        ):
            checkpoint_reason = "step_limit"
        if checkpoint_reason is not None:
            save_checkpoint(
                vm,
                args.checkpoint_out,
                image_sha256=linked_image_sha256,
                initramfs_sha256=initramfs_sha256,
            )
            checkpoint_written = True
            print(
                "BOOT_EXEC_CHECKPOINT_SAVED "
                f"path={args.checkpoint_out} steps={vm.steps} "
                f"reason={checkpoint_reason} function={vm.current_function} "
                f"block={vm.current_block} ip={vm.ip}",
                flush=True,
            )
        if args.checkpoint_function is not None:
            print(
                "BOOT_EXEC_CHECKPOINT_FUNCTION_PROGRESS "
                f"function={args.checkpoint_function} "
                f"hits={checkpoint_function_hits} steps={vm.steps}",
                flush=True,
            )
        inst = current_instruction(vm)
        print(
            "BOOT_EXEC_BLOCKED "
            f"stage=execute steps={vm.steps} function={vm.current_function} "
            f"block={vm.current_block} ip={vm.ip} sp=0x{vm.sp:x} "
            f"error={exc}"
        )
        if inst is not None:
            print(f"BOOT_EXEC_NEXT {inst!r}", flush=True)
            dump_instruction_slots(vm, inst)
        dump_current_call_frame(vm)
        dump_scheduler_indirect_call(vm)
        dump_linux_memory_state(vm)
        probe_linux_memory_helpers(vm)
        return 1

    restart_result_base = getattr(
        vm,
        "init_exec_restart_result_base",
        None,
    )
    if restart_result_base is not None:
        raw = vm.memory.read(restart_result_base, 64)
        raw32 = raw & 0xFFFFFFFF
        signed = raw32 - (1 << 32) if raw32 & (1 << 31) else raw32
        print(
            "BOOT_EXEC_INIT_EXEC_RESTART_RESULT "
            f"result={signed} raw=0x{raw:x} "
            f"result_ptr=0x{restart_result_base:x}",
            flush=True,
        )

        if signed == 0 and args.restart_run_init_after_checkpoint:
            current_addr = vm.program.symbol_addresses.get(
                "minimachine_current_task"
            )
            if current_addr is None:
                print(
                    "BOOT_EXEC_BLOCKED stage=exec-return-to-user "
                    "error=missing-minimachine_current_task",
                    flush=True,
                )
                return 1
            current = vm.memory.read(current_addr, 64)
            if not current:
                print(
                    "BOOT_EXEC_BLOCKED stage=exec-return-to-user "
                    "error=null-current-task",
                    flush=True,
                )
                return 1
            if "task_stack_page" not in vm.program.functions:
                print(
                    "BOOT_EXEC_BLOCKED stage=exec-return-to-user "
                    "error=missing-task_stack_page",
                    flush=True,
                )
                return 1
            if "minimachine_enter_userspace" not in vm.program.functions:
                print(
                    "BOOT_EXEC_BLOCKED stage=exec-return-to-user "
                    "error=missing-minimachine_enter_userspace",
                    flush=True,
                )
                return 1

            try:
                stack_base, = _call_linux_function_preserving_control(
                    vm,
                    "task_stack_page",
                    (current,),
                    result_count=1,
                    max_extra_steps=500_000,
                )
            except VMError as exc:
                print(
                    "BOOT_EXEC_BLOCKED stage=exec-return-to-user "
                    f"error=task-stack-page:{exc}",
                    flush=True,
                )
                return 1

            # Exact MiniMachine arch contract:
            # THREAD_SIZE=16 KiB and struct pt_regs is 12 x u64 = 96 bytes.
            regs = (stack_base + 16384 - 96) & ((1 << 64) - 1)
            status = vm.memory.read(regs + 80, 64)
            print(
                "BOOT_EXEC_INIT_RETURN_TO_USER "
                f"current=0x{current:x} stack_base=0x{stack_base:x} "
                f"regs=0x{regs:x} status=0x{status:x}",
                flush=True,
            )
            if not (status & 1):
                print(
                    "BOOT_EXEC_BLOCKED stage=exec-return-to-user "
                    f"error=pt-regs-not-user status=0x{status:x}",
                    flush=True,
                )
                return 1

            vm.enter_function(
                "minimachine_enter_userspace",
                (regs,),
                stack_top=0x0F00_0000,
                result_count=0,
            )
            try:
                vm.run(max_steps=0 if args.native_vm else args.max_steps)
            except VMError as exc:
                print(
                    "BOOT_EXEC_BLOCKED stage=userspace-after-exec "
                    f"steps={vm.steps} function={vm.current_function} "
                    f"block={vm.current_block} ip={vm.ip} error={exc}",
                    flush=True,
                )
                return 1

    print(
        "BOOT_EXEC_HALTED "
        f"steps={vm.steps} function={vm.current_function} "
        f"block={vm.current_block} ip={vm.ip}"
    )
    print(
        "BOOT_EXEC_STAGE "
        f"stage=execute-end execute_s={time.perf_counter() - execution_started:.3f} "
        f"total_s={time.perf_counter() - runner_started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
