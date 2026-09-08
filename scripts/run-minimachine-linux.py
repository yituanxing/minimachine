#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import fields, is_dataclass
from functools import cmp_to_key
from pathlib import Path
import re
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir
from src.minimachine.abi import (
    ARG_COUNT,
    CALLER_SP,
    FRAME_SIZE,
    HEADER_SIZE,
    RESULT_COUNT,
    RESULT_PTR,
    RET_PC,
    WORD,
    expand_function,
)
from src.minimachine.cli.linux import parse_args
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
from src.minimachine.platform.traps import arch_escapes, register_traps, trap_reasons
from src.minimachine.platform.diagnostics import (
    dump_current_call_frame,
    dump_instruction_slots,
    dump_linux_memory_state,
    dump_scheduler_indirect_call,
    probe_linux_memory_helpers,
)

from src.minimachine.platform.linux_control import (
    _call_linux_function_preserving_control,
    inject_live_root_init,
    inject_live_root_initramfs,
    install_hot_filp_trace,
    probe_live_rootfs,
    skip_prepare_namespace_after_hot_init,
)

from src.minimachine.platform.libc import build_user_libc_callback
from src.minimachine.platform.syscall import dispatch_user_syscall
from src.minimachine.program_cache import (
    ProgramCache,
    ProgramCacheError,
    load_program_cache,
    save_program_cache,
)
from src.minimachine.native_hot_cache import (
    NativeHotCacheError,
    load_native_hot_cache,
)
from src.minimachine.runtime import (
    accelerate_direct_runtime,
    collect_runtime_surface,
    direct_runtime_callback,
    helper_callback,
    install_runtime,
)
from src.minimachine.user_bundle import rebase_user_program_namespace
from src.minimachine.user_image import UserImageError, unpack_user_image
from src.minimachine.user_image_cache import (
    UserImageCacheError,
    load_user_image_cache,
)
from src.minimachine.verify import verify_muir, verify_p3
from src.minimachine.vm import HOST_CONTROL_TRANSFER, Program, VMError


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


def _user_external_prefix(symbol: str) -> str | None:
    if not symbol.startswith("__mm_"):
        return None
    marker = "_ext_"
    marker_at = symbol.find(marker, len("__mm_"))
    if marker_at < 0:
        return None
    return symbol[: marker_at + len(marker)]


def _user_external_original(symbol: str) -> str:
    prefix = _user_external_prefix(symbol)
    return symbol[len(prefix):] if prefix is not None else symbol


def _user_libc_callback(symbol: str, errno_address: int | None):
    return build_user_libc_callback(
        symbol,
        errno_address,
        user_syscall=user_syscall,
        _call_linux_function_preserving_control=_call_linux_function_preserving_control,
        _call_guest_descriptor_preserving_control=_call_guest_descriptor_preserving_control,
        _guest_function_name_from_descriptor=_guest_function_name_from_descriptor,
        _snapshot_p3_call_chain=_snapshot_p3_call_chain,
        _restore_p3_call_chain=_restore_p3_call_chain,
    )
def trace_user_external_descriptor(vm, original: str, stage: str) -> None:
    matches = [
        symbol
        for symbol in vm.program.symbol_addresses
        if _user_external_prefix(symbol) is not None
        and _user_external_original(symbol) == original
    ]
    symbol = matches[-1] if matches else ""
    descriptor = vm.program.symbol_addresses.get(symbol) if symbol else None
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


def sync_program_initial_range(vm, start: int, end: int) -> None:
    if end < start:
        raise VMError(
            f"invalid initial-memory sync range: 0x{start:x}..0x{end:x}"
        )
    if end == start:
        return
    payload = bytes(
        vm.program.initial_memory.read(address, 8)
        for address in range(start, end)
    )
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        bulk_write(start, payload)
    else:
        for offset, byte in enumerate(payload):
            vm.memory.write(start + offset, 8, byte)
    print(
        "BOOT_EXEC_USER_INITIAL_SYNC "
        f"start=0x{start:x} end=0x{end:x} bytes={len(payload)}",
        flush=True,
    )


def _snapshot_p3_call_chain(vm):
    snapshots = []
    frame = int(vm.sp)
    seen = set()
    for _ in range(4096):
        if not frame or frame in seen:
            break
        seen.add(frame)

        frame_size = int(vm.memory.read(frame + FRAME_SIZE, 64))
        argc = int(vm.memory.read(frame + ARG_COUNT, 64))
        result_count = int(vm.memory.read(frame + RESULT_COUNT, 64))
        if frame_size < HEADER_SIZE or frame_size > (1 << 24):
            break
        if argc > (1 << 20) or result_count > (1 << 20):
            break
        total = frame_size + argc * WORD + max(1, result_count) * WORD
        if total > (1 << 26):
            raise VMError(
                "MiniMachine P3 call-chain frame snapshot too large: "
                f"sp=0x{frame:x} bytes={total}"
            )
        payload = bytes(vm.memory.read(frame + offset, 8) for offset in range(total))
        snapshots.append((frame, payload))

        caller = int(vm.memory.read(frame + CALLER_SP, 64))
        if not caller or caller == frame:
            break
        frame = caller
    return tuple(snapshots)


def _restore_p3_call_chain(vm, snapshots) -> None:
    bulk_write = getattr(vm.memory, "bulk_write", None)
    for frame, payload in snapshots:
        if bulk_write is not None:
            bulk_write(frame, payload)
        else:
            for offset, byte in enumerate(payload):
                vm.memory.write(frame + offset, 8, byte)


def _activate_user_linux_task(vm, task: int, *, source: str) -> None:
    task = int(task)
    if not task:
        return

    previous = int(getattr(vm, "linux_current_task", 0) or 0)
    current_addr = vm.program.symbol_addresses.get("minimachine_current_task")
    kernel_previous = (
        int(vm.memory.read(current_addr, 64))
        if current_addr is not None
        else 0
    )
    if previous == task and (current_addr is None or kernel_previous == task):
        return

    vm.linux_current_task = task
    if current_addr is not None:
        vm.memory.write(current_addr, 64, task)
    print(
        "BOOT_EXEC_USER_TASK_ACTIVATE "
        f"source={source} task=0x{task:x} "
        f"vm_before=0x{previous:x} kernel_before=0x{kernel_previous:x}",
        flush=True,
    )


def install_user_external_surface(vm, user_image, envp: int) -> None:
    image = user_image.image
    if image is None:
        return

    external_prefixes = {
        prefix
        for symbol in (*image.external_data, *image.external_functions)
        if (prefix := _user_external_prefix(symbol)) is not None
    }
    if len(external_prefixes) > 1:
        raise VMError(
            "userspace image mixes external namespaces: "
            + ",".join(sorted(external_prefixes))
        )
    external_prefix = next(iter(external_prefixes), "__mm_user_ext_")

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

    errno_symbol = f"{external_prefix}__errno_cell"
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
        raw_callback = callback

        def callback(
            vm_arg,
            args,
            *,
            _callback=raw_callback,
            _source=original,
        ):
            active_task = int(
                getattr(vm_arg, "active_user_task", 0)
                or getattr(vm_arg, "linux_current_task", 0)
                or 0
            )
            if active_task:
                _activate_user_linux_task(
                    vm_arg,
                    active_task,
                    source=f"external:{_source}",
                )
            return _callback(vm_arg, args)
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


def _arm_preserved_task_transfer(vm, task: int, *, reason: str) -> bool:
    depth = int(getattr(vm, "_preserved_call_depth", 0))
    if depth <= 0:
        return False

    owners = list(getattr(vm, "_preserved_call_tasks", ()))
    if len(owners) != depth:
        # Legacy/unit-test callers may only set the depth flag. Preserve the
        # old all-level unwind behavior in that case.
        stop_depth = 0
    else:
        stop_depth = depth
        while stop_depth > 0 and int(owners[stop_depth - 1]) == int(task):
            stop_depth -= 1
        if stop_depth == depth:
            # The active semantic call belongs to another Linux task. The
            # current task may exec/exit while that caller is blocked; do not
            # unwind the unrelated outer call.
            return False

    vm._preserved_nonreturning_transfer = True
    vm._preserved_transfer_stop_depth = stop_depth
    vm.halted = True
    print(
        "BOOT_EXEC_PRESERVED_CONTROL_TRANSFER "
        f"reason={reason} task=0x{int(task):x} "
        f"depth={depth} stop_depth={stop_depth}",
        flush=True,
    )
    return True


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

        def arm_exit_transfer_if_needed() -> None:
            exiting_task = int(
                getattr(vm, "_user_exit_task", 0) or 0
            )
            if exiting_task and exiting_task == prev:
                status = int(
                    getattr(vm, "_user_exit_status", 0) or 0
                )
                armed = _arm_preserved_task_transfer(
                    vm,
                    prev,
                    reason="exit",
                )
                vm._user_exit_task = 0
                vm._user_exit_status = 0
                print(
                    "BOOT_EXEC_USER_EXIT_SWITCH "
                    f"prev=0x{prev:x} next=0x{next_task:x} "
                    f"status={status} "
                    f"depth={getattr(vm, '_preserved_call_depth', 0)} "
                    f"scoped_transfer={1 if armed else 0}",
                    flush=True,
                )

        # Save where __switch_to() must continue when this task is resumed,
        # together with the system-call result slot carrying the last task.
        saved_prev = (
            vm.memory.read(vm.sp + CALLER_SP, 64),
            vm.memory.read(vm.sp + RET_PC, 64),
            vm.memory.read(vm.sp + RESULT_PTR, 64),
        )
        contexts[prev] = saved_prev
        prev_resume_sp, prev_resume_pc, prev_result_ptr = saved_prev
        prev_code = vm.program.code_block.get(prev_resume_pc)
        print(
            "BOOT_EXEC_TASK_CONTEXT_SAVE "
            f"prev=0x{prev:x} next=0x{next_task:x} "
            f"frame_sp=0x{vm.sp:x} "
            f"resume_sp=0x{prev_resume_sp:x} "
            f"resume_pc=0x{prev_resume_pc:x} "
            f"resume_code={prev_code!r} "
            f"result_ptr=0x{prev_result_ptr:x}",
            flush=True,
        )

        saved = contexts.get(next_task)
        if saved is not None:
            resume_sp, resume_pc, result_ptr = saved
            resume_code = vm.program.code_block.get(resume_pc)
            print(
                "BOOT_EXEC_TASK_CONTEXT_RESTORE "
                f"prev=0x{prev:x} next=0x{next_task:x} "
                f"resume_sp=0x{resume_sp:x} "
                f"resume_pc=0x{resume_pc:x} "
                f"resume_code={resume_code!r} "
                f"result_ptr=0x{result_ptr:x}",
                flush=True,
            )
            vm.memory.write(result_ptr, 64, prev)
            vm.linux_current_task = next_task
            vm.active_user_task = next_task
            vm.sp = resume_sp
            vm.halted = False
            vm._set_code(resume_pc)
            arm_exit_transfer_if_needed()
            return HOST_CONTROL_TRANSFER

        if not fresh_sp:
            raise VMError(
                "first MiniMachine task switch lacks fresh stack: "
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

        vm.linux_current_task = next_task
        vm.active_user_task = next_task
        if not start_fn:
            pending = getattr(vm, "pending_user_fork_continuation", None)
            if pending is not None:
                continuations = getattr(
                    vm, "linux_user_fork_continuations", None
                )
                if continuations is None:
                    continuations = {}
                    vm.linux_user_fork_continuations = continuations
                continuations[next_task] = pending
                vm.pending_user_fork_continuation = None
                print(
                    "BOOT_EXEC_USER_FORK_CHILD_ARMED "
                    f"task=0x{next_task:x}",
                    flush=True,
                )
        print(
            "BOOT_EXEC_TASK_FIRST_RUN "
            f"task=0x{next_task:x} kind={'kernel' if start_fn else 'user'} "
            f"guest_sp=0x{fresh_sp:x} start_fn=0x{start_fn:x}",
            flush=True,
        )
        vm.enter_function(
            "minimachine_ret_from_fork",
            (prev, start_fn, start_arg),
            stack_top=shadow_top,
            result_count=0,
        )
        arm_exit_transfer_if_needed()
        return HOST_CONTROL_TRANSFER

    if service == 3:
        if len(args) != 2:
            raise VMError(
                "Linux user-mode handoff ecall expects service,pt_regs; "
                f"got {len(args)} args"
            )
        _, regs = args

        current_addr = vm.program.symbol_addresses.get(
            "minimachine_current_task"
        )
        kernel_current = (
            int(vm.memory.read(current_addr, 64))
            if current_addr is not None
            else 0
        )
        current_task = int(
            getattr(vm, "linux_current_task", 0)
            or kernel_current
            or 0
        )
        if current_task:
            vm.active_user_task = current_task
            _activate_user_linux_task(
                vm,
                current_task,
                source="userspace-handoff",
            )
            print(
                "BOOT_EXEC_ACTIVE_USER_TASK "
                f"source=userspace-handoff task=0x{current_task:x}",
                flush=True,
            )
        fork_continuations = getattr(
            vm, "linux_user_fork_continuations", {}
        )
        fork_continuation = (
            fork_continuations.pop(current_task, None)
            if current_task else None
        )
        if fork_continuation is not None:
            caller_sp, ret_pc, result_ptr = fork_continuation
            vm.memory.write(result_ptr, 64, 0)
            vm.sp = caller_sp
            vm.halted = False
            vm._set_code(ret_pc)
            print(
                "BOOT_EXEC_USER_FORK_CHILD_RETURN "
                f"task=0x{current_task:x} sp=0x{caller_sp:x} "
                f"pc=0x{ret_pc:x} result_ptr=0x{result_ptr:x}",
                flush=True,
            )
            return HOST_CONTROL_TRANSFER
        pc = vm.memory.read(regs + 0, 64)
        user_sp = vm.memory.read(regs + 8, 64)
        status = vm.memory.read(regs + 80, 64)
        if not (status & 1):
            raise VMError(
                "Linux user-mode handoff received non-user pt_regs: "
                f"regs=0x{regs:x} status=0x{status:x}"
            )

        header = vm.memory.bulk_read(pc, 12)
        if header[:4] != b"MMP3":
            raise VMError(
                "MiniMachine user entry is not an MMP3 payload: "
                f"pc=0x{pc:x} magic={header[:4]!r}"
            )
        size = int.from_bytes(header[8:12], "big")
        if size > 16 * 1024 * 1024:
            raise VMError(f"MiniMachine user payload too large: {size}")
        payload = vm.memory.bulk_read(pc, 12 + size)
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
            if payload == reference_payload:
                mismatch = None
            else:
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
                if mismatch is None:
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
        payload_hash = (
            guest_hash
            if reference_path
            else hashlib.sha256(payload).hexdigest()
        )
        image_cache = getattr(vm, "user_payload_image_cache", None)
        if image_cache is None:
            image_cache = {}
            vm.user_payload_image_cache = image_cache
        slim_user_cache = os.environ.get(
            "MINIMACHINE_USER_IMAGE_CACHE_SLIM",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        user_image = image_cache.get(payload_hash)
        if user_image is None:
            disk_cache_path = os.environ.get("MINIMACHINE_USER_IMAGE_CACHE")
            if slim_user_cache:
                append_cache_dir = getattr(
                    vm,
                    "_append_pack_cache_in_dir",
                    None,
                )
                if not disk_cache_path or append_cache_dir is None:
                    raise VMError(
                        "slim userspace image cache requires a disk cache "
                        "and native append pack cache input"
                    )
            if disk_cache_path:
                try:
                    user_image = load_user_image_cache(
                        Path(disk_cache_path),
                        payload_sha256=payload_hash,
                        require_slim_metadata=slim_user_cache,
                    )
                except UserImageCacheError as exc:
                    raise VMError(
                        f"cannot load MiniMachine userspace image cache: {exc}"
                    ) from exc
                print(
                    "BOOT_EXEC_USER_IMAGE_DISK_CACHE "
                    f"path={disk_cache_path} payload={payload_hash[:16]} hit=1",
                    flush=True,
                )
            else:
                try:
                    user_image = unpack_user_image(payload)
                except UserImageError as exc:
                    raise VMError(f"invalid MiniMachine user payload: {exc}") from exc
            image_cache[payload_hash] = user_image
            image_cache_hit = 0
        else:
            image_cache_hit = 1
        print(
            "BOOT_EXEC_USER_IMAGE_CACHE "
            f"payload={payload_hash[:16]} hit={image_cache_hit} "
            f"entries={len(image_cache)}",
            flush=True,
        )

        instance_table = getattr(vm, "user_exec_instances", None)
        if instance_table is None:
            instance_table = {}
            vm.user_exec_instances = instance_table
        instance_key = (current_task, payload_hash)
        instance = instance_table.get(instance_key)
        reuse_instance = instance is not None
        instance_namespace = (
            str(instance["namespace"])
            if instance is not None and instance.get("namespace")
            else None
        )

        namespace_cache = getattr(vm, "user_namespace_image_cache", None)
        if namespace_cache is None:
            namespace_cache = {}
            vm.user_namespace_image_cache = namespace_cache

        fast_namespace_rebase = True

        if instance_namespace is not None:
            namespace_key = (payload_hash, instance_namespace)
            cached_image = namespace_cache.get(namespace_key)
            if cached_image is None:
                cached_image = rebase_user_program_namespace(
                    user_image,
                    namespace=instance_namespace,
                    fast=fast_namespace_rebase,
                )
                namespace_cache[namespace_key] = cached_image
                namespace_cache_hit = 0
            else:
                namespace_cache_hit = 1
            user_image = cached_image
            print(
                "BOOT_EXEC_USER_NAMESPACE_CACHE "
                f"payload={payload_hash[:16]} namespace={instance_namespace} "
                f"hit={namespace_cache_hit} entries={len(namespace_cache)}",
                flush=True,
            )
        elif not reuse_instance and len(user_image.functions) > 1:
            image_symbols = set()
            if user_image.image is not None:
                image_symbols.update(
                    obj.name for obj in user_image.image.objects
                )
                image_symbols.update(
                    alias.name for alias in user_image.image.aliases
                )
                image_symbols.update(user_image.image.external_data)
                image_symbols.update(user_image.image.external_functions)
            candidate_symbols = {
                function.name for function in user_image.functions
            } | image_symbols
            collisions = sorted(
                symbol
                for symbol in candidate_symbols
                if (
                    symbol in vm.program.functions
                    or symbol in vm.program.symbol_addresses
                )
            )
            if collisions:
                instance_namespace = (
                    f"exec_{current_task:x}_{payload_hash[:12]}"
                )
                namespace_key = (payload_hash, instance_namespace)
                user_image = rebase_user_program_namespace(
                    user_image,
                    namespace=instance_namespace,
                    fast=fast_namespace_rebase,
                )
                namespace_cache[namespace_key] = user_image
                print(
                    "BOOT_EXEC_USER_INSTANCE_NAMESPACE "
                    f"task=0x{current_task:x} "
                    f"payload={payload_hash[:16]} "
                    f"namespace={instance_namespace} "
                    f"collisions={len(collisions)} "
                    f"fast_rebase={int(fast_namespace_rebase)}",
                    flush=True,
                )

        functions = list(user_image.functions)
        entry_name = user_image.entry

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

        # Dynamic P3 userspace descriptors/globals are VM metadata, not
        # Linux-owned RAM. Different Linux tasks need independent instances,
        # while a same-task exec can reuse its instance after restoring the
        # initial data/descriptor bytes.
        kernel_program_data_end = vm.program._next_data
        if reuse_instance:
            user_data_base = int(instance["data_base"])
            user_data_end = int(instance["data_end"])
            reset_size = user_data_end - user_data_base
            initial_blob = bytes(
                vm.program.initial_memory.read(user_data_base + i, 8)
                for i in range(reset_size)
            )
            bulk_write = getattr(vm.memory, "bulk_write", None)
            if bulk_write is not None:
                bulk_write(user_data_base, initial_blob)
            else:
                for offset, byte in enumerate(initial_blob):
                    vm.memory.write(user_data_base + offset, 8, byte)

            if user_image.image is not None:
                for symbol in user_image.image.external_data:
                    if _user_external_original(symbol) == "environ":
                        address = vm.program.symbol_addresses.get(symbol)
                        if address is not None:
                            vm.memory.write(address, 64, user_envp)
            print(
                "BOOT_EXEC_USER_INSTANCE_RESET "
                f"task=0x{current_task:x} "
                f"payload={payload_hash[:16]} "
                f"namespace={instance_namespace or 'base'} "
                f"base=0x{user_data_base:x} end=0x{user_data_end:x} "
                f"bytes={reset_size}",
                flush=True,
            )
        else:
            # The old post-payload arena shares the NOMMU numeric address
            # space with live Linux allocations. Allocate from the VM
            # synthetic heap, which is outside guest physical RAM.
            user_data_base = (int(vm.heap_next) + 15) & ~15
            if user_data_base <= kernel_program_data_end:
                raise VMError(
                    "MiniMachine synthetic userspace P3 arena is not above kernel "
                    f"Program data: base=0x{user_data_base:x} "
                    f"kernel_end=0x{kernel_program_data_end:x}"
                )
            if user_data_base >= vm.stack_top:
                raise VMError(
                    "MiniMachine synthetic userspace P3 arena exhausted: "
                    f"base=0x{user_data_base:x} stack_top=0x{vm.stack_top:x}"
                )
            vm.program._next_data = user_data_base
            print(
                "BOOT_EXEC_USER_DATA_ARENA "
                f"kernel_end=0x{kernel_program_data_end:x} "
                f"base=0x{user_data_base:x} limit=0x{vm.stack_top:x} "
                f"heap_before=0x{vm.heap_next:x} "
                f"capacity={vm.stack_top - user_data_base}",
                flush=True,
            )

            install_user_external_surface(
                vm,
                user_image,
                user_envp,
            )
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

        if not reuse_instance:
            # Preserve the collision-safe legacy behavior for single-function
            # probes. Multi-function programs are already rebased above when
            # another live task owns their base namespace.
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

            if slim_user_cache:
                if any(
                    block.instructions
                    for function in functions
                    for block in function.blocks
                ):
                    raise VMError(
                        "slim userspace image cache unexpectedly contains "
                        "P3 instruction bodies"
                    )
                print(
                    "BOOT_EXEC_USER_SLIM_CACHE "
                    f"payload={payload_hash[:16]} functions={len(functions)}",
                    flush=True,
                )
            else:
                for function in functions:
                    verify_p3(function)
            for function in functions:
                vm.program.add_function(function, verify=False)
            trace_user_external_descriptor(vm, "getcwd", "functions-added")

            if user_image.image is not None:
                image_data_base = vm.program._next_data
                try:
                    install_module_image(vm.program, user_image.image)
                except (ImageError, VMError) as exc:
                    raise VMError(
                        f"cannot install MiniMachine userspace data image: {exc}"
                    ) from exc
                sync_program_initial_range(
                    vm,
                    image_data_base,
                    vm.program._next_data,
                )
                print(
                    "BOOT_EXEC_USER_IMAGE_DATA "
                    f"objects={len(user_image.image.objects)} "
                    f"bytes={user_image.image.byte_size} "
                    f"relocs={user_image.image.relocation_count}",
                    flush=True,
                )
        trace_user_external_descriptor(vm, "getcwd", "module-image")
        if not reuse_instance:
            user_data_end = vm.program._next_data
            if user_data_end >= vm.stack_top:
                raise VMError(
                    "MiniMachine userspace P3 data arena exhausted: "
                    f"end=0x{user_data_end:x} "
                    f"stack_top=0x{vm.stack_top:x}"
                )
            # Future libc malloc/realloc allocations share the same synthetic
            # address region, so advance the VM heap past immutable P3 metadata.
            vm.heap_next = (user_data_end + 15) & ~15
            instance_table[instance_key] = {
                "namespace": instance_namespace,
                "data_base": user_data_base,
                "data_end": user_data_end,
                "entry": entry_name,
            }
            print(
                "BOOT_EXEC_USER_INSTANCE_INSTALLED "
                f"task=0x{current_task:x} "
                f"payload={payload_hash[:16]} "
                f"namespace={instance_namespace or 'base'} "
                f"entry={entry_name}",
                flush=True,
            )
        print(
            "BOOT_EXEC_USER_DATA_ARENA_USED "
            f"base=0x{user_data_base:x} end=0x{user_data_end:x} "
            f"bytes={user_data_end - user_data_base} "
            f"heap_next=0x{vm.heap_next:x} "
            f"remaining={vm.stack_top - vm.heap_next}",
            flush=True,
        )

        if hasattr(vm, "native_append_cache_context"):
            vm.native_append_cache_context = (
                f"{payload_hash}:{instance_namespace or 'base'}:"
                f"{user_data_base:x}:{user_data_end:x}"
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
        if getattr(vm, "_preserved_call_depth", 0):
            # Exec is non-returning only for semantic calls owned by the task
            # being replaced. If this task is running inside another task's
            # blocking wait/vfork call, unwind only the current-task suffix;
            # keep the outer waiter alive so its vm.run() can continue.
            _arm_preserved_task_transfer(
                vm,
                current_task,
                reason=f"exec:{entry_name}",
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
    return dispatch_user_syscall(
        vm,
        args,
        _call_linux_function_preserving_control=_call_linux_function_preserving_control,
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
    gc_disabled = os.environ.get("MINIMACHINE_DISABLE_GC", "").lower() in {
        "1", "true", "yes", "on"
    }
    if gc_disabled:
        gc.disable()
        print(
            "BOOT_EXEC_GC mode=disabled "
            f"thresholds={gc.get_threshold()}",
            flush=True,
        )
    args = parse_args()
    llvm_text = args.input.read_text()
    linked_image_sha256 = image_fingerprint(llvm_text)
    linker_contract = LinkerContract.load(args.linker_contract)

    if args.native_hot_cache_in is not None:
        if not args.native_vm:
            print(
                "BOOT_EXEC_BLOCKED stage=native-hot-cache "
                "error=requires --native-vm"
            )
            return 1
        try:
            cached = load_native_hot_cache(
                args.native_hot_cache_in,
                image_sha256=linked_image_sha256,
            )
        except NativeHotCacheError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=native-hot-cache error={exc}")
            return 1
        program = cached.program
        surface = cached.surface
        reasons = set(cached.reasons)
        blocked_functions = list(cached.blocked_functions)
        image = cached.image
        task_sched_class_offset = cached.task_sched_class_offset
        p3_function_count = cached.function_count
        print(
            "BOOT_EXEC_NATIVE_HOT_CACHE_LOADED "
            f"path={args.native_hot_cache_in} functions={p3_function_count}",
            flush=True,
        )
        print(
            "BOOT_EXEC_STAGE "
            f"stage=native-hot-cache elapsed_s={time.perf_counter() - runner_started:.3f}",
            flush=True,
        )
    elif args.program_cache_in is not None:
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

    checkpoint_layout_only = bool(args.native_vm and args.checkpoint_in is not None)

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
            if checkpoint_layout_only:
                initramfs_size = args.initramfs.stat().st_size
                digest = hashlib.sha256()
                with args.initramfs.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                initramfs_sha256 = digest.hexdigest()
                initramfs_data = None
            else:
                initramfs_data = args.initramfs.read_bytes()
                initramfs_size = len(initramfs_data)
                initramfs_sha256 = hashlib.sha256(initramfs_data).hexdigest()
        except OSError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=initramfs error={exc}")
            return 1
        if initramfs_size <= 0:
            print("BOOT_EXEC_BLOCKED stage=initramfs error=empty image")
            return 1
        try:
            if checkpoint_layout_only:
                initramfs_start = program.reserve_data_symbol(
                    "__initramfs_start",
                    initramfs_size,
                    align=4,
                )
                initramfs_size_addr = program.reserve_data_symbol(
                    "__initramfs_size",
                    8,
                    align=8,
                )
            else:
                assert initramfs_data is not None
                initramfs_start = program.define_data_symbol(
                    "__initramfs_start",
                    initramfs_data,
                    align=4,
                )
                initramfs_size_addr = program.define_data_symbol(
                    "__initramfs_size",
                    initramfs_size.to_bytes(8, "little"),
                    align=8,
                )
        except VMError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=initramfs error={exc}")
            return 1
        print(
            "BOOT_EXEC_INITRAMFS "
            f"path={args.initramfs} bytes={initramfs_size} "
            f"start=0x{initramfs_start:x} size_symbol=0x{initramfs_size_addr:x} "
            f"materialized={int(not checkpoint_layout_only)}",
            flush=True,
        )

    try:
        install_module_image(
            program,
            image,
            symbol_aliases=dict(linker_contract.aliases),
            linker_contract=linker_contract,
            materialize=not checkpoint_layout_only,
        )
        if checkpoint_layout_only:
            print(
                "BOOT_EXEC_IMAGE_LAYOUT "
                f"objects={len(image.objects)} bytes={image.byte_size} "
                f"relocs={image.relocation_count} materialized=0",
                flush=True,
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
        from src.minimachine.native_vm import (
            NativeVM,
            native_pack_cache_key,
        )
        native_init_started = time.perf_counter()
        pack_cache_key = native_pack_cache_key(
            image_sha256=linked_image_sha256,
            initramfs_sha256=initramfs_sha256,
        )
        print(
            "BOOT_EXEC_NATIVE_PACK_CACHE_KEY "
            f"sha256={pack_cache_key}",
            flush=True,
        )
        vm = NativeVM(
            program,
            pack_cache_in=args.native_pack_cache_in,
            pack_cache_out=args.native_pack_cache_out,
            pack_cache_key=pack_cache_key,
            append_pack_cache_in_dir=args.native_append_pack_cache_in_dir,
            append_pack_cache_out_dir=args.native_append_pack_cache_out_dir,
            load_initial_memory=args.checkpoint_in is None,
        )
        vm.native_report_every = max(0, args.native_report_every)
        vm.native_report_slots = tuple(args.native_report_slot)
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
    vm.linux_task_semantic_stacks = {}
    vm.linux_semantic_stack_next = (
        vm.stack_top - _LINUX_SEMANTIC_STACK_START_GAP
    )
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
        if (
            args.inject_initramfs_cpio is not None
            and (args.inject_init is not None or args.inject_file)
        ):
            print(
                "BOOT_EXEC_BLOCKED stage=rootfs-inject "
                "error=choose cpio injection or direct file injection, not both"
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
        elif args.inject_init is not None or args.inject_file:
            try:
                if args.inject_init is not None:
                    inject_live_root_init(
                        vm,
                        args.inject_init,
                        guest_path_text=args.inject_init_path,
                    )
                for host_text, guest_path in args.inject_file:
                    inject_live_root_init(
                        vm,
                        Path(host_text),
                        guest_path_text=guest_path,
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
        trace_memcpy = (
            getattr(vm, "trace_user_read_memcpy_code", None) == code
        )
        original_set_code(code)
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
    host_profile_summary = getattr(vm, "host_profile_summary", None)
    if host_profile_summary is not None:
        for line in host_profile_summary():
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
