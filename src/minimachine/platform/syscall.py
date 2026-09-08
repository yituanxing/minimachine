from __future__ import annotations

import os
import time

from ..vm import HOST_CONTROL_TRANSFER, VMError


def dispatch_user_syscall(
    vm,
    args: tuple[int, ...],
    *,
    _call_linux_function_preserving_control,
):
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
        29: ("__se_sys_ioctl", 3),
        49: ("__se_sys_chdir", 1),
        56: ("__se_sys_openat", 4),
        57: ("__se_sys_close", 1),
        61: ("__se_sys_getdents64", 3),
        63: ("__se_sys_read", 3),
        64: ("__se_sys_write", 3),
        93: ("__se_sys_exit", 1),
        94: ("__se_sys_exit_group", 1),
        142: ("__se_sys_reboot", 4),
        153: ("__se_sys_times", 1),
        157: ("sys_setsid", 0),
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
        260: ("__se_sys_wait4", 4),
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
            call_result = _call_linux_function_preserving_control(
                vm,
                "minimachine_user_syscall",
                tuple(args),
                result_count=1,
                max_extra_steps=8_000_000,
            )
            if call_result is HOST_CONTROL_TRANSFER:
                return HOST_CONTROL_TRANSFER
            result, = call_result
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
                call_result = _call_linux_function_preserving_control(
                    vm,
                    target,
                    tuple(argv[:argc]),
                    result_count=1,
                    max_extra_steps=8_000_000,
                    # wait4 may block and schedule another Linux task. Its
                    # task/current/context mutations are the syscall's real
                    # semantics and must survive the semantic-call wrapper.
                    preserve_linux_task_state=(nr == 260),
                )
                if call_result is HOST_CONTROL_TRANSFER:
                    print(
                        "BOOT_EXEC_USER_SYSCALL_FALLBACK_TRANSFER "
                        f"nr={nr} target={target}",
                        flush=True,
                    )
                    return HOST_CONTROL_TRANSFER
                result, = call_result
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

    signed_result = result - (1 << 64) if result & (1 << 63) else result
    active_task = int(
        getattr(vm, "active_user_task", 0)
        or getattr(vm, "linux_current_task", 0)
        or 0
    )
    if active_task and signed_result >= 0 and nr in {172, 173}:
        attr = "user_task_pids" if nr == 172 else "user_task_parent_pids"
        table = getattr(vm, attr, None)
        if table is None:
            table = {}
            setattr(vm, attr, table)
        table[active_task] = int(signed_result)

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
