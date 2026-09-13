from __future__ import annotations


_U64_MASK = (1 << 64) - 1
_ENOSYS = 38

# asm-generic/RISC-V syscall numbers which are intentionally absent from the
# first-stage minimachine_user_syscall switch but have a real Linux syscall
# implementation in the linked kernel. Keep the fallback here rather than
# reimplementing kernel ABI structures in the host.
ENOSYS_LINUX_FALLBACKS: dict[int, tuple[str, int]] = {
    80: ("__se_sys_newfstat", 2),
}


def _signed_u64(value: int) -> int:
    value = int(value) & _U64_MASK
    return value - (1 << 64) if value & (1 << 63) else value


def retry_enosys_linux_syscall(
    vm,
    args: tuple[int, ...],
    result,
    *,
    linux_call,
    host_control_transfer=None,
):
    """Retry a stage-1 -ENOSYS through the linked Linux syscall body.

    ``minimachine_user_syscall`` intentionally started with a very small
    syscall switch. Real userspace can therefore reach a syscall whose
    implementation is already present in the linked Linux image while the
    architecture shim still returns ``-ENOSYS``. In that case only, dispatch
    to Linux's own ``__se_sys_*`` wrapper so argument validation, guest ABI
    structures, VFS semantics, and errno behavior remain kernel-owned.
    """
    if result is None or result is host_control_transfer:
        return result
    if _signed_u64(int(result)) != -_ENOSYS or not args:
        return result

    nr = int(args[0])
    fallback = ENOSYS_LINUX_FALLBACKS.get(nr)
    if fallback is None:
        print(
            "BOOT_EXEC_USER_SYSCALL_ENOSYS_UNRESOLVED "
            f"nr={nr} reason=no-fallback",
            flush=True,
        )
        return result

    target, argc = fallback
    program = getattr(vm, "program", None)
    functions = getattr(program, "functions", {}) if program is not None else {}
    if target not in functions:
        print(
            "BOOT_EXEC_USER_SYSCALL_ENOSYS_UNRESOLVED "
            f"nr={nr} reason=missing-target target={target}",
            flush=True,
        )
        return result

    call_args = tuple(int(value) for value in args[1 : 1 + argc])
    print(
        "BOOT_EXEC_USER_SYSCALL_ENOSYS_FALLBACK "
        f"nr={nr} target={target} argc={argc}",
        flush=True,
    )
    retried = linux_call(
        vm,
        target,
        call_args,
        result_count=1,
    )
    if retried is None or retried is host_control_transfer:
        return retried
    if len(retried) != 1:
        raise RuntimeError(
            f"{target} returned {len(retried)} values, expected one"
        )
    value = int(retried[0]) & _U64_MASK
    print(
        "BOOT_EXEC_USER_SYSCALL_ENOSYS_RETRY "
        f"nr={nr} target={target} result={_signed_u64(value)}",
        flush=True,
    )
    return value
