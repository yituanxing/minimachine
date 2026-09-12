#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
POSIX_BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-posix.py"
_U64_MASK = (1 << 64) - 1
_RESTART_ERRNOS = {512, 513, 514, 516}
_SIGCHLD = 17
_SIGSET_BYTES = 8
_EXECVE_NR = 221
_EXECVE_PRESERVED_STEPS = 64_000_000
_WAIT_SCHEDULE_BASE_STEPS = 12_000_000
_WAIT_SCHEDULE_PRESERVED_STEPS = 180_000_000


def load_posix_bridge():
    spec = spec_from_file_location(
        "run_minimachine_linux_forkstack_posix", POSIX_BRIDGE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MiniMachine POSIX bridge: {POSIX_BRIDGE_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _current_task(runner, vm) -> int:
    current = int(getattr(vm, "linux_current_task", 0) or 0)
    if current:
        return current
    current_addr = vm.program.symbol_addresses.get("minimachine_current_task")
    if current_addr is None:
        return 0
    return int(vm.memory.read(current_addr, 64))


def _signed_u64(value: int) -> int:
    value = int(value) & _U64_MASK
    return value - (1 << 64) if value & (1 << 63) else value


def _drain_pending_sigchld(runner, vm) -> bool:
    """Consume one pending SIGCHLD before retrying a direct clone syscall.

    The base runner invokes ``__se_sys_clone`` directly instead of traversing
    the architecture syscall-exit path.  Linux copy_process() deliberately
    returns -ERESTARTNOINTR when TIF_SIGPENDING is set so that syscall-exit can
    deliver/dequeue the signal before restarting fork.  A completed child can
    therefore leave SIGCHLD pending across our semantic wait4 call and make a
    later fork restart forever.  Use Linux's own rt_sigtimedwait syscall with a
    zero timeout to dequeue exactly SIGCHLD; never turn ERESTART* into success.
    """
    base = (int(vm.heap_next) + 15) & ~15
    sigset_addr = base
    timeout_addr = base + 8
    vm.heap_next = base + 24

    vm.memory.write(sigset_addr, 64, 1 << (_SIGCHLD - 1))
    vm.memory.write(timeout_addr, 64, 0)
    vm.memory.write(timeout_addr + 8, 64, 0)

    raw = runner.user_syscall(
        vm,
        (137, sigset_addr, 0, timeout_addr, _SIGSET_BYTES, 0, 0),
    )
    if raw is getattr(runner, "HOST_CONTROL_TRANSFER", None) or raw is None:
        print(
            "BOOT_EXEC_USER_FORK_SIGNAL_DRAIN sig=17 result=control-transfer",
            flush=True,
        )
        return False

    signed = _signed_u64(int(raw))
    print(
        "BOOT_EXEC_USER_FORK_SIGNAL_DRAIN "
        f"sig=17 result={signed} mask=0x{sigset_addr:x} timeout=0x{timeout_addr:x}",
        flush=True,
    )
    return signed == _SIGCHLD


def install_fork_stack_bridge(runner) -> None:
    """Preserve the fork-time P3 stack until the child first returns to user mode.

    MiniMachine NOMMU fork uses CLONE_VM, so parent and child intentionally share
    guest memory.  The P3 continuation, however, must still observe the caller
    frames as they existed at fork return.  The base runner keeps only the
    continuation addresses; if the parent reaches waitpid first, those shared
    stack addresses can be reused before the child consumes them.
    """
    if getattr(runner, "_minimachine_fork_stack_bridge_installed", False):
        return

    base_callback = runner._user_libc_callback
    base_linux_ecall = runner.linux_ecall
    base_preserved_call = getattr(
        runner, "_call_linux_function_preserving_control", None
    )

    def preserved_call(vm, name, args, **kwargs):
        if name == "minimachine_user_syscall" and args and int(args[0]) == _EXECVE_NR:
            old_budget = int(kwargs.get("max_extra_steps", 0) or 0)
            if old_budget < _EXECVE_PRESERVED_STEPS:
                kwargs["max_extra_steps"] = _EXECVE_PRESERVED_STEPS
                print(
                    "BOOT_EXEC_USER_EXECVE_BUDGET "
                    f"old={old_budget} new={_EXECVE_PRESERVED_STEPS}",
                    flush=True,
                )
        elif name == "schedule":
            old_budget = int(kwargs.get("max_extra_steps", 0) or 0)
            if old_budget == _WAIT_SCHEDULE_BASE_STEPS:
                kwargs["max_extra_steps"] = _WAIT_SCHEDULE_PRESERVED_STEPS
                print(
                    "BOOT_EXEC_USER_WAIT_SCHEDULE_BUDGET "
                    f"old={old_budget} new={_WAIT_SCHEDULE_PRESERVED_STEPS}",
                    flush=True,
                )
        return base_preserved_call(vm, name, args, **kwargs)

    def callback_for(symbol: str, errno_address: int | None):
        callback = base_callback(symbol, errno_address)
        original = runner._user_external_original(symbol)

        if original == "atexit":
            def user_atexit(vm, args):
                if len(args) != 1:
                    raise runner.VMError("atexit expects one function pointer")
                handler = int(args[0])
                handlers = getattr(vm, "user_atexit_handlers", None)
                if handlers is None:
                    handlers = []
                    vm.user_atexit_handlers = handlers
                handlers.append(handler)
                print(
                    "BOOT_EXEC_USER_ATEXIT_REGISTER "
                    f"handler=0x{handler:x} count={len(handlers)}",
                    flush=True,
                )
                return 0

            return user_atexit

        if original not in {"fork", "vfork"} or callback is None:
            return callback

        def user_fork_with_snapshot(vm, args):
            snapshot = runner._snapshot_p3_call_chain(vm)
            vm.pending_user_fork_stack_snapshot = snapshot
            print(
                "BOOT_EXEC_USER_FORK_STACK_SNAPSHOT "
                f"kind={original} frames={len(snapshot)} "
                f"bytes={sum(len(payload) for _, payload in snapshot)}",
                flush=True,
            )
            signal_retry = 0
            while True:
                vm.pending_user_fork_stack_snapshot = snapshot
                try:
                    result = callback(vm, args)
                except Exception:
                    if getattr(vm, "pending_user_fork_continuation", None) is None:
                        vm.pending_user_fork_stack_snapshot = None
                    raise

                if result != _U64_MASK or errno_address is None:
                    if getattr(vm, "pending_user_fork_continuation", None) is None:
                        vm.pending_user_fork_stack_snapshot = None
                    return result

                errno_value = int(vm.memory.read(errno_address, 32))
                if errno_value not in _RESTART_ERRNOS or signal_retry >= 3:
                    if getattr(vm, "pending_user_fork_continuation", None) is None:
                        vm.pending_user_fork_stack_snapshot = None
                    return result

                if not _drain_pending_sigchld(runner, vm):
                    if getattr(vm, "pending_user_fork_continuation", None) is None:
                        vm.pending_user_fork_stack_snapshot = None
                    return result

                signal_retry += 1
                vm.memory.write(errno_address, 32, 0)
                print(
                    "BOOT_EXEC_USER_FORK_SIGNAL_RESTART "
                    f"kind={original} attempt={signal_retry} errno={errno_value}",
                    flush=True,
                )

        return user_fork_with_snapshot

    def linux_ecall(vm, args: tuple[int, ...]):
        service = int(args[0]) if args else -1

        if service == 2 and len(args) == 6:
            next_task = int(args[2])
            start_fn = int(args[4])
            snapshot = getattr(vm, "pending_user_fork_stack_snapshot", None)
            result = base_linux_ecall(vm, args)
            if not start_fn and snapshot is not None:
                continuations = getattr(vm, "linux_user_fork_continuations", {})
                if next_task in continuations:
                    snapshots = getattr(vm, "linux_user_fork_stack_snapshots", None)
                    if snapshots is None:
                        snapshots = {}
                        vm.linux_user_fork_stack_snapshots = snapshots
                    snapshots[next_task] = snapshot
                    vm.pending_user_fork_stack_snapshot = None
                    print(
                        "BOOT_EXEC_USER_FORK_STACK_ARMED "
                        f"task=0x{next_task:x} frames={len(snapshot)}",
                        flush=True,
                    )
            return result

        if service == 3:
            current_task = _current_task(runner, vm)
            snapshots = getattr(vm, "linux_user_fork_stack_snapshots", None)
            snapshot = snapshots.pop(current_task, None) if snapshots else None
            if snapshot is not None:
                runner._restore_p3_call_chain(vm, snapshot)
                print(
                    "BOOT_EXEC_USER_FORK_STACK_RESTORED "
                    f"task=0x{current_task:x} frames={len(snapshot)} "
                    f"bytes={sum(len(payload) for _, payload in snapshot)}",
                    flush=True,
                )

            result = base_linux_ecall(vm, args)

            # register_service/register_system mutate Program.initial_memory.
            # On a live checkpoint-restored VM those immutable descriptors are
            # not automatically copied into the concrete guest/native memory.
            # Refresh after each real userspace handoff so dynamically added
            # runtime helpers and pre-existing system services retain valid
            # descriptor entries before userspace executes them.
            fence_descriptor = vm.program.symbol_addresses.get("__mm_sys_fence")
            fence_initial = (
                vm.program.initial_memory.read(fence_descriptor, 64)
                if fence_descriptor is not None
                else 0
            )
            fence_before = (
                vm.memory.read(fence_descriptor, 64)
                if fence_descriptor is not None
                else 0
            )
            fence_registered = int("__mm_sys_fence" in vm.program.host_services)
            runner.refresh_host_service_descriptors(vm)
            fence_after = (
                vm.memory.read(fence_descriptor, 64)
                if fence_descriptor is not None
                else 0
            )
            print(
                "BOOT_EXEC_USER_HOST_DESCRIPTOR_REFRESH "
                f"fence_descriptor={f'0x{fence_descriptor:x}' if fence_descriptor is not None else 'missing'} "
                f"fence_registered={fence_registered} "
                f"initial=0x{fence_initial:x} before=0x{fence_before:x} "
                f"after=0x{fence_after:x}",
                flush=True,
            )
            return result

        return base_linux_ecall(vm, args)

    if base_preserved_call is not None:
        runner._call_linux_function_preserving_control = preserved_call
    runner._user_libc_callback = callback_for
    runner.linux_ecall = linux_ecall
    runner._minimachine_fork_stack_bridge_installed = True


def load_runner():
    bridge = load_posix_bridge()
    runner = bridge.load_runner()
    install_fork_stack_bridge(runner)
    return runner


def main() -> int:
    return load_runner().main()


if __name__ == "__main__":
    raise SystemExit(main())
