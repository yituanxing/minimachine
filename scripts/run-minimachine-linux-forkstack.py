#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
POSIX_BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-posix.py"


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

    def callback_for(symbol: str, errno_address: int | None):
        callback = base_callback(symbol, errno_address)
        original = runner._user_external_original(symbol)
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
            try:
                return callback(vm, args)
            except Exception:
                if getattr(vm, "pending_user_fork_continuation", None) is None:
                    vm.pending_user_fork_stack_snapshot = None
                raise

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

        return base_linux_ecall(vm, args)

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
