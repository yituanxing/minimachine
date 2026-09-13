#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.user_syscall_surface import retry_enosys_linux_syscall


FORKSTACK_BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-forkstack.py"


def load_forkstack_bridge():
    spec = spec_from_file_location(
        "run_minimachine_linux_syscall_forkstack", FORKSTACK_BRIDGE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load MiniMachine fork-stack bridge: {FORKSTACK_BRIDGE_PATH}"
        )
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_syscall_bridge(runner) -> None:
    """Extend the stage-1 architecture syscall shim with kernel-owned fallbacks."""
    if getattr(runner, "_minimachine_enosys_syscall_bridge_installed", False):
        return

    base_user_syscall = runner.user_syscall
    host_control_transfer = getattr(runner, "HOST_CONTROL_TRANSFER", None)

    def user_syscall(vm, args):
        result = base_user_syscall(vm, args)
        return retry_enosys_linux_syscall(
            vm,
            tuple(int(value) for value in args),
            result,
            linux_call=runner._call_linux_function_preserving_control,
            host_control_transfer=host_control_transfer,
        )

    runner.user_syscall = user_syscall
    runner._minimachine_enosys_syscall_bridge_installed = True


def load_runner():
    bridge = load_forkstack_bridge()
    runner = bridge.load_runner()
    install_syscall_bridge(runner)
    return runner


def main() -> int:
    return load_runner().main()


if __name__ == "__main__":
    raise SystemExit(main())
