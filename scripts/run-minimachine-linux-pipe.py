#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run-minimachine-linux.py"


def load_runner():
    spec = spec_from_file_location("run_minimachine_linux_pipe_base", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MiniMachine Linux runner: {RUNNER_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_pipe_callbacks(runner) -> None:
    base_callback = runner._user_libc_callback

    def callback_for(symbol: str, errno_address: int | None):
        original = runner._user_external_original(symbol)
        if original not in {"pipe", "pipe2"}:
            return base_callback(symbol, errno_address)

        def user_pipe(vm, args):
            expected = 1 if original == "pipe" else 2
            if len(args) != expected:
                raise runner.VMError(
                    f"{original} expects pipefd"
                    + (",flags" if original == "pipe2" else "")
                )
            pipefd = int(args[0])
            flags = int(args[1]) if original == "pipe2" else 0
            # RISC-V follows asm-generic: there is no legacy __NR_pipe;
            # libc pipe() is pipe2(pipefd, 0), syscall number 59.
            raw = runner.user_syscall(
                vm,
                (59, pipefd, flags, 0, 0, 0, 0),
            )
            if raw is runner.HOST_CONTROL_TRANSFER:
                return raw
            return runner.libc_linux_result(vm, raw)

        return user_pipe

    runner._user_libc_callback = callback_for


def main() -> int:
    runner = load_runner()
    install_pipe_callbacks(runner)
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
