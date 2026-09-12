#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PIPE_RUNNER = ROOT / "scripts" / "run-minimachine-linux-pipe.py"


def load_pipe_runner():
    spec = spec_from_file_location("run_minimachine_linux_pipe_for_forkvfork", PIPE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pipe runner: {PIPE_RUNNER}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_fork_vfork_compat(runner) -> None:
    base_callback = runner._user_libc_callback

    def callback_for(symbol: str, errno_address: int | None):
        original = runner._user_external_original(symbol)
        if original != "fork":
            return base_callback(symbol, errno_address)

        prefix = runner._user_external_prefix(symbol) or "__mm_user_ext_"
        print(
            "BOOT_EXEC_USER_FORK_COMPAT mode=vfork reason=shared-p3-stack",
            flush=True,
        )
        return base_callback(prefix + "vfork", errno_address)

    runner._user_libc_callback = callback_for


def main() -> int:
    bridge = load_pipe_runner()
    runner = bridge.load_runner()
    bridge.install_pipe_callbacks(runner)
    install_fork_vfork_compat(runner)
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
