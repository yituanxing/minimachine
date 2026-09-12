#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPE_BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-pipe.py"
AT_FDCWD = (1 << 64) - 100
AT_REMOVEDIR = 0x200


def load_pipe_bridge():
    spec = spec_from_file_location("run_minimachine_linux_posix_pipe", PIPE_BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MiniMachine pipe bridge: {PIPE_BRIDGE_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _libc_result(vm, raw, errno_address: int | None):
    if raw is None:
        return raw
    signed = raw - (1 << 64) if raw & (1 << 63) else raw
    if -4095 <= signed < 0:
        if errno_address is not None:
            vm.memory.write(errno_address, 32, (-signed) & 0xFFFFFFFF)
        return (1 << 64) - 1
    return raw


def install_posix_callbacks(runner) -> None:
    base_callback = runner._user_libc_callback

    syscall_specs = {
        "fstat64": (80, 2),
        "fsync": (82, 1),
        "ftruncate64": (46, 2),
        "pread64": (67, 4),
        "pwrite64": (68, 4),
        "mmap64": (222, 6),
        "munmap": (215, 2),
        "fchmod": (52, 2),
        "fchown": (55, 3),
    }

    def callback_for(symbol: str, errno_address: int | None):
        original = runner._user_external_original(symbol)

        if original in syscall_specs:
            nr, argc = syscall_specs[original]

            def direct_syscall(vm, args):
                if len(args) != argc:
                    raise runner.VMError(
                        f"{original} expects {argc} arguments; got {len(args)}"
                    )
                padded = tuple(int(x) for x in args) + (0,) * (6 - len(args))
                raw = runner.user_syscall(vm, (nr, *padded[:6]))
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)

            return direct_syscall

        if original == "unlink":
            def user_unlink(vm, args):
                if len(args) != 1:
                    raise runner.VMError("unlink expects path")
                raw = runner.user_syscall(
                    vm, (35, AT_FDCWD, int(args[0]), 0, 0, 0, 0)
                )
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)
            return user_unlink

        if original == "mkdir":
            def user_mkdir(vm, args):
                if len(args) != 2:
                    raise runner.VMError("mkdir expects path,mode")
                raw = runner.user_syscall(
                    vm, (34, AT_FDCWD, int(args[0]), int(args[1]), 0, 0, 0)
                )
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)
            return user_mkdir

        if original == "rmdir":
            def user_rmdir(vm, args):
                if len(args) != 1:
                    raise runner.VMError("rmdir expects path")
                raw = runner.user_syscall(
                    vm, (35, AT_FDCWD, int(args[0]), AT_REMOVEDIR, 0, 0, 0)
                )
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)
            return user_rmdir

        if original == "readlink":
            def user_readlink(vm, args):
                if len(args) != 3:
                    raise runner.VMError("readlink expects path,buf,size")
                raw = runner.user_syscall(
                    vm,
                    (78, AT_FDCWD, int(args[0]), int(args[1]), int(args[2]), 0, 0),
                )
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)
            return user_readlink

        if original == "symlink":
            def user_symlink(vm, args):
                if len(args) != 2:
                    raise runner.VMError("symlink expects target,linkpath")
                raw = runner.user_syscall(
                    vm,
                    (36, int(args[0]), AT_FDCWD, int(args[1]), 0, 0, 0),
                )
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)
            return user_symlink

        if original == "chmod":
            def user_chmod(vm, args):
                if len(args) != 2:
                    raise runner.VMError("chmod expects path,mode")
                raw = runner.user_syscall(
                    vm,
                    (53, AT_FDCWD, int(args[0]), int(args[1]), 0, 0, 0),
                )
                if raw is runner.HOST_CONTROL_TRANSFER:
                    return raw
                return _libc_result(vm, raw, errno_address)
            return user_chmod

        return base_callback(symbol, errno_address)

    runner._user_libc_callback = callback_for


def load_runner():
    bridge = load_pipe_bridge()
    runner = bridge.load_runner()
    bridge.install_pipe_callbacks(runner)
    install_posix_callbacks(runner)
    return runner


def main() -> int:
    return load_runner().main()


if __name__ == "__main__":
    raise SystemExit(main())
