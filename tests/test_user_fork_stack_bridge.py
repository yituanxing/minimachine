#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-forkstack.py"


def load_bridge():
    spec = spec_from_file_location("test_minimachine_forkstack_bridge", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bridge: {BRIDGE_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Memory:
    def __init__(self):
        self.values = {}

    def write(self, address, width, value):
        self.values[(int(address), int(width))] = int(value)

    def read(self, address, width):
        return self.values.get((int(address), int(width)), 0)


class ForkStackBridgeTests(unittest.TestCase):
    def test_child_first_user_return_restores_fork_time_stack(self):
        bridge = load_bridge()
        events = []
        fork_snapshot = [(0x1000, b"fork-frame"), (0x2000, b"caller-frame")]

        def external_original(symbol):
            return symbol.rsplit("_ext_", 1)[-1]

        def snapshot(vm):
            events.append(("snapshot", vm.linux_current_task))
            return list(fork_snapshot)

        def restore(vm, frames):
            events.append(("restore", vm.linux_current_task, list(frames)))

        def base_callback(symbol, errno_address):
            original = external_original(symbol)
            if original == "fork":
                def user_fork(vm, args):
                    self.assertEqual(args, ())
                    vm.pending_user_fork_continuation = (0x1110, 0x2220, 0x3330)
                    events.append(("fork-return-parent",))
                    return 17
                return user_fork
            return lambda vm, args: 99

        def base_linux_ecall(vm, args):
            service = int(args[0])
            if service == 2:
                next_task = int(args[2])
                pending = getattr(vm, "pending_user_fork_continuation", None)
                if pending is not None:
                    vm.linux_user_fork_continuations[next_task] = pending
                    vm.pending_user_fork_continuation = None
                events.append(("base-service2", next_task))
                return "switch"
            if service == 3:
                events.append(("base-service3", vm.linux_current_task))
                return "user"
            raise AssertionError(args)

        runner = SimpleNamespace(
            _user_libc_callback=base_callback,
            _user_external_original=external_original,
            _snapshot_p3_call_chain=snapshot,
            _restore_p3_call_chain=restore,
            linux_ecall=base_linux_ecall,
        )
        bridge.install_fork_stack_bridge(runner)

        vm = SimpleNamespace(
            linux_current_task=0x1000,
            pending_user_fork_continuation=None,
            linux_user_fork_continuations={},
        )

        fork_cb = runner._user_libc_callback("__mm_user_ext_fork", None)
        self.assertEqual(fork_cb(vm, ()), 17)
        self.assertEqual(vm.pending_user_fork_stack_snapshot, fork_snapshot)

        self.assertEqual(
            runner.linux_ecall(vm, (2, 0x1000, 0x2000, 0x3000, 0, 0)),
            "switch",
        )
        self.assertIsNone(vm.pending_user_fork_stack_snapshot)
        self.assertEqual(vm.linux_user_fork_stack_snapshots[0x2000], fork_snapshot)

        vm.linux_current_task = 0x2000
        self.assertEqual(runner.linux_ecall(vm, (3, 0xDEAD)), "user")
        self.assertNotIn(0x2000, vm.linux_user_fork_stack_snapshots)

        restore_index = next(
            index for index, event in enumerate(events) if event[0] == "restore"
        )
        service3_index = next(
            index for index, event in enumerate(events) if event[0] == "base-service3"
        )
        self.assertLess(restore_index, service3_index)
        self.assertEqual(events[restore_index][2], fork_snapshot)

    def test_restart_errno_drains_sigchld_then_retries_fork(self):
        bridge = load_bridge()
        errno_address = 0x3000
        memory = Memory()
        calls = []
        syscalls = []
        fork_snapshot = [(0x1000, b"fork-frame")]

        def external_original(symbol):
            return symbol.rsplit("_ext_", 1)[-1]

        def base_callback(symbol, callback_errno_address):
            self.assertEqual(callback_errno_address, errno_address)

            def user_fork(vm, args):
                calls.append(args)
                if len(calls) == 1:
                    vm.pending_user_fork_continuation = None
                    vm.memory.write(errno_address, 32, 513)
                    return (1 << 64) - 1
                vm.pending_user_fork_continuation = (0x1110, 0x2220, 0x3330)
                return 23

            return user_fork

        def user_syscall(vm, args):
            syscalls.append(tuple(args))
            self.assertEqual(args[0], 137)
            sigset_addr = args[1]
            timeout_addr = args[3]
            self.assertEqual(vm.memory.read(sigset_addr, 64), 1 << 16)
            self.assertEqual(vm.memory.read(timeout_addr, 64), 0)
            self.assertEqual(vm.memory.read(timeout_addr + 8, 64), 0)
            self.assertEqual(args[4], 8)
            return 17

        runner = SimpleNamespace(
            _user_libc_callback=base_callback,
            _user_external_original=external_original,
            _snapshot_p3_call_chain=lambda vm: list(fork_snapshot),
            _restore_p3_call_chain=lambda vm, frames: None,
            linux_ecall=lambda vm, args: None,
            user_syscall=user_syscall,
            HOST_CONTROL_TRANSFER=object(),
        )
        bridge.install_fork_stack_bridge(runner)
        vm = SimpleNamespace(
            linux_current_task=0x1000,
            pending_user_fork_continuation=None,
            linux_user_fork_continuations={},
            memory=memory,
            heap_next=0x4003,
        )

        fork_cb = runner._user_libc_callback("__mm_user_ext_fork", errno_address)
        self.assertEqual(fork_cb(vm, ()), 23)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(syscalls), 1)
        self.assertEqual(memory.read(errno_address, 32), 0)
        self.assertEqual(vm.pending_user_fork_stack_snapshot, fork_snapshot)
        self.assertEqual(vm.pending_user_fork_continuation, (0x1110, 0x2220, 0x3330))
        self.assertEqual(vm.heap_next, 0x4028)

    def test_non_fork_callback_is_unchanged(self):
        bridge = load_bridge()
        callback = lambda vm, args: 7
        runner = SimpleNamespace(
            _user_libc_callback=lambda symbol, errno_address: callback,
            _user_external_original=lambda symbol: "write",
            _snapshot_p3_call_chain=lambda vm: [],
            _restore_p3_call_chain=lambda vm, frames: None,
            linux_ecall=lambda vm, args: None,
        )
        bridge.install_fork_stack_bridge(runner)
        self.assertIs(runner._user_libc_callback("__mm_user_ext_write", None), callback)


if __name__ == "__main__":
    unittest.main()
