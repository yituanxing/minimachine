from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.minimachine.vm import Program


BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-pipe.py"


def load_bridge():
    spec = spec_from_file_location("run_minimachine_linux_pipe_test", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UserPipeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = load_bridge()
        self.runner = self.bridge.load_runner()
        self.bridge.install_pipe_callbacks(self.runner)
        self.vm = Program().new_vm()
        self.calls = []

        def fake_user_syscall(vm, args):
            self.assertIs(vm, self.vm)
            self.calls.append(args)
            pipefd = int(args[1])
            vm.memory.write(pipefd + 0, 32, 7)
            vm.memory.write(pipefd + 4, 32, 8)
            return 0

        # These two libc-facing tests isolate the ABI mapping. The separate
        # fallback test below validates syscall 59 -> __se_sys_pipe2.
        self.runner.user_syscall = fake_user_syscall

    def test_pipe_maps_to_pipe2_syscall_zero_flags(self):
        pipefd = self.vm.alloc_bytes(8, align=4)
        callback = self.runner._user_libc_callback("__mm_user_ext_pipe", None)
        self.assertIsNotNone(callback)
        assert callback is not None
        self.assertEqual(callback(self.vm, (pipefd,)), 0)
        self.assertEqual(self.calls, [(59, pipefd, 0, 0, 0, 0, 0)])
        self.assertEqual(self.vm.memory.read(pipefd + 0, 32), 7)
        self.assertEqual(self.vm.memory.read(pipefd + 4, 32), 8)

    def test_pipe2_preserves_flags(self):
        pipefd = self.vm.alloc_bytes(8, align=4)
        callback = self.runner._user_libc_callback("__mm_user_ext_pipe2", None)
        self.assertIsNotNone(callback)
        assert callback is not None
        self.assertEqual(callback(self.vm, (pipefd, 0x80800)), 0)
        self.assertEqual(self.calls, [(59, pipefd, 0x80800, 0, 0, 0, 0)])

    def test_pipe2_syscall_uses_guest_kernel_wrapper(self):
        runner = self.bridge.load_runner()
        base_calls = []
        guest_calls = []

        def fake_base(vm, args):
            base_calls.append(args)
            return (1 << 64) - 38

        def fake_call(vm, target, args, **kwargs):
            guest_calls.append((target, args, kwargs))
            return (0,)

        runner.user_syscall = fake_base
        runner._call_linux_function_preserving_control = fake_call
        self.bridge.install_pipe_syscall_fallback(runner)

        vm = Program().new_vm()
        vm.program.functions["__se_sys_pipe2"] = object()
        pipefd = vm.alloc_bytes(8, align=4)
        result = runner.user_syscall(vm, (59, pipefd, 0x80000, 0, 0, 0, 0))

        self.assertEqual(result, 0)
        self.assertEqual(base_calls, [])
        self.assertEqual(len(guest_calls), 1)
        target, args, kwargs = guest_calls[0]
        self.assertEqual(target, "__se_sys_pipe2")
        self.assertEqual(args, (pipefd, 0x80000))
        self.assertEqual(kwargs["result_count"], 1)
        self.assertEqual(kwargs["max_extra_steps"], 8_000_000)

    def test_pipe2_syscall_delegates_when_kernel_wrapper_missing(self):
        runner = self.bridge.load_runner()
        calls = []

        def fake_base(vm, args):
            calls.append(args)
            return (1 << 64) - 38

        runner.user_syscall = fake_base
        self.bridge.install_pipe_syscall_fallback(runner)
        vm = Program().new_vm()
        args = (59, 0x1000, 0, 0, 0, 0, 0)
        self.assertEqual(runner.user_syscall(vm, args), (1 << 64) - 38)
        self.assertEqual(calls, [args])

    def test_non_pipe_external_still_uses_base_runner(self):
        callback = self.runner._user_libc_callback("__mm_user_ext_getpid", None)
        self.assertIsNotNone(callback)


if __name__ == "__main__":
    unittest.main()
