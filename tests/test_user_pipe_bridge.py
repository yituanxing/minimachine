from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

from src.minimachine.vm import Program


ROOT = Path(__file__).resolve().parents[1]
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

    def test_non_pipe_external_still_uses_base_runner(self):
        callback = self.runner._user_libc_callback("__mm_user_ext_getpid", None)
        self.assertIsNotNone(callback)


if __name__ == "__main__":
    unittest.main()
