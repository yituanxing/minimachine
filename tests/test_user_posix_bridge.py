from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.minimachine.vm import Program


BRIDGE_PATH = ROOT / "scripts" / "run-minimachine-linux-posix.py"


def load_bridge():
    spec = spec_from_file_location("run_minimachine_linux_posix_test", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UserPosixBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = load_bridge()
        self.runner = self.bridge.load_runner()
        self.vm = Program().new_vm()
        self.calls = []

        def fake_user_syscall(vm, args):
            self.assertIs(vm, self.vm)
            self.calls.append(args)
            return 0

        self.runner.user_syscall = fake_user_syscall

    def callback(self, name):
        callback = self.runner._user_libc_callback(
            f"__mm_sqlite3534_ext_{name}", None
        )
        self.assertIsNotNone(callback)
        return callback

    def test_direct_vfs_syscall_numbers(self):
        cases = [
            ("fstat64", (3, 0x1000), 80),
            ("fsync", (3,), 82),
            ("ftruncate64", (3, 4096), 46),
            ("pread64", (3, 0x2000, 64, 12), 67),
            ("pwrite64", (3, 0x2000, 64, 12), 68),
            ("mmap64", (0, 4096, 3, 2, 3, 0), 222),
            ("munmap", (0x400000, 4096), 215),
            ("fchmod", (3, 0o600), 52),
            ("fchown", (3, 1000, 1000), 55),
        ]
        for name, args, nr in cases:
            with self.subTest(name=name):
                self.calls.clear()
                self.assertEqual(self.callback(name)(self.vm, args), 0)
                self.assertEqual(self.calls[0][0], nr)
                self.assertEqual(self.calls[0][1:1 + len(args)], args)
                self.assertEqual(len(self.calls[0]), 7)

    def test_at_fdcwd_wrappers(self):
        at_fdcwd = (1 << 64) - 100
        cases = [
            ("unlink", (0x1000,), (35, at_fdcwd, 0x1000, 0, 0, 0, 0)),
            ("mkdir", (0x1000, 0o755), (34, at_fdcwd, 0x1000, 0o755, 0, 0, 0)),
            ("rmdir", (0x1000,), (35, at_fdcwd, 0x1000, 0x200, 0, 0, 0)),
            ("readlink", (0x1000, 0x2000, 128), (78, at_fdcwd, 0x1000, 0x2000, 128, 0, 0)),
            ("symlink", (0x1000, 0x2000), (36, 0x1000, at_fdcwd, 0x2000, 0, 0, 0)),
            ("chmod", (0x1000, 0o600), (53, at_fdcwd, 0x1000, 0o600, 0, 0, 0)),
        ]
        for name, args, expected in cases:
            with self.subTest(name=name):
                self.calls.clear()
                self.assertEqual(self.callback(name)(self.vm, args), 0)
                self.assertEqual(self.calls, [expected])

    def test_pipe_is_inherited(self):
        pipefd = self.vm.alloc_bytes(8, align=4)
        callback = self.callback("pipe")
        self.assertEqual(callback(self.vm, (pipefd,)), 0)
        self.assertEqual(self.calls[0][0], 59)


if __name__ == "__main__":
    unittest.main()
