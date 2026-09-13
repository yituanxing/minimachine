from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.user_syscall_surface import retry_enosys_linux_syscall


_U64_MASK = (1 << 64) - 1


class UserSyscallSurfaceTests(unittest.TestCase):
    def vm_with(self, *functions: str):
        return SimpleNamespace(
            program=SimpleNamespace(functions={name: object() for name in functions})
        )

    def assert_retry(self, nr, target, argv):
        vm = self.vm_with(target)
        seen = []

        def linux_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            seen.append((name, args, kwargs))
            return (0,)

        padded = tuple(argv) + (0,) * (6 - len(argv))
        result = retry_enosys_linux_syscall(
            vm,
            (nr, *padded),
            (-38) & _U64_MASK,
            linux_call=linux_call,
        )
        self.assertEqual(result, 0)
        self.assertEqual(
            seen,
            [(target, tuple(argv), {"result_count": 1})],
        )

    def test_stat_syscall_family_routes_to_linux(self):
        dirfd = (-100) & _U64_MASK
        cases = (
            (79, "__se_sys_newfstatat", (dirfd, 0x12341000, 0x12342000, 0x100)),
            (80, "__se_sys_newfstat", (3, 0x12340000)),
        )
        for nr, target, argv in cases:
            with self.subTest(nr=nr, target=target):
                self.assert_retry(nr, target, argv)

    def test_file_lifecycle_syscall_family_routes_to_linux(self):
        dirfd = (-100) & _U64_MASK
        cases = (
            (35, "__se_sys_unlinkat", (dirfd, 0x22000000, 0)),
            (55, "__se_sys_fchown", (4, 1000, 1000)),
        )
        for nr, target, argv in cases:
            with self.subTest(nr=nr, target=target):
                self.assert_retry(nr, target, argv)

    def test_fd_io_syscall_family_routes_to_linux(self):
        cases = (
            (46, "__se_sys_ftruncate", (3, 4096)),
            (67, "__se_sys_pread64", (3, 0x20000000, 100, 0)),
            (68, "__se_sys_pwrite64", (3, 0x20000100, 100, 4096)),
            (82, "__se_sys_fsync", (3,)),
            (83, "__se_sys_fdatasync", (3,)),
        )
        for nr, target, argv in cases:
            with self.subTest(nr=nr, target=target):
                self.assert_retry(nr, target, argv)

    def test_nanosleep_routes_to_linux(self):
        self.assert_retry(101, "__se_sys_nanosleep", (0x23000000, 0x23000020))

    def test_minimachine_arch_opts_into_asm_generic_new_stat(self):
        text = (
            ROOT / "linux-overlay/arch/minimachine/include/asm/unistd.h"
        ).read_text()
        self.assertIn("#define __ARCH_WANT_NEW_STAT", text)

    def test_non_enosys_result_is_not_replayed(self):
        vm = self.vm_with("__se_sys_newfstat")
        calls = []
        result = retry_enosys_linux_syscall(
            vm,
            (80, 3, 0x1234, 0, 0, 0, 0),
            0,
            linux_call=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        self.assertEqual(result, 0)
        self.assertEqual(calls, [])

    def test_unknown_enosys_remains_enosys(self):
        vm = self.vm_with("__se_sys_newfstat")
        raw = (-38) & _U64_MASK
        result = retry_enosys_linux_syscall(
            vm,
            (999, 1, 2, 3, 4, 5, 6),
            raw,
            linux_call=lambda *args, **kwargs: self.fail("unexpected replay"),
        )
        self.assertEqual(result, raw)

    def test_missing_kernel_target_remains_enosys(self):
        vm = self.vm_with()
        raw = (-38) & _U64_MASK
        result = retry_enosys_linux_syscall(
            vm,
            (67, 3, 0x1234, 100, 0, 0, 0),
            raw,
            linux_call=lambda *args, **kwargs: self.fail("unexpected replay"),
        )
        self.assertEqual(result, raw)

    def test_control_transfer_is_preserved(self):
        token = object()
        vm = self.vm_with("__se_sys_pread64")
        result = retry_enosys_linux_syscall(
            vm,
            (67, 3, 0x1234, 100, 0, 0, 0),
            token,
            linux_call=lambda *args, **kwargs: self.fail("unexpected replay"),
            host_control_transfer=token,
        )
        self.assertIs(result, token)


if __name__ == "__main__":
    unittest.main()
