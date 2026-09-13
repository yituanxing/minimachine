from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.minimachine.user_syscall_surface import retry_enosys_linux_syscall


_U64_MASK = (1 << 64) - 1


class UserSyscallSurfaceTests(unittest.TestCase):
    def vm_with(self, *functions: str):
        return SimpleNamespace(
            program=SimpleNamespace(functions={name: object() for name in functions})
        )

    def test_fstat_enosys_retries_through_linux_newfstat(self):
        vm = self.vm_with("__se_sys_newfstat")
        seen = []

        def linux_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            seen.append((name, args, kwargs))
            return (0,)

        result = retry_enosys_linux_syscall(
            vm,
            (80, 3, 0x12340000, 0, 0, 0, 0),
            (-38) & _U64_MASK,
            linux_call=linux_call,
        )
        self.assertEqual(result, 0)
        self.assertEqual(
            seen,
            [("__se_sys_newfstat", (3, 0x12340000), {"result_count": 1})],
        )

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
            (80, 3, 0x1234, 0, 0, 0, 0),
            raw,
            linux_call=lambda *args, **kwargs: self.fail("unexpected replay"),
        )
        self.assertEqual(result, raw)

    def test_control_transfer_is_preserved(self):
        token = object()
        vm = self.vm_with("__se_sys_newfstat")
        result = retry_enosys_linux_syscall(
            vm,
            (80, 3, 0x1234, 0, 0, 0, 0),
            token,
            linux_call=lambda *args, **kwargs: self.fail("unexpected replay"),
            host_control_transfer=token,
        )
        self.assertIs(result, token)


if __name__ == "__main__":
    unittest.main()
