from __future__ import annotations

import unittest

from src.minimachine.passwd_surface import resolve_passwd_callback
from src.minimachine.vm import Program, VMError


def callback(name: str):
    result = resolve_passwd_callback(name, vm_error=VMError)
    assert result is not None
    return result


def read_cstring(vm, address: int) -> bytes:
    data = bytearray()
    while True:
        byte = vm.memory.read(address + len(data), 8)
        if byte == 0:
            return bytes(data)
        data.append(byte)


class PasswdSurfaceTests(unittest.TestCase):
    def test_root_record_uses_guest_memory_and_exact_layout(self):
        vm = Program().new_vm()
        record = callback("getpwuid")(vm, (0,))

        self.assertNotEqual(record, 0)
        self.assertEqual(read_cstring(vm, vm.memory.read(record + 0, 64)), b"root")
        self.assertEqual(read_cstring(vm, vm.memory.read(record + 8, 64)), b"x")
        self.assertEqual(vm.memory.read(record + 16, 32), 0)
        self.assertEqual(vm.memory.read(record + 20, 32), 0)
        self.assertEqual(read_cstring(vm, vm.memory.read(record + 24, 64)), b"root")
        self.assertEqual(read_cstring(vm, vm.memory.read(record + 32, 64)), b"/root")
        self.assertEqual(read_cstring(vm, vm.memory.read(record + 40, 64)), b"/bin/sh")

    def test_root_record_has_stable_static_storage(self):
        vm = Program().new_vm()
        first = callback("getpwuid")(vm, (0,))
        heap_after_first = vm.heap_next
        second = callback("getpwuid")(vm, (0,))
        self.assertEqual(second, first)
        self.assertEqual(vm.heap_next, heap_after_first)

    def test_unknown_uid_returns_null(self):
        vm = Program().new_vm()
        self.assertEqual(callback("getpwuid")(vm, (12345,)), 0)
        self.assertFalse(hasattr(vm, "user_passwd_records"))

    def test_uid_argument_uses_uid_t_width(self):
        vm = Program().new_vm()
        self.assertNotEqual(callback("getpwuid")(vm, (1 << 32,)), 0)

    def test_bad_arity_and_unrelated_surface_are_not_hidden(self):
        vm = Program().new_vm()
        with self.assertRaisesRegex(VMError, "expects uid"):
            callback("getpwuid")(vm, ())
        self.assertIsNone(resolve_passwd_callback("getpwnam", vm_error=VMError))


if __name__ == "__main__":
    unittest.main()
