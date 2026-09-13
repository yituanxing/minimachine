from __future__ import annotations

import unittest

from src.minimachine.ctype_surface import resolve_ctype_callback
from src.minimachine.vm import Program, VMError


def callback(name: str):
    result = resolve_ctype_callback(name, vm_error=VMError)
    assert result is not None
    return result


def signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & (1 << 31) else value


class CtypeSurfaceTests(unittest.TestCase):
    def test_classification_table_matches_frozen_sqlite_masks(self):
        vm = Program().new_vm()
        cell = callback("__ctype_b_loc")(vm, ())
        table = vm.memory.read(cell, 64)

        self.assertTrue(vm.memory.read(table + ord(" ") * 2, 16) & 0x2000)
        self.assertTrue(vm.memory.read(table + ord("\t") * 2, 16) & 0x2000)
        self.assertTrue(vm.memory.read(table + ord("C") * 2, 16) & 0x0100)
        self.assertTrue(vm.memory.read(table + ord("C") * 2, 16) & 0x0400)
        self.assertFalse(vm.memory.read(table + ord("C") * 2, 16) & 0x0200)
        self.assertTrue(vm.memory.read(table + ord("7") * 2, 16) & 0x0800)
        self.assertTrue(vm.memory.read(table + ord(";") * 2, 16) & 0x0004)

    def test_tolower_table_maps_ascii_and_preserves_eof(self):
        vm = Program().new_vm()
        cell = callback("__ctype_tolower_loc")(vm, ())
        table = vm.memory.read(cell, 64)

        self.assertEqual(vm.memory.read(table + ord("A") * 4, 32), ord("a"))
        self.assertEqual(vm.memory.read(table + ord("z") * 4, 32), ord("z"))
        self.assertEqual(signed32(vm.memory.read(table - 4, 32)), -1)

    def test_tables_use_stable_guest_storage(self):
        vm = Program().new_vm()
        b_first = callback("__ctype_b_loc")(vm, ())
        lower_first = callback("__ctype_tolower_loc")(vm, ())
        heap_after = vm.heap_next
        self.assertEqual(callback("__ctype_b_loc")(vm, ()), b_first)
        self.assertEqual(callback("__ctype_tolower_loc")(vm, ()), lower_first)
        self.assertEqual(vm.heap_next, heap_after)

    def test_negative_compatibility_range_is_allocated(self):
        vm = Program().new_vm()
        b_cell = callback("__ctype_b_loc")(vm, ())
        b_table = vm.memory.read(b_cell, 64)
        lower_cell = callback("__ctype_tolower_loc")(vm, ())
        lower_table = vm.memory.read(lower_cell, 64)

        self.assertEqual(vm.memory.read(b_table - 128 * 2, 16), 0)
        self.assertEqual(signed32(vm.memory.read(lower_table - 128 * 4, 32)), -128)

    def test_bad_arity_and_unrelated_symbol_are_not_hidden(self):
        vm = Program().new_vm()
        with self.assertRaisesRegex(VMError, "expects no arguments"):
            callback("__ctype_b_loc")(vm, (1,))
        self.assertIsNone(resolve_ctype_callback("__ctype_toupper_loc", vm_error=VMError))


if __name__ == "__main__":
    unittest.main()
