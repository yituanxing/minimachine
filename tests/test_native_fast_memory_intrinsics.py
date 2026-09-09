from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


class NativeFastMemoryIntrinsicTests(unittest.TestCase):
    def _native_vm(self, program: Program) -> NativeVM:
        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_FAST_MEMORY_INTRINSICS": "1"},
            clear=False,
        ):
            return NativeVM(program)

    def test_fast_memcpy_returns_dst_and_copies_across_page(self):
        symbol = "__mm_fast_memcpy"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("fast_memcpy fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        src = 0x4FFF0
        dst = 0x60020
        size = 257
        fn = muir.Function(
            "fast_memcpy_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(dst), muir.Imm(src), muir.Imm(size)),
                            result,
                        ),
                        muir.Ret(result),
                    ],
                )
            ],
            {"result"},
        )
        expanded, _ = expand_function(fn)
        program.add_function(lower_function(expanded))
        vm = self._native_vm(program)
        payload = bytes((i * 13 + 5) & 0xFF for i in range(size))
        vm.memory.bulk_write(src, payload)
        self.assertEqual(
            vm.run_function("fast_memcpy_caller", result_count=1),
            (dst,),
        )
        self.assertEqual(vm.memory.bulk_read(dst, size), payload)

    def test_fast_memset_returns_dst(self):
        symbol = "__mm_fast_memset"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("fast_memset fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        dst = 0x64000
        size = 193
        fn = muir.Function(
            "fast_memset_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(dst), muir.Imm(0x5A), muir.Imm(size)),
                            result,
                        ),
                        muir.Ret(result),
                    ],
                )
            ],
            {"result"},
        )
        expanded, _ = expand_function(fn)
        program.add_function(lower_function(expanded))
        vm = self._native_vm(program)
        self.assertEqual(
            vm.run_function("fast_memset_caller", result_count=1),
            (dst,),
        )
        self.assertEqual(vm.memory.bulk_read(dst, size), bytes([0x5A]) * size)

    def test_fast_strlen_matches_c_string_length(self):
        symbol = "__mm_fast_strlen"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("fast_strlen fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        ptr = 0x67FFB
        fn = muir.Function(
            "fast_strlen_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(ptr),),
                            result,
                        ),
                        muir.Ret(result),
                    ],
                )
            ],
            {"result"},
        )
        expanded, _ = expand_function(fn)
        program.add_function(lower_function(expanded))
        vm = self._native_vm(program)
        payload = b"minimachine-fast-string\0tail"
        vm.memory.bulk_write(ptr, payload)
        self.assertEqual(
            vm.run_function("fast_strlen_caller", result_count=1),
            (len(b"minimachine-fast-string"),),
        )


if __name__ == "__main__":
    unittest.main()
