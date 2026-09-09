from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


class NativeMemoryIntrinsicTests(unittest.TestCase):
    def _build_vm(self, symbol: str, args: tuple[muir.Value, ...], name: str):
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError(f"{symbol} fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            name,
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            args,
                            None,
                        ),
                        muir.Mov(muir.Width.I64, result, muir.Imm(1)),
                        muir.Ret(result),
                    ],
                )
            ],
            {"result"},
        )
        expanded, _ = expand_function(fn)
        program.add_function(lower_function(expanded))
        with patch.dict(
            os.environ,
            {
                "MINIMACHINE_NATIVE_MEMORY_INTRINSICS": "1",
                "MINIMACHINE_NATIVE_I128_VALUE_INTRINSICS": "1",
            },
            clear=False,
        ):
            return NativeVM(program)

    def test_memset_runs_inside_native_vm(self):
        dst = 0x28000
        vm = self._build_vm(
            "__mm_llvm_memset_p0_i64",
            (
                muir.Imm(dst),
                muir.Imm(0xA5),
                muir.Imm(130),
                muir.Imm(0),
            ),
            "memset_caller",
        )
        self.assertEqual(
            vm.run_function("memset_caller", result_count=1),
            (1,),
        )
        self.assertEqual(
            vm.memory.bulk_read(dst, 130),
            bytes([0xA5]) * 130,
        )

    def test_memcpy_runs_inside_native_vm_across_page_boundary(self):
        src = 0x2FFF0
        dst = 0x40020
        size = 257
        vm = self._build_vm(
            "__mm_llvm_memcpy_p0_p0_i64",
            (
                muir.Imm(dst),
                muir.Imm(src),
                muir.Imm(size),
                muir.Imm(0),
            ),
            "memcpy_caller",
        )
        payload = bytes((i * 29 + 7) & 0xFF for i in range(size))
        vm.memory.bulk_write(src, payload)
        self.assertEqual(
            vm.run_function("memcpy_caller", result_count=1),
            (1,),
        )
        self.assertEqual(vm.memory.bulk_read(dst, size), payload)


if __name__ == "__main__":
    unittest.main()
