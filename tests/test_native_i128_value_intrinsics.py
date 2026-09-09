from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


class NativeI128ValueIntrinsicTests(unittest.TestCase):
    def test_wide_const_i128_allocates_expected_little_endian_blob(self):
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("__mm_wide_const_128 fell back to Python")

        program.register_service("__mm_wide_const_128", should_not_run)
        result = muir.Slot("result")
        low = 0x0123456789ABCDEF
        high = 0xFEDCBA9876543210
        fn = muir.Function(
            "wide_const_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="__mm_wide_const_128"),
                            (muir.Imm(low), muir.Imm(high)),
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

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_I128_VALUE_INTRINSICS": "1"},
            clear=False,
        ):
            vm = NativeVM(program)

        ptr, = vm.run_function("wide_const_caller", result_count=1)
        self.assertEqual(ptr & 0xF, 0)
        self.assertEqual(vm.memory.read(ptr, 64), low)
        self.assertEqual(vm.memory.read(ptr + 8, 64), high)

    def test_icmp_eq_i128_compares_both_halves(self):
        lhs = 0x24000
        rhs = 0x24020
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("__mm_icmp_eq_128 fell back to Python")

        program.register_service("__mm_icmp_eq_128", should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            "icmp_eq_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="__mm_icmp_eq_128"),
                            (muir.Imm(lhs), muir.Imm(rhs)),
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

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_I128_VALUE_INTRINSICS": "1"},
            clear=False,
        ):
            vm = NativeVM(program)

        low = 0x1111222233334444
        high = 0xAAAABBBBCCCCDDDD
        vm.memory.write(lhs, 64, low)
        vm.memory.write(lhs + 8, 64, high)
        vm.memory.write(rhs, 64, low)
        vm.memory.write(rhs + 8, 64, high)
        self.assertEqual(
            vm.run_function("icmp_eq_caller", result_count=1),
            (1,),
        )

        vm.memory.write(rhs + 8, 64, high ^ 1)
        self.assertEqual(
            vm.run_function("icmp_eq_caller", result_count=1),
            (0,),
        )


if __name__ == "__main__":
    unittest.main()
