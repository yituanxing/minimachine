from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


MASK64 = (1 << 64) - 1


class NativeScalarIntrinsicTests(unittest.TestCase):
    def _run_binary(self, symbol: str, a: int, b: int) -> int:
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError(f"{symbol} fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            "scalar_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(a), muir.Imm(b)),
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
            {"MINIMACHINE_NATIVE_SCALAR_INTRINSICS": "1"},
            clear=False,
        ):
            vm = NativeVM(program)
        return vm.run_function("scalar_caller", result_count=1)[0]

    def test_unsigned_division_and_remainder(self):
        self.assertEqual(self._run_binary("__mm_udiv_64", 100, 7), 14)
        self.assertEqual(self._run_binary("__mm_urem_64", 100, 7), 2)
        self.assertEqual(self._run_binary("__mm_udiv_64", 100, 0), 0)
        self.assertEqual(self._run_binary("__mm_urem_32", 100, 0), 0)

    def test_signed_division_and_remainder_match_llvm_concretization(self):
        neg7 = (-7) & MASK64
        self.assertEqual(
            self._run_binary("__mm_sdiv_64", neg7, 3),
            (-2) & MASK64,
        )
        self.assertEqual(
            self._run_binary("__mm_srem_64", neg7, 3),
            (-1) & MASK64,
        )
        self.assertEqual(
            self._run_binary("__mm_sdiv_32", 0x80000000, 0xFFFFFFFF),
            0,
        )

    def test_is_constant_returns_false_without_python_callback(self):
        symbol = "__mm_llvm_is_constant_i32"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("is_constant fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            "is_constant_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(123),),
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
            {"MINIMACHINE_NATIVE_SCALAR_INTRINSICS": "1"},
            clear=False,
        ):
            vm = NativeVM(program)
        self.assertEqual(
            vm.run_function("is_constant_caller", result_count=1),
            (0,),
        )

    def test_prefetch_is_native_noop(self):
        symbol = "__mm_llvm_prefetch_p0"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("prefetch fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            "prefetch_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (
                                muir.Imm(0x1000),
                                muir.Imm(0),
                                muir.Imm(3),
                                muir.Imm(1),
                            ),
                            None,
                        ),
                        muir.Mov(muir.Width.I64, result, muir.Imm(7)),
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
            {"MINIMACHINE_NATIVE_SCALAR_INTRINSICS": "1"},
            clear=False,
        ):
            vm = NativeVM(program)
        self.assertEqual(
            vm.run_function("prefetch_caller", result_count=1),
            (7,),
        )


if __name__ == "__main__":
    unittest.main()
