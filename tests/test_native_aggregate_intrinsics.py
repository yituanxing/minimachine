from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


class NativeAggregateIntrinsicTests(unittest.TestCase):
    def _native_vm(self, program: Program) -> NativeVM:
        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_AGGREGATE_INTRINSICS": "1"},
            clear=False,
        ):
            return NativeVM(program)

    def test_load_aggregate_allocates_and_copies_blob(self):
        symbol = "__mm_load_aggregate"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("load_aggregate fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        source = 0x32000
        size = 257
        fn = muir.Function(
            "load_aggregate_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(source), muir.Imm(size)),
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

        payload = bytes((i * 17 + 3) & 0xFF for i in range(size))
        vm.memory.bulk_write(source, payload)
        ptr, = vm.run_function("load_aggregate_caller", result_count=1)
        self.assertEqual(ptr & 7, 0)
        self.assertEqual(vm.memory.bulk_read(ptr, size), payload)

    def test_store_aggregate_copies_blob(self):
        symbol = "__mm_store_aggregate"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("store_aggregate fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        destination = 0x36000
        blob = 0x36FF0
        size = 131
        fn = muir.Function(
            "store_aggregate_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (
                                muir.Imm(destination),
                                muir.Imm(blob),
                                muir.Imm(size),
                            ),
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
        vm = self._native_vm(program)

        payload = bytes((i * 31 + 11) & 0xFF for i in range(size))
        vm.memory.bulk_write(blob, payload)
        self.assertEqual(
            vm.run_function("store_aggregate_caller", result_count=1),
            (1,),
        )
        self.assertEqual(vm.memory.bulk_read(destination, size), payload)

    def test_returnaddress_depth_zero_stays_in_native_vm(self):
        symbol = "__mm_llvm_returnaddress"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("returnaddress fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            "returnaddress_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (muir.Imm(0),),
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
            vm.run_function("returnaddress_caller", result_count=1),
            (program.halt_code,),
        )


if __name__ == "__main__":
    unittest.main()
