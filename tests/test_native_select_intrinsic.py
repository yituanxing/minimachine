from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


class NativeSelectIntrinsicTests(unittest.TestCase):
    def _run(self, cond: int, on_true: int, on_false: int) -> int:
        symbol = "__mm_select_i32"
        program = Program()

        def should_not_run(_vm, _args):
            raise AssertionError("select fell back to Python")

        program.register_service(symbol, should_not_run)
        result = muir.Slot("result")
        fn = muir.Function(
            "select_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol=symbol),
                            (
                                muir.Imm(cond),
                                muir.Imm(on_true),
                                muir.Imm(on_false),
                            ),
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
            {"MINIMACHINE_NATIVE_SELECT_INTRINSIC": "1"},
            clear=False,
        ):
            vm = NativeVM(program)
        return vm.run_function("select_caller", result_count=1)[0]

    def test_select_uses_low_condition_bit(self):
        self.assertEqual(self._run(0, 0x11, 0x22), 0x22)
        self.assertEqual(self._run(1, 0x11, 0x22), 0x11)
        self.assertEqual(self._run(2, 0x11, 0x22), 0x22)
        self.assertEqual(self._run(3, 0x11, 0x22), 0x11)


if __name__ == "__main__":
    unittest.main()
