from __future__ import annotations

import ctypes
import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import CInst, COperand, NativeVM
from src.minimachine.vm import Program


class NativeCompactLayoutTests(unittest.TestCase):
    @staticmethod
    def _machine(fn: muir.Function):
        expanded, _ = expand_function(fn)
        return lower_function(expanded)

    def test_instruction_layout_is_136_bytes(self):
        self.assertEqual(ctypes.sizeof(COperand), 32)
        self.assertEqual(ctypes.sizeof(CInst), 136)

    def test_mov_sub_and_branch_execute_in_native_vm(self):
        x = muir.Slot("x")
        y = muir.Slot("y")
        copied = muir.Slot("copied")
        diff = muir.Slot("diff")
        fn = muir.Function(
            "compact_ops",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Mov(muir.Width.I64, copied, x),
                        muir.Sub(muir.Width.I64, diff, copied, y),
                        muir.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            diff,
                            muir.Imm(0),
                            muir.Target(label="equal"),
                            muir.Target(label="different"),
                        ),
                    ],
                ),
                muir.Block("equal", [muir.Ret(muir.Imm(7))]),
                muir.Block("different", [muir.Ret(diff)]),
            ],
            {"x", "y", "copied", "diff"},
            ("x", "y"),
        )
        vm = NativeVM(Program([self._machine(fn)]))
        self.assertEqual(vm.run_function("compact_ops", (9, 9)), (7,))
        self.assertEqual(vm.run_function("compact_ops", (13, 5)), (8,))

    def test_packed_cache_round_trip_uses_compact_layout(self):
        from pathlib import Path
        import tempfile

        fn = muir.Function(
            "compact_cached",
            [muir.Block("entry", [muir.Ret(muir.Imm(42))])],
            set(),
            (),
        )
        program = Program([self._machine(fn)])
        key = "22" * 32

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compact-pack.bin"
            cold = NativeVM(
                program,
                pack_cache_out=path,
                pack_cache_key=key,
            )
            self.assertEqual(cold.run_function("compact_cached", ()), (42,))
            size = path.stat().st_size
            self.assertGreater(size, 128)

            hot = NativeVM(
                program,
                pack_cache_in=path,
                pack_cache_key=key,
            )
            self.assertEqual(hot.run_function("compact_cached", ()), (42,))


if __name__ == "__main__":
    unittest.main()
