import unittest

from src.minimachine import muir
from src.minimachine.abi import HEADER_SIZE, expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.vm import Program


def machine(function: muir.Function):
    expanded, _ = expand_function(function)
    return lower_function(expanded)


class SlotColorLayoutTests(unittest.TestCase):
    def _program(self):
        dec = muir.Function(
            "dec",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("out"),
                            muir.Slot("x"),
                            muir.Imm(1),
                        ),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"x", "out"},
            ("x",),
        )
        caller = muir.Function(
            "caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("tmp"),
                            muir.Slot("x"),
                            muir.Imm(0),
                        ),
                        muir.Call(
                            muir.Callee(symbol="dec"),
                            (muir.Slot("tmp"),),
                            muir.Slot("r"),
                        ),
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("answer"),
                            muir.Slot("r"),
                            muir.Imm(1),
                        ),
                        muir.Ret(muir.Slot("answer")),
                    ],
                )
            ],
            {"x", "tmp", "r", "answer"},
            ("x",),
        )
        return Program([machine(dec), machine(caller)])

    def test_colored_layout_preserves_call_semantics_and_shrinks_frames(self):
        program = self._program()
        before = program.functions["caller"]
        before_result = program.new_vm().run_function("caller", (43,))
        self.assertEqual(before_result, (41,))

        stats = program.enable_slot_coloring()
        after = program.functions["caller"]
        after_result = program.new_vm().run_function("caller", (43,))

        self.assertEqual(after_result, before_result)
        self.assertLess(after.frame_size, before.frame_size)
        self.assertLess(
            len(set(after.slot_offsets.values())),
            len(after.slot_offsets),
        )
        self.assertLess(stats["physical_slots"], stats["logical_slots"])

    def test_host_fast_path_descriptor_stays_header_only(self):
        program = self._program()
        program.replace_function_with_service(
            "dec", lambda vm, args: args[0] + 100
        )
        descriptor = program.symbol_addresses["dec"]
        self.assertEqual(
            program.initial_memory.read(descriptor + 8, 64),
            HEADER_SIZE,
        )

        program.enable_slot_coloring()

        self.assertEqual(
            program.initial_memory.read(descriptor + 8, 64),
            HEADER_SIZE,
        )
        self.assertEqual(
            program.new_vm().run_function("caller", (43,)),
            (142,),
        )


if __name__ == "__main__":
    unittest.main()
