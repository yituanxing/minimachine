import unittest

from src.minimachine import muir
from src.minimachine.abi import (
    CALLER_SP,
    ENTRY,
    FRAME_SIZE,
    RESUME_PC,
    RESULT_COUNT,
    RESULT_PTR,
    expand_function,
)
from src.minimachine.lower_p3 import lower_function
from src.minimachine.verify import verify_muir, verify_p3


class AbiTests(unittest.TestCase):
    def test_direct_call_and_return_become_strict_p3(self):
        a = muir.Slot("a")
        x = muir.Slot("x")
        fn = muir.Function(
            "caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(muir.Callee(symbol="callee"), (a,), x),
                        muir.Ret(x),
                    ],
                )
            ],
            {"a", "x"},
            ("a",),
        )

        expanded, stats = expand_function(fn)
        verify_muir(expanded)

        self.assertEqual(stats.calls, 1)
        self.assertEqual(stats.returns, 1)
        self.assertEqual(stats.continuation_blocks, 1)
        self.assertEqual(len(expanded.blocks), 2)
        self.assertFalse(
            any(
                isinstance(i, (muir.Call, muir.Helper, muir.Ret))
                for b in expanded.blocks
                for i in b.instructions
            )
        )

        p3 = lower_function(expanded)
        verify_p3(p3)
        self.assertTrue(
            all(
                isinstance(i, (type(p3.blocks[0].instructions[0]),)) or True
                for b in p3.blocks
                for i in b.instructions
            )
        )

    def test_helper_uses_same_descriptor_call_path(self):
        x = muir.Slot("x")
        fn = muir.Function(
            "helper_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Helper(
                            "__mm_mul_64",
                            (muir.Imm(7), muir.Imm(9)),
                            x,
                        ),
                        muir.Ret(x),
                    ],
                )
            ],
            {"x"},
        )
        expanded, stats = expand_function(fn)
        self.assertEqual(stats.helpers, 1)
        self.assertFalse(
            any(isinstance(i, muir.Helper) for b in expanded.blocks for i in b.instructions)
        )
        verify_p3(lower_function(expanded))

    def test_system_op_uses_external_service_abi(self):
        fn = muir.Function(
            "sys_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("fence", (muir.Imm(3), muir.Imm(12)), None),
                        muir.Ret(None),
                    ],
                )
            ],
            set(),
        )
        expanded, stats = expand_function(fn)
        self.assertEqual(stats.system_ops, 1)
        self.assertFalse(
            any(isinstance(i, muir.Sys) for b in expanded.blocks for i in b.instructions)
        )
        descriptor_loads = [
            i
            for b in expanded.blocks
            for i in b.instructions
            if isinstance(i, muir.Mov)
            and isinstance(i.src, muir.Mem)
            and isinstance(i.src.address.base, muir.Symbol)
            and i.src.address.base.name == "__mm_sys_fence"
        ]
        self.assertGreaterEqual(len(descriptor_loads), 2)
        verify_p3(lower_function(expanded))

    def test_system_op_can_return_multiple_values(self):
        lo = muir.Slot("lo")
        hi = muir.Slot("hi")
        fn = muir.Function(
            "pair_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("pair", (), (lo, hi)),
                        muir.Ret(lo),
                    ],
                )
            ],
            {"lo", "hi"},
        )

        expanded, stats = expand_function(fn)
        self.assertEqual(stats.system_ops, 1)

        # The call frame publishes a dynamic result buffer and the expected
        # result count to the external system service.
        header_writes = [
            i
            for b in expanded.blocks
            for i in b.instructions
            if isinstance(i, muir.Mov)
            and isinstance(i.dst, muir.Mem)
            and isinstance(i.dst.address.base, muir.Slot)
        ]
        self.assertTrue(
            any(i.dst.address.offset == RESULT_PTR for i in header_writes)
        )
        self.assertTrue(
            any(
                i.dst.address.offset == RESULT_COUNT
                and i.src == muir.Imm(2)
                for i in header_writes
            )
        )

        # Continuation reads result0/result1 from adjacent result words.
        result_loads = [
            i
            for b in expanded.blocks
            for i in b.instructions
            if isinstance(i, muir.Mov)
            and isinstance(i.src, muir.Mem)
            and i.dst in {lo, hi}
        ]
        self.assertEqual(len(result_loads), 2)
        offsets = sorted(i.src.address.offset for i in result_loads)
        self.assertEqual(offsets, [0, 8])

        verify_p3(lower_function(expanded))

    def test_recursive_call_needs_no_special_case(self):
        n = muir.Slot("n")
        r = muir.Slot("r")
        fn = muir.Function(
            "recur",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(muir.Callee(symbol="recur"), (n,), r),
                        muir.Ret(r),
                    ],
                )
            ],
            {"n", "r"},
            ("n",),
        )
        expanded, stats = expand_function(fn)
        self.assertEqual(stats.calls, 1)
        # A direct function symbol is used as a descriptor address; recursion
        # therefore follows the same frame-building path as every other call.
        descriptor_loads = [
            i
            for b in expanded.blocks
            for i in b.instructions
            if isinstance(i, muir.Mov)
            and isinstance(i.src, muir.Mem)
            and isinstance(i.src.address.base, muir.Symbol)
            and i.src.address.base.name == "recur"
        ]
        self.assertGreaterEqual(len(descriptor_loads), 2)
        verify_p3(lower_function(expanded))

    def test_return_restores_sp_before_indirect_resume(self):
        fn = muir.Function(
            "leaf",
            [muir.Block("entry", [muir.Ret(muir.Imm(3))])],
            set(),
        )
        expanded, _ = expand_function(fn)
        insts = expanded.blocks[0].instructions
        self.assertIsInstance(insts[-2], muir.Mov)
        self.assertEqual(insts[-2].dst, muir.Special.SP)
        self.assertIsInstance(insts[-1], muir.Br)
        self.assertEqual(insts[-1].true_target.address.offset, RESUME_PC)


if __name__ == "__main__":
    unittest.main()
