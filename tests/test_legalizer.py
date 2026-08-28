import unittest

from src.minimachine import muir
from src.minimachine.legalize import legalize_module
from src.minimachine.verify import verify_muir


class LegalizerTests(unittest.TestCase):
    def lower_one(self, text):
        functions, stats = legalize_module(text)
        self.assertEqual(len(functions), 1)
        verify_muir(functions[0])
        return functions[0], stats

    def test_add_and_icmp_branch_fuse(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i64 %a, i64 %b) {
            entry:
              %sum = add i64 %a, %b
              %c = icmp ult i64 %sum, 10
              br i1 %c, label %yes, label %no
            yes:
              ret i64 %sum
            no:
              ret i64 %a
            }
            """
        )
        entry = fn.blocks[0].instructions
        self.assertEqual(stats.fused_icmp_br, 1)
        self.assertEqual(sum(isinstance(x, muir.Sub) for x in entry), 2)
        self.assertIsInstance(entry[-1], muir.Br)
        self.assertEqual(entry[-1].cond, muir.Cond.ULT)

    def test_phi_becomes_edge_moves(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i1 %c, i64 %a, i64 %b) {
            entry:
              br i1 %c, label %left, label %right
            left:
              br label %merge
            right:
              br label %merge
            merge:
              %x = phi i64 [ %a, %left ], [ %b, %right ]
              ret i64 %x
            }
            """
        )
        by_name = {b.label: b for b in fn.blocks}
        self.assertEqual(stats.phi_edge_moves, 2)
        self.assertIsInstance(by_name["left"].instructions[-2], muir.Mov)
        self.assertIsInstance(by_name["right"].instructions[-2], muir.Mov)
        self.assertFalse(
            any(type(i).__name__.lower() == "phi" for b in fn.blocks for i in b.instructions)
        )

    def test_constant_gep_folds_into_load(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(ptr %p) {
            entry:
              %q = getelementptr i64, ptr %p, i64 3
              %x = load i64, ptr %q, align 8
              ret i64 %x
            }
            """
        )
        entry = fn.blocks[0].instructions
        self.assertEqual(stats.folded_gep_mem, 1)
        mov = entry[0]
        self.assertIsInstance(mov, muir.Mov)
        self.assertIsInstance(mov.src, muir.Mem)
        self.assertEqual(mov.src.address.offset, 24)

    def test_call_is_muir_pseudo_not_machine_instruction(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i64 %a) {
            entry:
              %x = call i64 @foo(i64 %a)
              ret i64 %x
            }
            """
        )
        self.assertEqual(stats.regular_calls, 1)
        self.assertIsInstance(fn.blocks[0].instructions[0], muir.Call)
        self.assertEqual(fn.blocks[0].instructions[0].callee.symbol, "foo")

    def test_system_asm_maps_to_explicit_sys_contract(self):
        fn, stats = self.lower_one(
            """
            define i64 @sys_contract() {
            entry:
              call void asm sideeffect "fence rw,rw", "~{memory}"()
              %s = call i64 asm sideeffect "csrr $0, 0x100", "=r,~{memory}"()
              call void asm sideeffect "sfence.vma", "~{memory}"()
              call void asm sideeffect "wfi", ""()
              ret i64 %s
            }
            """
        )
        sysops = [
            i
            for b in fn.blocks
            for i in b.instructions
            if isinstance(i, muir.Sys)
        ]
        self.assertEqual(
            [i.op for i in sysops],
            [
                "fence",
                "state_read_status",
                "tlb_flush_all",
                "wait_interrupt",
            ],
        )
        self.assertEqual(stats.lowered_system_fence, 1)
        self.assertEqual(stats.lowered_system_state, 1)
        self.assertEqual(stats.lowered_system_tlb, 1)
        self.assertEqual(stats.lowered_system_wait, 1)

    def test_simple_amo_maps_to_atomic_system_contract(self):
        fn, stats = self.lower_one(
            """
            define i32 @atomic_add(ptr %p) {
            entry:
              %old = call i32 asm sideeffect "amoadd.w.aqrl $1, $2, $0", "=*A,=r,r,*A,~{memory}"(ptr elementtype(i32) %p, i32 1, ptr elementtype(i32) %p)
              ret i32 %old
            }
            """
        )
        sysops = [
            i
            for b in fn.blocks
            for i in b.instructions
            if isinstance(i, muir.Sys)
        ]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "atomic_add_i32_acq_rel")
        self.assertEqual(sysops[0].args, (muir.Slot("p"), muir.Imm(1)))
        self.assertEqual(sysops[0].result, muir.Slot("old"))
        self.assertEqual(stats.lowered_system_atomic, 1)

    def test_bitwise_routes_to_explicit_helper(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i64 %a, i64 %b) {
            entry:
              %x = and i64 %a, %b
              ret i64 %x
            }
            """
        )
        self.assertEqual(stats.temporary_helpers, 1)
        helper = fn.blocks[0].instructions[0]
        self.assertIsInstance(helper, muir.Helper)
        self.assertEqual(helper.symbol, "__mm_and_64")


if __name__ == "__main__":
    unittest.main()
