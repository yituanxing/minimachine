import unittest

from src.minimachine import muir
from src.minimachine.legalize import legalize_module
from src.minimachine.verify import verify_muir


class LLVMFenceTests(unittest.TestCase):
    def test_standard_fence_orderings_use_existing_sys_contract(self):
        text = r'''
        define void @f() {
        entry:
          fence acquire
          fence release
          fence acq_rel
          fence seq_cst
          fence syncscope("singlethread") seq_cst
          ret void
        }
        '''
        functions, stats = legalize_module(text)
        self.assertEqual(len(functions), 1)
        verify_muir(functions[0])
        sysops = [
            inst
            for block in functions[0].blocks
            for inst in block.instructions
            if isinstance(inst, muir.Sys)
        ]
        self.assertEqual(len(sysops), 5)
        self.assertTrue(all(inst.op == "fence" for inst in sysops))
        self.assertTrue(
            all(inst.args == (muir.Imm(12), muir.Imm(12)) for inst in sysops)
        )
        self.assertEqual(stats.lowered_system_fence, 5)

    def test_atomic_load_store_ordering_does_not_pollute_pointer_expression(self):
        text = r'''
        define i32 @atomic_io(i32 %v, ptr %p) {
        entry:
          store atomic i32 %v, ptr %p monotonic, align 4
          %x = load atomic i32, ptr %p acquire, align 4
          store atomic i32 %x, ptr getelementptr inbounds ([2 x i32], ptr @slots, i64 0, i64 1) syncscope("singlethread") release, align 4
          ret i32 %x
        }
        '''
        functions, _stats = legalize_module(text)
        self.assertEqual(len(functions), 1)
        verify_muir(functions[0])
        moves = [
            inst
            for block in functions[0].blocks
            for inst in block.instructions
            if isinstance(inst, muir.Mov)
        ]
        self.assertGreaterEqual(len(moves), 3)
        self.assertTrue(
            any(
                isinstance(inst.dst, muir.Mem)
                and isinstance(inst.dst.address.base, muir.Symbol)
                and inst.dst.address.base.name == "slots"
                and inst.dst.address.offset == 4
                for inst in moves
            )
        )


if __name__ == "__main__":
    unittest.main()
