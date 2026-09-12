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


if __name__ == "__main__":
    unittest.main()
