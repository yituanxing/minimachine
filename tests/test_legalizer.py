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

    def test_undef_indirectbr_refines_to_declared_successor(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i1 %cond) {
            entry:
              indirectbr ptr undef, [label %left, label %right]
            left:
              ret i64 11
            right:
              ret i64 22
            }
            """
        )
        self.assertEqual(stats.lowered_indirectbr, 1)
        self.assertEqual(stats.lowered_undef_indirectbr, 1)
        entry = fn.blocks[0]
        self.assertIsInstance(entry.instructions[-1], muir.Br)
        self.assertEqual(entry.instructions[-1].true_target.label, "left")
        self.assertEqual(entry.instructions[-1].false_target.label, "left")

    def test_switch_lowers_to_direct_branch_chain(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i32 %x) {
            entry:
              switch i32 %x, label %other [
                i32 0, label %zero
                i32 7, label %seven
              ]
            zero:
              ret i64 10
            seven:
              ret i64 17
            other:
              ret i64 99
            }
            """
        )
        self.assertEqual(stats.lowered_switch, 1)
        by_name = {b.label: b for b in fn.blocks}
        self.assertIsInstance(by_name["entry"].instructions[-1], muir.Br)
        self.assertEqual(by_name["entry"].instructions[-1].cond, muir.Cond.EQ)
        self.assertEqual(
            sum(label.startswith("__switch_entry_") for label in by_name),
            4,
        )
        self.assertFalse(
            any(
                isinstance(i, muir.ArchEscape)
                for b in fn.blocks
                for i in b.instructions
            )
        )

    def test_switch_with_profile_metadata_lowers_normally(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i8 %x) {
            entry:
              switch i8 %x, label %other [
                i8 3, label %three
                i8 19, label %nineteen
              ], !prof !1
            three:
              ret i64 3
            nineteen:
              ret i64 19
            other:
              ret i64 0
            }
            !1 = !{!"branch_weights", i32 1, i32 2, i32 3}
            """
        )
        self.assertEqual(stats.lowered_switch, 1)
        by_name = {b.label: b for b in fn.blocks}
        self.assertIn("three", by_name)
        self.assertIn("nineteen", by_name)
        self.assertIn("other", by_name)

    def test_switch_phi_copy_is_placed_on_selected_edge(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i32 %x, i64 %a) {
            entry:
              switch i32 %x, label %other [
                i32 1, label %chosen
              ]
            chosen:
              %v = phi i64 [ %a, %entry ]
              ret i64 %v
            other:
              ret i64 0
            }
            """
        )
        by_name = {b.label: b for b in fn.blocks}
        self.assertFalse(
            any(
                isinstance(i, muir.Mov) and i.dst == muir.Slot("v")
                for i in by_name["entry"].instructions
            )
        )
        edge_blocks = [
            b
            for b in fn.blocks
            if b.label.startswith("__switch_entry_case0_edge")
        ]
        self.assertEqual(len(edge_blocks), 1)
        self.assertTrue(
            any(
                isinstance(i, muir.Mov) and i.dst == muir.Slot("v")
                for i in edge_blocks[0].instructions
            )
        )
        self.assertEqual(stats.phi_edge_moves, 1)

    def test_pointer_phi_ignores_nested_i8_gep_type(self):
        fn, stats = self.lower_one(
            """
            @chars = internal constant [7 x i8] c"\\0A()&|;\\00"

            define ptr @f(i1 %c, ptr %p) {
            entry:
              br i1 %c, label %left, label %right
            left:
              br label %merge
            right:
              br label %merge
            merge:
              %x = phi ptr [ %p, %left ], [ getelementptr inbounds ([7 x i8], ptr @chars, i64 0, i64 6), %right ]
              ret ptr %x
            }
            """
        )
        moves = [
            inst
            for block in fn.blocks
            for inst in block.instructions
            if isinstance(inst, muir.Mov) and inst.dst == muir.Slot("x")
        ]
        self.assertEqual(len(moves), 2)
        self.assertTrue(all(inst.width is muir.Width.I64 for inst in moves))
        self.assertEqual(stats.phi_edge_moves, 2)

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

    def test_conditional_phi_backedge_is_split_before_copy(self):
        fn, stats = self.lower_one(
            """
            define i64 @f(i64 %start) {
            entry:
              br label %loop
            loop:
              %i = phi i64 [ %start, %entry ], [ %next, %back ]
              br label %back
            back:
              %next = add nsw i64 %i, -1
              %done = icmp eq i64 %i, 0
              br i1 %done, label %exit, label %loop
            exit:
              ret i64 %i
            }
            """
        )
        by_name = {b.label: b for b in fn.blocks}
        back = by_name["back"]
        self.assertIsInstance(back.instructions[-1], muir.Br)
        self.assertFalse(
            any(
                isinstance(inst, muir.Mov)
                and inst.dst == muir.Slot("i")
                for inst in back.instructions[:-1]
            )
        )

        edge_blocks = [
            block
            for block in fn.blocks
            if block.label.startswith("__phi_back_to_loop_edge")
        ]
        self.assertEqual(len(edge_blocks), 1)
        edge = edge_blocks[0]
        self.assertTrue(
            any(
                isinstance(inst, muir.Mov)
                and inst.dst == muir.Slot("i")
                and inst.src == muir.Slot("next")
                for inst in edge.instructions
            )
        )
        self.assertIsInstance(edge.instructions[-1], muir.Br)
        self.assertEqual(edge.instructions[-1].true_target.label, "loop")
        self.assertEqual(edge.instructions[-1].false_target.label, "loop")
        self.assertIn(
            edge.label,
            {
                back.instructions[-1].true_target.label,
                back.instructions[-1].false_target.label,
            },
        )
        self.assertEqual(stats.phi_edge_moves, 2)

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

    def test_llvm_is_constant_compiletime_assert_guard_folds_only_guard(self):
        fn, _stats = self.lower_one(
            """
            declare i1 @llvm.is.constant.i64(i64)
            declare void @__compiletime_assert_1()

            define i32 @f(i64 %size, ptr %dst, ptr %src) {
            entry:
              %c = call i1 @llvm.is.constant.i64(i64 %size)
              br i1 %c, label %ok, label %bad
            bad:
              call void @__compiletime_assert_1()
              unreachable
            ok:
              switch i64 %size, label %bad2 [
                i64 1, label %one
                i64 8, label %eight
              ]
            one:
              ret i32 1
            eight:
              ret i32 8
            bad2:
              ret i32 -1
            }
            """
        )
        by_name = {b.label: b for b in fn.blocks}
        guard_moves = [
            inst
            for inst in by_name["entry"].instructions
            if isinstance(inst, muir.Mov) and inst.dst == muir.Slot("c")
        ]
        self.assertEqual(len(guard_moves), 1)
        self.assertEqual(guard_moves[0].src, muir.Imm(1))
        self.assertFalse(
            any(
                isinstance(inst, muir.Helper)
                and inst.symbol.startswith("__mm_llvm_is_constant")
                for block in fn.blocks
                for inst in block.instructions
            )
        )

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

    def test_jump_label_callbr_becomes_static_branch_sysop(self):
        fn, stats = self.lower_one(
            """
            define i1 @static_key(ptr %key) {
            entry:
              callbr void asm sideeffect "nop # __jump_table", "i,!i"(ptr %key) to label %fallthrough [label %taken]
            fallthrough:
              ret i1 false
            taken:
              ret i1 true
            }
            """
        )
        entry = fn.blocks[0].instructions
        self.assertIsInstance(entry[0], muir.Sys)
        self.assertEqual(entry[0].op, "static_branch")
        self.assertEqual(entry[0].args, (muir.Slot("key"),))
        self.assertIsInstance(entry[1], muir.Br)
        self.assertEqual(entry[1].true_target.label, "fallthrough")
        self.assertEqual(entry[1].false_target.label, "taken")
        self.assertEqual(stats.lowered_static_branch, 1)

    def test_float_ops_lower_to_ieee_bit_helpers(self):
        fn, stats = self.lower_one(
            """
            define i32 @fp(i32 %x, ptr %p) {
            entry:
              %a = sitofp i32 %x to double
              %b = fmul double %a, 2.500000e+00
              %c = fcmp ogt double %b, 0.000000e+00
              store double %b, ptr %p
              %d = load double, ptr %p
              %e = fptosi double %d to i32
              %r = select i1 %c, i32 %e, i32 0
              ret i32 %r
            }
            """
        )
        helpers = [
            inst.symbol
            for block in fn.blocks
            for inst in block.instructions
            if isinstance(inst, muir.Helper)
        ]
        self.assertIn("__mm_sitofp_32_64", helpers)
        self.assertIn("__mm_fmul_64", helpers)
        self.assertIn("__mm_fcmp_ogt_64", helpers)
        self.assertIn("__mm_fptosi_64_32", helpers)
        self.assertEqual(stats.lowered_float_ops, 2)
        self.assertEqual(stats.lowered_float_casts, 2)

    def test_float_phi_and_return_constant_are_raw_bits(self):
        fn, _ = self.lower_one(
            """
            define double @fp_phi(i1 %c) {
            entry:
              br i1 %c, label %yes, label %no
            yes:
              br label %join
            no:
              br label %join
            join:
              %v = phi double [ 1.500000e+00, %yes ], [ 2.500000e+00, %no ]
              ret double %v
            }
            """
        )
        self.assertTrue(
            any(
                isinstance(inst, muir.Mov)
                for block in fn.blocks
                for inst in block.instructions
            )
        )

    def test_busybox_forget_const_inline_asm_is_identity(self):
        fn, stats = self.lower_one(
            r"""
            @ptr_to_globals = external global i8

            define ptr @forget_const() {
            entry:
              %p = call ptr asm sideeffect "# forget that p points to const", "=r,0"(ptr @ptr_to_globals)
              ret ptr %p
            }
            """
        )
        moves = [
            inst
            for block in fn.blocks
            for inst in block.instructions
            if isinstance(inst, muir.Mov)
            and inst.dst == muir.Slot("p")
        ]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].src, muir.Symbol("ptr_to_globals"))
        self.assertEqual(stats.lowered_identity_asm, 1)
        self.assertFalse(
            any(
                isinstance(inst, muir.ArchEscape)
                for block in fn.blocks
                for inst in block.instructions
            )
        )

    def test_ecall_preserves_ptrtoint_global_argument(self):
        lowered, _ = self.lower_one(
            r"""
            @console_buffer = internal global [256 x i8] zeroinitializer, align 1

            define i64 @probe(i64 %count) {
            entry:
              %got = call i64 asm sideeffect "ecall", "=r,r,r,r,~{memory}"(i64 4, i64 ptrtoint (ptr @console_buffer to i64), i64 %count)
              ret i64 %got
            }
            """
        )
        sysops = [
            inst
            for block in lowered.blocks
            for inst in block.instructions
            if isinstance(inst, muir.Sys)
        ]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "ecall")
        self.assertEqual(sysops[0].args[0], muir.Imm(4))
        self.assertEqual(sysops[0].args[1], muir.Symbol("console_buffer"))
        self.assertEqual(sysops[0].args[2], muir.Slot("count"))

    def test_aggregate_ecall_uses_multi_result_sysop(self):
        fn, stats = self.lower_one(
            """
            define i64 @sbi(i64 %a0, i64 %a1) {
            entry:
              %pair = call { i64, i64 } asm sideeffect "ecall", "={x10},={x11},{x10},{x11}"(i64 %a0, i64 %a1)
              %lo = extractvalue { i64, i64 } %pair, 0
              %hi = extractvalue { i64, i64 } %pair, 1
              %sum = add i64 %lo, %hi
              ret i64 %sum
            }
            """
        )
        entry = fn.blocks[0].instructions
        sysops = [i for i in entry if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "ecall")
        self.assertIsInstance(sysops[0].result, tuple)
        self.assertEqual(len(sysops[0].result), 2)
        self.assertEqual(stats.lowered_system_ecall, 1)

        extracted = [
            i for i in entry
            if isinstance(i, muir.Mov)
            and i.dst in {muir.Slot("lo"), muir.Slot("hi")}
        ]
        self.assertEqual(len(extracted), 2)
        self.assertFalse(
            any(
                isinstance(i, muir.Helper) and i.symbol == "__mm_extractvalue"
                for i in entry
            )
        )

    def test_faultable_load_uses_multi_result_system_contract(self):
        fn, stats = self.lower_one(
            """
            define i32 @uget(ptr %p) {
            entry:
              %pair = call { i64, i32 } asm sideeffect "1:; lw $1, $2;2:; .pushsection __ex_table", "=r,=&r,*m,0"(ptr elementtype(i32) %p, i64 0)
              %err = extractvalue { i64, i32 } %pair, 0
              %val = extractvalue { i64, i32 } %pair, 1
              ret i32 %val
            }
            """
        )
        sysops = [
            i for b in fn.blocks for i in b.instructions
            if isinstance(i, muir.Sys)
        ]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "faultable_load_i32")
        self.assertEqual(sysops[0].args, (muir.Slot("p"), muir.Imm(0)))
        self.assertIsInstance(sysops[0].result, tuple)
        self.assertEqual(len(sysops[0].result), 2)
        self.assertEqual(stats.lowered_system_faultable, 1)
        self.assertFalse(
            any(
                isinstance(i, muir.Helper) and i.symbol == "__mm_extractvalue"
                for b in fn.blocks for i in b.instructions
            )
        )

    def test_lrsc_cmpxchg_uses_multi_result_system_contract(self):
        fn, stats = self.lower_one(
            """
            define i32 @cmpx(ptr %p, i64 %expected, i32 %desired) {
            entry:
              %pair = call { i32, i32 } asm sideeffect "0: lr.w $0, $2; bne $0, $3, 1f; sc.w.rl $1, $4, $2; bnez $1, 0b; fence rw, rw;1:;", "=&r,=&r,=*A,rJ,rJ,*A,~{memory}"(ptr elementtype(i32) %p, i64 %expected, i32 %desired, ptr elementtype(i32) %p)
              %old = extractvalue { i32, i32 } %pair, 0
              %status = extractvalue { i32, i32 } %pair, 1
              ret i32 %old
            }
            """
        )
        sysops = [
            i for b in fn.blocks for i in b.instructions
            if isinstance(i, muir.Sys)
        ]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(
            sysops[0].op,
            "atomic_cmpxchg_i32_release_post_full_fence",
        )
        self.assertEqual(
            sysops[0].args,
            (muir.Slot("p"), muir.Slot("expected"), muir.Slot("desired")),
        )
        self.assertIsInstance(sysops[0].result, tuple)
        self.assertEqual(len(sysops[0].result), 2)
        self.assertEqual(stats.lowered_system_lrsc, 1)

    def test_cpu_feature_alternative_callbr_becomes_sys_branch(self):
        fn, stats = self.lower_one(
            """
            define i1 @feature() {
            entry:
              callbr void asm sideeffect "886 : nop;887 : .pushsection .alternative;888 : j ${1:l};889 :", "i,!i"(i64 34) to label %fallthrough [label %taken]
            fallthrough:
              ret i1 false
            taken:
              ret i1 true
            }
            """
        )
        entry = fn.blocks[0].instructions
        self.assertIsInstance(entry[0], muir.Sys)
        self.assertEqual(entry[0].op, "cpu_feature")
        self.assertEqual(entry[0].args, (muir.Imm(34),))
        self.assertIsInstance(entry[1], muir.Br)
        self.assertEqual(entry[1].true_target.label, "fallthrough")
        self.assertEqual(entry[1].false_target.label, "taken")
        self.assertEqual(stats.lowered_cpu_feature_branch, 1)

    def test_icache_fence_becomes_system_contract(self):
        fn, stats = self.lower_one(
            """
            define void @flush_icache() {
            entry:
              call void asm sideeffect "fence.i", "~{memory}"()
              ret void
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "icache_sync")
        self.assertEqual(stats.lowered_system_icache, 1)

    def test_alternative_tlb_flush_becomes_system_contract(self):
        fn, stats = self.lower_one(
            """
            define void @flush_one(i64 %addr) {
            entry:
              call void asm sideeffect "886 : sfence.vma $0;887 : .pushsection .alternative;888 : sfence.vma;889 :", "r,~{memory}"(i64 %addr)
              ret void
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "tlb_flush_address")
        self.assertEqual(sysops[0].args, (muir.Slot("addr"),))
        self.assertEqual(stats.lowered_alternative_tlb, 1)

    def test_generic_csr_expression_becomes_system_service(self):
        fn, stats = self.lower_one(
            """
            define i64 @pmu() {
            entry:
              %v = call i64 asm sideeffect "csrr $0, 0xc00 + 16 + 8 + 4 + 2 + 1", "=r,~{memory}"()
              ret i64 %v
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "csr_read")
        self.assertEqual(sysops[0].args, (muir.Imm(0xc00 + 31),))
        self.assertEqual(sysops[0].result, muir.Slot("v"))
        self.assertEqual(stats.lowered_generic_csr, 1)

    def test_conditional_lrsc_preserves_condition_and_order(self):
        fn, stats = self.lower_one(
            """
            define i32 @inc_not_negative(ptr %p) {
            entry:
              %pair = call { i32, i32 } asm sideeffect "0: lr.w $0, $2; bltz $0, 1f; addi $1, $0, 1; sc.w.rl $1, $1, $2; bnez $1, 0b; fence rw, rw;1:;", "=&r,=&r,=*A,*A,~{memory}"(ptr elementtype(i32) %p, ptr elementtype(i32) %p)
              %old = extractvalue { i32, i32 } %pair, 0
              ret i32 %old
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(
            sysops[0].op,
            "atomic_add1_if_nonnegative_i32_release_post_full_fence",
        )
        self.assertEqual(sysops[0].args, (muir.Slot("p"),))
        self.assertIsInstance(sysops[0].result, tuple)
        self.assertEqual(len(sysops[0].result), 2)
        self.assertEqual(stats.lowered_system_lrsc, 1)

    def test_amoswap_post_acquire_fence_is_preserved(self):
        fn, stats = self.lower_one(
            """
            define i64 @swap(ptr %p, i64 %x) {
            entry:
              %old = call i64 asm sideeffect "amoswap.d $0, $2, $1; fence r , rw;", "=r,=*A,r,*A,~{memory}"(ptr elementtype(i64) %p, i64 %x, ptr elementtype(i64) %p)
              ret i64 %old
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(
            sysops[0].op,
            "atomic_swap_i64_relaxed_post_acquire_fence",
        )
        self.assertEqual(sysops[0].args, (muir.Slot("p"), muir.Slot("x")))
        self.assertEqual(stats.lowered_system_atomic, 1)

    def test_faultable_amo_uses_multi_result_system_contract(self):
        fn, stats = self.lower_one(
            """
            define i32 @futex_add(ptr %p, i32 %x) {
            entry:
              %pair = call { i32, i32 } asm sideeffect "1: amoadd.w.aqrl $1,$3,$2;2:; .pushsection __ex_table", "=r,=&r,=*m,Jr,0,*m,~{memory}"(ptr elementtype(i32) %p, i32 %x, i32 0, ptr elementtype(i32) %p)
              %err = extractvalue { i32, i32 } %pair, 0
              %old = extractvalue { i32, i32 } %pair, 1
              ret i32 %old
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "faultable_atomic_add_i32_acq_rel")
        self.assertEqual(sysops[0].args, (muir.Slot("p"), muir.Slot("x")))
        self.assertIsInstance(sysops[0].result, tuple)
        self.assertEqual(len(sysops[0].result), 2)
        self.assertEqual(stats.lowered_faultable_atomic, 1)

    def test_faultable_cmpxchg_preserves_all_results(self):
        fn, stats = self.lower_one(
            """
            define i32 @futex_cmpx(ptr %p, i64 %oldv, i32 %newv) {
            entry:
              %triple = call { i32, i32, i64 } asm sideeffect "1: lr.w.aqrl $1,$2; bne $1,$4,3f;2: sc.w.aqrl $3,$5,$2; bnez $3,1b;3:; .pushsection __ex_table", "=r,=&r,=*m,=&r,Jr,Jr,0,*m,~{memory}"(ptr elementtype(i32) %p, i64 %oldv, i32 %newv, i32 0, ptr elementtype(i32) %p)
              %err = extractvalue { i32, i32, i64 } %triple, 0
              %val = extractvalue { i32, i32, i64 } %triple, 1
              %status = extractvalue { i32, i32, i64 } %triple, 2
              ret i32 %val
            }
            """
        )
        sysops = [i for b in fn.blocks for i in b.instructions if isinstance(i, muir.Sys)]
        self.assertEqual(len(sysops), 1)
        self.assertEqual(sysops[0].op, "faultable_atomic_cmpxchg_i32_acq_rel")
        self.assertEqual(
            sysops[0].args,
            (muir.Slot("p"), muir.Slot("oldv"), muir.Slot("newv")),
        )
        self.assertIsInstance(sysops[0].result, tuple)
        self.assertEqual(len(sysops[0].result), 3)
        self.assertEqual(stats.lowered_faultable_atomic, 1)

    def test_sext_preserves_exact_source_bit_width(self):
        fn, stats = self.lower_one(
            """
            define i64 @sx(i1 %x) {
            entry:
              %y = sext i1 %x to i64
              ret i64 %y
            }
            """
        )
        mov = next(
            i for b in fn.blocks for i in b.instructions
            if isinstance(i, muir.Mov) and i.dst == muir.Slot("y")
        )
        self.assertEqual(mov.extend, "sext")
        self.assertEqual(mov.src_bits, 1)

    def test_insertvalue_can_overwrite_poison_literal_field(self):
        fn, stats = self.lower_one(
            """
            define i64 @replace(i64 %x) {
            entry:
              %v = insertvalue [2 x i64] [i64 0, i64 poison], i64 %x, 1
              %r = extractvalue [2 x i64] %v, 1
              ret i64 %r
            }
            """
        )
        helpers = [
            i for b in fn.blocks for i in b.instructions
            if isinstance(i, muir.Helper)
        ]
        self.assertTrue(any(i.symbol == "__mm_insertvalue" for i in helpers))
        self.assertTrue(any(i.symbol == "__mm_extractvalue" for i in helpers))

    def test_insertvalue_rejects_live_poison_literal_field(self):
        with self.assertRaisesRegex(ValueError, "live poison/undef"):
            legalize_module(
                """
                define i64 @bad(i64 %x) {
                entry:
                  %v = insertvalue [2 x i64] [i64 0, i64 poison], i64 %x, 0
                  %r = extractvalue [2 x i64] %v, 1
                  ret i64 %r
                }
                """
            )

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
