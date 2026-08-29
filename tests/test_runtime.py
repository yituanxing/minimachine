import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.image import install_module_image, parse_module_image
from src.minimachine.legalize import legalize_module
from src.minimachine.lower_p3 import lower_function
from src.minimachine.runtime import (
    accelerate_direct_runtime,
    collect_runtime_surface,
    direct_runtime_callback,
    install_runtime,
)
from src.minimachine.vm import Program


def executable(functions):
    p3_functions = []
    for fn in functions:
        expanded, _ = expand_function(fn)
        p3_functions.append(lower_function(expanded))
    program = Program(p3_functions)
    install_runtime(program, collect_runtime_surface(functions))
    return program


class RuntimeTests(unittest.TestCase):
    def test_direct_runtime_acceleration_rebinds_existing_memset(self):
        memset_fn = muir.Function(
            "memset",
            [muir.Block("entry", [muir.Ret(muir.Slot("dst"))])],
            {"dst", "value", "size"},
            ("dst", "value", "size"),
        )
        expanded, _ = expand_function(memset_fn)
        program = Program([lower_function(expanded)])
        descriptor = program.symbol_addresses["memset"]
        p3_entry = program.initial_memory.read(descriptor, 64)

        accelerated = accelerate_direct_runtime(program)
        host_entry = program.initial_memory.read(descriptor, 64)

        self.assertEqual(accelerated, ("memset",))
        self.assertNotEqual(host_entry, p3_entry)
        self.assertEqual(program.host_code[host_entry], "__mm_fast_memset")

    def test_scalar_helpers_execute_end_to_end(self):
        functions, _ = legalize_module(
            """
            define i64 @scalar(i64 %a, i64 %b) {
            entry:
              %x = and i64 %a, %b
              %y = shl i64 %x, 3
              %z = mul i64 %y, 5
              %c = icmp ult i64 %z, 1000
              %r = select i1 %c, i64 %z, i64 1000
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        self.assertEqual(vm.run_function("scalar", (0x3F, 0x0F)), (600,))

    def test_signed_division_matches_llvm_truncation(self):
        functions, _ = legalize_module(
            """
            define i64 @sdiv_test(i64 %a, i64 %b) {
            entry:
              %q = sdiv i64 %a, %b
              ret i64 %q
            }
            """
        )
        program = executable(functions)
        result = program.new_vm().run_function(
            "sdiv_test",
            ((-7) & ((1 << 64) - 1), 3),
        )
        self.assertEqual(result, (((-2) & ((1 << 64) - 1)),))

    def test_supervisor_state_system_service_executes(self):
        fn = muir.Function(
            "state_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("state_write_status", (muir.Imm(0x12),), None),
                        muir.Sys("state_set_status", (muir.Imm(0x80),), None),
                        muir.Sys("state_read_status", (), muir.Slot("out")),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"out"},
        )
        program = executable([fn])
        self.assertEqual(
            program.new_vm().run_function("state_user"),
            (0x92,),
        )

    def test_atomic_add_mutates_memory_and_returns_old(self):
        fn = muir.Function(
            "atomic_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "atomic_add_i32_relaxed",
                            (muir.Slot("p"), muir.Slot("v")),
                            muir.Slot("old"),
                        ),
                        muir.Ret(muir.Slot("old")),
                    ],
                )
            ],
            {"p", "v", "old"},
            ("p", "v"),
        )
        program = executable([fn])
        vm = program.new_vm()
        vm.memory.write(0x2000, 32, 5)
        self.assertEqual(vm.run_function("atomic_user", (0x2000, 3)), (5,))
        self.assertEqual(vm.memory.read(0x2000, 32), 8)

    def test_faultable_load_returns_error_then_value(self):
        fn = muir.Function(
            "uget",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "faultable_load_i32",
                            (muir.Slot("p"), muir.Imm(0)),
                            (muir.Slot("err"), muir.Slot("value")),
                        ),
                        muir.Ret(muir.Slot("value")),
                    ],
                )
            ],
            {"p", "err", "value"},
            ("p",),
        )
        program = executable([fn])
        vm = program.new_vm()
        vm.memory.write(0x3000, 32, 0x12345678)
        self.assertEqual(vm.run_function("uget", (0x3000,)), (0x12345678,))

    def test_static_key_controls_real_p3_branch(self):
        fn = muir.Function(
            "static_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "static_branch",
                            (muir.Slot("key"),),
                            muir.Slot("enabled"),
                        ),
                        muir.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Slot("enabled"),
                            muir.Imm(0),
                            muir.Target(label="off"),
                            muir.Target(label="on"),
                        ),
                    ],
                ),
                muir.Block("off", [muir.Ret(muir.Imm(10))]),
                muir.Block("on", [muir.Ret(muir.Imm(20))]),
            ],
            {"key", "enabled"},
            ("key",),
        )
        program = executable([fn])

        off_vm = program.new_vm()
        self.assertEqual(off_vm.run_function("static_user", (7,)), (10,))

        on_vm = program.new_vm()
        on_vm.static_keys[7] = 1
        self.assertEqual(on_vm.run_function("static_user", (7,)), (20,))

    def test_ecall_requires_explicit_handler(self):
        fn = muir.Function(
            "ecall_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "ecall",
                            (muir.Slot("x"),),
                            (muir.Slot("a"), muir.Slot("b")),
                        ),
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("d"),
                            muir.Slot("b"),
                            muir.Slot("a"),
                        ),
                        muir.Ret(muir.Slot("d")),
                    ],
                )
            ],
            {"x", "a", "b", "d"},
            ("x",),
        )
        program = executable([fn])
        vm = program.new_vm()
        with self.assertRaisesRegex(Exception, "ecall reached without"):
            vm.run_function("ecall_user", (9,))

        vm = program.new_vm()
        vm.ecall_handler = lambda _vm, args: (args[0], args[0] + 4)
        self.assertEqual(vm.run_function("ecall_user", (9,)), (4,))

    def test_vector_state_snapshot_and_restore_execute(self):
        fn = muir.Function(
            "vector_state_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "vector_state_snapshot",
                            (),
                            (
                                muir.Slot("a"),
                                muir.Slot("b"),
                                muir.Slot("c"),
                                muir.Slot("d"),
                                muir.Slot("e"),
                            ),
                        ),
                        muir.Sys(
                            "vector_state_restore",
                            (
                                muir.Imm(11),
                                muir.Imm(12),
                                muir.Imm(13),
                                muir.Imm(14),
                            ),
                            None,
                        ),
                        muir.Sys(
                            "vector_state_snapshot",
                            (),
                            (
                                muir.Slot("a2"),
                                muir.Slot("b2"),
                                muir.Slot("c2"),
                                muir.Slot("d2"),
                                muir.Slot("e2"),
                            ),
                        ),
                        muir.Ret(muir.Slot("e2")),
                    ],
                )
            ],
            {"a","b","c","d","e","a2","b2","c2","d2","e2"},
        )
        program = executable([fn])
        vm = program.new_vm()
        vm.vector_state = (1, 2, 3, 4, 64)
        self.assertEqual(vm.run_function("vector_state_user"), (64,))
        self.assertEqual(vm.vector_state, (11, 12, 13, 14, 64))

    def test_llvm_fixed_register_reads_execute(self):
        functions, stats = legalize_module(
            """
            define i64 @read_sp() {
            entry:
              %v = call i64 @llvm.read_register.i64(metadata !0)
              ret i64 %v
            }

            define i64 @read_tp() {
            entry:
              %v = call i64 @llvm.read_register.i64(metadata !1)
              ret i64 %v
            }

            !0 = !{!"sp"}
            !1 = !{!"tp"}
            """
        )
        self.assertEqual(stats.lowered_read_sp, 1)
        self.assertEqual(stats.lowered_read_thread_pointer, 1)

        program = executable(functions)

        sp_vm = program.new_vm()
        sp_result = sp_vm.run_function("read_sp")
        self.assertEqual(sp_result[0], sp_vm.stack_top - program.functions["read_sp"].frame_size - 8)

        tp_vm = program.new_vm()
        tp_vm.system_state["thread_pointer"] = 0xABCDEF
        self.assertEqual(tp_vm.run_function("read_tp"), (0xABCDEF,))

    def test_anonymous_array_load_store_round_trip_executes(self):
        functions, _ = legalize_module(
            """
            define void @copy_array(ptr %src, ptr %dst) {
            entry:
              %v = load [2 x i64], ptr %src
              store [2 x i64] %v, ptr %dst
              ret void
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        source = 0x3A00
        dest = 0x3B00
        vm.memory.write(source + 0, 64, 0x1122334455667788)
        vm.memory.write(source + 8, 64, 0x99AABBCCDDEEFF00)
        self.assertEqual(
            vm.run_function("copy_array", (source, dest), result_count=0),
            (),
        )
        self.assertEqual(
            vm.memory.read(dest + 0, 64),
            0x1122334455667788,
        )
        self.assertEqual(
            vm.memory.read(dest + 8, 64),
            0x99AABBCCDDEEFF00,
        )

    def test_aggregate_load_store_round_trip_executes(self):
        functions, _ = legalize_module(
            """
            %Pair = type { i32, i32 }

            define void @copy_pair(ptr %src, ptr %dst) {
            entry:
              %v = load %Pair, ptr %src
              store %Pair %v, ptr %dst
              ret void
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        source = 0x4000
        dest = 0x5000
        raw = bytes([1, 2, 3, 4, 0xAA, 0xBB, 0xCC, 0xDD])
        for i, byte in enumerate(raw):
            vm.memory.write(source + i, 8, byte)

        self.assertEqual(
            vm.run_function("copy_pair", (source, dest), result_count=0),
            (),
        )
        copied = bytes(vm.memory.read(dest + i, 8) for i in range(len(raw)))
        self.assertEqual(copied, raw)

    def test_extractvalue_uses_struct_padding_offset(self):
        functions, _ = legalize_module(
            """
            %Padded = type { i8, i32 }

            define i32 @get_second(ptr %src) {
            entry:
              %v = load %Padded, ptr %src
              %x = extractvalue %Padded %v, 1
              ret i32 %x
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        source = 0x6000
        vm.memory.write(source + 0, 8, 0x7F)
        vm.memory.write(source + 4, 32, 0xA1B2C3D4)
        self.assertEqual(
            vm.run_function("get_second", (source,)),
            (0xA1B2C3D4,),
        )

    def test_nested_insertvalue_round_trip_executes(self):
        functions, _ = legalize_module(
            """
            %Outer = type { i8, [2 x i32] }

            define void @put_nested(ptr %dst, i32 %x) {
            entry:
              %v = insertvalue %Outer zeroinitializer, i32 %x, 1, 1
              store %Outer %v, ptr %dst
              ret void
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        dest = 0x7000
        self.assertEqual(
            vm.run_function("put_nested", (dest, 0x11223344), result_count=0),
            (),
        )
        self.assertEqual(vm.memory.read(dest + 0, 8), 0)
        self.assertEqual(vm.memory.read(dest + 4, 32), 0)
        self.assertEqual(vm.memory.read(dest + 8, 32), 0x11223344)

    def test_insertvalue_preserves_nonzero_literal_field(self):
        functions, _ = legalize_module(
            """
            define i64 @replace_poison(i64 %x) {
            entry:
              %v = insertvalue [2 x i64] [i64 1, i64 poison], i64 %x, 1
              %r = extractvalue [2 x i64] %v, 0
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function("replace_poison", (99,)),
            (1,),
        )

    def test_funnel_shift_intrinsics_execute_with_distinct_halves(self):
        functions, _ = legalize_module(
            """
            declare i32 @llvm.fshl.i32(i32, i32, i32)
            declare i32 @llvm.fshr.i32(i32, i32, i32)

            define i32 @left(i32 %a, i32 %b, i32 %s) {
            entry:
              %r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %s)
              ret i32 %r
            }

            define i32 @right(i32 %a, i32 %b, i32 %s) {
            entry:
              %r = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %s)
              ret i32 %r
            }
            """
        )
        program = executable(functions)
        a = 0x12345678
        b = 0x9ABCDEF0
        self.assertEqual(program.new_vm().run_function("left", (a, b, 0)), (a,))
        self.assertEqual(program.new_vm().run_function("right", (a, b, 0)), (b,))
        self.assertEqual(
            program.new_vm().run_function("left", (a, b, 4)),
            (0x23456789,),
        )
        self.assertEqual(
            program.new_vm().run_function("right", (a, b, 4)),
            (0x89ABCDEF,),
        )
        # Shift amount is modulo the element width.
        self.assertEqual(
            program.new_vm().run_function("left", (a, b, 36)),
            (0x23456789,),
        )

    def test_odd_width_i24_memory_executes(self):
        functions, _ = legalize_module(
            """
            define i24 @roundtrip(ptr %p, i24 %x) {
            entry:
              store i24 %x, ptr %p
              %r = load i24, ptr %p
              ret i24 %r
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        self.assertEqual(
            vm.run_function("roundtrip", (0x8100, 0xABCDEF)),
            (0xABCDEF,),
        )
        self.assertEqual(vm.memory.read(0x8100, 8), 0xEF)
        self.assertEqual(vm.memory.read(0x8101, 8), 0xCD)
        self.assertEqual(vm.memory.read(0x8102, 8), 0xAB)

    def test_bit_count_intrinsics_execute(self):
        functions, _ = legalize_module(
            """
            declare i64 @llvm.cttz.i64(i64, i1 immarg)
            declare i32 @llvm.ctlz.i32(i32, i1 immarg)
            declare i64 @llvm.ctpop.i64(i64)

            define i64 @tz(i64 %x) {
            entry:
              %r = call i64 @llvm.cttz.i64(i64 %x, i1 false)
              ret i64 %r
            }

            define i32 @lz(i32 %x) {
            entry:
              %r = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
              ret i32 %r
            }

            define i64 @pop(i64 %x) {
            entry:
              %r = call i64 @llvm.ctpop.i64(i64 %x)
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        self.assertEqual(program.new_vm().run_function("tz", (0x1000,)), (12,))
        self.assertEqual(program.new_vm().run_function("tz", (0,)), (64,))
        self.assertEqual(program.new_vm().run_function("lz", (1,)), (31,))
        self.assertEqual(program.new_vm().run_function("lz", (0,)), (32,))
        self.assertEqual(
            program.new_vm().run_function("pop", (0xF0F0F0F0F0F0F0F0,)),
            (32,),
        )

    def test_static_alloca_store_load_executes(self):
        functions, _ = legalize_module(
            """
            define i64 @local_roundtrip(i64 %x) {
            entry:
              %p = alloca i64, align 8
              store i64 %x, ptr %p
              %y = load i64, ptr %p
              ret i64 %y
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function("local_roundtrip", (0x123456789ABCDEF0,)),
            (0x123456789ABCDEF0,),
        )

    def test_single_use_gep_stored_as_pointer_value_executes(self):
        functions, _ = legalize_module(
            """
            define ptr @gep_pointer_value_store(ptr %base) {
            entry:
              %slot = alloca ptr, align 8
              %next = getelementptr i8, ptr %base, i64 1
              store ptr %next, ptr %slot, align 8
              %got = load ptr, ptr %slot, align 8
              ret ptr %got
            }
            """
        )
        program = executable(functions)
        base = 0x12345000
        self.assertEqual(
            program.new_vm().run_function("gep_pointer_value_store", (base,)),
            (base + 1,),
        )

    def test_linker_symbol_difference_materializes_at_runtime(self):
        llvm = """
            @begin = external global i8
            @end = external global i8

            define i64 @symbol_distance() {
            entry:
              %distance = sdiv i64 sub (i64 ptrtoint (ptr @end to i64), i64 ptrtoint (ptr @begin to i64)), 8
              ret i64 %distance
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(
            program,
            parse_module_image(llvm),
            external_symbols={"begin": 0x12000, "end": 0x12080},
        )

        self.assertEqual(
            program.new_vm().run_function("symbol_distance"),
            (16,),
        )

    def test_standalone_icmp_materializes_linker_symbol_difference(self):
        llvm = """
            @begin = external global i8
            @end = external global i8

            define i64 @standalone_symbol_distance_check() {
            entry:
              %same = icmp sgt i64 sub (i64 ptrtoint (ptr @end to i64), i64 ptrtoint (ptr @begin to i64)), 0
              %result = zext i1 %same to i64
              ret i64 %result
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(
            program,
            parse_module_image(llvm),
            external_symbols={"begin": 0x12000, "end": 0x12080},
        )

        self.assertEqual(
            program.new_vm().run_function("standalone_symbol_distance_check"),
            (1,),
        )

    def test_phi_inline_icmp_materializes_linker_symbol_difference(self):
        llvm = """
            @begin = external global i8
            @end = external global i8

            define i64 @phi_symbol_distance_check() {
            entry:
              br i1 true, label %yes, label %no

            yes:
              br label %join

            no:
              br label %join

            join:
              %same = phi i1 [ icmp sgt (i64 sub (i64 ptrtoint (ptr @end to i64), i64 ptrtoint (ptr @begin to i64)), i64 0), %yes ], [ false, %no ]
              %result = zext i1 %same to i64
              ret i64 %result
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(
            program,
            parse_module_image(llvm),
            external_symbols={"begin": 0x12000, "end": 0x12080},
        )

        self.assertEqual(
            program.new_vm().run_function("phi_symbol_distance_check"),
            (1,),
        )

    def test_inline_icmp_materializes_linker_symbol_difference(self):
        llvm = """
            @begin = external global i8
            @end = external global i8

            define i64 @inline_symbol_distance_check() {
            entry:
              br i1 icmp eq (i64 sub (i64 ptrtoint (ptr @end to i64), i64 ptrtoint (ptr @begin to i64)), i64 128), label %yes, label %no

            yes:
              ret i64 1

            no:
              ret i64 0
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(
            program,
            parse_module_image(llvm),
            external_symbols={"begin": 0x12000, "end": 0x12080},
        )

        self.assertEqual(
            program.new_vm().run_function("inline_symbol_distance_check"),
            (1,),
        )

    def test_inline_icmp_gep_branch_and_phi_execute(self):
        llvm = """
            %struct.sched_class = type { i64, i64 }
            @classes = global [2 x %struct.sched_class] zeroinitializer, align 8

            define i64 @inline_sched_class_checks() {
            entry:
              br i1 icmp eq (ptr getelementptr inbounds ([2 x %struct.sched_class], ptr @classes, i64 0, i64 1), ptr getelementptr inbounds ([2 x %struct.sched_class], ptr @classes, i64 0, i64 1)), label %ok, label %bad, !prof !19

            ok:
              br label %join

            bad:
              br label %join

            join:
              %same = phi i1 [ icmp eq (ptr getelementptr inbounds ([2 x %struct.sched_class], ptr @classes, i64 0, i64 1), ptr getelementptr inbounds ([2 x %struct.sched_class], ptr @classes, i64 0, i64 1)), %ok ], [ false, %bad ]
              %result = zext i1 %same to i64
              ret i64 %result
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(program, parse_module_image(llvm))

        self.assertEqual(
            program.new_vm().run_function("inline_sched_class_checks"),
            (1,),
        )

    def test_nested_linker_symbol_constant_expression_executes(self):
        llvm = """
            @endmarker = external global i8

            define i64 @align_endmarker() {
            entry:
              %aligned = and i64 add (i64 ptrtoint (ptr @endmarker to i64), i64 4095), -4096
              ret i64 %aligned
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(
            program,
            parse_module_image(llvm),
            external_symbols={"endmarker": 0x12345},
        )

        self.assertEqual(
            program.new_vm().run_function("align_endmarker"),
            (0x13000,),
        )

    def test_constant_gep_stored_as_pointer_value_executes(self):
        llvm = """
            %struct.sample = type { i8, i64, i64 }

            @sample = global %struct.sample zeroinitializer, align 8

            define ptr @field_pointer() {
            entry:
              %slot = alloca ptr, align 8
              store ptr getelementptr inbounds (%struct.sample, ptr @sample, i32 0, i32 2), ptr %slot, align 8
              %got = load ptr, ptr %slot, align 8
              ret ptr %got
            }
        """
        functions, _ = legalize_module(llvm)
        program = executable(functions)
        install_module_image(program, parse_module_image(llvm))

        self.assertEqual(
            program.new_vm().run_function("field_pointer"),
            (program.symbol_addresses["sample"] + 16,),
        )

    def test_dynamic_alloca_and_gep_execute(self):
        functions, _ = legalize_module(
            """
            define i32 @dynamic_local(i64 %n, i64 %idx, i32 %x) {
            entry:
              %base = alloca i32, i64 %n, align 16
              %p = getelementptr i32, ptr %base, i64 %idx
              store i32 %x, ptr %p
              %y = load i32, ptr %p
              ret i32 %y
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        self.assertEqual(
            vm.run_function("dynamic_local", (8, 5, 0xAABBCCDD)),
            (0xAABBCCDD,),
        )

    def test_word_varargs_start_copy_end_execute(self):
        functions, _ = legalize_module(
            """
            declare void @llvm.va_start(ptr)
            declare void @llvm.va_copy(ptr, ptr)
            declare void @llvm.va_end(ptr)

            define i64 @first_vararg(i64 %fixed, ...) {
            entry:
              %ap = alloca ptr, align 8
              %cp = alloca ptr, align 8
              call void @llvm.va_start(ptr %ap)
              call void @llvm.va_copy(ptr %cp, ptr %ap)
              %cursor = load ptr, ptr %cp
              %value = load i64, ptr %cursor
              call void @llvm.va_end(ptr %cp)
              call void @llvm.va_end(ptr %ap)
              ret i64 %value
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function(
                "first_vararg",
                (0x1111, 0x2222333344445555),
            ),
            (0x2222333344445555,),
        )

    def test_wide_i128_math_executes_through_blob_values(self):
        functions, _ = legalize_module(
            """
            define i64 @wide_math(i64 %x) {
            entry:
              %a = zext i64 %x to i128
              %b = shl i128 %a, 68
              %c = or i128 %b, 15
              %d = lshr i128 %c, 64
              %r = trunc i128 %d to i64
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function("wide_math", (0x123,)),
            (0x1230,),
        )

    def test_i128_load_preserves_high_word(self):
        functions, _ = legalize_module(
            """
            define i64 @load_high(ptr %p) {
            entry:
              %w = load i128, ptr %p, align 16
              %h = lshr i128 %w, 64
              %r = trunc i128 %h to i64
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        address = 0x9000
        vm.memory.write(address + 0, 64, 0x0123456789ABCDEF)
        vm.memory.write(address + 8, 64, 0xFEDCBA9876543210)
        self.assertEqual(
            vm.run_function("load_high", (address,)),
            (0xFEDCBA9876543210,),
        )

    def test_i65_signed_overflow_aggregate_executes(self):
        functions, _ = legalize_module(
            """
            declare { i65, i1 } @llvm.sadd.with.overflow.i65(i65, i65)

            define i1 @overflow65(i64 %x, i64 %y) {
            entry:
              %a = zext i64 %x to i65
              %b = zext i64 %y to i65
              %pair = call { i65, i1 } @llvm.sadd.with.overflow.i65(i65 %a, i65 %b)
              %ov = extractvalue { i65, i1 } %pair, 1
              ret i1 %ov
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function(
                "overflow65",
                (((1 << 64) - 1), 1),
            ),
            (1,),
        )

    def test_signed_saturating_add_executes(self):
        functions, _ = legalize_module(
            """
            declare i32 @llvm.sadd.sat.i32(i32, i32)

            define i32 @sat_add(i32 %a, i32 %b) {
            entry:
              %r = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b)
              ret i32 %r
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function("sat_add", (0x7fffffff, 1)),
            (0x7fffffff,),
        )
        self.assertEqual(
            program.new_vm().run_function("sat_add", (0x80000000, 0xffffffff)),
            (0x80000000,),
        )

    def test_pointer_scaled_helper_executes(self):
        fn = muir.Function(
            "ptr",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Helper(
                            "__mm_ptr_add_scaled_24",
                            (muir.Slot("base"), muir.Slot("index")),
                            muir.Slot("out"),
                        ),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"base", "index", "out"},
            ("base", "index"),
        )
        program = executable([fn])
        self.assertEqual(
            program.new_vm().run_function("ptr", (1000, 7)),
            (1168,),
        )


    def test_portable_string_runtime_services_execute(self):
        functions, _ = legalize_module(
            """
            declare i64 @strlen(ptr)
            declare i32 @strcmp(ptr, ptr)
            declare i32 @strncmp(ptr, ptr, i64)

            define i64 @runtime_strlen(ptr %p) {
            entry:
              %n = call i64 @strlen(ptr %p)
              ret i64 %n
            }

            define i32 @runtime_strcmp(ptr %a, ptr %b) {
            entry:
              %r = call i32 @strcmp(ptr %a, ptr %b)
              ret i32 %r
            }

            define i32 @runtime_strncmp(ptr %a, ptr %b, i64 %n) {
            entry:
              %r = call i32 @strncmp(ptr %a, ptr %b, i64 %n)
              ret i32 %r
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        a = 0xA000
        b = 0xA100
        for i, byte in enumerate(b"mini" + bytes([0])):
            vm.memory.write(a + i, 8, byte)
        for i, byte in enumerate(b"mino" + bytes([0])):
            vm.memory.write(b + i, 8, byte)

        self.assertEqual(vm.run_function("runtime_strlen", (a,)), (4,))
        self.assertEqual(vm.run_function("runtime_strcmp", (a, a)), (0,))
        self.assertEqual(vm.run_function("runtime_strncmp", (a, b, 3)), (0,))
        self.assertNotEqual(vm.run_function("runtime_strncmp", (a, b, 4)), (0,))

    def test_portable_memmove_runtime_handles_overlap(self):
        callback = direct_runtime_callback("memmove")
        self.assertIsNotNone(callback)

        vm = Program().new_vm()
        address = 0xB000
        for i, byte in enumerate(b"abcd" + bytes([0])):
            vm.memory.write(address + i, 8, byte)

        assert callback is not None
        self.assertEqual(
            callback(vm, (address + 1, address, 4)),
            address + 1,
        )
        self.assertEqual(
            bytes(vm.memory.read(address + i, 8) for i in range(5)),
            b"aabcd",
        )

    def test_objectsize_unknown_fallback_executes(self):
        functions, _ = legalize_module(
            """
            declare i64 @llvm.objectsize.i64.p0(ptr, i1, i1, i1)

            define i64 @obj_max(ptr %p) {
            entry:
              %r = call i64 @llvm.objectsize.i64.p0(ptr %p, i1 false, i1 true, i1 false)
              ret i64 %r
            }

            define i64 @obj_min(ptr %p) {
            entry:
              %r = call i64 @llvm.objectsize.i64.p0(ptr %p, i1 true, i1 true, i1 false)
              ret i64 %r
            }

            define i64 @obj_null() {
            entry:
              %r = call i64 @llvm.objectsize.i64.p0(ptr null, i1 false, i1 false, i1 false)
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        self.assertEqual(
            program.new_vm().run_function("obj_max", (0x4000,)),
            ((1 << 64) - 1,),
        )
        self.assertEqual(
            program.new_vm().run_function("obj_min", (0x4000,)),
            (0,),
        )
        self.assertEqual(
            program.new_vm().run_function("obj_null"),
            (0,),
        )

    def test_expect_and_is_constant_intrinsics_execute(self):
        functions, _ = legalize_module(
            """
            declare i64 @llvm.expect.i64(i64, i64)
            declare i1 @llvm.is.constant.i64(i64)

            define i64 @expect_user(i64 %x) {
            entry:
              %r = call i64 @llvm.expect.i64(i64 %x, i64 7)
              ret i64 %r
            }

            define i1 @is_constant_user(i64 %x) {
            entry:
              %r = call i1 @llvm.is.constant.i64(i64 %x)
              ret i1 %r
            }
            """
        )
        program = executable(functions)
        self.assertEqual(program.new_vm().run_function("expect_user", (42,)), (42,))
        self.assertEqual(program.new_vm().run_function("is_constant_user", (42,)), (0,))

if __name__ == "__main__":
    unittest.main()
