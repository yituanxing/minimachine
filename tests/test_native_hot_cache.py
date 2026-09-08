import tempfile
import unittest
from pathlib import Path

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.image import ModuleImage
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_hot_cache import (
    NativeHotCacheError,
    load_native_hot_cache,
    save_native_hot_cache,
    slim_program_cache,
)
from src.minimachine.program_cache import ProgramCache
from src.minimachine.runtime import RuntimeSurface, install_runtime
from src.minimachine.vm import Program


def machine(function: muir.Function):
    expanded, _ = expand_function(function)
    return lower_function(expanded)


class NativeHotCacheTests(unittest.TestCase):
    def cache(self):
        fn = muir.Function(
            "entry",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(label="done"),
                            muir.Target(label="done"),
                        )
                    ],
                ),
                muir.Block("done", [muir.Ret(None)]),
            ],
            set(),
        )
        program = Program([machine(fn)])
        image = ModuleImage(
            objects=(),
            aliases=(),
            external_data=(),
            external_functions=(),
            skipped_linker_metadata=(),
        )
        return ProgramCache(
            image_sha256="abc",
            program=program,
            surface=RuntimeSurface(
                helpers=frozenset({"__mm_llvm_va_end"}),
                system_ops=frozenset({"fence"}),
            ),
            reasons=frozenset(),
            blocked_functions=(),
            image=image,
            task_sched_class_offset=88,
        )

    def test_slim_cache_drops_base_instruction_bodies_but_keeps_linkage(self):
        cache = self.cache()
        original_blocks = dict(cache.program.block_code)
        original_codes = dict(cache.program.code_block)
        original_descriptor = cache.program.symbol_addresses["entry"]
        original_entry = cache.program.initial_memory.read(
            original_descriptor, 64
        )

        slim = slim_program_cache(cache)

        self.assertEqual(slim.program.block_code, original_blocks)
        self.assertEqual(slim.program.code_block, original_codes)
        linked = slim.program.functions["entry"]
        self.assertEqual(linked.function.blocks[0].label, "entry")
        self.assertEqual(linked.function.blocks[0].instructions, [])
        self.assertEqual(
            slim.program.initial_memory.read(original_descriptor, 64),
            original_entry,
        )
        install_runtime(slim.program, slim.surface)
        self.assertIn("__mm_llvm_va_end", slim.program.host_services)
        self.assertIn("__mm_sys_fence", slim.program.host_services)

    def test_round_trip_checks_linked_image(self):
        cache = self.cache()
        with tempfile.TemporaryDirectory() as td:
            for name in ("native-hot.pkl.gz", "native-hot.pkl"):
                path = Path(td) / name
                save_native_hot_cache(cache, path)
                restored = load_native_hot_cache(path, image_sha256="abc")
                self.assertEqual(restored.function_count, 1)
                self.assertEqual(restored.task_sched_class_offset, 88)
                with self.assertRaises(NativeHotCacheError):
                    load_native_hot_cache(path, image_sha256="different")


if __name__ == "__main__":
    unittest.main()
