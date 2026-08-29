import tempfile
import unittest
from pathlib import Path

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.image import ModuleImage
from src.minimachine.lower_p3 import lower_function
from src.minimachine.program_cache import (
    ProgramCache,
    ProgramCacheError,
    load_program_cache,
    save_program_cache,
)
from src.minimachine.runtime import RuntimeSurface, install_runtime
from src.minimachine.vm import Program


def machine(function: muir.Function):
    expanded, _ = expand_function(function)
    return lower_function(expanded)


class ProgramCacheTests(unittest.TestCase):
    def cache(self):
        fn = muir.Function(
            "entry",
            [muir.Block("entry", [muir.Ret(None)])],
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
            reasons=frozenset({"trap-a"}),
            blocked_functions=(("blocked", 2),),
            image=image,
            task_sched_class_offset=88,
        )

    def test_round_trip_preserves_linked_program(self):
        cache = self.cache()
        descriptor = cache.program.symbol_addresses["entry"]
        entry_code = cache.program.initial_memory.read(descriptor, 64)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "program-cache.pkl.gz"
            save_program_cache(cache, path)
            restored = load_program_cache(path, image_sha256="abc")

        self.assertEqual(restored.function_count, 1)
        self.assertEqual(restored.blocked_functions, (("blocked", 2),))
        self.assertEqual(restored.task_sched_class_offset, 88)
        self.assertEqual(restored.reasons, frozenset({"trap-a"}))
        self.assertEqual(
            restored.program.initial_memory.read(
                restored.program.symbol_addresses["entry"], 64
            ),
            entry_code,
        )

        # Runtime callbacks are intentionally not serialized; they are rebound
        # from the current MiniMachine implementation after cache load.
        self.assertEqual(restored.program.host_services, {})
        install_runtime(restored.program, restored.surface)
        self.assertIn("__mm_llvm_va_end", restored.program.host_services)
        self.assertIn("__mm_sys_fence", restored.program.host_services)

    def test_rejects_other_linked_image(self):
        cache = self.cache()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "program-cache.pkl.gz"
            save_program_cache(cache, path)
            with self.assertRaises(ProgramCacheError):
                load_program_cache(path, image_sha256="different")


if __name__ == "__main__":
    unittest.main()
