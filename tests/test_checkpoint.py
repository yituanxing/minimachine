import tempfile
import unittest
from pathlib import Path

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from src.minimachine.lower_p3 import lower_function
from src.minimachine.vm import Program


def machine(function: muir.Function):
    expanded, _ = expand_function(function)
    return lower_function(expanded)


class CheckpointTests(unittest.TestCase):
    def test_vm_checkpoint_round_trip(self):
        fn = muir.Function(
            "entry",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        program = Program([machine(fn)])
        vm = program.new_vm()
        vm.enter_function("entry", (), stack_top=vm.stack_top, result_count=0)
        vm.steps = 12345
        vm.system_state["clock"] = 9
        vm.csr[7] = 11
        vm.static_keys[0x1234] = 1
        vm.cpu_features.add(5)
        vm.vector_state = (1, 2, 3, 4, 5)
        vm.heap_next += 0x100
        vm.memory.write(0x20000, 64, 0xDEADBEEF)
        vm.linux_task_contexts = {0x10: (0x20, 0x30, 0x40)}
        vm.linux_task_shadow_stacks = {0x10: 0x5000}
        vm.linux_shadow_stack_next = 0x6000
        vm.linux_task_sched_class_offset = 88

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "linux.chk.gz"
            save_checkpoint(vm, path, image_sha256="abc")

            restored = program.new_vm()
            load_checkpoint(restored, path, image_sha256="abc")

        self.assertEqual(restored.steps, 12345)
        self.assertEqual(restored.current_function, "entry")
        self.assertEqual(restored.current_block, "entry")
        self.assertEqual(restored.ip, 0)
        self.assertEqual(restored.memory.read(0x20000, 64), 0xDEADBEEF)
        self.assertEqual(restored.system_state, {"clock": 9})
        self.assertEqual(restored.csr, {7: 11})
        self.assertEqual(restored.static_keys, {0x1234: 1})
        self.assertEqual(restored.cpu_features, {5})
        self.assertEqual(restored.vector_state, (1, 2, 3, 4, 5))
        self.assertEqual(restored.linux_task_contexts, {0x10: (0x20, 0x30, 0x40)})
        self.assertEqual(restored.linux_task_shadow_stacks, {0x10: 0x5000})
        self.assertEqual(restored.linux_shadow_stack_next, 0x6000)
        self.assertEqual(restored.linux_task_sched_class_offset, 88)

    def test_checkpoint_rejects_other_linked_image(self):
        fn = muir.Function(
            "entry",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        program = Program([machine(fn)])
        vm = program.new_vm()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "linux.chk.gz"
            save_checkpoint(vm, path, image_sha256="abc")
            with self.assertRaises(CheckpointError):
                load_checkpoint(
                    program.new_vm(),
                    path,
                    image_sha256="different",
                )


if __name__ == "__main__":
    unittest.main()
