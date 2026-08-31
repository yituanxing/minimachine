from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run-minimachine-linux.py"


def load_runner():
    spec = spec_from_file_location("run_minimachine_linux_dynamic_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DynamicUserExternalSurfaceTests(unittest.TestCase):
    def test_runtime_registered_external_descriptor_is_live_immediately(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        image = SimpleNamespace(
            external_functions=("__mm_user_ext_malloc",),
            external_data=("__mm_user_ext_environ",),
        )
        user_image = SimpleNamespace(image=image)

        runner.install_user_external_surface(vm, user_image, 0x12345678)

        descriptor = program.symbol_addresses["__mm_user_ext_malloc"]
        initial_entry = program.initial_memory.read(descriptor, 64)
        initial_frame = program.initial_memory.read(descriptor + 8, 64)
        self.assertNotEqual(initial_entry, 0)
        self.assertEqual(vm.memory.read(descriptor, 64), initial_entry)
        self.assertEqual(vm.memory.read(descriptor + 8, 64), initial_frame)

        environ = program.symbol_addresses["__mm_user_ext_environ"]
        self.assertEqual(vm.memory.read(environ, 64), 0x12345678)


    def test_native_vm_can_call_service_registered_after_vm_creation(self):
        program = Program()
        vm = NativeVM(program)

        program.register_service("__mm_user_ext_probe", lambda _vm, _args: 77)
        result = muir.Slot("result")
        fn = muir.Function(
            "__mm_user_probe_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="__mm_user_ext_probe"),
                            (),
                            result,
                        ),
                        muir.Ret(result),
                    ],
                )
            ],
            {"result"},
        )
        expanded, _ = expand_function(fn)
        program.add_function(lower_function(expanded))

        descriptor = program.symbol_addresses["__mm_user_ext_probe"]
        self.assertNotEqual(program.initial_memory.read(descriptor, 64), 0)
        self.assertEqual(vm.memory.read(descriptor, 64), 0)

        self.assertEqual(
            vm.run_function("__mm_user_probe_caller", result_count=1),
            (77,),
        )
        self.assertEqual(
            vm.memory.read(descriptor, 64),
            program.initial_memory.read(descriptor, 64),
        )

if __name__ == "__main__":
    unittest.main()
