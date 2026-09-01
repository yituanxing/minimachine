from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from src.minimachine.vm import Program


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run-minimachine-linux.py"


def load_runner():
    spec = spec_from_file_location("run_minimachine_linux_namespace_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UserExternalNamespaceTests(unittest.TestCase):
    def test_namespaced_external_surface_uses_its_own_errno_and_data(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        image = SimpleNamespace(
            external_functions=(
                "__mm_lua54_ext_malloc",
                "__mm_lua54_ext___errno_location",
            ),
            external_data=(
                "__mm_lua54_ext_environ",
                "__mm_lua54_ext_optarg",
                "__mm_lua54_ext_optind",
                "__mm_lua54_ext_optopt",
            ),
        )
        user_image = SimpleNamespace(image=image)

        runner.install_user_external_surface(vm, user_image, 0x12345678)

        self.assertEqual(
            runner._user_external_prefix("__mm_lua54_ext_malloc"),
            "__mm_lua54_ext_",
        )
        self.assertEqual(
            runner._user_external_original("__mm_lua54_ext_malloc"),
            "malloc",
        )
        self.assertEqual(
            vm.memory.read(
                program.symbol_addresses["__mm_lua54_ext_environ"],
                64,
            ),
            0x12345678,
        )
        self.assertIn("__mm_lua54_ext_malloc", program.symbol_addresses)
        self.assertIn("__mm_lua54_ext___errno_cell", program.symbol_addresses)

        errno_cb = runner._user_libc_callback(
            "__mm_lua54_ext___errno_location",
            program.symbol_addresses["__mm_lua54_ext___errno_cell"],
        )
        self.assertIsNotNone(errno_cb)
        assert errno_cb is not None
        self.assertEqual(
            errno_cb(vm, ()),
            program.symbol_addresses["__mm_lua54_ext___errno_cell"],
        )


if __name__ == "__main__":
    unittest.main()
