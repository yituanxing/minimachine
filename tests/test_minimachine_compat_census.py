from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

from src.minimachine import muir, p3
from src.minimachine.user_image import UserProgramImage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "minimachine-compat-census.py"


def load_census_module():
    spec = spec_from_file_location("minimachine_compat_census", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load census script: {SCRIPT}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniMachineCompatCensusTests(unittest.TestCase):
    def test_reports_unserialized_system_descriptor(self):
        census = load_census_module()
        fn = p3.Function(
            "main",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Mov(
                            muir.Width.I64,
                            muir.Slot("entry"),
                            p3.Mem(
                                muir.Address(muir.Symbol("__mm_sys_fence"), 0),
                                muir.Width.I64,
                            ),
                        )
                    ],
                )
            ],
            {"entry"},
        )
        program = UserProgramImage("main", (fn,))
        result = census.census_program(Path("synthetic.bflt"), program)

        self.assertEqual(result.system_descriptors, ("__mm_sys_fence",))
        self.assertEqual(result.serialized_system_ops, ())
        self.assertEqual(result.missing_system_descriptors, ("__mm_sys_fence",))

    def test_non_system_symbols_do_not_pollute_system_surface(self):
        census = load_census_module()
        fn = p3.Function(
            "main",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="helper"),
                            muir.Target(symbol="helper"),
                        )
                    ],
                )
            ],
            set(),
        )
        helper = p3.Function("helper", [p3.Block("entry", [])], set())
        program = UserProgramImage("main", (fn, helper))
        result = census.census_program(Path("synthetic.bflt"), program)

        self.assertEqual(result.system_descriptors, ())
        self.assertEqual(result.missing_system_descriptors, ())


if __name__ == "__main__":
    unittest.main()
