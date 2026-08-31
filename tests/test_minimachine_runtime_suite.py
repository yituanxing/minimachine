from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/minimachine-runtime-suite.py"


def load_suite():
    spec = spec_from_file_location("minimachine_runtime_suite", SUITE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniMachineRuntimeSuiteTests(unittest.TestCase):
    def test_matrix_is_cumulative_and_frontier_focused(self):
        suite = load_suite()
        self.assertEqual(
            [case.case_id for case in suite.CASES],
            [
                "builtin",
                "shell-state",
                "external-status",
                "subshell",
                "uname",
                "vfs-file",
                "directory",
                "pipeline",
            ],
        )
        self.assertEqual(tuple(suite.PROFILES["smoke"]), suite.CASES[:2])
        self.assertEqual(tuple(suite.PROFILES["core"]), suite.CASES)
        self.assertEqual(
            [case.level for case in suite.CASES],
            [
                "L0-shell",
                "L0-shell",
                "L1-process",
                "L1-process",
                "L1-process",
                "L2-vfs",
                "L2-vfs",
                "L3-pipe",
            ],
        )

    def test_generated_init_has_per_case_start_pass_and_suite_endpoint(self):
        suite = load_suite()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "init.sh"
            suite.write_init(path, "core")
            text = path.read_text()
        self.assertTrue(text.startswith("#!/bin/sh\n"))
        for case in suite.CASES:
            self.assertIn(
                f"MMRT_CASE_START id=%s level=%s\\n' {case.case_id} {case.level}",
                text,
            )
            self.assertIn(
                f"MMRT_CASE_PASS id=%s\\n' {case.case_id}",
                text,
            )
            for marker in case.markers:
                self.assertNotIn(marker, case.command)
        self.assertIn("MMRT_SUITE_PASS profile=%s cases=%s", text)
        self.assertIn('exit "$rc"', text)

    def test_verify_accepts_complete_log_and_rejects_blocked_log(self):
        suite = load_suite()
        lines = [
            "BOOT_EXEC_USER_HANDOFF function=__mm_user_main",
        ]
        for case in suite.CASES:
            lines.append(f"MMRT_CASE_START id={case.case_id} level={case.level}")
            lines.extend(case.markers)
            lines.append(f"MMRT_CASE_PASS id={case.case_id}")
        lines.extend(
            [
                f"MMRT_SUITE_PASS profile=core cases={len(suite.CASES)}",
                "BOOT_EXEC_USER_EXIT status=0",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime.log"
            path.write_text("\n".join(lines) + "\n")
            suite.verify_log(path, "core")
            path.write_text(
                "\n".join(lines + ["BOOT_EXEC_BLOCKED stage=execute"]) + "\n"
            )
            with self.assertRaises(RuntimeError):
                suite.verify_log(path, "core")


if __name__ == "__main__":
    unittest.main()
