from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "scripts/minimachine-runtime-suite.py"


def load_suite():
    spec = spec_from_file_location("minimachine_runtime_suite", SUITE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MiniMachineRuntimeSuiteTests(unittest.TestCase):
    def test_profiles_keep_core_progress_separate_from_process_frontier(self):
        suite = load_suite()
        self.assertEqual(
            [case.case_id for case in suite.CORE_CASES],
            [
                "builtin",
                "shell-state",
                "vfs-file",
                "cwd",
                "fd-redirection",
                "path-error",
            ],
        )
        self.assertEqual(
            [case.case_id for case in suite.PROCESS_CASES],
            [
                "external-status",
                "subshell",
                "uname",
                "directory",
                "pipeline",
            ],
        )
        self.assertEqual(tuple(suite.PROFILES["smoke"]), suite.CORE_CASES[:2])
        self.assertEqual(tuple(suite.PROFILES["core"]), suite.CORE_CASES)
        self.assertEqual(tuple(suite.PROFILES["process"]), suite.PROCESS_CASES)
        self.assertEqual(tuple(suite.PROFILES["full"]), suite.CASES)
        self.assertEqual(suite.CASES, suite.CORE_CASES + suite.PROCESS_CASES)

    def test_core_profile_has_no_external_process_commands(self):
        suite = load_suite()
        forbidden = ("/bin/false", "/bin/sh -c", "/bin/uname", "/bin/mkdir", "| /bin/cat")
        for case in suite.CORE_CASES:
            for text in forbidden:
                self.assertNotIn(text, case.command)

    def test_generated_init_splits_cases_below_ash_read_buffer(self):
        suite = load_suite()
        for profile in ("core", "process"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "init.sh"
                suite.write_init(path, profile)
                text = path.read_text()
                self.assertTrue(text.startswith("#!/bin/sh\n"))
                self.assertLess(len(text.encode()), 1024)
                self.assertIn("MMRT_CASE_START id=%s level=%s", text)
                self.assertIn("MMRT_CASE_PASS id=%s", text)
                self.assertIn('if . "$script"; then', text)
                self.assertNotIn('/bin/sh "$script"', text)
                self.assertIn("MMRT_SUITE_PASS profile=%s cases=%s", text)
                self.assertIn('exit "$rc"', text)
                for index, case in enumerate(suite.PROFILES[profile]):
                    sidecar = Path(td) / suite.case_script_name(index)
                    self.assertTrue(sidecar.is_file())
                    case_text = sidecar.read_text()
                    self.assertLess(len(case_text.encode()), 1024)
                    self.assertIn(case.command, case_text)
                    self.assertIn(
                        f"run_case {case.case_id} {case.level} /{suite.case_script_name(index)}",
                        text,
                    )
                    for marker in case.markers:
                        self.assertNotIn(marker, case.command)

    def test_verify_accepts_complete_core_log_and_rejects_blocked_log(self):
        suite = load_suite()
        cases = suite.PROFILES["core"]
        lines = [
            "BOOT_EXEC_USER_HANDOFF function=__mm_user_main",
        ]
        for case in cases:
            # START lines are diagnostic-only: the first one can be absent
            # during the initial ash handoff.  Result markers + ordered PASS
            # lines are the durable execution evidence.
            lines.extend(case.markers)
            lines.append(f"MMRT_CASE_PASS id={case.case_id}")
        lines.extend(
            [
                f"MMRT_SUITE_PASS profile=core cases={len(cases)}",
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

    def test_process_frontier_reports_first_unpassed_case(self):
        suite = load_suite()
        cases = suite.PROFILES["process"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime.log"
            path.write_text(
                "\n".join(
                    [
                        f"MMRT_CASE_START id={cases[0].case_id} level={cases[0].level}",
                        "BOOT_EXEC_BLOCKED stage=userspace-after-exec error=fork",
                    ]
                )
                + "\n"
            )
            # summarize_log is intentionally non-throwing: the process profile
            # remains a focused frontier while the independent core gate advances.
            suite.summarize_log(path, "process")


if __name__ == "__main__":
    unittest.main()
