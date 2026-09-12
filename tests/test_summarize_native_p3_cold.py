from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize-native-p3-cold.py"
SPEC = importlib.util.spec_from_file_location("summarize_native_p3_cold", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class CombinedLogSummaryTests(unittest.TestCase):
    def test_github_prefixed_markers_are_parsed(self):
        prefixes = [prefix for _name, _log, prefix in SUMMARY.STAGES]
        lines = [
            "multicall-cold\tinput\tLOWERING_SHA=4fb5699f12345678",
            "multicall-cold\tinput\tLINUX_SHA=7ca7a47612345678",
            "multicall-cold\tinput\tNATIVE_PACK_SHA=501d444012345678",
            "multicall-cold\tinput\tP3_REFRESH_RUN_ID=101",
            "multicall-cold\tinput\tLINUX_RUN_ID=102",
            "multicall-cold\tinput\tBUSYBOX_SHA=abcdef0123456789",
            "multicall-cold\tinput\tBUSYBOX_RUN_ID=103",
        ]
        for index, prefix in enumerate(prefixes):
            wall = 80.0 - index * 5.0
            lines.extend(
                [
                    f"multicall-cold\tstage\tBOOT_EXEC_CHECKPOINT_SAVED steps=815095846 function=kernel_execve hit=1",
                    f"multicall-cold\tstage\t{prefix}_WALL={wall:.2f} USER=1.20 SYS=0.30 MAXRSS_KB=12345",
                    f"multicall-cold\tstage\techo \"{prefix}_STATUS=$status\"",
                    f"multicall-cold\tstage\t{prefix}_STATUS=0",
                ]
            )
        lines.extend(
            [
                "real-software\trun\tMINIMACHINE_REAL_SOFTWARE_WALL=3.25 USER=2.00 SYS=0.10 MAXRSS_KB=4567",
                "real-software\trun\tMINIMACHINE_REAL_SOFTWARE_STATUS=0",
                "real-software\treport\tMINIMACHINE_REAL_FRONTIER=external-software-passed",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary = SUMMARY.summarize_combined_log(path)

        self.assertEqual(set(summary["stages"]), {name for name, _log, _prefix in SUMMARY.STAGES})
        self.assertEqual(summary["stages"]["real_cold"]["wall_seconds"], 80.0)
        self.assertEqual(summary["stages"]["fast_memory"]["wall_seconds"], 45.0)
        self.assertEqual(summary["stages"]["fast_memory"]["kernel_execve_steps"], 815095846)
        self.assertTrue(
            all(item["steps_equal"] for item in summary["comparisons"].values())
        )
        self.assertEqual(summary["generation"]["p3_refresh_run_id"], "101")
        self.assertEqual(summary["real_software"]["status"], 0)
        self.assertEqual(
            summary["real_software"]["frontier"], "external-software-passed"
        )


if __name__ == "__main__":
    unittest.main()
