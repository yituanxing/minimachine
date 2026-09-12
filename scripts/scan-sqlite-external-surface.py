#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BRIDGE = ROOT / "scripts" / "run-minimachine-linux-pipe.py"


def load_bridge():
    spec = spec_from_file_location("sqlite_external_surface_bridge", BRIDGE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {BRIDGE}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: scan-sqlite-external-surface.py REPORT.json")
    report = json.loads(Path(sys.argv[1]).read_text())
    bridge = load_bridge()
    runner = bridge.load_runner()
    bridge.install_pipe_callbacks(runner)

    supported: list[str] = []
    unsupported: list[str] = []
    for original in sorted(report["external_functions"]):
        symbol = f"__mm_sqlite3534_ext_{original}"
        callback = runner._user_libc_callback(symbol, None)
        if callback is None or getattr(callback, "__name__", "") == "unimplemented":
            unsupported.append(original)
        else:
            supported.append(original)

    print(
        "SQLITE_EXTERNAL_SURFACE "
        f"total={len(supported) + len(unsupported)} "
        f"supported={len(supported)} unsupported={len(unsupported)}"
    )
    for name in unsupported:
        print(f"SQLITE_EXTERNAL_UNSUPPORTED {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
