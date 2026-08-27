#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a compact frozen MiniMachine corpus from Kbuild save-temps."
    )
    p.add_argument("--kbuild", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    entries = [
        line.strip()
        for line in args.manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    for index, rel_bc in enumerate(entries, 1):
        bc = args.kbuild / rel_bc
        stem = bc.with_suffix("")
        i_file = stem.with_suffix(".i")
        if not bc.is_file() or not i_file.is_file():
            raise SystemExit(f"missing frozen material for {rel_bc}")

        rel_stem = Path(rel_bc).with_suffix("")
        out_bc = args.output / rel_stem.with_suffix(".bc")
        out_i = args.output / rel_stem.with_suffix(".i")
        link_or_copy(bc, out_bc)
        link_or_copy(i_file, out_i)

        records.append(
            {
                "index": index,
                "bc": rel_stem.with_suffix(".bc").as_posix(),
                "i": rel_stem.with_suffix(".i").as_posix(),
                "bc_bytes": bc.stat().st_size,
                "i_bytes": i_file.stat().st_size,
                "bc_sha256": sha256(bc),
                "i_sha256": sha256(i_file),
            }
        )

        if index % 25 == 0 or index == len(entries):
            print(f"EXPORT {index}/{len(entries)}")

    with (args.output / "manifest.jsonl").open("w") as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"EXPORTED {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
