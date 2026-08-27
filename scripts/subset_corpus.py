#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description="Create a deterministic prefix corpus from a frozen corpus.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    return p.parse_args()


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src,dst)
    except OSError:
        shutil.copy2(src,dst)


def main():
    a=parse_args()
    records=[json.loads(x) for x in (a.input/"manifest.jsonl").read_text().splitlines() if x.strip()]
    records=sorted(records,key=lambda r:r["bc"])[:a.count]
    if len(records)!=a.count:
        raise SystemExit(f"requested {a.count}, only {len(records)} records available")
    a.output.mkdir(parents=True,exist_ok=True)
    for r in records:
        link_or_copy(a.input/r["bc"], a.output/r["bc"])
    (a.output/"manifest.jsonl").write_text(
        "".join(json.dumps(r,sort_keys=True)+"\n" for r in records)
    )
    print(f"SUBSET {len(records)}/{a.count}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
