#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOTS = [
    "init", "kernel", "mm", "fs", "net", "lib", "arch", "drivers",
    "block", "ipc", "security", "crypto", "io_uring", "virt", "sound",
    "certs",
]


def parse_args():
    p=argparse.ArgumentParser(description="Select a deterministic diverse focused corpus.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=16)
    return p.parse_args()


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    args=parse_args()
    records=[json.loads(x) for x in (args.input/"manifest.jsonl").read_text().splitlines() if x.strip()]
    by_path={r["bc"]:r for r in records}
    paths=sorted(by_path)
    chosen=[]
    used=set()

    for root in ROOTS:
        prefix=root+"/"
        for p in paths:
            if p not in used and p.startswith(prefix):
                chosen.append(p); used.add(p); break
        if len(chosen)>=args.count:
            break

    for p in paths:
        if len(chosen)>=args.count: break
        if p not in used:
            chosen.append(p); used.add(p)

    args.output.mkdir(parents=True, exist_ok=True)
    out_records=[]
    for p in chosen:
        src=args.input/p
        dst=args.output/p
        link_or_copy(src,dst)
        out_records.append(by_path[p])

    (args.output/"manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True)+"\n" for r in out_records)
    )
    (args.output/"focused.txt").write_text("".join(p+"\n" for p in chosen))
    print(f"FOCUSED {len(chosen)}/{args.count}")
    for p in chosen:
        print(p)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
