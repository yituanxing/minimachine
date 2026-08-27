#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.legalize import LegalizeError, legalize_module
from src.minimachine.llvm_text import LLVMTextError
from src.minimachine.verify import VerifyError, verify_muir


def parse_args():
    p = argparse.ArgumentParser(description="Legalize LLVM bitcode into MiniMachine μIR.")
    p.add_argument("input", type=Path)
    p.add_argument("--llvm-major", default="18")
    p.add_argument("--json", type=Path)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--jobs", type=int, default=0)
    return p.parse_args()


def files_under(path: Path):
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.bc"))


def main() -> int:
    args = parse_args()
    llvm_dis = f"llvm-dis-{args.llvm_major}"
    files = files_under(args.input)
    if not files:
        raise SystemExit(f"no bitcode under {args.input}")

    records = []
    total_stats = {}
    pass_count = 0
    jobs = args.jobs or max(1, os.cpu_count() or 1)

    def one(path: Path):
        rel = path.relative_to(args.input).as_posix() if args.input.is_dir() else path.name
        try:
            proc = subprocess.run(
                [llvm_dis, "-o", "-", str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            functions, stats = legalize_module(proc.stdout)
            for fn in functions:
                verify_muir(fn)
            return {
                "file": rel,
                "status": "PASS",
                "functions": len(functions),
                "stats": stats.as_dict(),
            }
        except (subprocess.CalledProcessError, LegalizeError, LLVMTextError, VerifyError, ValueError) as e:
            return {"file": rel, "status": "FAIL", "error": str(e)}

    print(f"LEGALIZE_START files={len(files)} jobs={jobs}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        future_map = {pool.submit(one, path): path for path in files}
        done = 0
        for future in concurrent.futures.as_completed(future_map):
            rec = future.result()
            records.append(rec)
            done += 1
            if rec["status"] == "PASS":
                pass_count += 1
                for k, v in rec["stats"].items():
                    total_stats[k] = total_stats.get(k, 0) + v
            if done % 25 == 0 or rec["status"] == "FAIL" or done == len(files):
                tail = f" FAIL {rec['file']} :: {rec['error']}" if rec["status"] == "FAIL" else ""
                print(f"LEGALIZE {done}/{len(files)} pass={pass_count} fail={done-pass_count}{tail}", flush=True)

    records.sort(key=lambda x: x["file"])

    summary = {
        "files": len(files),
        "pass": pass_count,
        "fail": len(files) - pass_count,
        "stats": total_stats,
        "records": records,
    }

    print(
        "LEGALIZE_SUMMARY "
        f"pass={summary['pass']}/{summary['files']} "
        f"fail={summary['fail']} "
        f"llvm_instructions={total_stats.get('llvm_instructions', 0)} "
        f"muir_instructions={total_stats.get('muir_instructions', 0)} "
        f"fused_icmp_br={total_stats.get('fused_icmp_br', 0)} "
        f"phi_edge_moves={total_stats.get('phi_edge_moves', 0)} "
        f"folded_gep_mem={total_stats.get('folded_gep_mem', 0)} "
        f"temporary_helpers={total_stats.get('temporary_helpers', 0)}"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.strict and summary["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
