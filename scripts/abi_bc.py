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

from src.minimachine import muir
from src.minimachine.abi import AbiError, expand_function
from src.minimachine.legalize import LegalizeError, legalize_module
from src.minimachine.llvm_text import LLVMTextError
from src.minimachine.lower_p3 import MachineLoweringError, lower_function
from src.minimachine.verify import VerifyError, verify_muir, verify_p3


def parse_args():
    p = argparse.ArgumentParser(description="Run LLVM -> μIR -> ABI -> strict-P3 structural gates.")
    p.add_argument("input", type=Path)
    p.add_argument("--llvm-major", default="18")
    p.add_argument("--json", type=Path)
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def files_under(path: Path):
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.bc"))


def _has_system_escape(fn: muir.Function) -> tuple[int, int]:
    traps = 0
    arch = 0
    for block in fn.blocks:
        for inst in block.instructions:
            if isinstance(inst, muir.Trap):
                traps += 1
            elif isinstance(inst, muir.ArchEscape):
                arch += 1
    return traps, arch


def main() -> int:
    args = parse_args()
    llvm_dis = f"llvm-dis-{args.llvm_major}"
    files = files_under(args.input)
    if not files:
        raise SystemExit(f"no bitcode under {args.input}")

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
            functions, legal_stats = legalize_module(proc.stdout)
            abi_stats = {
                "calls": 0,
                "helpers": 0,
                "returns": 0,
                "continuation_blocks": 0,
                "argument_loads": 0,
            }
            p3_pass = 0
            p3_skip_escape = 0
            trap_sites = 0
            arch_sites = 0
            p3_instructions = 0

            for fn in functions:
                verify_muir(fn)
                expanded, stats = expand_function(fn)
                for key, value in stats.as_dict().items():
                    abi_stats[key] += value

                traps, arch = _has_system_escape(expanded)
                trap_sites += traps
                arch_sites += arch

                if traps or arch:
                    p3_skip_escape += 1
                    continue

                lowered = lower_function(expanded)
                verify_p3(lowered)
                p3_pass += 1
                p3_instructions += sum(len(b.instructions) for b in lowered.blocks)

            return {
                "file": rel,
                "status": "PASS",
                "functions": len(functions),
                "legalizer": legal_stats.as_dict(),
                "abi": abi_stats,
                "p3_function_pass": p3_pass,
                "p3_function_skip_escape": p3_skip_escape,
                "trap_sites": trap_sites,
                "arch_escape_sites": arch_sites,
                "p3_instructions": p3_instructions,
            }
        except (
            subprocess.CalledProcessError,
            LegalizeError,
            LLVMTextError,
            VerifyError,
            AbiError,
            MachineLoweringError,
            ValueError,
        ) as e:
            return {"file": rel, "status": "FAIL", "error": str(e)}

    records = []
    pass_count = 0
    totals = {
        "functions": 0,
        "calls": 0,
        "helpers": 0,
        "returns": 0,
        "continuation_blocks": 0,
        "argument_loads": 0,
        "p3_function_pass": 0,
        "p3_function_skip_escape": 0,
        "trap_sites": 0,
        "arch_escape_sites": 0,
        "p3_instructions": 0,
    }

    print(f"ABI_START files={len(files)} jobs={jobs}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(one, path): path for path in files}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            rec = future.result()
            records.append(rec)
            done += 1
            if rec["status"] == "PASS":
                pass_count += 1
                totals["functions"] += rec["functions"]
                for key in ("calls", "helpers", "returns", "continuation_blocks", "argument_loads"):
                    totals[key] += rec["abi"][key]
                for key in (
                    "p3_function_pass",
                    "p3_function_skip_escape",
                    "trap_sites",
                    "arch_escape_sites",
                    "p3_instructions",
                ):
                    totals[key] += rec[key]

            if done % 25 == 0 or rec["status"] == "FAIL" or done == len(files):
                tail = f" FAIL {rec['file']} :: {rec['error']}" if rec["status"] == "FAIL" else ""
                print(
                    f"ABI {done}/{len(files)} pass={pass_count} fail={done-pass_count}{tail}",
                    flush=True,
                )

    records.sort(key=lambda x: x["file"])
    summary = {
        "files": len(files),
        "pass": pass_count,
        "fail": len(files) - pass_count,
        "totals": totals,
        "records": records,
    }

    print(
        "ABI_SUMMARY "
        f"pass={summary['pass']}/{summary['files']} "
        f"fail={summary['fail']} "
        f"functions={totals['functions']} "
        f"calls={totals['calls']} "
        f"helpers={totals['helpers']} "
        f"returns={totals['returns']} "
        f"p3_function_pass={totals['p3_function_pass']} "
        f"p3_function_skip_escape={totals['p3_function_skip_escape']} "
        f"trap_sites={totals['trap_sites']} "
        f"arch_escape_sites={totals['arch_escape_sites']} "
        f"p3_instructions={totals['p3_instructions']}"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.strict and summary["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
