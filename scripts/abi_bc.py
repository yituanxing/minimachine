#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import json
import os
import re
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
from src.minimachine.runtime import helper_callback, system_callback
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


def _asm_template(text: str) -> str:
    m = re.search(r'\basm\b(?:\s+\w+)*\s+"((?:\\.|[^"])*)"', text)
    if not m:
        return "<unparsed-asm>"
    template = m.group(1)
    template = template.replace(r"\0A", ";").replace(r"\09", " ")
    template = re.sub(r"\$\{?\d+(?::[^}]*)?\}?", "$N", template)
    template = re.sub(r"\s+", " ", template).strip()
    return template[:240]


def _escape_key(inst: muir.ArchEscape) -> str:
    if "asm" in inst.text:
        template = _asm_template(inst.text)
        if template == "<unparsed-asm>":
            raw = re.sub(r"\s+", " ", inst.text).strip()[:220]
            return f"{inst.kind}|<unparsed-asm>|{raw}"
        return f"{inst.kind}|{template}"
    return f"{inst.kind}|<control>"


def _escape_family(inst: muir.ArchEscape) -> str:
    if inst.kind == "indirectbr":
        return "indirect_control"
    template = _asm_template(inst.text)
    t = template.lower().strip()

    if inst.kind == "callbr":
        if "__jump_table" in t:
            return "jump_label"
        if ".alternative" in t or ".altinstructions" in t:
            return "alternative_patch"
        return "asm_goto_other"

    if t == "<unparsed-asm>":
        return "unparsed"
    if re.search(r"(^|[;\s])(amo\w*|lr\.[wd]|sc\.[wd])\b", t):
        return "atomic"
    if re.search(r"(^|[;\s])fence(?:\.i)?\b", t):
        return "fence"
    if "sfence.vma" in t or "sinval.vma" in t:
        return "tlb_vm"
    if re.search(r"(^|[;\s])csr(?:r|w|s|c|rw|rs|rc)\b", t) or "sstatus" in t or "satp" in t:
        return "csr_privilege"
    if "ecall" in t:
        return "ecall"
    if "ebreak" in t:
        return "debug_trap"
    if t == "pause" or t == "nop":
        return "hint"
    if "__ex_table" in t or ".fixup" in t or ".l__gpr_num_" in t or ".irp" in t:
        return "faultable_access"
    if re.match(r"^(?:l[bhwdu]|s[bhwd])\b", t):
        return "plain_memory"
    if re.match(r"^(?:div|rem|mul)\b", t):
        return "integer_arch"
    if not t:
        return "empty_asm"
    return "other"


def _has_system_escape(fn: muir.Function) -> tuple[int, int, Counter[str], Counter[str], Counter[str], dict[str, str]]:
    traps = 0
    arch = 0
    kinds: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    families: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for block in fn.blocks:
        for inst in block.instructions:
            if isinstance(inst, muir.Trap):
                traps += 1
            elif isinstance(inst, muir.ArchEscape):
                arch += 1
                kinds[inst.kind] += 1
                key = _escape_key(inst)
                groups[key] += 1
                families[_escape_family(inst)] += 1
                samples.setdefault(key, re.sub(r"\s+", " ", inst.text).strip()[:12000])
    return traps, arch, kinds, groups, families, samples


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
                "system_ops": 0,
                "returns": 0,
                "continuation_blocks": 0,
                "argument_loads": 0,
            }
            p3_pass = 0
            p3_skip_escape = 0
            trap_sites = 0
            arch_sites = 0
            p3_instructions = 0
            escape_kinds: Counter[str] = Counter()
            escape_groups: Counter[str] = Counter()
            escape_families: Counter[str] = Counter()
            escape_samples: dict[str, str] = {}
            helper_symbols: Counter[str] = Counter()
            system_ops_used: Counter[str] = Counter()

            for fn in functions:
                for block in fn.blocks:
                    for inst in block.instructions:
                        if isinstance(inst, muir.Helper):
                            helper_symbols[inst.symbol] += 1
                        elif isinstance(inst, muir.Sys):
                            system_ops_used[inst.op] += 1
                verify_muir(fn)
                expanded, stats = expand_function(fn)
                for key, value in stats.as_dict().items():
                    abi_stats[key] += value

                traps, arch, kinds, groups, families, samples = _has_system_escape(expanded)
                trap_sites += traps
                arch_sites += arch
                escape_kinds.update(kinds)
                escape_groups.update(groups)
                escape_families.update(families)
                for key, sample in samples.items():
                    escape_samples.setdefault(key, sample)

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
                "escape_kinds": dict(escape_kinds),
                "escape_groups": dict(escape_groups),
                "escape_families": dict(escape_families),
                "escape_samples": escape_samples,
                "helper_symbols": dict(helper_symbols),
                "system_ops_used": dict(system_ops_used),
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
        "system_ops": 0,
        "returns": 0,
        "continuation_blocks": 0,
        "argument_loads": 0,
        "p3_function_pass": 0,
        "p3_function_skip_escape": 0,
        "trap_sites": 0,
        "arch_escape_sites": 0,
        "p3_instructions": 0,
        "escape_kinds": {},
        "escape_groups": {},
        "escape_families": {},
        "escape_samples": {},
        "helper_symbols": {},
        "system_ops_used": {},
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
                for key in ("calls", "helpers", "system_ops", "returns", "continuation_blocks", "argument_loads"):
                    totals[key] += rec["abi"][key]
                for key in (
                    "p3_function_pass",
                    "p3_function_skip_escape",
                    "trap_sites",
                    "arch_escape_sites",
                    "p3_instructions",
                ):
                    totals[key] += rec[key]
                for key, value in rec["escape_kinds"].items():
                    totals["escape_kinds"][key] = totals["escape_kinds"].get(key, 0) + value
                for key, value in rec["escape_groups"].items():
                    totals["escape_groups"][key] = totals["escape_groups"].get(key, 0) + value
                for key, value in rec["escape_families"].items():
                    totals["escape_families"][key] = totals["escape_families"].get(key, 0) + value
                for key, sample in rec["escape_samples"].items():
                    totals["escape_samples"].setdefault(key, sample)
                for key, value in rec["helper_symbols"].items():
                    totals["helper_symbols"][key] = totals["helper_symbols"].get(key, 0) + value
                for key, value in rec["system_ops_used"].items():
                    totals["system_ops_used"][key] = totals["system_ops_used"].get(key, 0) + value

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
        f"system_ops={totals['system_ops']} "
        f"returns={totals['returns']} "
        f"p3_function_pass={totals['p3_function_pass']} "
        f"p3_function_skip_escape={totals['p3_function_skip_escape']} "
        f"trap_sites={totals['trap_sites']} "
        f"arch_escape_sites={totals['arch_escape_sites']} "
        f"p3_instructions={totals['p3_instructions']} "
        f"escape_groups={len(totals['escape_groups'])}"
    )
    print("ESCAPE_KINDS " + " ".join(
        f"{k}={v}" for k, v in sorted(totals["escape_kinds"].items(), key=lambda x: (-x[1], x[0]))
    ))
    print("ESCAPE_FAMILIES " + " ".join(
        f"{k}={v}" for k, v in sorted(totals["escape_families"].items(), key=lambda x: (-x[1], x[0]))
    ))
    print(
        f"RUNTIME_SURFACE helper_kinds={len(totals['helper_symbols'])} "
        f"system_op_kinds={len(totals['system_ops_used'])}"
    )
    resolved_helpers = {
        key for key in totals["helper_symbols"]
        if helper_callback(key) is not None
    }
    unresolved_helpers = {
        key for key in totals["helper_symbols"]
        if key not in resolved_helpers
    }
    resolved_systems = {
        key for key in totals["system_ops_used"]
        if system_callback(key) is not None
    }
    unresolved_systems = {
        key for key in totals["system_ops_used"]
        if key not in resolved_systems
    }
    helper_sites_total = sum(totals["helper_symbols"].values())
    helper_sites_resolved = sum(
        value for key, value in totals["helper_symbols"].items()
        if key in resolved_helpers
    )
    system_sites_total = sum(totals["system_ops_used"].values())
    system_sites_resolved = sum(
        value for key, value in totals["system_ops_used"].items()
        if key in resolved_systems
    )
    totals["runtime_coverage"] = {
        "helper_kinds_resolved": len(resolved_helpers),
        "helper_kinds_total": len(totals["helper_symbols"]),
        "helper_sites_resolved": helper_sites_resolved,
        "helper_sites_total": helper_sites_total,
        "system_kinds_resolved": len(resolved_systems),
        "system_kinds_total": len(totals["system_ops_used"]),
        "system_sites_resolved": system_sites_resolved,
        "system_sites_total": system_sites_total,
        "unresolved_helpers": sorted(unresolved_helpers),
        "unresolved_systems": sorted(unresolved_systems),
    }
    print(
        "RUNTIME_EXECUTABLE "
        f"helper_kinds={len(resolved_helpers)}/{len(totals['helper_symbols'])} "
        f"helper_sites={helper_sites_resolved}/{helper_sites_total} "
        f"system_kinds={len(resolved_systems)}/{len(totals['system_ops_used'])} "
        f"system_sites={system_sites_resolved}/{system_sites_total}"
    )
    for key, value in sorted(
        (
            (key, totals["helper_symbols"][key])
            for key in unresolved_helpers
        ),
        key=lambda x: (-x[1], x[0]),
    )[:40]:
        print(f"UNRESOLVED_HELPER {value}x {key}")
    for key, value in sorted(
        (
            (key, totals["system_ops_used"][key])
            for key in unresolved_systems
        ),
        key=lambda x: (-x[1], x[0]),
    )[:40]:
        print(f"UNRESOLVED_SYS {value}x {key}")
    for key, value in sorted(totals["helper_symbols"].items(), key=lambda x: (-x[1], x[0]))[:40]:
        print(f"HELPER_KIND {value}x {key}")
    for key, value in sorted(totals["system_ops_used"].items(), key=lambda x: (-x[1], x[0]))[:40]:
        print(f"SYS_KIND {value}x {key}")
    for key, value in sorted(totals["escape_groups"].items(), key=lambda x: (-x[1], x[0]))[:30]:
        print(f"ESCAPE_GROUP {value}x {key}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.strict and summary["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
