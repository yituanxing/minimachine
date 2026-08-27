#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path


OPCODES = {
    "ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume",
    "catchswitch", "catchret", "cleanupret", "unreachable",
    "fneg", "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv",
    "fdiv", "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or",
    "xor", "extractelement", "insertelement", "shufflevector",
    "extractvalue", "insertvalue", "alloca", "load", "store", "fence",
    "cmpxchg", "atomicrmw", "getelementptr", "trunc", "zext", "sext",
    "fptrunc", "fpext", "fptoui", "fptosi", "uitofp", "sitofp",
    "ptrtoint", "inttoptr", "bitcast", "addrspacecast", "icmp", "fcmp",
    "phi", "select", "freeze", "call", "va_arg", "landingpad", "catchpad",
    "cleanuppad",
}

ASSIGN_RE = re.compile(r"^%[^=]+?=\s*(?:tail\s+|musttail\s+|notail\s+)?([a-z][a-z0-9_.]*)\b")
BARE_RE = re.compile(r"^(?:tail\s+|musttail\s+|notail\s+)?([a-z][a-z0-9_.]*)\b")
ICMP_RE = re.compile(r"\bicmp\s+([a-z]+)\b")
WIDTH_RE = re.compile(r"(?<![%@.$A-Za-z0-9_])i([1-9][0-9]*)(?![A-Za-z0-9_])")
ADDRSPACE_RE = re.compile(r"\baddrspace\(([0-9]+)\)")
INTRINSIC_RE = re.compile(r"@llvm\.([A-Za-z0-9_.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Census LLVM bitcode for MiniMachine.")
    p.add_argument("input", type=Path)
    p.add_argument("--llvm-major", default="18")
    p.add_argument("--json", type=Path)
    return p.parse_args()


def opcode_of(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith(";") or text.endswith(":"):
        return None
    m = ASSIGN_RE.match(text)
    if not m:
        m = BARE_RE.match(text)
    if not m:
        return None
    op = m.group(1)
    return op if op in OPCODES else None


def disassemble(dis: str, bc: Path) -> str:
    p = subprocess.run(
        [dis, "-o", "-", str(bc)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout


def main() -> int:
    args = parse_args()
    dis = f"llvm-dis-{args.llvm_major}"

    files = sorted(args.input.rglob("*.bc"))
    if not files:
        raise SystemExit(f"no bitcode files under {args.input}")

    opcodes: collections.Counter[str] = collections.Counter()
    icmp: collections.Counter[str] = collections.Counter()
    widths: collections.Counter[str] = collections.Counter()
    addrspaces: collections.Counter[str] = collections.Counter()
    intrinsics: collections.Counter[str] = collections.Counter()
    total_instructions = 0
    inline_asm = 0
    direct_calls = 0
    indirect_calls = 0
    volatile_load_store = 0
    atomic_ops = 0

    per_file = []

    for index, bc in enumerate(files, 1):
        text = disassemble(dis, bc)
        file_ops: collections.Counter[str] = collections.Counter()
        file_instructions = 0

        for raw in text.splitlines():
            op = opcode_of(raw)
            if not op:
                continue

            total_instructions += 1
            file_instructions += 1
            opcodes[op] += 1
            file_ops[op] += 1

            for width in WIDTH_RE.findall(raw):
                widths[f"i{width}"] += 1
            for space in ADDRSPACE_RE.findall(raw):
                addrspaces[space] += 1
            for name in INTRINSIC_RE.findall(raw):
                intrinsics[name] += 1

            if op == "icmp":
                m = ICMP_RE.search(raw)
                if m:
                    icmp[m.group(1)] += 1

            if op in {"load", "store"} and " volatile " in f" {raw} ":
                volatile_load_store += 1
            if op in {"atomicrmw", "cmpxchg", "fence"} or " atomic " in f" {raw} ":
                atomic_ops += 1

            if op == "call":
                if " asm " in f" {raw} " or " asm sideeffect " in f" {raw} ":
                    inline_asm += 1
                # Direct IR calls name a global symbol after the call signature.
                if re.search(r"\bcall\b.*@[A-Za-z0-9_.$]+", raw):
                    direct_calls += 1
                else:
                    indirect_calls += 1

        per_file.append(
            {
                "file": bc.relative_to(args.input).as_posix(),
                "instructions": file_instructions,
                "opcode_kinds": len(file_ops),
            }
        )

        if index % 50 == 0 or index == len(files):
            print(f"CENSUS {index}/{len(files)}")

    report = {
        "files": len(files),
        "instructions": total_instructions,
        "opcode_kinds": len(opcodes),
        "opcodes": dict(opcodes.most_common()),
        "icmp_predicates": dict(icmp.most_common()),
        "integer_width_mentions": dict(widths.most_common()),
        "address_spaces": dict(addrspaces.most_common()),
        "intrinsics": dict(intrinsics.most_common()),
        "inline_asm_calls": inline_asm,
        "direct_calls": direct_calls,
        "indirect_calls": indirect_calls,
        "volatile_load_store": volatile_load_store,
        "atomic_ops": atomic_ops,
        "per_file": per_file,
    }

    print(f"FILES {report['files']}")
    print(f"LLVM_INSTRUCTIONS {report['instructions']}")
    print(f"OPCODE_KINDS {report['opcode_kinds']}")
    print(f"INLINE_ASM_CALLS {inline_asm}")
    print(f"INDIRECT_CALLS {indirect_calls}")
    print(f"ATOMIC_OPS {atomic_ops}")
    print("TOP_OPCODES")
    for name, count in opcodes.most_common(20):
        print(f"  {name}={count}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
