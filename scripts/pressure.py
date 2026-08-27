#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ASSIGN_RE = re.compile(r"^(%[-A-Za-z$._0-9]+)\s*=\s*(?:tail\s+|musttail\s+|notail\s+)?([a-z][a-z0-9_.]*)\b")
BARE_RE = re.compile(r"^(?:tail\s+|musttail\s+|notail\s+)?([a-z][a-z0-9_.]*)\b")
SSA_RE = re.compile(r"%[-A-Za-z$._0-9]+")
INTRINSIC_RE = re.compile(r"@llvm\.([A-Za-z0-9_.]+)")
GEP_INDEX_RE = re.compile(r",\s+i\d+\s+([^,\s]+)")

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

NATIVE = {"load", "store", "sub", "br"}
STRUCTURED = {
    "ret", "alloca", "add", "phi", "select", "trunc", "zext", "sext",
    "ptrtoint", "inttoptr", "bitcast", "addrspacecast", "extractvalue",
    "insertvalue",
}
HELPER = {
    "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr",
    "and", "or", "xor",
}
ARCH = {"callbr"}
DROP = {"unreachable"}
SPECIAL = {"freeze", "indirectbr"}

DROP_INTRINSIC_PREFIXES = (
    "lifetime.start", "lifetime.end", "assume", "dbg.",
)
HELPER_INTRINSIC_PREFIXES = (
    "memcpy.", "memmove.", "memset.", "bswap.", "bitreverse.", "ctpop.",
    "ctlz.", "cttz.", "fshl.", "fshr.", "smax.", "smin.", "umax.", "umin.",
    "uadd.sat.", "usub.sat.", "sadd.sat.", "ssub.sat.",
    "uadd.with.overflow.", "usub.with.overflow.",
    "sadd.with.overflow.", "ssub.with.overflow.",
    "umul.with.overflow.", "smul.with.overflow.",
)
ARCH_INTRINSIC_PREFIXES = (
    "read_register.", "write_register.", "frameaddress.", "returnaddress",
    "stacksave", "stackrestore",
)


@dataclass
class Inst:
    raw: str
    op: str
    result: str | None
    uses: list[str]


def parse_inst(line: str) -> Inst | None:
    text = line.strip()
    if not text or text.startswith(";") or text.endswith(":"):
        return None
    m = ASSIGN_RE.match(text)
    if m:
        result, op = m.group(1), m.group(2)
    else:
        m2 = BARE_RE.match(text)
        if not m2:
            return None
        result, op = None, m2.group(1)
    if op not in OPCODES:
        return None

    uses = SSA_RE.findall(text)
    if result is not None and uses and uses[0] == result:
        uses = uses[1:]
    return Inst(text, op, result, uses)


def functions(text: str) -> list[list[Inst]]:
    result: list[list[Inst]] = []
    current: list[Inst] | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("define "):
            current = []
            continue
        if current is not None and stripped == "}":
            result.append(current)
            current = None
            continue
        if current is not None:
            inst = parse_inst(line)
            if inst:
                current.append(inst)
    return result


def intrinsic_class(name: str) -> str:
    if name.startswith(DROP_INTRINSIC_PREFIXES):
        return "drop"
    if name.startswith(HELPER_INTRINSIC_PREFIXES):
        return "helper_expand"
    if name.startswith(ARCH_INTRINSIC_PREFIXES):
        return "arch_escape"
    return "special_runtime"


def analyze_function(insts: list[Inst]) -> tuple[collections.Counter[str], collections.Counter[str], collections.Counter[str]]:
    cats: collections.Counter[str] = collections.Counter()
    metrics: collections.Counter[str] = collections.Counter()
    specials: collections.Counter[str] = collections.Counter()

    defs = {inst.result: i for i, inst in enumerate(insts) if inst.result}
    users: dict[str, list[int]] = collections.defaultdict(list)
    for i, inst in enumerate(insts):
        for use in inst.uses:
            if use in defs:
                users[use].append(i)

    for i, inst in enumerate(insts):
        op = inst.op

        if op == "icmp":
            metrics["icmp_total"] += 1
            us = users.get(inst.result or "", [])
            if len(us) == 1 and insts[us[0]].op == "br":
                cats["p3_native_or_fused"] += 1
                metrics["icmp_fused_to_br"] += 1
            else:
                cats["structured_lowering"] += 1
                metrics["icmp_materialized"] += 1
            continue

        if op == "getelementptr":
            metrics["gep_total"] += 1
            indices = GEP_INDEX_RE.findall(inst.raw)
            constant = bool(indices) and all(re.fullmatch(r"-?[0-9]+", x) for x in indices)
            if constant:
                metrics["gep_constant_indices"] += 1
            us = users.get(inst.result or "", [])
            foldable = constant and len(us) == 1 and insts[us[0]].op in {"load", "store"}
            if foldable:
                cats["p3_native_or_fused"] += 1
                metrics["gep_foldable_into_mem"] += 1
            else:
                cats["structured_lowering"] += 1
                if len(us) == 1 and insts[us[0]].op in {"load", "store"}:
                    metrics["gep_dynamic_single_mem_use"] += 1
            continue

        if op in NATIVE:
            cats["p3_native_or_fused"] += 1
            continue

        if op == "call":
            if " asm " in f" {inst.raw} " or " asm sideeffect " in f" {inst.raw} ":
                cats["arch_escape"] += 1
                metrics["inline_asm"] += 1
                continue
            m = INTRINSIC_RE.search(inst.raw)
            if m:
                name = m.group(1)
                cls = intrinsic_class(name)
                cats[cls] += 1
                metrics[f"intrinsic_{cls}"] += 1
                if cls == "special_runtime":
                    specials[name] += 1
                continue
            cats["structured_lowering"] += 1
            if re.search(r"\bcall\b.*@[A-Za-z0-9_.$]+", inst.raw):
                metrics["direct_call"] += 1
            else:
                metrics["indirect_call"] += 1
            continue

        if op in STRUCTURED:
            cats["structured_lowering"] += 1
            continue
        if op in HELPER:
            cats["helper_expand"] += 1
            continue
        if op in ARCH:
            cats["arch_escape"] += 1
            continue
        if op in DROP:
            cats["drop"] += 1
            continue
        if op in SPECIAL:
            cats["special_runtime"] += 1
            specials[op] += 1
            continue

        # These should be rare in the C/kernel corpus and are deliberately
        # surfaced rather than silently assigned to the machine.
        cats["unsupported"] += 1
        specials[f"unsupported:{op}"] += 1

    return cats, metrics, specials


def analyze_file(dis: str, root: Path, path: Path):
    p = subprocess.run(
        [dis, "-o", "-", str(path)],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    cats: collections.Counter[str] = collections.Counter()
    metrics: collections.Counter[str] = collections.Counter()
    specials: collections.Counter[str] = collections.Counter()
    inst_count = 0
    for fn in functions(p.stdout):
        inst_count += len(fn)
        c, m, s = analyze_function(fn)
        cats.update(c)
        metrics.update(m)
        specials.update(s)
    return path.relative_to(root).as_posix(), inst_count, cats, metrics, specials


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure LLVM -> P3 lowering pressure.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--llvm-major", default="18")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    files = sorted(args.input.rglob("*.bc"))
    if not files:
        raise SystemExit(f"no bitcode under {args.input}")

    jobs = args.jobs or min(32, max(1, __import__("os").cpu_count() or 1))
    dis = f"llvm-dis-{args.llvm_major}"

    total_cats: collections.Counter[str] = collections.Counter()
    total_metrics: collections.Counter[str] = collections.Counter()
    total_specials: collections.Counter[str] = collections.Counter()
    total_insts = 0
    per_file = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(analyze_file, dis, args.input, p) for p in files]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            rel, n, cats, metrics, specials = fut.result()
            total_insts += n
            total_cats.update(cats)
            total_metrics.update(metrics)
            total_specials.update(specials)
            per_file.append({"file": rel, "instructions": n})
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"PRESSURE {done}/{len(files)}")

    classified = sum(total_cats.values())
    report = {
        "files": len(files),
        "instructions": total_insts,
        "classified_instructions": classified,
        "jobs": jobs,
        "categories": dict(total_cats.most_common()),
        "category_percent": {
            k: round(v * 100.0 / classified, 4) if classified else 0.0
            for k, v in total_cats.items()
        },
        "metrics": dict(total_metrics.most_common()),
        "specials": dict(total_specials.most_common()),
        "per_file": sorted(per_file, key=lambda x: x["file"]),
    }

    print(f"FILES {len(files)}")
    print(f"INSTRUCTIONS {total_insts}")
    print("CATEGORIES")
    for k, v in total_cats.most_common():
        pct = v * 100.0 / classified if classified else 0
        print(f"  {k}={v} ({pct:.2f}%)")
    print("KEY_METRICS")
    for key in (
        "icmp_total", "icmp_fused_to_br", "icmp_materialized",
        "gep_total", "gep_constant_indices", "gep_foldable_into_mem",
        "gep_dynamic_single_mem_use", "direct_call", "indirect_call",
        "inline_asm",
    ):
        print(f"  {key}={total_metrics[key]}")
    print(f"UNSUPPORTED {total_cats['unsupported']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
