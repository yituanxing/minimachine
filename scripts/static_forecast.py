#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rng_add(acc, count, lo, hi):
    acc[0] += count * lo
    acc[1] += count * hi


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Static MiniMachine engineering forecast.")
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--pressure", type=Path, required=True)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    census = json.loads(args.census.read_text())
    pressure = json.loads(args.pressure.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None

    op = census["opcodes"]
    cats = pressure["categories"]
    met = pressure["metrics"]
    total = census["instructions"]

    # Static P3 code-size proxy. This deliberately estimates emitted instruction
    # sites, not dynamic execution. Fused icmp and foldable GEP add zero because
    # the consuming BR/MOV is already counted.
    p3 = [0, 0]
    for name in ("load", "store", "sub", "br"):
        rng_add(p3, op.get(name, 0), 1, 1)

    # Structured legalization.
    rng_add(p3, op.get("add", 0), 1, 2)
    rng_add(p3, met.get("icmp_materialized", 0), 2, 5)
    rng_add(p3, met.get("gep_total", 0) - met.get("gep_foldable_into_mem", 0), 1, 4)
    rng_add(p3, op.get("phi", 0), 2, 4)
    rng_add(p3, op.get("select", 0), 3, 5)
    rng_add(p3, op.get("ret", 0), 1, 3)
    rng_add(p3, met.get("direct_call", 0) + met.get("indirect_call", 0), 2, 6)
    for name in ("zext", "sext", "trunc", "ptrtoint", "inttoptr", "bitcast", "addrspacecast"):
        rng_add(p3, op.get(name, 0), 0, 1)
    for name in ("extractvalue", "insertvalue"):
        rng_add(p3, op.get(name, 0), 1, 4)

    # Helper call sites. Shared helper bodies are intentionally excluded.
    helper_sites = 0
    for name in ("mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr", "and", "or", "xor"):
        helper_sites += op.get(name, 0)
    helper_sites += met.get("intrinsic_helper_expand", 0)
    rng_add(p3, helper_sites, 2, 5)

    # Rare special-runtime sites.
    rng_add(p3, cats.get("special_runtime", 0), 1, 4)

    kinds = census["opcode_kinds"]
    unsupported = cats.get("unsupported", 0)

    # Source LOC model. Recommended route assumes LLVM libraries parse/read IR.
    # Legalizer range scales mildly with observed semantic surface; unsupported
    # semantics carry an explicit uncertainty penalty.
    loc = {
        "llvm_adapter_reader": [1500, 3000],
        "legalizer": [
            2500 + 80 * kinds + 200 * unsupported,
            3500 + 160 * kinds + 800 * unsupported,
        ],
        "muir_and_verifier": [2500, 4500],
        "muir_to_p3": [1000, 2500],
        "abi_call_frame": [2000, 4000],
        "runtime_helpers": [3000, 7000],
        "reference_vm": [600, 1500],
    }
    core_loc = [sum(x[0] for x in loc.values()), sum(x[1] for x in loc.values())]
    standalone_parser_penalty = [5000, 10000]
    tests_infra = [8000, 20000]
    linux_nommu_up = [5000, 10000]
    linux_mmu_smp = [12000, 25000]

    result = {
        "corpus": {
            "files": census["files"],
            "instructions": total,
            "opcode_kinds": kinds,
            "instructions_per_tu": round(total / census["files"], 2),
        },
        "route": {
            "categories": cats,
            "direct_plus_structured_percent": round(
                pct(cats.get("p3_native_or_fused", 0) + cats.get("structured_lowering", 0), total), 4
            ),
            "through_helper_percent": round(
                pct(
                    cats.get("p3_native_or_fused", 0)
                    + cats.get("structured_lowering", 0)
                    + cats.get("helper_expand", 0),
                    total,
                ),
                4,
            ),
            "unsupported": unsupported,
        },
        "static_p3_emission_proxy": {
            "instructions_low": p3[0],
            "instructions_high": p3[1],
            "ratio_to_llvm_low": round(p3[0] / total, 3) if total else 0,
            "ratio_to_llvm_high": round(p3[1] / total, 3) if total else 0,
            "notes": [
                "Static emitted-site proxy, not dynamic cycles.",
                "Shared runtime helper bodies are excluded.",
                "Arch escapes are excluded because they belong to arch/minimachine.",
                "Fused icmp and foldable GEP add zero beyond their consuming BR/MOV.",
            ],
        },
        "source_loc_forecast": {
            "components": {k: {"low": v[0], "high": v[1]} for k, v in loc.items()},
            "recommended_core": {"low": core_loc[0], "high": core_loc[1]},
            "standalone_llvm_parser_penalty": {
                "low": standalone_parser_penalty[0],
                "high": standalone_parser_penalty[1],
            },
            "tests_and_infra": {"low": tests_infra[0], "high": tests_infra[1]},
            "linux_arch_nommu_up": {"low": linux_nommu_up[0], "high": linux_nommu_up[1]},
            "linux_arch_mmu_smp_mature": {"low": linux_mmu_smp[0], "high": linux_mmu_smp[1]},
        },
        "key_pressure": {
            "icmp_total": met.get("icmp_total", 0),
            "icmp_fused_to_br": met.get("icmp_fused_to_br", 0),
            "icmp_fusion_percent": round(pct(met.get("icmp_fused_to_br", 0), met.get("icmp_total", 0)), 4),
            "gep_total": met.get("gep_total", 0),
            "gep_foldable_into_mem": met.get("gep_foldable_into_mem", 0),
            "gep_mem_fold_percent": round(pct(met.get("gep_foldable_into_mem", 0), met.get("gep_total", 0)), 4),
            "inline_asm": met.get("inline_asm", 0),
            "indirect_call": met.get("indirect_call", 0),
        },
    }

    if baseline:
        result["growth_vs_baseline"] = {
            "baseline_files": baseline["files"],
            "file_multiplier": round(census["files"] / baseline["files"], 3),
            "instruction_multiplier": round(total / baseline["instructions"], 3),
            "opcode_kinds": {
                "baseline": baseline["opcode_kinds"],
                "current": kinds,
                "delta": kinds - baseline["opcode_kinds"],
            },
            "unsupported": {
                "baseline": baseline.get("categories", {}).get("unsupported", 0),
                "current": unsupported,
            },
        }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    md = []
    md.append("# MiniMachine full-corpus static forecast")
    md.append("")
    md.append(f"- TUs: **{census['files']}**")
    md.append(f"- O2 normalized LLVM instructions: **{total:,}**")
    md.append(f"- LLVM opcode kinds: **{kinds}**")
    md.append(f"- generic unsupported: **{unsupported}**")
    if baseline:
        g = result["growth_vs_baseline"]
        md.append(
            f"- vs full500: files **{g['file_multiplier']}x**, instructions **{g['instruction_multiplier']}x**, "
            f"opcode kinds **{g['opcode_kinds']['baseline']} -> {g['opcode_kinds']['current']}**"
        )
    md.append("")
    md.append("## Lowering route")
    md.append("")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        md.append(f"- {k}: **{v:,}** ({pct(v, total):.2f}%)")
    md.append(f"- direct + structured: **{result['route']['direct_plus_structured_percent']:.2f}%**")
    md.append(f"- through helper: **{result['route']['through_helper_percent']:.2f}%**")
    md.append("")
    md.append("## P3 pressure")
    md.append("")
    kp = result["key_pressure"]
    md.append(
        f"- icmp -> BR fusion: **{kp['icmp_fused_to_br']:,}/{kp['icmp_total']:,} "
        f"({kp['icmp_fusion_percent']:.2f}%)**"
    )
    md.append(
        f"- GEP folded into memory operand: **{kp['gep_foldable_into_mem']:,}/{kp['gep_total']:,} "
        f"({kp['gep_mem_fold_percent']:.2f}%)**"
    )
    md.append(f"- inline asm sites: **{kp['inline_asm']:,}**")
    md.append(f"- indirect calls: **{kp['indirect_call']:,}**")
    md.append("")
    md.append("## Static emitted P3 proxy")
    md.append("")
    pp = result["static_p3_emission_proxy"]
    md.append(
        f"- estimated emitted P3 instruction sites: **{pp['instructions_low']:,} - "
        f"{pp['instructions_high']:,}**"
    )
    md.append(
        f"- ratio to normalized LLVM: **{pp['ratio_to_llvm_low']:.2f}x - "
        f"{pp['ratio_to_llvm_high']:.2f}x**"
    )
    md.append("- This is not a runtime-cycle prediction.")
    md.append("")
    md.append("## Source LOC forecast")
    md.append("")
    for k, v in result["source_loc_forecast"]["components"].items():
        md.append(f"- {k}: **{v['low']:,} - {v['high']:,} LOC**")
    md.append(
        f"- recommended core total: **{core_loc[0]:,} - {core_loc[1]:,} LOC**"
    )
    md.append(
        f"- if we write our own LLVM parser: add **{standalone_parser_penalty[0]:,} - "
        f"{standalone_parser_penalty[1]:,} LOC**"
    )
    md.append(
        f"- tests/infra: **{tests_infra[0]:,} - {tests_infra[1]:,} LOC**"
    )
    md.append(
        f"- first Linux NOMMU/UP arch port: **{linux_nommu_up[0]:,} - {linux_nommu_up[1]:,} LOC**"
    )
    md.append(
        f"- mature MMU/SMP arch port: **{linux_mmu_smp[0]:,} - {linux_mmu_smp[1]:,} LOC**"
    )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(
        "The ISA should grow only if a new primitive materially reduces the legalizer/runtime/static-emission "
        "cost. Opcode-count growth by itself is not a reason to add a machine instruction."
    )

    text = "\n".join(md) + "\n"
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
