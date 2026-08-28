#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import re
import subprocess
from collections import defaultdict
from pathlib import Path


LOCAL_LINKAGE = {"internal", "private"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--llvm-major", type=int, default=18)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=80)
    return p.parse_args()


def symbol_name(token: str) -> str:
    assert token.startswith("@")
    if token.startswith('@"') and token.endswith('"'):
        return token[2:-1]
    return token[1:]


def classify_linkage(words: list[str]) -> str:
    for word in words:
        if word in {
            "private", "internal", "available_externally", "linkonce",
            "weak", "common", "appending", "extern_weak", "linkonce_odr",
            "weak_odr", "external",
        }:
            return word
    return "external"


def scan_one(path: Path, root: Path, llvm_dis: str):
    proc = subprocess.run(
        [llvm_dis, "-o", "-", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    defs = []
    rel = path.relative_to(root).as_posix()
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if line.startswith("define "):
            m = re.search(r'@(?:"[^"]+"|[-A-Za-z$._0-9]+)\s*\(', line)
            if not m:
                continue
            token = m.group(0).split("(", 1)[0].strip()
            prefix = line[:m.start()].split()
            linkage = classify_linkage(prefix)
            defs.append((symbol_name(token), "function", linkage, line))
            continue

        if not line.startswith("@") or " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        name = symbol_name(lhs.strip())
        words = rhs.split()
        linkage = classify_linkage(words[:6])

        # Pure external global declarations are not definitions.
        if re.search(r"\bexternal\s+(?:global|constant)\b", rhs):
            continue

        kind = "alias" if re.search(r"\balias\b", rhs) else "global"
        defs.append((name, kind, linkage, line))

    digest = hashlib.sha256(proc.stdout.encode()).hexdigest()
    return rel, digest, defs


def main() -> int:
    args = parse_args()
    files = sorted(args.root.rglob("*.bc"))
    llvm_dis = f"llvm-dis-{args.llvm_major}"

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(scan_one, path, args.root, llvm_dis)
            for path in files
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    by_symbol = defaultdict(list)
    by_digest = defaultdict(list)
    for rel, digest, defs in records:
        by_digest[digest].append(rel)
        for name, kind, linkage, line in defs:
            by_symbol[name].append((rel, kind, linkage, line))

    exact_duplicate_modules = [
        files for files in by_digest.values() if len(files) > 1
    ]

    strong_duplicates = []
    local_duplicates = 0
    weak_duplicates = 0
    for name, entries in by_symbol.items():
        if len(entries) < 2:
            continue
        nonlocal_entries = [
            e for e in entries if e[2] not in LOCAL_LINKAGE
        ]
        if len(nonlocal_entries) >= 2:
            # Weak/linkonce duplicates are linker-resolvable; strong external
            # duplicates are the corpus-quality problem we care about.
            strongish = [
                e for e in nonlocal_entries
                if e[2] not in {"weak", "weak_odr", "linkonce", "linkonce_odr", "available_externally"}
            ]
            if len(strongish) >= 2:
                strong_duplicates.append((name, strongish))
            else:
                weak_duplicates += 1
        else:
            local_duplicates += 1

    print(
        f"DUPLICATE_SYMBOLS modules={len(files)} "
        f"exact_duplicate_module_groups={len(exact_duplicate_modules)} "
        f"strong_duplicate_symbols={len(strong_duplicates)} "
        f"weak_duplicate_symbols={weak_duplicates} "
        f"local_duplicate_symbols={local_duplicates}"
    )

    for group in sorted(exact_duplicate_modules, key=lambda x: (-len(x), x))[:args.limit]:
        print("DUP_MODULE " + " ; ".join(group))

    for name, entries in sorted(
        strong_duplicates,
        key=lambda x: (-len(x[1]), x[0]),
    )[:args.limit]:
        print(f"DUP_SYMBOL {name} defs={len(entries)}")
        for rel, kind, linkage, line in entries:
            print(f"  {rel} kind={kind} linkage={linkage}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
