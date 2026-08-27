#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


FOCUSED_ROOTS = [
    "init",
    "kernel",
    "mm",
    "fs",
    "net",
    "lib",
    "arch",
    "drivers",
    "block",
    "ipc",
    "security",
    "crypto",
    "io_uring",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build deterministic manifests from Clang -save-temps Kbuild output."
    )
    p.add_argument("--kbuild", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def discover(kbuild: Path) -> list[str]:
    tus: list[str] = []
    for bc in kbuild.rglob("*.bc"):
        # A corpus entry must look like a real C compile from -save-temps=obj:
        # the preprocessed source and final object must exist beside the bitcode.
        stem = bc.with_suffix("")
        if not stem.with_suffix(".i").is_file():
            continue
        if not stem.with_suffix(".o").is_file():
            continue
        tus.append(bc.relative_to(kbuild).as_posix())
    return sorted(set(tus))


def focused16(all_tus: list[str]) -> list[str]:
    chosen: list[str] = []
    used: set[str] = set()

    # First take one TU from each major Linux subsystem when present.
    for root in FOCUSED_ROOTS:
        prefix = root + "/"
        for path in all_tus:
            if path in used:
                continue
            if path.startswith(prefix):
                chosen.append(path)
                used.add(path)
                break
        if len(chosen) == 16:
            return chosen

    # Then fill deterministically.
    for path in all_tus:
        if path not in used:
            chosen.append(path)
            used.add(path)
        if len(chosen) == 16:
            break
    return chosen


def write(path: Path, entries: list[str]) -> None:
    path.write_text("".join(f"{x}\n" for x in entries))


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_tus = discover(args.kbuild)
    if not all_tus:
        raise SystemExit(
            "no .bc/.i/.o translation units found; was the kernel built "
            "with Clang KCFLAGS=-save-temps=obj?"
        )

    write(args.output / "full-all.txt", all_tus)
    write(args.output / "full100.txt", all_tus[:100])
    write(args.output / "full500.txt", all_tus[:500])
    write(args.output / "focused16.txt", focused16(all_tus))

    print(f"discovered={len(all_tus)}")
    print(f"focused16={min(16, len(all_tus))}")
    print(f"full100={min(100, len(all_tus))}")
    print(f"full500={min(500, len(all_tus))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
