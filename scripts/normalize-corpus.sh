#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-riscv.env"

input=${1:-"$root/build/linux-$LINUX_VERSION-riscv/corpus"}
output=${2:-"$root/build/linux-$LINUX_VERSION-riscv/normalized"}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

clang="clang-$LLVM_MAJOR"
opt="opt-$LLVM_MAJOR"
for tool in "$clang" "$opt"; do
    command -v "$tool" >/dev/null 2>&1 || {
        printf 'missing required tool: %s\n' "$tool" >&2
        exit 2
    }
done

manifest="$input/manifest.jsonl"
test -s "$manifest" || {
    printf 'missing corpus manifest: %s\n' "$manifest" >&2
    exit 2
}

passes=$("$opt" --print-passes 2>/dev/null || true)
if printf '%s\n' "$passes" | grep -Eq '^[[:space:]]*lowerswitch([[:space:]]|$)'; then
    lower_switch=lowerswitch
elif printf '%s\n' "$passes" | grep -Eq '^[[:space:]]*lower-switch([[:space:]]|$)'; then
    lower_switch=lower-switch
else
    printf 'cannot find LLVM lower-switch pass in %s --print-passes\n' "$opt" >&2
    exit 2
fi

rm -rf "$output"
mkdir -p "$output"

export MM_INPUT="$input"
export MM_OUTPUT="$output"
export MM_MANIFEST="$manifest"
export MM_CLANG="$clang"
export MM_OPT="$opt"
export MM_LOWER_SWITCH="$lower_switch"
export MM_JOBS="$jobs"

python3 - <<'PY'
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

src_root = Path(os.environ["MM_INPUT"])
out_root = Path(os.environ["MM_OUTPUT"])
manifest = Path(os.environ["MM_MANIFEST"])
clang = os.environ["MM_CLANG"]
opt = os.environ["MM_OPT"]
lower_switch = os.environ["MM_LOWER_SWITCH"]
jobs = max(1, int(os.environ["MM_JOBS"]))

entries = [
    json.loads(line)["bc"]
    for line in manifest.read_text().splitlines()
    if line.strip()
]

def one(rel: str) -> str:
    src = src_root / rel
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".o2.tmp.bc")
    try:
        subprocess.run(
            [
                clang,
                "--target=riscv64-linux-gnu",
                "-mabi=lp64",
                "-O2",
                "-emit-llvm",
                "-c",
                "-x",
                "ir",
                str(src),
                "-o",
                str(tmp),
            ],
            check=True,
        )
        subprocess.run(
            [opt, f"-passes={lower_switch}", str(tmp), "-o", str(dst)],
            check=True,
        )
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return rel

print(f"NORMALIZE_START total={len(entries)} jobs={jobs}")
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
    futures = [pool.submit(one, rel) for rel in entries]
    done = 0
    for future in concurrent.futures.as_completed(futures):
        future.result()
        done += 1
        if done % 25 == 0 or done == len(entries):
            print(f"NORMALIZE {done}/{len(entries)}", flush=True)
PY

cp "$manifest" "$output/manifest.jsonl"
printf 'NORMALIZED %s clang_o2=1 pass=%s jobs=%s\n' \
    "$(wc -l < "$manifest" | tr -d ' ')" "$lower_switch" "$jobs"
