#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-riscv.env"

input=${1:-"$root/build/linux-$LINUX_VERSION-riscv/corpus"}
output=${2:-"$root/build/linux-$LINUX_VERSION-riscv/normalized"}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

opt="opt-$LLVM_MAJOR"
llvm_dis="llvm-dis-$LLVM_MAJOR"
llvm_as="llvm-as-$LLVM_MAJOR"
for tool in "$opt" "$llvm_dis" "$llvm_as"; do
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
export MM_OPT="$opt"
export MM_LLVM_DIS="$llvm_dis"
export MM_LLVM_AS="$llvm_as"
export MM_LOWER_SWITCH="$lower_switch"
export MM_JOBS="$jobs"

python3 - <<'PY'
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

src_root = Path(os.environ["MM_INPUT"])
out_root = Path(os.environ["MM_OUTPUT"])
manifest = Path(os.environ["MM_MANIFEST"])
opt = os.environ["MM_OPT"]
llvm_dis = os.environ["MM_LLVM_DIS"]
llvm_as = os.environ["MM_LLVM_AS"]
lower_switch = os.environ["MM_LOWER_SWITCH"]
jobs = max(1, int(os.environ["MM_JOBS"]))

entries = [
    json.loads(line)["bc"]
    for line in manifest.read_text().splitlines()
    if line.strip()
]

# Use LLVM's middle-end directly. This deliberately avoids Clang's backend and
# integrated assembler: some real kernel TUs contain module-level inline asm
# such as .incbin, which must remain opaque during MiniMachine normalization.
pipeline = f"default<O2>,{lower_switch}"

def one(rel: str):
    src = src_root / rel
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    first = subprocess.run(
        [opt, f"-passes={pipeline}", str(src), "-o", str(dst)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if first.returncode == 0:
        return rel, None

    # Some kernel build TUs are primarily module-level assembler containers
    # (for example kernel/configs.bc with .incbin kernel/config_data.gz).
    # Keep them in the 2053-TU corpus, but make the build/arch escape explicit:
    # strip only module asm in an analysis copy, then optimize any ordinary IR.
    dis = subprocess.run(
        [llvm_dis, "-o", "-", str(src)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = dis.stdout.splitlines()
    module_asm = [line for line in lines if line.lstrip().startswith("module asm ")]
    if not module_asm:
        raise RuntimeError(
            f"normalization failed for {rel}:\n{first.stderr}"
        )

    filtered = "\n".join(
        line for line in lines if not line.lstrip().startswith("module asm ")
    ) + "\n"

    with tempfile.TemporaryDirectory(prefix="minimachine-normalize-") as td:
        td = Path(td)
        ll = td / "stripped.ll"
        bc = td / "stripped.bc"
        ll.write_text(filtered)
        subprocess.run([llvm_as, str(ll), "-o", str(bc)], check=True)
        subprocess.run(
            [opt, f"-passes={pipeline}", str(bc), "-o", str(dst)],
            check=True,
        )

    return rel, {
        "file": rel,
        "reason": "module_inline_asm",
        "module_asm_lines": len(module_asm),
    }

print(f"NORMALIZE_START total={len(entries)} jobs={jobs} pipeline={pipeline}")
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
    futures = [pool.submit(one, rel) for rel in entries]
    done = 0
    escapes = []
    for future in concurrent.futures.as_completed(futures):
        rel, escape = future.result()
        if escape:
            escapes.append(escape)
            print(
                f"NORMALIZE_ESCAPE file={rel} reason={escape['reason']} "
                f"module_asm_lines={escape['module_asm_lines']}",
                flush=True,
            )
        done += 1
        if done % 25 == 0 or done == len(entries):
            print(f"NORMALIZE {done}/{len(entries)}", flush=True)

(out_root / "normalization-escapes.jsonl").write_text(
    "".join(json.dumps(x, sort_keys=True) + "\n" for x in sorted(escapes, key=lambda x: x["file"]))
)
print(f"NORMALIZE_ESCAPES {len(escapes)}", flush=True)
PY

cp "$manifest" "$output/manifest.jsonl"
printf 'NORMALIZED %s pipeline=default<O2>,%s jobs=%s\n' \
    "$(wc -l < "$manifest" | tr -d ' ')" "$lower_switch" "$jobs"
