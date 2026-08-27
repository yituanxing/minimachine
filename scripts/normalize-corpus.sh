#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-riscv.env"

input=${1:-"$root/build/linux-$LINUX_VERSION-riscv/corpus"}
output=${2:-"$root/build/linux-$LINUX_VERSION-riscv/normalized"}

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

mapfile -t entries < <(
    python3 - "$manifest" <<'PY'
import json
import sys
for line in open(sys.argv[1]):
    if line.strip():
        print(json.loads(line)["bc"])
PY
)

total=${#entries[@]}
index=0
for rel in "${entries[@]}"; do
    index=$((index + 1))
    src="$input/$rel"
    dst="$output/$rel"
    tmp="$dst.o2.tmp.bc"
    mkdir -p "$(dirname -- "$dst")"

    # Clang -save-temps=obj intentionally gives us pre-optimization bitcode.
    # Re-run only LLVM optimization on that frozen bitcode (no C frontend)
    # so our canonical corpus matches the -O2 shape used for machine design.
    "$clang" --target=riscv64-linux-gnu -mabi=lp64 -O2 -emit-llvm -c -x ir \
        "$src" -o "$tmp"

    # Lower switch after O2 because switches may be introduced/canonicalized
    # by optimization. Keep SSA/PHI: reg2mem was measured to be too costly.
    "$opt" -passes="$lower_switch" "$tmp" -o "$dst"
    rm -f "$tmp"

    if test $((index % 25)) -eq 0 || test "$index" -eq "$total"; then
        printf 'NORMALIZE %s/%s\n' "$index" "$total"
    fi
done

cp "$manifest" "$output/manifest.jsonl"
printf 'NORMALIZED %s clang_o2=1 pass=%s\n' "$total" "$lower_switch"
