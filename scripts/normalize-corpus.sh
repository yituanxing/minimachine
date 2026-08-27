#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-riscv.env"

input=${1:-"$root/build/linux-$LINUX_VERSION-riscv/corpus"}
output=${2:-"$root/build/linux-$LINUX_VERSION-riscv/normalized"}

opt="opt-$LLVM_MAJOR"
command -v "$opt" >/dev/null 2>&1 || {
    printf 'missing required tool: %s\n' "$opt" >&2
    exit 2
}

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

python3 - "$manifest" <<'PY' | while IFS= read -r rel; do
import json
import sys
for line in open(sys.argv[1]):
    if line.strip():
        print(json.loads(line)["bc"])
PY
    src="$input/$rel"
    dst="$output/$rel"
    mkdir -p "$(dirname -- "$dst")"
    "$opt" -passes="$lower_switch" "$src" -o "$dst"
done

cp "$manifest" "$output/manifest.jsonl"
count=$(find "$output" -type f -name '*.bc' | wc -l | tr -d ' ')
printf 'NORMALIZED %s pass=%s\n' "$count" "$lower_switch"
