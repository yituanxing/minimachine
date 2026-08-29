#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-minimachine.env"

cache_root=${CACHE_ROOT:-"$root/.cache/minimachine"}
build_root=${BUILD_ROOT:-"$root/build/linux-$LINUX_VERSION-minimachine-boot"}
archive="$cache_root/source/$LINUX_ARCHIVE"
source_root="$build_root/source"
src="$source_root/linux-$LINUX_VERSION"
out="$build_root/out"
log="$build_root/boot-build.log"

need() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'missing required tool: %s\n' "$1" >&2
        exit 2
    }
}

need curl
need sha256sum
need tar
need make
need "clang-$LLVM_MAJOR"
need "ld.lld-$LLVM_MAJOR"
need "llvm-link-$LLVM_MAJOR"
need "opt-$LLVM_MAJOR"
need "llvm-dis-$LLVM_MAJOR"

mkdir -p "$cache_root/source" "$build_root"

if test -s "$archive" &&
   printf '%s  %s\n' "$LINUX_SHA256" "$archive" | sha256sum -c - >/dev/null 2>&1; then
    printf 'BOOT_SOURCE_ARCHIVE hit %s\n' "$archive"
else
    rm -f "$archive" "$archive.tmp"
    printf 'BOOT_SOURCE_ARCHIVE fill %s\n' "$archive"
    curl -fsSL "$LINUX_URL" -o "$archive.tmp"
    printf '%s  %s\n' "$LINUX_SHA256" "$archive.tmp" | sha256sum -c -
    mv "$archive.tmp" "$archive"
fi

rm -rf "$source_root"
mkdir -p "$source_root" "$out"
tar -xJf "$archive" -C "$source_root"

mkdir -p "$src/arch/minimachine"
cp -a "$root/linux-overlay/arch/minimachine/." "$src/arch/minimachine/"

common_make=(
    make -C "$src"
    O="$out"
    ARCH="$ARCH"
    LLVM="-$LLVM_MAJOR"
    CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"
    KBUILD_BUILD_USER=minimachine
    KBUILD_BUILD_HOST=minimachine
    KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z
    LDFLAGS_vmlinux=--error-limit=0
)

printf 'BOOT_GATE configure ARCH=%s LLVM=%s\n' "$ARCH" "$LLVM_MAJOR"
"${common_make[@]}" "$KCONFIG_TARGET"

# Default MiniMachine Linux baseline: use the architecture defconfig as-is.
# Do not prune subsystems here. Dynamic replay must expose the real next
# blocker under the normal default configuration.
"${common_make[@]}" olddefconfig

config="$out/.config"
test -s "$config"
printf 'BOOT_GATE_DEFAULT_CONFIG target=%s enabled=%s\n' "$KCONFIG_TARGET" "$(grep -c '^CONFIG_[A-Za-z0-9_]*=y$' "$config")"
printf 'BOOT_GATE config_sha256=%s\n' "$(sha256sum "$config" | awk '{print $1}')"
printf 'BOOT_GATE build target=vmlinux\n'
set +e
"${common_make[@]}" KCFLAGS="-save-temps=obj" -j"$(nproc)" vmlinux >"$log" 2>&1
status=$?
set -e

if test "$status" -ne 0; then
    printf 'BOOT_GATE_BLOCKED status=%d log=%s\n' "$status" "$log"
    first=$(grep -n -m1 -E '(^|: )(fatal error:|error:|undefined reference|No rule to make target|No such file|not found)' "$log" || true)
    if test -n "$first"; then
        printf 'BOOT_GATE_FIRST %s\n' "$first"
    fi
    undefined_symbols="$(
        grep -E 'undefined symbol:|undefined reference to' "$log" |
        sed -E 's/.*undefined symbol: ([^[:space:]]+).*/\1/; s/.*undefined reference to [\x27\x60]([^\x27\x60]+)[\x27\x60].*/\1/' |
        sort -u |
        tr '\n' ' '
    )"
    if test -n "$undefined_symbols"; then
        printf 'BOOT_GATE_UNDEFINED symbols=%s\n' "$undefined_symbols"
    fi
    tail -n 240 "$log"
    exit "$status"
fi

test -s "$out/vmlinux"
printf 'BOOT_GATE_VMLINUX_PASS bytes=%s sha256=%s\n'     "$(stat -c%s "$out/vmlinux")"     "$(sha256sum "$out/vmlinux" | awk '{print $1}')"

llvm_root="$build_root/llvm"
rm -rf "$llvm_root"
mkdir -p "$llvm_root"

bc_files=()
while IFS= read -r bc; do
    case "$bc" in
        */kernel/bounds.bc|*/arch/minimachine/kernel/asm-offsets.bc)
            ;;
        *)
            bc_files+=("$bc")
            ;;
    esac
done < <(find "$out" -type f -name '*.bc' | sort)

if test "${#bc_files[@]}" -eq 0; then
    printf 'BOOT_GATE_LLVM_BLOCKED no kernel bitcode produced\n' >&2
    exit 1
fi

printf 'BOOT_GATE_LLVM link bc_files=%d\n' "${#bc_files[@]}"
"llvm-link-$LLVM_MAJOR" "${bc_files[@]}" -o "$llvm_root/linked.bc"
"opt-$LLVM_MAJOR" -passes=verify -disable-output "$llvm_root/linked.bc"
"llvm-dis-$LLVM_MAJOR" "$llvm_root/linked.bc" -o "$llvm_root/linked.ll"
printf 'BOOT_GATE_LLVM_LINK_PASS bytes=%s\n' "$(stat -c%s "$llvm_root/linked.bc")"

printf 'BOOT_GATE_ARTIFACT_READY linked_ll=%s vmlinux=%s\n' \
    "$(stat -c%s "$llvm_root/linked.ll")" "$(stat -c%s "$out/vmlinux")"
