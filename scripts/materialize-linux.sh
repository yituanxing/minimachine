#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-riscv.env"

cache_root=${CACHE_ROOT:-"$root/.cache/minimachine"}
build_root=${BUILD_ROOT:-"$root/build/linux-$LINUX_VERSION-riscv"}
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
if test "$jobs" -lt 1; then
    jobs=1
fi

archive="$cache_root/source/$LINUX_ARCHIVE"
src="$cache_root/source/linux-$LINUX_VERSION"
out="$build_root/kbuild"
corpus="$build_root/corpus"
manifest_dir="$build_root/manifests"

mkdir -p "$cache_root/source" "$build_root" "$manifest_dir"

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
need python3

clang_id=$("clang-$LLVM_MAJOR" --version | head -n 1)
lld_id=$("ld.lld-$LLVM_MAJOR" --version | head -n 1)

if test -s "$archive" &&
   printf '%s  %s\n' "$LINUX_SHA256" "$archive" | sha256sum -c - >/dev/null 2>&1; then
    printf 'SOURCE_ARCHIVE hit %s\n' "$archive"
else
    rm -f "$archive" "$archive.tmp"
    printf 'SOURCE_ARCHIVE fill %s\n' "$archive"
    curl -fsSL "$LINUX_URL" -o "$archive.tmp"
    printf '%s  %s\n' "$LINUX_SHA256" "$archive.tmp" | sha256sum -c -
    mv "$archive.tmp" "$archive"
fi
printf '%s  %s\n' "$LINUX_SHA256" "$archive" | sha256sum -c -

if test ! -f "$src/Makefile"; then
    rm -rf "$src"
    tmp_extract="$cache_root/source/.extract-$LINUX_VERSION"
    rm -rf "$tmp_extract"
    mkdir -p "$tmp_extract"
    tar -xJf "$archive" -C "$tmp_extract"
    mv "$tmp_extract/linux-$LINUX_VERSION" "$src"
    rmdir "$tmp_extract"
    printf 'SOURCE_TREE materialized %s\n' "$src"
else
    printf 'SOURCE_TREE hit %s\n' "$src"
fi

identity="$build_root/build-identity.txt"
expected_identity=$(cat <<EOF
linux_version=$LINUX_VERSION
linux_sha256=$LINUX_SHA256
arch=$ARCH
kconfig_target=$KCONFIG_TARGET
llvm_major=$LLVM_MAJOR
clang_id=$clang_id
lld_id=$lld_id
llvm_ias=1
build_user=minimachine
build_host=minimachine
build_timestamp=1970-01-01T00:00:00Z
EOF
)

if test -f "$identity" && test "$(cat "$identity")" != "$expected_identity"; then
    printf 'BUILD_IDENTITY changed; dropping generated Kbuild/corpus state\n'
    rm -rf "$out" "$corpus" "$manifest_dir"
    mkdir -p "$manifest_dir"
fi
printf '%s\n' "$expected_identity" > "$identity"

if test ! -s "$out/.config"; then
    mkdir -p "$out"
    printf 'KCONFIG materialize %s/%s\n' "$ARCH" "$KCONFIG_TARGET"
    make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" LLVM_IAS=1 \
        KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine \
        KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z "$KCONFIG_TARGET"
else
    printf 'KCONFIG hit %s\n' "$out/.config"
fi

# Freeze exact tool/config identity before compiling.
{
    printf 'clang=%s\n' "$clang_id"
    printf 'ld.lld=%s\n' "$lld_id"
    printf 'jobs=%s\n' "$jobs"
    printf 'llvm_ias=1\n'
    printf 'config_sha256='
    sha256sum "$out/.config" | awk '{print $1}'
} > "$build_root/toolchain-identity.txt"

printf 'KBUILD full vmlinux with Clang save-temps (jobs=%s)\n' "$jobs"
# -save-temps=obj makes each real C compilation preserve .i, .bc and .s
# beside its object in the O= tree. This is the expensive stage.
make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" LLVM_IAS=1 \
    KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine \
    KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z \
    KCFLAGS="-save-temps=obj" -j"$jobs" vmlinux

python3 "$root/scripts/select-corpus.py" \
    --kbuild "$out" \
    --output "$manifest_dir"

all_count=$(wc -l < "$manifest_dir/full-all.txt" | tr -d ' ')
printf 'MATERIALIZED_TUS %s\n' "$all_count"
printf 'MANIFEST focused16=%s full100=%s full500=%s full-all=%s\n' \
    "$(wc -l < "$manifest_dir/focused16.txt" | tr -d ' ')" \
    "$(wc -l < "$manifest_dir/full100.txt" | tr -d ' ')" \
    "$(wc -l < "$manifest_dir/full500.txt" | tr -d ' ')" \
    "$all_count"

# Optional compact corpus export. Set EXPORT_CORPUS=1 to copy only the
# selected .i/.bc files into build/.../corpus for caching/artifacts.
if test "${EXPORT_CORPUS:-0}" = 1; then
    rm -rf "$corpus"
    mkdir -p "$corpus"
    python3 "$root/scripts/export-corpus.py" \
        --kbuild "$out" \
        --manifest "$manifest_dir/full500.txt" \
        --output "$corpus"
fi
