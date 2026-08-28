#!/usr/bin/env bash
set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$root/configs/linux-6.6.143-minimachine.env"

cache_root=${CACHE_ROOT:-"$root/.cache/minimachine"}
build_root=${BUILD_ROOT:-"$root/build/linux-$LINUX_VERSION-minimachine-target"}
archive="$cache_root/source/$LINUX_ARCHIVE"
source_root="$build_root/source"
src="$source_root/linux-$LINUX_VERSION"
out="$build_root/kconfig"

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

mkdir -p "$cache_root/source" "$build_root"

if test -s "$archive" &&
   printf '%s  %s\n' "$LINUX_SHA256" "$archive" | sha256sum -c - >/dev/null 2>&1; then
    printf 'TARGET_SOURCE_ARCHIVE hit %s\n' "$archive"
else
    rm -f "$archive" "$archive.tmp"
    printf 'TARGET_SOURCE_ARCHIVE fill %s\n' "$archive"
    curl -fsSL "$LINUX_URL" -o "$archive.tmp"
    printf '%s  %s\n' "$LINUX_SHA256" "$archive.tmp" | sha256sum -c -
    mv "$archive.tmp" "$archive"
fi
printf '%s  %s\n' "$LINUX_SHA256" "$archive" | sha256sum -c -

rm -rf "$source_root" "$out"
mkdir -p "$source_root" "$out"
tar -xJf "$archive" -C "$source_root"

if test ! -d "$root/linux-overlay/arch/minimachine"; then
    printf 'missing linux-overlay/arch/minimachine\n' >&2
    exit 2
fi
mkdir -p "$src/arch/minimachine"
cp -a "$root/linux-overlay/arch/minimachine/." "$src/arch/minimachine/"

printf 'TARGET_KCONFIG start ARCH=%s LLVM=%s target=%s\n'     "$ARCH" "$LLVM_MAJOR" "$KCONFIG_TARGET"

make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"     KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine     KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z     "$KCONFIG_TARGET"

# Re-resolve defaults once. This catches target Kconfig symbols that only
# become visible after the initial defconfig merge.
make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"     KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine     KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z     olddefconfig

config="$out/.config"

require_line() {
    grep -Fxq "$1" "$config" || {
        printf 'TARGET_KCONFIG missing: %s\n' "$1" >&2
        exit 1
    }
}

require_line 'CONFIG_64BIT=y'
require_line 'CONFIG_MINIMACHINE=y'
require_line '# CONFIG_MMU is not set'
require_line '# CONFIG_SMP is not set'
require_line 'CONFIG_NR_CPUS=1'
require_line 'CONFIG_BINFMT_FLAT=y'

config_sha=$(sha256sum "$config" | awk '{print $1}')
printf 'TARGET_KCONFIG_PASS arch=%s bits=64 mmu=0 smp=0 nr_cpus=1 flat=1 config_sha256=%s\n'     "$ARCH" "$config_sha"

printf 'TARGET_PREPARE start ARCH=%s\n' "$ARCH"
make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"     KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine     KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z     prepare
printf 'TARGET_PREPARE_PASS arch=%s\n' "$ARCH"

printf 'TARGET_FIRST_TU start source=init/main.c\n'
make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"     KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine     KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z     KCFLAGS="-save-temps=obj" init/main.o

first_bc="$out/init/main.bc"
test -s "$first_bc"
python3 "$root/scripts/legalize_bc.py" "$first_bc" --llvm-major "$LLVM_MAJOR" --strict --json "$build_root/init-main-legalize.json"
python3 "$root/scripts/abi_bc.py" "$first_bc" --llvm-major "$LLVM_MAJOR" --strict --json "$build_root/init-main-abi.json"
printf 'TARGET_FIRST_TU_PASS source=init/main.c bitcode=%s\n' "$first_bc"

printf 'TARGET_INIT_CLUSTER start path=init/\n'
make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"     KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine     KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z     KCFLAGS="-save-temps=obj" init/
python3 "$root/scripts/legalize_bc.py" "$out/init" --llvm-major "$LLVM_MAJOR" --strict --json "$build_root/init-cluster-legalize.json"
python3 "$root/scripts/abi_bc.py" "$out/init" --llvm-major "$LLVM_MAJOR" --strict --json "$build_root/init-cluster-abi.json"
printf 'TARGET_INIT_CLUSTER_PASS path=init/\n'

printf 'TARGET_KERNEL_CLUSTER start path=kernel/\n'
kernel_build_log="$build_root/kernel-build.log"
if ! make -C "$src" O="$out" ARCH="$ARCH" LLVM="-$LLVM_MAJOR" CLANG_TARGET_FLAGS="$LLVM_CARRIER_TRIPLE"     KBUILD_BUILD_USER=minimachine KBUILD_BUILD_HOST=minimachine     KBUILD_BUILD_TIMESTAMP=1970-01-01T00:00:00Z     KCFLAGS="-save-temps=obj" kernel/ >"$kernel_build_log" 2>&1; then
    printf 'TARGET_KERNEL_CLUSTER_BUILD_FAIL log=%s\n' "$kernel_build_log"
    tail -n 200 "$kernel_build_log"
    exit 1
fi
printf 'TARGET_KERNEL_CLUSTER_BUILD_PASS path=kernel/\n'
# kernel/bounds.bc is a prepare-time generator whose inline .ascii markers
# produce bounds.h; it is not linked into the runtime kernel image.
rm -f "$out/kernel/bounds.bc"
python3 "$root/scripts/legalize_bc.py" "$out/kernel" --llvm-major "$LLVM_MAJOR" --strict --json "$build_root/kernel-cluster-legalize.json"
python3 "$root/scripts/abi_bc.py" "$out/kernel" --llvm-major "$LLVM_MAJOR" --strict --json "$build_root/kernel-cluster-abi.json"
printf 'TARGET_KERNEL_CLUSTER_PASS path=kernel/\n'
