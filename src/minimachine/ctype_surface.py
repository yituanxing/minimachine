from __future__ import annotations


CTYPE_EXTERNALS = frozenset({"__ctype_b_loc", "__ctype_tolower_loc"})
_CTYPE_MIN = -128
_CTYPE_MAX = 255
_CTYPE_COUNT = _CTYPE_MAX - _CTYPE_MIN + 1

# glibc's ctype bit values as observed by little-endian userspace.  The table
# itself is locale data, not host state; MiniMachine intentionally exposes a
# deterministic C-locale surface to guest programs.
_ISUPPER = 0x0100
_ISLOWER = 0x0200
_ISALPHA = 0x0400
_ISDIGIT = 0x0800
_ISXDIGIT = 0x1000
_ISSPACE = 0x2000
_ISPRINT = 0x4000
_ISGRAPH = 0x8000
_ISBLANK = 0x0001
_ISCNTRL = 0x0002
_ISPUNCT = 0x0004
_ISALNUM = 0x0008


def _class_mask(code: int) -> int:
    if code < 0 or code > 127:
        return 0

    mask = 0
    if code < 32 or code == 127:
        mask |= _ISCNTRL
    if code in (9, 10, 11, 12, 13, 32):
        mask |= _ISSPACE
    if code in (9, 32):
        mask |= _ISBLANK
    if 32 <= code <= 126:
        mask |= _ISPRINT
    if 33 <= code <= 126:
        mask |= _ISGRAPH

    if 48 <= code <= 57:
        mask |= _ISDIGIT | _ISXDIGIT | _ISALNUM
    elif 65 <= code <= 90:
        mask |= _ISUPPER | _ISALPHA | _ISALNUM
        if code <= 70:
            mask |= _ISXDIGIT
    elif 97 <= code <= 122:
        mask |= _ISLOWER | _ISALPHA | _ISALNUM
        if code <= 102:
            mask |= _ISXDIGIT
    elif 33 <= code <= 126 and code != 32:
        mask |= _ISPUNCT

    return mask


def _tolower(code: int) -> int:
    if 65 <= code <= 90:
        return code + 32
    return code


def _tables(vm) -> tuple[int, int]:
    cached = getattr(vm, "user_ctype_tables", None)
    if cached is not None:
        return cached

    b_storage = vm.alloc_bytes(_CTYPE_COUNT * 2, align=2)
    lower_storage = vm.alloc_bytes(_CTYPE_COUNT * 4, align=4)
    for code in range(_CTYPE_MIN, _CTYPE_MAX + 1):
        slot = code - _CTYPE_MIN
        vm.memory.write(b_storage + slot * 2, 16, _class_mask(code))
        vm.memory.write(lower_storage + slot * 4, 32, _tolower(code) & 0xFFFFFFFF)

    # glibc returns a pointer to a pointer.  Bias the exposed tables so index 0
    # is the C character 0 while still leaving the -128..-1 compatibility
    # range allocated immediately before it.
    b_table = b_storage + (-_CTYPE_MIN) * 2
    lower_table = lower_storage + (-_CTYPE_MIN) * 4
    b_cell = vm.alloc_bytes(8, align=8)
    lower_cell = vm.alloc_bytes(8, align=8)
    vm.memory.write(b_cell, 64, b_table)
    vm.memory.write(lower_cell, 64, lower_table)
    cached = (b_cell, lower_cell)
    vm.user_ctype_tables = cached
    return cached


def resolve_ctype_callback(original: str, *, vm_error):
    """Resolve glibc-compatible C-locale ctype table accessors.

    These accessors return guest addresses only.  The layout matches the ABI
    expected by the frozen userspace LLVM: ``__ctype_b_loc`` exposes 16-bit
    classification masks and ``__ctype_tolower_loc`` exposes signed 32-bit
    case-conversion entries, both via pointer-to-pointer cells.
    """
    if original not in CTYPE_EXTERNALS:
        return None

    if original == "__ctype_b_loc":
        def ctype_b_loc(vm, args):
            if args:
                raise vm_error("__ctype_b_loc expects no arguments")
            cell, _ = _tables(vm)
            print(f"BOOT_EXEC_USER_CTYPE kind=b cell=0x{cell:x}", flush=True)
            return cell

        return ctype_b_loc

    if original == "__ctype_tolower_loc":
        def ctype_tolower_loc(vm, args):
            if args:
                raise vm_error("__ctype_tolower_loc expects no arguments")
            _, cell = _tables(vm)
            print(f"BOOT_EXEC_USER_CTYPE kind=tolower cell=0x{cell:x}", flush=True)
            return cell

        return ctype_tolower_loc

    raise AssertionError(f"unhandled ctype external: {original}")
