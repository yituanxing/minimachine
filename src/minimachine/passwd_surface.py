from __future__ import annotations


PASSWD_EXTERNALS = frozenset({"getpwuid"})
_PASSWD_SIZE = 48


def _write_bytes(vm, address: int, payload: bytes) -> None:
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        bulk_write(address, payload)
        return
    for offset, byte in enumerate(payload):
        vm.memory.write(address + offset, 8, byte)


def _alloc_cstring(vm, payload: bytes) -> int:
    data = bytes(payload) + b"\0"
    address = vm.alloc_bytes(len(data), align=1)
    _write_bytes(vm, address, data)
    return address


def _zero_bytes(vm, address: int, size: int) -> None:
    bulk_fill = getattr(vm.memory, "bulk_fill", None)
    if bulk_fill is not None:
        bulk_fill(address, 0, size)
        return
    for offset in range(size):
        vm.memory.write(address + offset, 8, 0)


def _records(vm) -> dict[int, int]:
    records = getattr(vm, "user_passwd_records", None)
    if records is None:
        records = {}
        vm.user_passwd_records = records
    return records


def _install_root_record(vm) -> int:
    records = _records(vm)
    existing = records.get(0)
    if existing is not None:
        return int(existing)

    # 64-bit Linux/glibc-compatible struct passwd layout used by the frozen
    # SQLite bundle:
    #   char *pw_name;      +0
    #   char *pw_passwd;    +8
    #   uid_t pw_uid;       +16 (u32)
    #   gid_t pw_gid;       +20 (u32)
    #   char *pw_gecos;     +24
    #   char *pw_dir;       +32
    #   char *pw_shell;     +40
    name = _alloc_cstring(vm, b"root")
    passwd = _alloc_cstring(vm, b"x")
    gecos = _alloc_cstring(vm, b"root")
    home = _alloc_cstring(vm, b"/root")
    shell = _alloc_cstring(vm, b"/bin/sh")

    record = vm.alloc_bytes(_PASSWD_SIZE, align=8)
    _zero_bytes(vm, record, _PASSWD_SIZE)
    vm.memory.write(record + 0, 64, name)
    vm.memory.write(record + 8, 64, passwd)
    vm.memory.write(record + 16, 32, 0)
    vm.memory.write(record + 20, 32, 0)
    vm.memory.write(record + 24, 64, gecos)
    vm.memory.write(record + 32, 64, home)
    vm.memory.write(record + 40, 64, shell)
    records[0] = record
    return record


def resolve_passwd_callback(original: str, *, vm_error):
    """Resolve the small passwd database surface required by real software.

    MiniMachine currently has no NSS or /etc/passwd service layer.  The Linux
    guest nevertheless has a real uid/gid identity, and the current init runs
    as uid 0.  Expose that identity through a stable guest-side struct passwd
    rather than leaking host pointers or inventing arbitrary users.  Unknown
    uids correctly return NULL.
    """
    if original not in PASSWD_EXTERNALS:
        return None

    if original == "getpwuid":
        def getpwuid(vm, args):
            if len(args) != 1:
                raise vm_error("getpwuid expects uid")
            uid = int(args[0]) & 0xFFFFFFFF
            if uid != 0:
                return 0
            record = _install_root_record(vm)
            print(
                "BOOT_EXEC_USER_GETPWUID "
                f"uid={uid} passwd=0x{record:x} home='/root' shell='/bin/sh'",
                flush=True,
            )
            return record

        return getpwuid

    raise AssertionError(f"unhandled passwd external: {original}")
