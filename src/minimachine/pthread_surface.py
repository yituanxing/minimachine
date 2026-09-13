from __future__ import annotations


PTHREAD_MUTEX_NORMAL = 0
PTHREAD_MUTEX_RECURSIVE = 1
PTHREAD_MUTEX_ERRORCHECK = 2
PTHREAD_MUTEX_ADAPTIVE = 3

_VALID_MUTEX_TYPES = {
    PTHREAD_MUTEX_NORMAL,
    PTHREAD_MUTEX_RECURSIVE,
    PTHREAD_MUTEX_ERRORCHECK,
    PTHREAD_MUTEX_ADAPTIVE,
}

EPERM = 1
EBUSY = 16
EINVAL = 22
EDEADLK = 35

PTHREAD_MUTEX_EXTERNALS = frozenset(
    {
        "pthread_mutex_init",
        "pthread_mutex_destroy",
        "pthread_mutex_lock",
        "pthread_mutex_trylock",
        "pthread_mutex_unlock",
        "pthread_mutexattr_init",
        "pthread_mutexattr_destroy",
        "pthread_mutexattr_settype",
    }
)


def _owner(vm) -> int:
    return int(
        getattr(vm, "active_user_task", 0)
        or getattr(vm, "linux_current_task", 0)
        or 1
    )


def _attrs(vm) -> dict[int, int]:
    table = getattr(vm, "user_pthread_mutex_attrs", None)
    if table is None:
        table = {}
        vm.user_pthread_mutex_attrs = table
    return table


def _mutexes(vm) -> dict[int, dict[str, int]]:
    table = getattr(vm, "user_pthread_mutexes", None)
    if table is None:
        table = {}
        vm.user_pthread_mutexes = table
    return table


def _new_mutex(kind: int = PTHREAD_MUTEX_NORMAL) -> dict[str, int]:
    return {"type": int(kind), "owner": 0, "depth": 0}


def _mutex_state(vm, address: int) -> dict[str, int]:
    # Static PTHREAD_MUTEX_INITIALIZER objects reach lock/unlock without an
    # explicit pthread_mutex_init() call.  Lazily materialize the default
    # normal-mutex state on first use.
    table = _mutexes(vm)
    state = table.get(address)
    if state is None:
        state = _new_mutex()
        table[address] = state
    return state


def resolve_pthread_mutex_callback(original: str, *, vm_error):
    """Return a MiniMachine callback for the pthread mutex/attr surface.

    The model is deliberately small but stateful.  It is sufficient for
    single-threaded real software that still uses libc pthread mutexes, while
    refusing to invent successful blocking semantics when two execution owners
    actually contend.  pthread_create/join are intentionally outside this
    surface until MiniMachine has a real pthread scheduling contract.
    """
    if original not in PTHREAD_MUTEX_EXTERNALS:
        return None

    if original == "pthread_mutexattr_init":
        def mutexattr_init(vm, args):
            if len(args) != 1:
                raise vm_error("pthread_mutexattr_init expects attr*")
            attr = int(args[0])
            if not attr:
                return EINVAL
            _attrs(vm)[attr] = PTHREAD_MUTEX_NORMAL
            return 0

        return mutexattr_init

    if original == "pthread_mutexattr_destroy":
        def mutexattr_destroy(vm, args):
            if len(args) != 1:
                raise vm_error("pthread_mutexattr_destroy expects attr*")
            attr = int(args[0])
            if not attr:
                return EINVAL
            _attrs(vm).pop(attr, None)
            return 0

        return mutexattr_destroy

    if original == "pthread_mutexattr_settype":
        def mutexattr_settype(vm, args):
            if len(args) != 2:
                raise vm_error("pthread_mutexattr_settype expects attr*,type")
            attr, kind = map(int, args)
            if not attr or kind not in _VALID_MUTEX_TYPES:
                return EINVAL
            attrs = _attrs(vm)
            if attr not in attrs:
                return EINVAL
            attrs[attr] = kind
            return 0

        return mutexattr_settype

    if original == "pthread_mutex_init":
        def mutex_init(vm, args):
            if len(args) != 2:
                raise vm_error("pthread_mutex_init expects mutex*,attr*")
            mutex, attr = map(int, args)
            if not mutex:
                return EINVAL
            kind = PTHREAD_MUTEX_NORMAL
            if attr:
                attrs = _attrs(vm)
                if attr not in attrs:
                    return EINVAL
                kind = int(attrs[attr])
            _mutexes(vm)[mutex] = _new_mutex(kind)
            return 0

        return mutex_init

    if original == "pthread_mutex_destroy":
        def mutex_destroy(vm, args):
            if len(args) != 1:
                raise vm_error("pthread_mutex_destroy expects mutex*")
            mutex = int(args[0])
            if not mutex:
                return EINVAL
            table = _mutexes(vm)
            state = table.get(mutex)
            if state is not None and state["depth"]:
                return EBUSY
            table.pop(mutex, None)
            return 0

        return mutex_destroy

    if original in {"pthread_mutex_lock", "pthread_mutex_trylock"}:
        def mutex_lock(vm, args):
            if len(args) != 1:
                raise vm_error(f"{original} expects mutex*")
            mutex = int(args[0])
            if not mutex:
                return EINVAL
            state = _mutex_state(vm, mutex)
            owner = _owner(vm)

            if state["depth"] == 0:
                state["owner"] = owner
                state["depth"] = 1
                return 0

            if state["owner"] == owner and state["type"] == PTHREAD_MUTEX_RECURSIVE:
                state["depth"] += 1
                return 0

            if original == "pthread_mutex_trylock":
                return EBUSY

            if state["owner"] == owner and state["type"] == PTHREAD_MUTEX_ERRORCHECK:
                return EDEADLK

            # pthread_mutex_lock blocks rather than returning EBUSY.  Until
            # MiniMachine has a pthread scheduler/wakeup model, stopping here is
            # more truthful than silently succeeding and hiding a concurrency
            # bug in the guest.
            raise vm_error(
                "pthread_mutex_lock would block: "
                f"mutex=0x{mutex:x} owner=0x{state['owner']:x} "
                f"requester=0x{owner:x} type={state['type']} depth={state['depth']}"
            )

        return mutex_lock

    if original == "pthread_mutex_unlock":
        def mutex_unlock(vm, args):
            if len(args) != 1:
                raise vm_error("pthread_mutex_unlock expects mutex*")
            mutex = int(args[0])
            if not mutex:
                return EINVAL
            state = _mutexes(vm).get(mutex)
            owner = _owner(vm)
            if state is None or state["depth"] == 0 or state["owner"] != owner:
                return EPERM
            state["depth"] -= 1
            if state["depth"] == 0:
                state["owner"] = 0
            return 0

        return mutex_unlock

    raise AssertionError(f"unhandled pthread mutex external: {original}")
