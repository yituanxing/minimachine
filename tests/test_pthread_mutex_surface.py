from __future__ import annotations

import unittest

from src.minimachine.pthread_surface import (
    EBUSY,
    EINVAL,
    EPERM,
    PTHREAD_MUTEX_RECURSIVE,
    resolve_pthread_mutex_callback,
)
from src.minimachine.vm import Program, VMError


def callback(name: str):
    result = resolve_pthread_mutex_callback(name, vm_error=VMError)
    assert result is not None
    return result


class PthreadMutexSurfaceTests(unittest.TestCase):
    def test_recursive_mutex_attr_contract(self):
        vm = Program().new_vm()
        vm.active_user_task = 0x1000
        attr = 0x2000
        mutex = 0x3000

        self.assertEqual(callback("pthread_mutexattr_init")(vm, (attr,)), 0)
        self.assertEqual(
            callback("pthread_mutexattr_settype")(
                vm, (attr, PTHREAD_MUTEX_RECURSIVE)
            ),
            0,
        )
        self.assertEqual(callback("pthread_mutex_init")(vm, (mutex, attr)), 0)
        self.assertEqual(callback("pthread_mutex_lock")(vm, (mutex,)), 0)
        self.assertEqual(callback("pthread_mutex_lock")(vm, (mutex,)), 0)
        self.assertEqual(vm.user_pthread_mutexes[mutex]["depth"], 2)
        self.assertEqual(callback("pthread_mutex_unlock")(vm, (mutex,)), 0)
        self.assertEqual(callback("pthread_mutex_unlock")(vm, (mutex,)), 0)
        self.assertEqual(vm.user_pthread_mutexes[mutex]["depth"], 0)
        self.assertEqual(callback("pthread_mutex_destroy")(vm, (mutex,)), 0)
        self.assertEqual(callback("pthread_mutexattr_destroy")(vm, (attr,)), 0)

    def test_trylock_and_owner_checks(self):
        vm = Program().new_vm()
        mutex = 0x4000

        vm.active_user_task = 0xA000
        self.assertEqual(callback("pthread_mutex_lock")(vm, (mutex,)), 0)

        vm.active_user_task = 0xB000
        self.assertEqual(callback("pthread_mutex_trylock")(vm, (mutex,)), EBUSY)
        self.assertEqual(callback("pthread_mutex_unlock")(vm, (mutex,)), EPERM)
        with self.assertRaisesRegex(VMError, "would block"):
            callback("pthread_mutex_lock")(vm, (mutex,))

        vm.active_user_task = 0xA000
        self.assertEqual(callback("pthread_mutex_unlock")(vm, (mutex,)), 0)

        vm.active_user_task = 0xB000
        self.assertEqual(callback("pthread_mutex_trylock")(vm, (mutex,)), 0)
        self.assertEqual(callback("pthread_mutex_unlock")(vm, (mutex,)), 0)

    def test_static_initializer_is_lazy_default_mutex(self):
        vm = Program().new_vm()
        vm.active_user_task = 0x1234
        mutex = 0x5000

        self.assertFalse(hasattr(vm, "user_pthread_mutexes"))
        self.assertEqual(callback("pthread_mutex_lock")(vm, (mutex,)), 0)
        self.assertEqual(vm.user_pthread_mutexes[mutex]["owner"], 0x1234)
        self.assertEqual(callback("pthread_mutex_unlock")(vm, (mutex,)), 0)

    def test_invalid_attr_type_is_rejected(self):
        vm = Program().new_vm()
        attr = 0x6000
        self.assertEqual(callback("pthread_mutexattr_init")(vm, (attr,)), 0)
        self.assertEqual(
            callback("pthread_mutexattr_settype")(vm, (attr, 99)), EINVAL
        )

    def test_pthread_thread_creation_is_not_faked(self):
        self.assertIsNone(
            resolve_pthread_mutex_callback("pthread_create", vm_error=VMError)
        )
        self.assertIsNone(
            resolve_pthread_mutex_callback("pthread_join", vm_error=VMError)
        )


if __name__ == "__main__":
    unittest.main()
