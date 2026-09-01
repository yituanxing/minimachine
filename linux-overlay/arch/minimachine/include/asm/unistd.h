#ifndef _ASM_MINIMACHINE_UNISTD_H
#define _ASM_MINIMACHINE_UNISTD_H

/*
 * MiniMachine uses the asm-generic syscall number space.  The generic
 * syscall table reserves __NR_clone, but the kernel implementation is only
 * built for architectures that explicitly opt in to sys_clone.
 *
 * NOMMU userspace needs this so libc fork can be adapted to Linux
 * CLONE_VM|CLONE_VFORK while still using the real task/fd/VFS machinery.
 */
#define __ARCH_WANT_SYS_CLONE

#include <uapi/asm/unistd.h>

#define NR_syscalls __NR_syscalls

#endif /* _ASM_MINIMACHINE_UNISTD_H */
