#ifndef _ASM_MINIMACHINE_UNISTD_H
#define _ASM_MINIMACHINE_UNISTD_H

/*
 * MiniMachine uses the asm-generic syscall number space.  Some generic
 * syscall families are compiled only for architectures that explicitly opt
 * in to them, so the architecture ABI declarations must match the userspace
 * ABI carried by our RISC-V-built flat binaries.
 *
 * NOMMU userspace needs sys_clone so libc fork can be adapted to
 * CLONE_VM|CLONE_VFORK while still using the real task/fd/VFS machinery.
 * The 64-bit asm-generic stat ABI uses newfstatat/newfstat for syscall
 * numbers 79/80; opt in here so the linked Linux image actually contains
 * those implementations instead of returning -ENOSYS for valid userspace.
 */
#define __ARCH_WANT_SYS_CLONE
#define __ARCH_WANT_NEW_STAT

#include <uapi/asm/unistd.h>

#define NR_syscalls __NR_syscalls

#endif /* _ASM_MINIMACHINE_UNISTD_H */
