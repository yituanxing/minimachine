/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
#ifndef _UAPI_ASM_MINIMACHINE_PTRACE_H
#define _UAPI_ASM_MINIMACHINE_PTRACE_H

#include <linux/types.h>

/*
 * Semantic user execution state.  MiniMachine exposes Linux-visible machine
 * state by role rather than by a physical register-file numbering scheme.
 */
struct user_regs_struct {
	__u64 pc;
	__u64 sp;
	__u64 args[6];
	__u64 result;
	__u64 syscall_nr;
	__u64 status;
};

#endif /* _UAPI_ASM_MINIMACHINE_PTRACE_H */
