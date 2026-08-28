/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
#ifndef _UAPI_ASM_MINIMACHINE_SIGCONTEXT_H
#define _UAPI_ASM_MINIMACHINE_SIGCONTEXT_H

#include <linux/types.h>

/*
 * Provisional first-port signal frame storage.  Signal delivery/return is not
 * enabled as a frozen MiniMachine userspace ABI yet; reserve a fixed 32-word
 * machine-state area without borrowing another ISA's register naming.
 */
#define MINIMACHINE_SIGCONTEXT_WORDS 32

struct sigcontext {
	__u64 state[MINIMACHINE_SIGCONTEXT_WORDS];
};

#endif /* _UAPI_ASM_MINIMACHINE_SIGCONTEXT_H */
