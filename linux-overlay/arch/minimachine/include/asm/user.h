#ifndef _ASM_MINIMACHINE_USER_H
#define _ASM_MINIMACHINE_USER_H

#include <asm/page.h>
#include <asm/ptrace.h>

/*
 * Traditional core/ptrace user record.  MiniMachine keeps this aligned with
 * its semantic user register ABI rather than inventing a hardware GPR file.
 */
struct user {
	struct user_regs_struct regs;
	unsigned long u_tsize;
	unsigned long u_dsize;
	unsigned long u_ssize;
	unsigned long start_code;
	unsigned long start_data;
	unsigned long start_stack;
	long signal;
	unsigned long u_ar0;
	unsigned long magic;
	char u_comm[32];
};

#endif /* _ASM_MINIMACHINE_USER_H */
