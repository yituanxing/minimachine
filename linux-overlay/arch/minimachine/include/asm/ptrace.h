#ifndef _ASM_MINIMACHINE_PTRACE_H
#define _ASM_MINIMACHINE_PTRACE_H

#include <uapi/asm/ptrace.h>

#ifndef __ASSEMBLY__

#include <linux/types.h>

#define MINIMACHINE_STATUS_USER		(1UL << 0)
#define MINIMACHINE_STATUS_IRQ_ENABLE	(1UL << 1)

/*
 * Keep the kernel trap frame shape identical to the public semantic register
 * prefix.  Entry code/runtime will populate this at the trap boundary.
 */
struct pt_regs {
	unsigned long pc;
	unsigned long sp;
	unsigned long args[6];
	unsigned long result;
	unsigned long syscall_nr;
	unsigned long status;
};

static inline int user_mode(const struct pt_regs *regs)
{
	return !!(regs->status & MINIMACHINE_STATUS_USER);
}

static inline unsigned long instruction_pointer(const struct pt_regs *regs)
{
	return regs->pc;
}

static inline void instruction_pointer_set(struct pt_regs *regs,
					   unsigned long value)
{
	regs->pc = value;
}

#define profile_pc(regs) instruction_pointer(regs)

static inline unsigned long user_stack_pointer(const struct pt_regs *regs)
{
	return regs->sp;
}

static inline void user_stack_pointer_set(struct pt_regs *regs,
					  unsigned long value)
{
	regs->sp = value;
}

static inline unsigned long kernel_stack_pointer(const struct pt_regs *regs)
{
	return regs->sp;
}

static inline unsigned long regs_return_value(const struct pt_regs *regs)
{
	return regs->result;
}

static inline void regs_set_return_value(struct pt_regs *regs,
					 unsigned long value)
{
	regs->result = value;
}

static inline unsigned long regs_get_kernel_argument(struct pt_regs *regs,
						      unsigned int n)
{
	return n < 6 ? regs->args[n] : 0;
}

static inline int regs_irqs_disabled(struct pt_regs *regs)
{
	return !(regs->status & MINIMACHINE_STATUS_IRQ_ENABLE);
}

#endif /* !__ASSEMBLY__ */

#endif /* _ASM_MINIMACHINE_PTRACE_H */
