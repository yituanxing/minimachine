#ifndef _ASM_MINIMACHINE_SYSCALL_H
#define _ASM_MINIMACHINE_SYSCALL_H

#include <linux/err.h>
#include <linux/sched.h>
#include <uapi/linux/audit.h>
#include <asm/ptrace.h>

static inline int syscall_get_nr(struct task_struct *task,
				 struct pt_regs *regs)
{
	return (int)regs->syscall_nr;
}

static inline void syscall_rollback(struct task_struct *task,
				    struct pt_regs *regs)
{
	regs->result = regs->orig_arg0;
}

static inline long syscall_get_error(struct task_struct *task,
				     struct pt_regs *regs)
{
	unsigned long value = regs->result;

	return IS_ERR_VALUE(value) ? (long)value : 0;
}

static inline long syscall_get_return_value(struct task_struct *task,
					    struct pt_regs *regs)
{
	return (long)regs->result;
}

static inline void syscall_set_return_value(struct task_struct *task,
					    struct pt_regs *regs,
					    int error, long value)
{
	regs->result = error ? (unsigned long)error : (unsigned long)value;
}

static inline void syscall_get_arguments(struct task_struct *task,
					 struct pt_regs *regs,
					 unsigned long *args)
{
	memcpy(args, regs->args, sizeof(regs->args));
}

/*
 * MiniMachine does not yet have an assigned ELF/AUDIT architecture number.
 * Keep the provisional audit identity explicit rather than borrowing another
 * ISA's machine number.
 */
static inline int syscall_get_arch(struct task_struct *task)
{
	return EM_NONE | __AUDIT_ARCH_64BIT | __AUDIT_ARCH_LE;
}

static inline bool arch_syscall_is_vdso_sigreturn(struct pt_regs *regs)
{
	return false;
}

#endif /* _ASM_MINIMACHINE_SYSCALL_H */
