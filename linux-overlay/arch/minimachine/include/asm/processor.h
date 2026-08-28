#ifndef _ASM_MINIMACHINE_PROCESSOR_H
#define _ASM_MINIMACHINE_PROCESSOR_H

#ifndef __ASSEMBLY__

#include <asm/barrier.h>
#include <asm/page.h>
#include <asm/ptrace.h>

struct task_struct;

#define TASK_SIZE		(~0UL)
#define STACK_TOP		TASK_SIZE
#define STACK_TOP_MAX		STACK_TOP
#define TASK_UNMAPPED_BASE	0UL

/*
 * Minimal saved kernel execution state.  P3 has no architectural register
 * file contract here; a suspended Linux task only needs a resume point and
 * kernel stack ownership at this stage.
 */
struct thread_struct {
	unsigned long kernel_sp;
	unsigned long resume_pc;
};

#define INIT_THREAD { }

#define task_pt_regs(task) \
	((struct pt_regs *)(task_stack_page(task) + THREAD_SIZE) - 1)

#define KSTK_EIP(task)	(task_pt_regs(task)->pc)
#define KSTK_ESP(task)	(task_pt_regs(task)->sp)

#define cpu_relax() barrier()

extern unsigned long __get_wchan(struct task_struct *task);

#endif /* !__ASSEMBLY__ */

#endif /* _ASM_MINIMACHINE_PROCESSOR_H */
