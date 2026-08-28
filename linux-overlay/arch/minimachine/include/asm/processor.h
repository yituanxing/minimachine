#ifndef _ASM_MINIMACHINE_PROCESSOR_H
#define _ASM_MINIMACHINE_PROCESSOR_H

#ifndef __ASSEMBLY__

#include <asm/barrier.h>

struct task_struct;

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

#define cpu_relax() barrier()

extern unsigned long __get_wchan(struct task_struct *task);

#endif /* !__ASSEMBLY__ */

#endif /* _ASM_MINIMACHINE_PROCESSOR_H */
