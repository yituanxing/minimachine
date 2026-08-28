#ifndef _ASM_MINIMACHINE_THREAD_INFO_H
#define _ASM_MINIMACHINE_THREAD_INFO_H

#include <asm/page.h>

/*
 * Keep thread_info at the beginning of task_struct.  The MiniMachine runtime
 * owns the current-task pointer, so no carrier-ISA stack-pointer assembly is
 * needed to find the current task.
 */
#define THREAD_SIZE_ORDER 2
#define THREAD_SHIFT      (PAGE_SHIFT + THREAD_SIZE_ORDER)
#define THREAD_SIZE       (PAGE_SIZE << THREAD_SIZE_ORDER)
#define THREAD_ALIGN      THREAD_SIZE

#ifndef __ASSEMBLY__

struct thread_info {
	unsigned long flags;
	int preempt_count;
	unsigned int cpu;
	unsigned long syscall_work;
};

#define INIT_THREAD_INFO(tsk)			\
{						\
	.flags		= 0,			\
	.preempt_count	= INIT_PREEMPT_COUNT,	\
	.cpu		= 0,			\
	.syscall_work	= 0,			\
}

#endif /* !__ASSEMBLY__ */

/* Low bits are return-to-user/syscall work, matching Linux's generic API. */
#define TIF_SYSCALL_TRACE	0
#define TIF_NOTIFY_RESUME	1
#define TIF_SIGPENDING		2
#define TIF_NEED_RESCHED	3
#define TIF_RESTORE_SIGMASK	4
#define TIF_SECCOMP		5
#define TIF_SYSCALL_AUDIT	6
#define TIF_SYSCALL_TRACEPOINT	7
#define TIF_SYSCALL_EMU		8
#define TIF_NOTIFY_SIGNAL	9
#define TIF_UPROBE		10
#define TIF_POLLING_NRFLAG	16
#define TIF_MEMDIE		18

#define _TIF_SYSCALL_TRACE	(1UL << TIF_SYSCALL_TRACE)
#define _TIF_NOTIFY_RESUME	(1UL << TIF_NOTIFY_RESUME)
#define _TIF_SIGPENDING		(1UL << TIF_SIGPENDING)
#define _TIF_NEED_RESCHED	(1UL << TIF_NEED_RESCHED)
#define _TIF_RESTORE_SIGMASK	(1UL << TIF_RESTORE_SIGMASK)
#define _TIF_SECCOMP		(1UL << TIF_SECCOMP)
#define _TIF_SYSCALL_AUDIT	(1UL << TIF_SYSCALL_AUDIT)
#define _TIF_SYSCALL_TRACEPOINT	(1UL << TIF_SYSCALL_TRACEPOINT)
#define _TIF_SYSCALL_EMU	(1UL << TIF_SYSCALL_EMU)
#define _TIF_NOTIFY_SIGNAL	(1UL << TIF_NOTIFY_SIGNAL)
#define _TIF_UPROBE		(1UL << TIF_UPROBE)
#define _TIF_POLLING_NRFLAG	(1UL << TIF_POLLING_NRFLAG)
#define _TIF_MEMDIE		(1UL << TIF_MEMDIE)

#define _TIF_WORK_MASK \
	(_TIF_NOTIFY_RESUME | _TIF_SIGPENDING | _TIF_NEED_RESCHED | \
	 _TIF_RESTORE_SIGMASK | _TIF_NOTIFY_SIGNAL | _TIF_UPROBE)

#define _TIF_SYSCALL_WORK \
	(_TIF_SYSCALL_TRACE | _TIF_SECCOMP | _TIF_SYSCALL_AUDIT | \
	 _TIF_SYSCALL_TRACEPOINT | _TIF_SYSCALL_EMU)

#endif /* _ASM_MINIMACHINE_THREAD_INFO_H */
