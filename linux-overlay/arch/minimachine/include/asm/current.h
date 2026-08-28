#ifndef _ASM_MINIMACHINE_CURRENT_H
#define _ASM_MINIMACHINE_CURRENT_H

#ifndef __ASSEMBLY__

struct task_struct;

/*
 * Updated by the MiniMachine context-switch runtime.  The first UP target has
 * exactly one executing CPU, so a single current-task slot is the correct ABI.
 */
extern struct task_struct *minimachine_current_task;

#define current minimachine_current_task
#define get_current() minimachine_current_task

#endif /* !__ASSEMBLY__ */

#endif /* _ASM_MINIMACHINE_CURRENT_H */
