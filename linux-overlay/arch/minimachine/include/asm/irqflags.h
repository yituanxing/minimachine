#ifndef _ASM_MINIMACHINE_IRQFLAGS_H
#define _ASM_MINIMACHINE_IRQFLAGS_H

/*
 * MiniMachine interrupt-state ABI.
 *
 * Keep Linux policy in asm-generic/irqflags.h and expose only the three
 * machine operations the target must eventually lower to MiniMachine SYS:
 * read state, restore state, and idle with interrupts enabled.
 */
unsigned long minimachine_irq_save_flags(void);
void minimachine_irq_restore(unsigned long flags);
void minimachine_cpu_idle(void);

#define arch_local_save_flags minimachine_irq_save_flags
#define arch_local_irq_restore minimachine_irq_restore

static inline void arch_safe_halt(void)
{
	minimachine_cpu_idle();
}

#include <asm-generic/irqflags.h>

#endif /* _ASM_MINIMACHINE_IRQFLAGS_H */
