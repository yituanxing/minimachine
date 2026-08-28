#ifndef _ASM_MINIMACHINE_ATOMIC_H
#define _ASM_MINIMACHINE_ATOMIC_H

#include <linux/types.h>
#include <linux/irqflags.h>
#include <asm-generic/atomic.h>

/*
 * First MiniMachine Linux target is uniprocessor.  Interrupt exclusion is
 * therefore sufficient to make 64-bit read/modify/write operations atomic.
 */
#define MINIMACHINE_ATOMIC64_OP(op, c_op)                              \
static inline void arch_atomic64_##op(s64 i, atomic64_t *v)            \
{                                                                      \
	unsigned long flags;                                               \
	raw_local_irq_save(flags);                                         \
	v->counter = v->counter c_op i;                                    \
	raw_local_irq_restore(flags);                                      \
}

#define MINIMACHINE_ATOMIC64_OP_RETURN(op, c_op)                       \
static inline s64 arch_atomic64_##op##_return(s64 i, atomic64_t *v)    \
{                                                                      \
	unsigned long flags;                                               \
	s64 ret;                                                           \
	raw_local_irq_save(flags);                                         \
	ret = (v->counter = v->counter c_op i);                            \
	raw_local_irq_restore(flags);                                      \
	return ret;                                                        \
}

#define MINIMACHINE_ATOMIC64_FETCH_OP(op, c_op)                        \
static inline s64 arch_atomic64_fetch_##op(s64 i, atomic64_t *v)       \
{                                                                      \
	unsigned long flags;                                               \
	s64 ret;                                                           \
	raw_local_irq_save(flags);                                         \
	ret = v->counter;                                                  \
	v->counter = ret c_op i;                                           \
	raw_local_irq_restore(flags);                                      \
	return ret;                                                        \
}

#define ATOMIC64_INIT(i)                { (i) }

#define arch_atomic64_read(v)          READ_ONCE((v)->counter)
#define arch_atomic64_set(v, i)        WRITE_ONCE((v)->counter, (i))

MINIMACHINE_ATOMIC64_OP(add, +)
MINIMACHINE_ATOMIC64_OP(sub, -)
MINIMACHINE_ATOMIC64_OP(and, &)
MINIMACHINE_ATOMIC64_OP(or, |)
MINIMACHINE_ATOMIC64_OP(xor, ^)

MINIMACHINE_ATOMIC64_OP_RETURN(add, +)
MINIMACHINE_ATOMIC64_OP_RETURN(sub, -)

MINIMACHINE_ATOMIC64_FETCH_OP(add, +)
MINIMACHINE_ATOMIC64_FETCH_OP(sub, -)
MINIMACHINE_ATOMIC64_FETCH_OP(and, &)
MINIMACHINE_ATOMIC64_FETCH_OP(or, |)
MINIMACHINE_ATOMIC64_FETCH_OP(xor, ^)

#define arch_atomic64_add_return arch_atomic64_add_return
#define arch_atomic64_sub_return arch_atomic64_sub_return

#define arch_atomic64_fetch_add arch_atomic64_fetch_add
#define arch_atomic64_fetch_sub arch_atomic64_fetch_sub
#define arch_atomic64_fetch_and arch_atomic64_fetch_and
#define arch_atomic64_fetch_or  arch_atomic64_fetch_or
#define arch_atomic64_fetch_xor arch_atomic64_fetch_xor

#undef MINIMACHINE_ATOMIC64_FETCH_OP
#undef MINIMACHINE_ATOMIC64_OP_RETURN
#undef MINIMACHINE_ATOMIC64_OP

#endif /* _ASM_MINIMACHINE_ATOMIC_H */
