#ifndef _ASM_MINIMACHINE_FLAT_H
#define _ASM_MINIMACHINE_FLAT_H

/*
 * First MiniMachine userspace ABI uses the generic bFLT relocation model.
 * User pointers are plain 32-bit bFLT relocation words even though the
 * kernel itself is 64-bit; no ISA-specific relocation encoding is needed.
 */
#include <asm-generic/flat.h>

#define FLAT_PLAT_INIT(regs) do { } while (0)

#endif /* _ASM_MINIMACHINE_FLAT_H */
