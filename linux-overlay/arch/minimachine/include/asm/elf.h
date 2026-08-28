#ifndef _ASM_MINIMACHINE_ELF_H
#define _ASM_MINIMACHINE_ELF_H

#include <linux/elf-em.h>
#include <linux/types.h>

/*
 * The first NOMMU MiniMachine userspace ABI is BINFMT_FLAT.  Do not pretend
 * that MiniMachine is an existing ELF machine while the native ELF ABI is
 * still unfrozen.
 */
#define elf_check_arch(hdr)	(false)

#define ELF_CLASS	ELFCLASS64
#define ELF_DATA	ELFDATA2LSB
#define ELF_ARCH	EM_NONE

#define ELF_EXEC_PAGESIZE	4096
#define ELF_HWCAP		0
#define ELF_PLATFORM		NULL
#define ELF_PLAT_INIT(regs, load_addr) do { } while (0)

typedef unsigned long elf_greg_t;
typedef elf_greg_t elf_gregset_t[1];
typedef unsigned long elf_fpregset_t;

#endif /* _ASM_MINIMACHINE_ELF_H */
