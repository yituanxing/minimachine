#ifndef _ASM_MINIMACHINE_PGTABLE_H
#define _ASM_MINIMACHINE_PGTABLE_H

/*
 * First MiniMachine Linux target is NOMMU.  These are only the dummy page
 * table contracts generic Linux expects to compile; no hardware translation
 * structure exists.
 */
#include <asm-generic/pgtable-nopud.h>
#include <asm/page.h>

#define pgd_present(pgd)	(1)
#define pgd_none(pgd)		(0)
#define pgd_bad(pgd)		(0)
#define pgd_clear(pgdp)		do { } while (0)
#define pmd_offset(a, b)	((void *)0)

#define PAGE_NONE	__pgprot(0)
#define PAGE_SHARED	__pgprot(0)
#define PAGE_COPY	__pgprot(0)
#define PAGE_READONLY	__pgprot(0)
#define PAGE_KERNEL	__pgprot(0)

extern void paging_init(void);
#define swapper_pg_dir ((pgd_t *)0)

extern void *empty_zero_page;
#define ZERO_PAGE(vaddr)	(virt_to_page(empty_zero_page))

#define VMALLOC_START	0UL
#define VMALLOC_END	(~0UL)
#define KMAP_START	0UL
#define KMAP_END	(~0UL)

#endif /* _ASM_MINIMACHINE_PGTABLE_H */
