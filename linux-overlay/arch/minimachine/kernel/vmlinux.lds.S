/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Minimal semantic linker script for the first MiniMachine Linux boot.
 *
 * This intentionally avoids a carrier-ISA OUTPUT_ARCH/OUTPUT_FORMAT.  The
 * bootstrap ELF is only a transport artifact; MiniMachine execution is built
 * from LLVM/P3.  The first dynamic milestone enters at start_kernel directly.
 */

#define MM_PAGE_SIZE 4096
#define MM_THREAD_SIZE 16384
#define MM_CACHE_BYTES 64

#include <asm-generic/vmlinux.lds.h>

ENTRY(start_kernel)

jiffies = jiffies_64;

SECTIONS
{
	. = 0x100000;

	_text = .;
	_stext = .;
	HEAD_TEXT_SECTION
	.text : {
		TEXT_TEXT
		SCHED_TEXT
		LOCK_TEXT
		IRQENTRY_TEXT
		SOFTIRQENTRY_TEXT
		KPROBES_TEXT
	}
	_etext = .;

	. = ALIGN(MM_PAGE_SIZE);
	__init_begin = .;
	INIT_TEXT_SECTION(MM_PAGE_SIZE)
	INIT_DATA_SECTION(MM_PAGE_SIZE)
	PERCPU_SECTION(MM_CACHE_BYTES)
	. = ALIGN(MM_PAGE_SIZE);
	__init_end = .;

	_sdata = .;
	RO_DATA(MM_PAGE_SIZE)
	RW_DATA(MM_CACHE_BYTES, MM_PAGE_SIZE, MM_THREAD_SIZE)
	_edata = .;

	EXCEPTION_TABLE(8)
	BSS_SECTION(MM_CACHE_BYTES, MM_PAGE_SIZE, MM_CACHE_BYTES)
	_end = .;

	STABS_DEBUG
	DWARF_DEBUG
	ELF_DETAILS

	DISCARDS
}
