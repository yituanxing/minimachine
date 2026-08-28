// SPDX-License-Identifier: GPL-2.0-only
/*
 * Boot-first MiniMachine Linux architecture runtime.
 *
 * This file intentionally implements only the semantic contracts required to
 * enter and execute generic Linux.  Device/platform detail remains a VM host
 * service problem and will be split into focused files after first boot.
 */

#include <linux/delay.h>
#include <linux/init.h>
#include <linux/irq.h>
#include <linux/kernel.h>
#include <linux/memblock.h>
#include <linux/mm.h>
#include <linux/ptrace.h>
#include <linux/reboot.h>
#include <linux/sched.h>
#include <linux/sched/debug.h>
#include <linux/sched/task.h>
#include <linux/sched/task_stack.h>
#include <linux/seq_file.h>
#include <linux/string.h>

#include <asm/page.h>
#include <asm/processor.h>
#include <asm/ptrace.h>

#define MINIMACHINE_BOOT_RAM_SIZE (64UL * 1024 * 1024)

extern char _end[];

unsigned long memory_start;
unsigned long memory_end = MINIMACHINE_BOOT_RAM_SIZE;
void *empty_zero_page;

/*
 * asm-generic/vmlinux.lds.h defines init_stack at the beginning of the
 * .data..init_task region.  Materialize the full THREAD_SIZE reservation in
 * LLVM as well so the MiniMachine data image cannot overlap the initial stack.
 */
unsigned char minimachine_init_stack_storage[THREAD_SIZE]
	__section(".data..init_task") __aligned(THREAD_SIZE) __used;

struct task_struct *minimachine_current_task = &init_task;

static const char setup_arch_entered[] =
	"Linux/MiniMachine: setup_arch entered\n";
static const char setup_arch_ready[] =
	"Linux/MiniMachine: setup_arch ready\n";

static __always_inline void minimachine_boot_console(const char *text,
						      unsigned long len)
{
	/*
	 * The RISC-V spelling is only the Clang transport syntax.  MiniMachine's
	 * legalizer converts this to the architecture-neutral SYS ecall contract;
	 * the P3 VM supplies the host console service.
	 */
	asm volatile("ecall" : : "r"(1UL), "r"(text), "r"(len) : "memory");
}

static unsigned long minimachine_irq_state;

unsigned long minimachine_irq_save_flags(void)
{
	return minimachine_irq_state;
}

void minimachine_irq_restore(unsigned long flags)
{
	minimachine_irq_state = !!flags;
}

void minimachine_cpu_idle(void)
{
	barrier();
}

void __delay(unsigned long loops)
{
	while (loops--)
		barrier();
}

void __const_udelay(unsigned long xloops)
{
	/* No physical device timing exists in the first semantic VM platform. */
	(void)xloops;
	barrier();
}

void __udelay(unsigned long usecs)
{
	(void)usecs;
	barrier();
}

void __ndelay(unsigned long nsecs)
{
	(void)nsecs;
	barrier();
}

void calibrate_delay(void)
{
	/* Deterministic placeholder until the VM clock service is connected. */
	loops_per_jiffy = 1;
}

void __init paging_init(void)
{
	unsigned long max_zone_pfn[MAX_NR_ZONES] = { 0 };

	high_memory = (void *)memory_end;
	empty_zero_page = memblock_alloc(PAGE_SIZE, PAGE_SIZE);
	if (!empty_zero_page)
		panic("MiniMachine: unable to allocate zero page");

	max_zone_pfn[ZONE_NORMAL] = memory_end >> PAGE_SHIFT;
	free_area_init(max_zone_pfn);
}

void __init mem_init(void)
{
	memblock_free_all();
}

void __init setup_arch(char **cmdline_p)
{
	minimachine_boot_console(setup_arch_entered,
				 sizeof(setup_arch_entered) - 1);

	memory_start = PAGE_ALIGN((unsigned long)_end);

	/*
	 * Reserve the carrier image and expose a single flat RAM bank.  PAGE_OFFSET
	 * is zero for the first NOMMU target, so VM addresses are direct.
	 */
	memblock_add(0, memory_end);
	memblock_reserve(0, memory_start);

	strscpy(boot_command_line, CONFIG_CMDLINE, COMMAND_LINE_SIZE);
	*cmdline_p = boot_command_line;
	parse_early_param();

	paging_init();

	minimachine_boot_console(setup_arch_ready,
				 sizeof(setup_arch_ready) - 1);
}

void __init init_IRQ(void)
{
	/* Timer/console interrupts are introduced as VM host services later. */
}

void __init time_init(void)
{
	/* The first dynamic milestone only needs deterministic early boot. */
}

struct task_struct *__switch_to(struct task_struct *prev,
				struct task_struct *next)
{
	minimachine_current_task = next;
	return prev;
}

int copy_thread(struct task_struct *p, const struct kernel_clone_args *args)
{
	struct pt_regs *regs = task_pt_regs(p);

	memset(regs, 0, sizeof(*regs));
	regs->sp = args->stack;
	regs->pc = args->fn ? (unsigned long)args->fn : 0;
	regs->args[0] = (unsigned long)args->fn_arg;
	regs->status = MINIMACHINE_STATUS_IRQ_ENABLE;

	p->thread.kernel_sp = (unsigned long)regs;
	p->thread.resume_pc = regs->pc;
	return 0;
}

void flush_thread(void)
{
}

unsigned long __get_wchan(struct task_struct *task)
{
	(void)task;
	return 0;
}

unsigned long wrong_size_cmpxchg(volatile void *ptr)
{
	panic("MiniMachine: invalid cmpxchg width at %p", (const void *)ptr);
}

void show_regs(struct pt_regs *regs)
{
	pr_info("MiniMachine regs: pc=%lx sp=%lx status=%lx\n",
		regs->pc, regs->sp, regs->status);
}

void show_stack(struct task_struct *task, unsigned long *sp,
		const char *loglvl)
{
	(void)sp;
	printk("%sMiniMachine stack: task=%s\n",
	       loglvl ? loglvl : KERN_INFO,
	       task ? task->comm : "<none>");
}

static void *cpuinfo_start(struct seq_file *m, loff_t *pos)
{
	(void)m;
	return *pos == 0 ? (void *)1 : NULL;
}

static void *cpuinfo_next(struct seq_file *m, void *v, loff_t *pos)
{
	(void)m;
	(void)v;
	++*pos;
	return NULL;
}

static void cpuinfo_stop(struct seq_file *m, void *v)
{
	(void)m;
	(void)v;
}

static int cpuinfo_show(struct seq_file *m, void *v)
{
	(void)v;
	seq_puts(m, "processor\t: 0\n");
	seq_puts(m, "model name\t: MiniMachine semantic VM\n\n");
	return 0;
}

const struct seq_operations cpuinfo_op = {
	.start = cpuinfo_start,
	.next = cpuinfo_next,
	.stop = cpuinfo_stop,
	.show = cpuinfo_show,
};

void ptrace_disable(struct task_struct *child)
{
	(void)child;
}

long arch_ptrace(struct task_struct *child, long request,
		 unsigned long addr, unsigned long data)
{
	return ptrace_request(child, request, addr, data);
}

void machine_restart(char *cmd)
{
	(void)cmd;
	for (;;)
		minimachine_cpu_idle();
}

void machine_halt(void)
{
	for (;;)
		minimachine_cpu_idle();
}

void machine_power_off(void)
{
	for (;;)
		minimachine_cpu_idle();
}
