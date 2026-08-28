// SPDX-License-Identifier: GPL-2.0-only
#include <linux/kbuild.h>

/*
 * No MiniMachine assembly consumes C structure offsets yet.  Keep this file
 * present so Linux can complete prepare; real pt_regs/thread offsets are added
 * together with the first entry/context assembly contract.
 */
int main(void)
{
	return 0;
}
