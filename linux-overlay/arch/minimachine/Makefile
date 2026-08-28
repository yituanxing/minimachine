# SPDX-License-Identifier: GPL-2.0-only

KBUILD_DEFCONFIG := defconfig

BITS := 64
UTS_MACHINE := minimachine64
export BITS

# The native MiniMachine compiler path is connected in a later milestone.
# For the Kconfig/target-boundary gate we intentionally add no foreign ISA
# -march/-mabi flags here.
KBUILD_CFLAGS += -fno-builtin
