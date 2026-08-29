// SPDX-License-Identifier: GPL-2.0-only
/*
 * Boot-first MiniMachine tty/console.
 *
 * ttyMM0 is a real Linux tty endpoint.  Writes are forwarded to the semantic
 * VM console service, so PID 1 and the eventual shell use normal Linux file
 * descriptors instead of bypassing the kernel.
 */
#include <linux/console.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/tty.h>
#include <linux/tty_driver.h>

struct minimachine_tty_port {
    struct tty_port port;
};

static struct minimachine_tty_port minimachine_tty;
static struct tty_driver *minimachine_tty_driver;

static __always_inline void minimachine_console_write_host(const char *buf,
                                                            unsigned long len)
{
    asm volatile("ecall"
                 :
                 : "r"(1UL), "r"(buf), "r"(len)
                 : "memory");
}

static int minimachine_tty_open(struct tty_struct *tty, struct file *file)
{
    tty->driver_data = &minimachine_tty;
    return tty_port_open(&minimachine_tty.port, tty, file);
}

static void minimachine_tty_close(struct tty_struct *tty, struct file *file)
{
    struct minimachine_tty_port *port = tty->driver_data;

    tty_port_close(&port->port, tty, file);
}

static ssize_t minimachine_tty_write(struct tty_struct *tty,
                                     const u8 *buf, size_t count)
{
    (void)tty;
    minimachine_console_write_host((const char *)buf, count);
    return count;
}

static unsigned int minimachine_tty_write_room(struct tty_struct *tty)
{
    (void)tty;
    return 4096;
}

static void minimachine_tty_hangup(struct tty_struct *tty)
{
    struct minimachine_tty_port *port = tty->driver_data;

    tty_port_hangup(&port->port);
}

static const struct tty_operations minimachine_tty_ops = {
    .open = minimachine_tty_open,
    .close = minimachine_tty_close,
    .write = minimachine_tty_write,
    .write_room = minimachine_tty_write_room,
    .hangup = minimachine_tty_hangup,
};

static const struct tty_port_operations minimachine_port_ops = {
};

static void minimachine_console_write(struct console *console,
                                      const char *buf, unsigned int count)
{
    (void)console;
    minimachine_console_write_host(buf, count);
}

static struct tty_driver *minimachine_console_device(struct console *console,
                                                     int *index)
{
    (void)console;
    *index = 0;
    return minimachine_tty_driver;
}

static int minimachine_console_setup(struct console *console, char *options)
{
    (void)options;
    console->index = 0;
    return 0;
}

static struct console minimachine_console = {
    .name = "ttyMM",
    .write = minimachine_console_write,
    .device = minimachine_console_device,
    .setup = minimachine_console_setup,
    .flags = CON_PRINTBUFFER,
    .index = -1,
};

static int __init minimachine_tty_init(void)
{
    int ret;

    minimachine_tty_driver = tty_alloc_driver(
        1, TTY_DRIVER_RESET_TERMIOS | TTY_DRIVER_REAL_RAW);
    if (IS_ERR(minimachine_tty_driver))
        return PTR_ERR(minimachine_tty_driver);

    tty_port_init(&minimachine_tty.port);
    minimachine_tty.port.ops = &minimachine_port_ops;

    minimachine_tty_driver->driver_name = "minimachine-console";
    minimachine_tty_driver->name = "ttyMM";
    minimachine_tty_driver->major = 0;
    minimachine_tty_driver->minor_start = 0;
    minimachine_tty_driver->type = TTY_DRIVER_TYPE_CONSOLE;
    minimachine_tty_driver->init_termios = tty_std_termios;
    tty_set_operations(minimachine_tty_driver, &minimachine_tty_ops);
    tty_port_link_device(&minimachine_tty.port, minimachine_tty_driver, 0);

    ret = tty_register_driver(minimachine_tty_driver);
    if (ret) {
        tty_port_destroy(&minimachine_tty.port);
        tty_driver_kref_put(minimachine_tty_driver);
        return ret;
    }

    register_console(&minimachine_console);
    pr_info("MiniMachine: ttyMM0 console ready major=%d\n",
            minimachine_tty_driver->major);
    return 0;
}
device_initcall(minimachine_tty_init);
