#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MM_PAGE_SHIFT 16
#define MM_PAGE_SIZE (1u << MM_PAGE_SHIFT)
#define MM_BUCKETS 8192u

#define MM_OP_MOV 1
#define MM_OP_SUB 2
#define MM_OP_BR  3

#define MM_V_IMM 1
#define MM_V_SLOT 2
#define MM_V_SP   3
#define MM_V_MEM  4

#define MM_EXT_NONE 0
#define MM_EXT_ZEXT 1
#define MM_EXT_SEXT 2
#define MM_EXT_TRUNC 3

#define MM_COND_EQ 1
#define MM_COND_ULT 2
#define MM_COND_SLT 3

#define MM_T_CODE 1
#define MM_T_SLOT 2
#define MM_T_MEM  3

#define MM_STATUS_LIMIT 0
#define MM_STATUS_HALT 1
#define MM_STATUS_HOST 2
#define MM_STATUS_WATCH 3
#define MM_STATUS_ERROR 4

#define MM_ERR_NONE 0
#define MM_ERR_BAD_BLOCK 1
#define MM_ERR_FALLOFF 2
#define MM_ERR_BAD_OPCODE 3
#define MM_ERR_BAD_VALUE 4
#define MM_ERR_BAD_TARGET 5
#define MM_ERR_OOM 6

typedef struct MMPage {
    uint64_t no;
    uint8_t *data;
    struct MMPage *next;
} MMPage;

typedef struct {
    uint8_t kind;
    uint8_t width;
    uint8_t base_kind;
    uint8_t _pad;
    int64_t offset;
    uint64_t value;
    uint64_t base_value;
} MMOperand;

typedef struct {
    uint8_t opcode;
    uint8_t width;
    uint8_t cond;
    uint8_t extend;
    uint8_t src_bits;
    uint8_t _pad[3];
    MMOperand dst;
    MMOperand src;
    MMOperand a;
    MMOperand b;
    MMOperand t;
    MMOperand f;
} MMInst;

typedef struct {
    uint64_t code;
    uint32_t first;
    uint32_t count;
} MMBlock;

typedef struct {
    int status;
    int error;
    uint64_t target_code;
    uint64_t block_code;
    uint64_t sp;
    uint64_t steps;
    uint32_t ip;
} MMRunResult;

typedef struct {
    const MMInst *insts;
    size_t inst_count;
    const MMBlock *blocks;
    size_t block_count;
    const uint64_t *host_codes;
    size_t host_count;
    uint64_t *watch_codes;
    size_t watch_count;
    uint64_t halt_code;

    MMPage *pages[MM_BUCKETS];
    uint64_t sp;
    uint64_t block_code;
    uint32_t ip;
    uint64_t steps;
    int oom;
    uint64_t cached_page_no;
    MMPage *cached_page;
    int cached_page_valid;
} MMVM;

static inline uint64_t mask_bits(unsigned bits) {
    return bits == 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1);
}

static inline int64_t sext64(uint64_t v, unsigned bits) {
    if (bits == 64) return (int64_t)v;
    uint64_t m = UINT64_C(1) << (bits - 1);
    v &= mask_bits(bits);
    return (int64_t)((v ^ m) - m);
}

static inline size_t page_hash(uint64_t no) {
    no ^= no >> 33;
    no *= UINT64_C(0xff51afd7ed558ccd);
    no ^= no >> 33;
    return (size_t)no & (MM_BUCKETS - 1);
}

static MMPage *get_page(MMVM *vm, uint64_t no, int create) {
    if (vm->cached_page_valid && vm->cached_page_no == no)
        return vm->cached_page;

    size_t h = page_hash(no);
    MMPage *p = vm->pages[h];
    while (p) {
        if (p->no == no) {
            vm->cached_page_no = no;
            vm->cached_page = p;
            vm->cached_page_valid = 1;
            return p;
        }
        p = p->next;
    }
    if (!create) return NULL;
    p = (MMPage *)calloc(1, sizeof(*p));
    if (!p) { vm->oom = 1; return NULL; }
    p->data = (uint8_t *)calloc(1, MM_PAGE_SIZE);
    if (!p->data) { free(p); vm->oom = 1; return NULL; }
    p->no = no;
    p->next = vm->pages[h];
    vm->pages[h] = p;
    vm->cached_page_no = no;
    vm->cached_page = p;
    vm->cached_page_valid = 1;
    return p;
}

static inline uint8_t mem_read8(MMVM *vm, uint64_t addr) {
    MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 0);
    return p ? p->data[addr & (MM_PAGE_SIZE - 1)] : 0;
}

static inline void mem_write8(MMVM *vm, uint64_t addr, uint8_t v) {
    MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 1);
    if (p) p->data[addr & (MM_PAGE_SIZE - 1)] = v;
}

static uint64_t mem_read(MMVM *vm, uint64_t addr, unsigned bits) {
    unsigned n = bits >> 3;
    uint64_t page_off = addr & (MM_PAGE_SIZE - 1);

    if (page_off + n <= MM_PAGE_SIZE) {
        MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 0);
        if (!p) return 0;
        const uint8_t *src = p->data + page_off;
        uint64_t v = 0;
        memcpy(&v, src, n);
        return v & mask_bits(bits);
    }

    uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i)
        v |= ((uint64_t)mem_read8(vm, addr + i)) << (8 * i);
    return v;
}

static void mem_write(MMVM *vm, uint64_t addr, unsigned bits, uint64_t v) {
    unsigned n = bits >> 3;
    uint64_t page_off = addr & (MM_PAGE_SIZE - 1);
    v &= mask_bits(bits);

    if (page_off + n <= MM_PAGE_SIZE) {
        MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 1);
        if (p) memcpy(p->data + page_off, &v, n);
        return;
    }

    for (unsigned i = 0; i < n; ++i)
        mem_write8(vm, addr + i, (uint8_t)(v >> (8 * i)));
}

static int find_block(MMVM *vm, uint64_t code, size_t *idx) {
    size_t lo = 0, hi = vm->block_count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t c = vm->blocks[mid].code;
        if (c == code) { *idx = mid; return 1; }
        if (c < code) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}

static int contains_code(const uint64_t *codes, size_t n, uint64_t code) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t c = codes[mid];
        if (c == code) return 1;
        if (c < code) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}

static int read_value(MMVM *vm, const MMOperand *o, uint64_t *out) {
    switch (o->kind) {
        case MM_V_IMM:
            *out = o->value;
            return 1;
        case MM_V_SLOT:
            *out = mem_read(vm, vm->sp + o->value, 64);
            return 1;
        case MM_V_SP:
            *out = vm->sp;
            return 1;
        case MM_V_MEM: {
            uint64_t base;
            MMOperand b = {0};
            b.kind = o->base_kind;
            b.value = o->base_value;
            if (!read_value(vm, &b, &base)) return 0;
            *out = mem_read(vm, base + (uint64_t)o->offset, o->width);
            return 1;
        }
        default:
            return 0;
    }
}

static int write_operand(MMVM *vm, const MMOperand *o,
                         uint64_t value, unsigned bits) {
    value &= mask_bits(bits);
    switch (o->kind) {
        case MM_V_SLOT:
            mem_write(vm, vm->sp + o->value, 64, value);
            return !vm->oom;
        case MM_V_SP:
            vm->sp = value;
            return 1;
        case MM_V_MEM: {
            uint64_t base;
            MMOperand b = {0};
            b.kind = o->base_kind;
            b.value = o->base_value;
            if (!read_value(vm, &b, &base)) return 0;
            mem_write(vm, base + (uint64_t)o->offset, o->width, value);
            return !vm->oom;
        }
        default:
            return 0;
    }
}

static int target_code(MMVM *vm, const MMOperand *o, uint64_t *out) {
    switch (o->kind) {
        case MM_T_CODE:
            *out = o->value;
            return 1;
        case MM_T_SLOT:
            *out = mem_read(vm, vm->sp + o->value, 64);
            return 1;
        case MM_T_MEM: {
            uint64_t base;
            MMOperand b = {0};
            b.kind = o->base_kind;
            b.value = o->base_value;
            if (!read_value(vm, &b, &base)) return 0;
            *out = mem_read(vm, base + (uint64_t)o->offset, 64);
            return 1;
        }
        default:
            return 0;
    }
}

MMVM *mm_vm_create(const MMInst *insts, size_t inst_count,
                   const MMBlock *blocks, size_t block_count,
                   const uint64_t *host_codes, size_t host_count,
                   uint64_t halt_code) {
    MMVM *vm = (MMVM *)calloc(1, sizeof(*vm));
    if (!vm) return NULL;
    vm->insts = insts;
    vm->inst_count = inst_count;
    vm->blocks = blocks;
    vm->block_count = block_count;
    vm->host_codes = host_codes;
    vm->host_count = host_count;
    vm->halt_code = halt_code;
    return vm;
}

int mm_vm_replace_program(MMVM *vm,
                          const MMInst *insts, size_t inst_count,
                          const MMBlock *blocks, size_t block_count,
                          const uint64_t *host_codes, size_t host_count,
                          uint64_t halt_code) {
    vm->insts = insts;
    vm->inst_count = inst_count;
    vm->blocks = blocks;
    vm->block_count = block_count;
    vm->host_codes = host_codes;
    vm->host_count = host_count;
    vm->halt_code = halt_code;
    return 1;
}

void mm_vm_destroy(MMVM *vm) {
    if (!vm) return;
    for (size_t i = 0; i < MM_BUCKETS; ++i) {
        MMPage *p = vm->pages[i];
        while (p) {
            MMPage *n = p->next;
            free(p->data);
            free(p);
            p = n;
        }
    }
    free(vm->watch_codes);
    free(vm);
}

int mm_vm_load_bytes(MMVM *vm,
                     const uint64_t *addrs,
                     const uint8_t *vals,
                     size_t n) {
    for (size_t i = 0; i < n; ++i) {
        mem_write8(vm, addrs[i], vals[i]);
        if (vm->oom) return 0;
    }
    return 1;
}

uint64_t mm_vm_mem_read(MMVM *vm, uint64_t addr, unsigned bits) {
    return mem_read(vm, addr, bits);
}

void mm_vm_mem_write(MMVM *vm, uint64_t addr,
                     unsigned bits, uint64_t value) {
    mem_write(vm, addr, bits, value);
}

int mm_vm_set_watches(MMVM *vm, const uint64_t *codes, size_t n) {
    uint64_t *copy = NULL;
    if (n) {
        copy = (uint64_t *)malloc(n * sizeof(uint64_t));
        if (!copy) return 0;
        memcpy(copy, codes, n * sizeof(uint64_t));
    }
    free(vm->watch_codes);
    vm->watch_codes = copy;
    vm->watch_count = n;
    return 1;
}

void mm_vm_set_state(MMVM *vm, uint64_t block_code, uint32_t ip,
                     uint64_t sp, uint64_t steps) {
    vm->block_code = block_code;
    vm->ip = ip;
    vm->sp = sp;
    vm->steps = steps;
}

MMRunResult mm_vm_run(MMVM *vm, uint64_t max_steps) {
    MMRunResult r = {0};
    r.status = MM_STATUS_LIMIT;

    while (vm->steps < max_steps) {
        size_t bi;
        if (!find_block(vm, vm->block_code, &bi)) {
            r.status = MM_STATUS_ERROR;
            r.error = MM_ERR_BAD_BLOCK;
            break;
        }
        const MMBlock *block = &vm->blocks[bi];
        if (vm->ip >= block->count) {
            r.status = MM_STATUS_ERROR;
            r.error = MM_ERR_FALLOFF;
            break;
        }

        const MMInst *in = &vm->insts[block->first + vm->ip];
        vm->steps++;

        if (in->opcode == MM_OP_MOV) {
            uint64_t raw;
            if (!read_value(vm, &in->src, &raw)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_VALUE;
                break;
            }
            unsigned dst_bits = in->width;
            uint64_t value = raw & mask_bits(dst_bits);
            if (in->extend != MM_EXT_NONE) {
                unsigned src_bits = in->src_bits ? in->src_bits : in->src.width;
                raw &= mask_bits(src_bits);
                if (in->extend == MM_EXT_SEXT)
                    value = (uint64_t)sext64(raw, src_bits) & mask_bits(dst_bits);
                else
                    value = raw & mask_bits(dst_bits);
            }
            if (!write_operand(vm, &in->dst, value, dst_bits)) {
                r.status = MM_STATUS_ERROR;
                r.error = vm->oom ? MM_ERR_OOM : MM_ERR_BAD_VALUE;
                break;
            }
            vm->ip++;
            continue;
        }

        if (in->opcode == MM_OP_SUB) {
            uint64_t a, b;
            if (!read_value(vm, &in->a, &a) ||
                !read_value(vm, &in->b, &b)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_VALUE;
                break;
            }
            uint64_t value = (a - b) & mask_bits(in->width);
            if (!write_operand(vm, &in->dst, value, in->width)) {
                r.status = MM_STATUS_ERROR;
                r.error = vm->oom ? MM_ERR_OOM : MM_ERR_BAD_VALUE;
                break;
            }
            vm->ip++;
            continue;
        }

        if (in->opcode == MM_OP_BR) {
            uint64_t a, b, target;
            if (!read_value(vm, &in->a, &a) ||
                !read_value(vm, &in->b, &b)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_VALUE;
                break;
            }

            unsigned bits = in->width;
            a &= mask_bits(bits);
            b &= mask_bits(bits);
            int take;
            if (in->cond == MM_COND_EQ)
                take = (a == b);
            else if (in->cond == MM_COND_ULT)
                take = (a < b);
            else if (in->cond == MM_COND_SLT)
                take = (sext64(a, bits) < sext64(b, bits));
            else {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_VALUE;
                break;
            }

            if (!target_code(vm, take ? &in->t : &in->f, &target)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_TARGET;
                break;
            }

            if (target == vm->halt_code) {
                r.status = MM_STATUS_HALT;
                r.target_code = target;
                break;
            }
            if (contains_code(vm->host_codes, vm->host_count, target)) {
                r.status = MM_STATUS_HOST;
                r.target_code = target;
                break;
            }
            if (contains_code(vm->watch_codes, vm->watch_count, target)) {
                vm->block_code = target;
                vm->ip = 0;
                r.status = MM_STATUS_WATCH;
                r.target_code = target;
                break;
            }

            size_t target_index;
            if (!find_block(vm, target, &target_index)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_TARGET;
                r.target_code = target;
                break;
            }
            vm->block_code = target;
            vm->ip = 0;
            continue;
        }

        r.status = MM_STATUS_ERROR;
        r.error = MM_ERR_BAD_OPCODE;
        break;
    }

    r.block_code = vm->block_code;
    r.ip = vm->ip;
    r.sp = vm->sp;
    r.steps = vm->steps;
    return r;
}
