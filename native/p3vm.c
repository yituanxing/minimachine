#include <stdint.h>
#include <stdio.h>
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
} MMSegment;

typedef struct {
    MMSegment *segments;
    size_t segment_count;
    size_t segment_capacity;
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
    uint64_t cached_block_code;
    size_t cached_segment_index;
    size_t cached_block_index;
    int cached_block_valid;
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


static int mem_read_bytes(MMVM *vm, uint64_t addr, uint8_t *out, uint64_t n) {
    while (n) {
        uint64_t off = addr & (MM_PAGE_SIZE - 1);
        uint64_t chunk = MM_PAGE_SIZE - off;
        if (chunk > n) chunk = n;
        MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 0);
        if (p) memcpy(out, p->data + off, (size_t)chunk);
        else memset(out, 0, (size_t)chunk);
        addr += chunk;
        out += chunk;
        n -= chunk;
    }
    return 1;
}

static int mem_write_bytes(MMVM *vm, uint64_t addr, const uint8_t *in, uint64_t n) {
    while (n) {
        uint64_t off = addr & (MM_PAGE_SIZE - 1);
        uint64_t chunk = MM_PAGE_SIZE - off;
        if (chunk > n) chunk = n;
        MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 1);
        if (!p) return 0;
        memcpy(p->data + off, in, (size_t)chunk);
        addr += chunk;
        in += chunk;
        n -= chunk;
    }
    return !vm->oom;
}

static int mem_fill_bytes(MMVM *vm, uint64_t addr, uint8_t value, uint64_t n) {
    while (n) {
        uint64_t off = addr & (MM_PAGE_SIZE - 1);
        uint64_t chunk = MM_PAGE_SIZE - off;
        if (chunk > n) chunk = n;
        MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 1);
        if (!p) return 0;
        memset(p->data + off, value, (size_t)chunk);
        addr += chunk;
        n -= chunk;
    }
    return !vm->oom;
}

static int mem_copy_bytes(MMVM *vm, uint64_t dst, uint64_t src,
                          uint64_t n, int move_semantics) {
    if (!n || dst == src) return 1;
    uint8_t *tmp = (uint8_t *)malloc(MM_PAGE_SIZE);
    if (!tmp) { vm->oom = 1; return 0; }

    int backwards = (
        move_semantics &&
        dst > src &&
        (dst - src) < n
    );

    if (backwards) {
        uint64_t remaining = n;
        while (remaining) {
            uint64_t chunk = remaining > MM_PAGE_SIZE ? MM_PAGE_SIZE : remaining;
            uint64_t start = remaining - chunk;
            mem_read_bytes(vm, src + start, tmp, chunk);
            if (!mem_write_bytes(vm, dst + start, tmp, chunk)) {
                free(tmp);
                return 0;
            }
            remaining = start;
        }
    } else {
        uint64_t done = 0;
        while (done < n) {
            uint64_t chunk = n - done;
            if (chunk > MM_PAGE_SIZE) chunk = MM_PAGE_SIZE;
            mem_read_bytes(vm, src + done, tmp, chunk);
            if (!mem_write_bytes(vm, dst + done, tmp, chunk)) {
                free(tmp);
                return 0;
            }
            done += chunk;
        }
    }

    free(tmp);
    return !vm->oom;
}

static int mem_compare_bytes(MMVM *vm, uint64_t a, uint64_t b, uint64_t n) {
    uint8_t abuf[4096];
    uint8_t bbuf[4096];
    uint64_t done = 0;
    while (done < n) {
        uint64_t chunk = n - done;
        if (chunk > sizeof(abuf)) chunk = sizeof(abuf);
        mem_read_bytes(vm, a + done, abuf, chunk);
        mem_read_bytes(vm, b + done, bbuf, chunk);
        if (memcmp(abuf, bbuf, (size_t)chunk) != 0) {
            for (uint64_t i = 0; i < chunk; ++i) {
                if (abuf[i] != bbuf[i])
                    return (int)abuf[i] - (int)bbuf[i];
            }
        }
        done += chunk;
    }
    return 0;
}

static uint64_t mem_strlen_bytes(MMVM *vm, uint64_t addr) {
    uint64_t length = 0;
    for (;;) {
        uint64_t off = addr & (MM_PAGE_SIZE - 1);
        uint64_t chunk = MM_PAGE_SIZE - off;
        MMPage *p = get_page(vm, addr >> MM_PAGE_SHIFT, 0);
        if (!p) return length;
        void *hit = memchr(p->data + off, 0, (size_t)chunk);
        if (hit) {
            return length + (uint64_t)((uint8_t *)hit - (p->data + off));
        }
        addr += chunk;
        length += chunk;
    }
}

static int find_block_in_segment(const MMSegment *segment,
                                 uint64_t code, size_t *idx) {
    size_t lo = 0, hi = segment->block_count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint64_t c = segment->blocks[mid].code;
        if (c == code) {
            *idx = mid;
            return 1;
        }
        if (c < code) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}

static int find_block(MMVM *vm, uint64_t code,
                      size_t *segment_index, size_t *idx) {
    if (vm->segment_count == 1) {
        *segment_index = 0;
        return find_block_in_segment(&vm->segments[0], code, idx);
    }

    for (size_t si = 0; si < vm->segment_count; ++si) {
        const MMSegment *segment = &vm->segments[si];
        if (!segment->block_count)
            continue;
        uint64_t first_code = segment->blocks[0].code;
        uint64_t last_code = segment->blocks[segment->block_count - 1].code;
        if (code < first_code || code > last_code)
            continue;
        if (find_block_in_segment(segment, code, idx)) {
            *segment_index = si;
            return 1;
        }
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
    vm->segments = (MMSegment *)calloc(1, sizeof(*vm->segments));
    if (!vm->segments) {
        free(vm);
        return NULL;
    }
    vm->segment_capacity = 1;
    vm->segment_count = 1;
    vm->segments[0].insts = insts;
    vm->segments[0].inst_count = inst_count;
    vm->segments[0].blocks = blocks;
    vm->segments[0].block_count = block_count;
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
    if (!vm || !vm->segments)
        return 0;
    vm->segments[0].insts = insts;
    vm->segments[0].inst_count = inst_count;
    vm->segments[0].blocks = blocks;
    vm->segments[0].block_count = block_count;
    vm->segment_count = 1;
    vm->host_codes = host_codes;
    vm->host_count = host_count;
    vm->halt_code = halt_code;
    vm->cached_block_valid = 0;
    return 1;
}

int mm_vm_set_host_codes(MMVM *vm,
                         const uint64_t *host_codes,
                         size_t host_count) {
    if (!vm)
        return 0;
    vm->host_codes = host_codes;
    vm->host_count = host_count;
    return 1;
}

int mm_vm_add_segment(MMVM *vm,
                      const MMInst *insts, size_t inst_count,
                      const MMBlock *blocks, size_t block_count) {
    if (!vm || !blocks || !block_count)
        return 0;
    if (vm->segment_count == vm->segment_capacity) {
        size_t next_capacity = vm->segment_capacity ? vm->segment_capacity * 2 : 2;
        MMSegment *next = (MMSegment *)realloc(
            vm->segments, next_capacity * sizeof(*next)
        );
        if (!next)
            return 0;
        vm->segments = next;
        vm->segment_capacity = next_capacity;
    }
    MMSegment *segment = &vm->segments[vm->segment_count++];
    segment->insts = insts;
    segment->inst_count = inst_count;
    segment->blocks = blocks;
    segment->block_count = block_count;
    vm->cached_block_valid = 0;
    return 1;
}

static void clear_pages(MMVM *vm) {
    if (!vm) return;
    for (size_t i = 0; i < MM_BUCKETS; ++i) {
        MMPage *p = vm->pages[i];
        while (p) {
            MMPage *n = p->next;
            free(p->data);
            free(p);
            p = n;
        }
        vm->pages[i] = NULL;
    }
    vm->cached_page = NULL;
    vm->cached_page_no = 0;
    vm->cached_page_valid = 0;
    vm->oom = 0;
}

void mm_vm_destroy(MMVM *vm) {
    if (!vm) return;
    clear_pages(vm);
    free(vm->watch_codes);
    free(vm->segments);
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

size_t mm_vm_mem_page_count(MMVM *vm) {
    if (!vm) return 0;
    size_t count = 0;
    for (size_t i = 0; i < MM_BUCKETS; ++i) {
        for (MMPage *p = vm->pages[i]; p; p = p->next)
            count++;
    }
    return count;
}

size_t mm_vm_mem_export_pages(MMVM *vm,
                              uint64_t *page_nos,
                              uint8_t *data,
                              size_t capacity_pages) {
    if (!vm) return 0;
    size_t out = 0;
    for (size_t i = 0; i < MM_BUCKETS; ++i) {
        for (MMPage *p = vm->pages[i]; p; p = p->next) {
            if (out >= capacity_pages)
                return out;
            page_nos[out] = p->no;
            memcpy(data + out * MM_PAGE_SIZE, p->data, MM_PAGE_SIZE);
            out++;
        }
    }
    return out;
}

int mm_vm_mem_restore_pages(MMVM *vm,
                            const uint64_t *page_nos,
                            const uint8_t *data,
                            size_t page_count,
                            size_t page_size) {
    if (!vm || page_size != MM_PAGE_SIZE)
        return 0;
    clear_pages(vm);
    for (size_t i = 0; i < page_count; ++i) {
        MMPage *p = get_page(vm, page_nos[i], 1);
        if (!p || vm->oom) {
            clear_pages(vm);
            return 0;
        }
        memcpy(p->data, data + i * MM_PAGE_SIZE, MM_PAGE_SIZE);
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

int mm_vm_mem_write_blob(MMVM *vm, uint64_t dst,
                         const uint8_t *data, uint64_t n) {
    if (!vm || (!data && n))
        return 0;
    return mem_write_bytes(vm, dst, data, n);
}

int mm_vm_mem_fill(MMVM *vm, uint64_t dst, uint8_t value, uint64_t n) {
    return mem_fill_bytes(vm, dst, value, n);
}

int mm_vm_mem_copy(MMVM *vm, uint64_t dst, uint64_t src, uint64_t n) {
    return mem_copy_bytes(vm, dst, src, n, 0);
}

int mm_vm_mem_move(MMVM *vm, uint64_t dst, uint64_t src, uint64_t n) {
    return mem_copy_bytes(vm, dst, src, n, 1);
}

int mm_vm_mem_compare(MMVM *vm, uint64_t a, uint64_t b, uint64_t n) {
    return mem_compare_bytes(vm, a, b, n);
}

uint64_t mm_vm_mem_strlen(MMVM *vm, uint64_t addr) {
    return mem_strlen_bytes(vm, addr);
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
    vm->cached_block_valid = 0;
}

MMRunResult mm_vm_run(MMVM *vm, uint64_t max_steps) {
    MMRunResult r = {0};
    r.status = MM_STATUS_LIMIT;

    while (vm->steps < max_steps) {
        size_t si, bi;
        if (vm->cached_block_valid &&
            vm->cached_block_code == vm->block_code) {
            si = vm->cached_segment_index;
            bi = vm->cached_block_index;
        } else {
            if (!find_block(vm, vm->block_code, &si, &bi)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_BLOCK;
                break;
            }
            vm->cached_block_code = vm->block_code;
            vm->cached_segment_index = si;
            vm->cached_block_index = bi;
            vm->cached_block_valid = 1;
        }
        const MMSegment *segment = &vm->segments[si];
        const MMBlock *block = &segment->blocks[bi];
        if (vm->ip >= block->count) {
            r.status = MM_STATUS_ERROR;
            r.error = MM_ERR_FALLOFF;
            break;
        }

        const MMInst *in = &segment->insts[block->first + vm->ip];
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
            int trace_xxreadtoken_move =
                in->src.kind == MM_V_SLOT &&
                in->dst.kind == MM_V_SLOT &&
                in->src.value == 112 &&
                in->dst.value == 248 &&
                (raw == UINT64_C(0x18acbea) || (raw & UINT64_C(0xff)) == UINT64_C(0xea));
            if (trace_xxreadtoken_move) {
                fprintf(
                    stderr,
                    "BOOT_EXEC_NATIVE_MOV64_DIAG_BEFORE "
                    "sp=0x%llx width=%u raw=0x%llx value=0x%llx "
                    "src_off=%llu dst_off=%llu src_mem=0x%llx dst_mem=0x%llx\n",
                    (unsigned long long)vm->sp,
                    dst_bits,
                    (unsigned long long)raw,
                    (unsigned long long)value,
                    (unsigned long long)in->src.value,
                    (unsigned long long)in->dst.value,
                    (unsigned long long)mem_read(vm, vm->sp + in->src.value, 64),
                    (unsigned long long)mem_read(vm, vm->sp + in->dst.value, 64)
                );
                fflush(stderr);
            }
            if (!write_operand(vm, &in->dst, value, dst_bits)) {
                r.status = MM_STATUS_ERROR;
                r.error = vm->oom ? MM_ERR_OOM : MM_ERR_BAD_VALUE;
                break;
            }
            if (trace_xxreadtoken_move) {
                fprintf(
                    stderr,
                    "BOOT_EXEC_NATIVE_MOV64_DIAG_AFTER "
                    "sp=0x%llx width=%u dst_mem=0x%llx\n",
                    (unsigned long long)vm->sp,
                    dst_bits,
                    (unsigned long long)mem_read(vm, vm->sp + in->dst.value, 64)
                );
                fflush(stderr);
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

            size_t target_segment, target_index;
            if (!find_block(vm, target, &target_segment, &target_index)) {
                r.status = MM_STATUS_ERROR;
                r.error = MM_ERR_BAD_TARGET;
                r.target_code = target;
                break;
            }
            vm->block_code = target;
            vm->ip = 0;
            vm->cached_block_code = target;
            vm->cached_segment_index = target_segment;
            vm->cached_block_index = target_index;
            vm->cached_block_valid = 1;
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
