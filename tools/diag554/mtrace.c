/*
 * mtrace.c - LD_PRELOAD allocation tracer that records the addresses glibc
 * malloc returns without perturbing the traced program's heap.
 *
 * Build:  gcc -O2 -fPIC -shared -o libmtrace.so mtrace.c -lgcc_s
 * Use:    MTRACE_OUT=/path/prefix [MTRACE_BT=1] LD_PRELOAD=./libmtrace.so prog
 * Decode: python3 mtrace_decode.py /path/prefix [--symbolize] [--out f.jsonl]
 *
 * Design constraints (the goal is an unperturbed heap):
 *   - The shim never calls malloc/calloc/realloc/free or any libc function
 *     that might (no stdio, no dlsym, no execinfo backtrace, no atexit).
 *     It uses only mmap/munmap/open/read/write/close/getpid/getpagesize,
 *     dl_iterate_phdr (once, in the constructor), _Unwind_Backtrace/
 *     _Unwind_GetIP, and the glibc entry points __libc_malloc, __libc_free,
 *     __libc_calloc, __libc_realloc, __libc_memalign, __libc_valloc,
 *     __libc_pvalloc.
 *   - Records go into one 2 GiB MAP_NORESERVE anonymous mapping, created on
 *     the first intercepted call.  The mapping is requested at a hint address
 *     far from both the brk heap and the top-down mmap area, so it does not
 *     consume the address range that later mmap()s (large malloc chunks,
 *     thread stacks, dlopen) would otherwise receive.
 *   - Backtraces (MTRACE_BT=1) use _Unwind_Backtrace from libgcc_s and are
 *     captured BEFORE the real allocator is entered, so that any allocation
 *     made by the unwinder itself is ordered before the traced one and, via a
 *     thread-local guard, is recorded without a backtrace instead of recursing.
 *
 * Environment variables (read in a constructor by scanning environ):
 *   MTRACE_OUT=<prefix>   Output prefix.  Unset: pure pass-through.
 *   MTRACE_BT=1           Capture up to 16 return addresses for allocation
 *                         calls (not free).  Calls made before the
 *                         constructor ran carry no backtrace.
 *   MTRACE_DISABLE=1      Pure pass-through even if MTRACE_OUT is set.
 *   MTRACE_PAD_SIZE=<n>  Leak one __libc_malloc(n) block in the constructor
 *                         (n > 0).  Shifts later top-carved heap chunks by
 *                         the chunk size while keeping relative layout.
 *                         Works with or without MTRACE_OUT.
 *   MTRACE_PID=1          Write <prefix>.<pid>.bin / .maps.  Use this when the
 *                         preload is inherited by more than one process (for
 *                         example `timeout`, `strace -f`, or a harness that
 *                         spawns helpers): otherwise every process that exits
 *                         normally writes to the same files and the last one
 *                         to exit wins.
 *
 * Output (written by a destructor, using open/write/close only):
 *   <prefix>.bin   16-byte header, then `count` fixed-size records.
 *   <prefix>.maps  byte copy of /proc/self/maps taken at exit.
 *
 * .bin layout (all integers little-endian):
 *   header, 16 bytes:
 *     char     magic[8]   = "MTRACE01"
 *     uint64   count      number of records that follow
 *   record, 176 bytes (natural alignment, no padding):
 *     off   0  uint64 seq       global order (index of the record)
 *     off   8  uint32 op        1 malloc, 2 free, 3 calloc, 4 realloc,
 *                               5 memalign, 6 posix_memalign,
 *                               7 aligned_alloc, 8 valloc, 9 pvalloc
 *     off  12  uint32 nframes   number of valid entries in frames[]
 *     off  16  uint64 size      requested bytes (calloc: nmemb*size, saturated
 *                               to UINT64_MAX on overflow; free: 0)
 *     off  24  uint64 align     requested alignment (memalign family),
 *                               page size for valloc/pvalloc, else 0
 *     off  32  uint64 ptr       returned pointer (free: the freed pointer;
 *                               0 on failure)
 *     off  40  uint64 old_ptr   realloc input pointer, else 0
 *     off  48  uint64 frames[16]  return addresses, innermost first
 *
 * Ordering: allocation records take their sequence number after the real
 * call returns (the pointer is then known); free records take it before the
 * real call, so with several threads a free is always ordered before any
 * later allocation that reuses its memory.  free(NULL) is not recorded.
 *
 * Not recorded: allocations made after this library's destructor ran, and
 * anything in processes that end via _exit/abort/quick_exit/signal (no
 * destructors).  Calls past the 2 GiB buffer capacity are counted but dropped;
 * a notice is written to stderr at exit.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

extern void* __libc_malloc(size_t);
extern void __libc_free(void*);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
extern void* __libc_memalign(size_t, size_t);
extern void* __libc_valloc(size_t);
extern void* __libc_pvalloc(size_t);
extern char** environ;

#define MAX_FRAMES 16
#ifndef BUF_BYTES
#define BUF_BYTES (2ULL << 30) /* overridable for tests */
#endif
/* Hint only; the kernel falls back to its normal placement if unavailable. */
#define BUF_HINT ((void*)0x100000000000ULL)

enum {
    OP_MALLOC = 1,
    OP_FREE = 2,
    OP_CALLOC = 3,
    OP_REALLOC = 4,
    OP_MEMALIGN = 5,
    OP_POSIX_MEMALIGN = 6,
    OP_ALIGNED_ALLOC = 7,
    OP_VALLOC = 8,
    OP_PVALLOC = 9,
};

struct rec {
    uint64_t seq;
    uint32_t op;
    uint32_t nframes;
    uint64_t size;
    uint64_t align;
    uint64_t ptr;
    uint64_t old_ptr;
    uint64_t frames[MAX_FRAMES];
};
_Static_assert(sizeof(struct rec) == 176, "record layout");

struct frames {
    uint64_t f[MAX_FRAMES];
    uint32_t n;
};

enum { M_UNINIT = 0, M_INITING = 1, M_ON = 2, M_OFF = 3 };

static int g_mode = M_UNINIT;
static uint64_t g_idx;     /* next record index; may exceed g_cap */
static uint64_t g_cap;     /* record capacity */
static struct rec* g_recs; /* mapping base */

/* Configuration, filled by scan_config().  The prefix leaves room in g_path
 * for ".<pid>.maps". */
static char g_out[4000];
static int g_have_out;
static int g_cfg_bt;
static int g_cfg_disable;
static int g_cfg_pid;
static uint64_t g_cfg_pad;
static int g_cfg_known; /* environ was available when scanned */

static int g_bt_on;                    /* set by the constructor only */
static uintptr_t g_text_lo, g_text_hi; /* this library's own mapping */
static pid_t g_pid;

static __thread int t_in_bt __attribute__((tls_model("initial-exec")));

static char g_iobuf[16384];
static char g_path[4096];

/* ------------------------------------------------------------------ */
/* Configuration                                                      */
/* ------------------------------------------------------------------ */

static int env_match(const char* e, const char* name, const char** val) {
    while (*name) {
        if (*e != *name)
            return 0;
        e++;
        name++;
    }
    if (*e != '=')
        return 0;
    *val = e + 1;
    return 1;
}

static void scan_config(void) {
    char** env = environ;
    if (!env)
        return;
    g_cfg_known = 1;
    g_have_out = 0;
    g_cfg_bt = 0;
    g_cfg_disable = 0;
    g_cfg_pid = 0;
    g_cfg_pad = 0;
    for (; *env; env++) {
        const char* v;
        if (env_match(*env, "MTRACE_OUT", &v)) {
            size_t n = 0;
            while (v[n] && n < sizeof(g_out) - 1) {
                g_out[n] = v[n];
                n++;
            }
            g_out[n] = 0;
            /* A truncated prefix would write to the wrong place: disable. */
            g_have_out = (v[n] == 0 && n > 0);
        } else if (env_match(*env, "MTRACE_BT", &v)) {
            g_cfg_bt = (v[0] == '1' && v[1] == 0);
        } else if (env_match(*env, "MTRACE_DISABLE", &v)) {
            g_cfg_disable = (v[0] == '1' && v[1] == 0);
        } else if (env_match(*env, "MTRACE_PAD_SIZE", &v)) {
            uint64_t n = 0;
            for (const char* c = v; *c >= '0' && *c <= '9'; c++)
                n = n * 10 + (uint64_t)(*c - '0');
            g_cfg_pad = n;
        } else if (env_match(*env, "MTRACE_PID", &v)) {
            g_cfg_pid = (v[0] == '1' && v[1] == 0);
        }
    }
}

static int phdr_cb(struct dl_phdr_info* info, size_t sz, void* arg) {
    uintptr_t anchor = (uintptr_t)&phdr_cb;
    uintptr_t lo = UINTPTR_MAX, hi = 0;
    int found = 0;
    (void)sz;
    (void)arg;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr)* p = &info->dlpi_phdr[i];
        if (p->p_type != PT_LOAD)
            continue;
        uintptr_t s = info->dlpi_addr + p->p_vaddr;
        uintptr_t e = s + p->p_memsz;
        if (s < lo)
            lo = s;
        if (e > hi)
            hi = e;
        if (anchor >= s && anchor < e)
            found = 1;
    }
    if (found) {
        g_text_lo = lo;
        g_text_hi = hi;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Lazy initialisation                                                */
/* ------------------------------------------------------------------ */

static int init_slow(void) {
    int expected = M_UNINIT;
    if (__atomic_compare_exchange_n(&g_mode, &expected, M_INITING, 0, __ATOMIC_ACQUIRE,
                                    __ATOMIC_RELAXED)) {
        int saved_errno = errno;
        int next = M_ON;
        scan_config(); /* no-op while environ is still NULL */
        if (g_cfg_known && (!g_have_out || g_cfg_disable)) {
            next = M_OFF;
        } else {
            void* p = mmap(BUF_HINT, BUF_BYTES, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) {
                static const char msg[] =
                    "mtrace: cannot map the record buffer; tracing disabled\n";
                ssize_t w = write(2, msg, sizeof(msg) - 1);
                (void)w;
                next = M_OFF;
            } else {
                g_recs = (struct rec*)p;
                g_cap = BUF_BYTES / sizeof(struct rec);
            }
        }
        errno = saved_errno;
        __atomic_store_n(&g_mode, next, __ATOMIC_RELEASE);
        return next;
    }
    int m;
    while ((m = __atomic_load_n(&g_mode, __ATOMIC_ACQUIRE)) == M_INITING)
        __builtin_ia32_pause();
    return m;
}

static inline int is_on(void) {
    int m = __atomic_load_n(&g_mode, __ATOMIC_ACQUIRE);
    if (__builtin_expect(m == M_ON, 1))
        return 1;
    if (m == M_OFF)
        return 0;
    return init_slow() == M_ON;
}

/* ------------------------------------------------------------------ */
/* Backtrace capture                                                  */
/* ------------------------------------------------------------------ */

#ifdef MTRACE_SELFTEST_NESTED
void* malloc(size_t);
void free(void*);
#endif

static _Unwind_Reason_Code bt_cb(struct _Unwind_Context* ctx, void* arg) {
    struct frames* fr = (struct frames*)arg;
    uintptr_t ip = (uintptr_t)_Unwind_GetIP(ctx);
    if (!ip)
        return _URC_END_OF_STACK;
#ifdef MTRACE_SELFTEST_NESTED
    /* Test build only: allocate from inside the unwinder to exercise the
     * reentrancy guard. */
    if (fr->n == 0) {
        void* volatile nested = malloc(77);
        free(nested);
    }
#endif
    /* Drop the shim's own leading frames (wrapper, helpers). */
    if (fr->n == 0 && ip >= g_text_lo && ip < g_text_hi)
        return _URC_NO_REASON;
    fr->f[fr->n++] = ip;
    return fr->n >= MAX_FRAMES ? _URC_END_OF_STACK : _URC_NO_REASON;
}

static __attribute__((noinline)) void capture(struct frames* fr) {
    t_in_bt = 1;
    _Unwind_Backtrace(bt_cb, fr);
    t_in_bt = 0;
}

static inline void bt_begin(struct frames* fr) {
    fr->n = 0;
    if (g_bt_on && !t_in_bt)
        capture(fr);
}

/* ------------------------------------------------------------------ */
/* Recording                                                          */
/* ------------------------------------------------------------------ */

static void record(uint32_t op, uint64_t size, uint64_t align, const void* ptr, const void* old_ptr,
                   const struct frames* fr) {
    uint64_t i = __atomic_fetch_add(&g_idx, 1, __ATOMIC_RELAXED);
    if (i >= g_cap)
        return;
    struct rec* r = &g_recs[i];
    r->seq = i;
    r->op = op;
    r->size = size;
    r->align = align;
    r->ptr = (uint64_t)(uintptr_t)ptr;
    r->old_ptr = (uint64_t)(uintptr_t)old_ptr;
    uint32_t n = fr ? fr->n : 0;
    r->nframes = n;
    for (uint32_t k = 0; k < n; k++)
        r->frames[k] = fr->f[k];
}

/* ------------------------------------------------------------------ */
/* Interposed entry points                                            */
/* ------------------------------------------------------------------ */

void* malloc(size_t size) {
    if (__builtin_expect(!is_on(), 0))
        return __libc_malloc(size);
    struct frames fr;
    bt_begin(&fr);
    void* p = __libc_malloc(size);
    record(OP_MALLOC, size, 0, p, NULL, &fr);
    return p;
}

void free(void* ptr) {
    if (__builtin_expect(!ptr, 0) || __builtin_expect(!is_on(), 0)) {
        __libc_free(ptr);
        return;
    }
    record(OP_FREE, 0, 0, ptr, NULL, NULL);
    __libc_free(ptr);
}

void* calloc(size_t nmemb, size_t size) {
    if (__builtin_expect(!is_on(), 0))
        return __libc_calloc(nmemb, size);
    struct frames fr;
    bt_begin(&fr);
    void* p = __libc_calloc(nmemb, size);
    size_t total;
    if (__builtin_mul_overflow(nmemb, size, &total))
        total = SIZE_MAX;
    record(OP_CALLOC, total, 0, p, NULL, &fr);
    return p;
}

void* realloc(void* ptr, size_t size) {
    if (__builtin_expect(!is_on(), 0))
        return __libc_realloc(ptr, size);
    struct frames fr;
    bt_begin(&fr);
    void* p = __libc_realloc(ptr, size);
    record(OP_REALLOC, size, 0, p, ptr, &fr);
    return p;
}

void* memalign(size_t alignment, size_t size) {
    if (__builtin_expect(!is_on(), 0))
        return __libc_memalign(alignment, size);
    struct frames fr;
    bt_begin(&fr);
    void* p = __libc_memalign(alignment, size);
    record(OP_MEMALIGN, size, alignment, p, NULL, &fr);
    return p;
}

int posix_memalign(void** memptr, size_t alignment, size_t size) {
    /* Same validation as glibc: a power-of-two multiple of sizeof(void *). */
    size_t units = alignment / sizeof(void*);
    int valid = alignment != 0 && alignment % sizeof(void*) == 0 && (units & (units - 1)) == 0;
    if (__builtin_expect(!is_on(), 0)) {
        if (!valid)
            return EINVAL;
        void* m = __libc_memalign(alignment, size);
        if (!m)
            return ENOMEM;
        *memptr = m;
        return 0;
    }
    struct frames fr;
    bt_begin(&fr);
    if (!valid) {
        record(OP_POSIX_MEMALIGN, size, alignment, NULL, NULL, &fr);
        return EINVAL;
    }
    void* m = __libc_memalign(alignment, size);
    record(OP_POSIX_MEMALIGN, size, alignment, m, NULL, &fr);
    if (!m)
        return ENOMEM;
    *memptr = m;
    return 0;
}

void* aligned_alloc(size_t alignment, size_t size) {
    int valid = alignment != 0 && (alignment & (alignment - 1)) == 0;
    if (__builtin_expect(!is_on(), 0)) {
        if (!valid) {
            errno = EINVAL;
            return NULL;
        }
        return __libc_memalign(alignment, size);
    }
    struct frames fr;
    bt_begin(&fr);
    if (!valid) {
        record(OP_ALIGNED_ALLOC, size, alignment, NULL, NULL, &fr);
        errno = EINVAL;
        return NULL;
    }
    void* p = __libc_memalign(alignment, size);
    record(OP_ALIGNED_ALLOC, size, alignment, p, NULL, &fr);
    return p;
}

void* valloc(size_t size) {
    if (__builtin_expect(!is_on(), 0))
        return __libc_valloc(size);
    struct frames fr;
    bt_begin(&fr);
    void* p = __libc_valloc(size);
    record(OP_VALLOC, size, (uint64_t)getpagesize(), p, NULL, &fr);
    return p;
}

void* pvalloc(size_t size) {
    if (__builtin_expect(!is_on(), 0))
        return __libc_pvalloc(size);
    struct frames fr;
    bt_begin(&fr);
    void* p = __libc_pvalloc(size);
    record(OP_PVALLOC, size, (uint64_t)getpagesize(), p, NULL, &fr);
    return p;
}

/* ------------------------------------------------------------------ */
/* Constructor / destructor                                           */
/* ------------------------------------------------------------------ */

__attribute__((constructor)) static void mtrace_ctor(void) {
    scan_config();
    g_pid = getpid();
    if (g_cfg_pad > 0) {
        void* volatile pad = __libc_malloc((size_t)g_cfg_pad);
        (void)pad;
    }
    int want = g_have_out && !g_cfg_disable;
    if (!want) {
        int m = __atomic_exchange_n(&g_mode, M_OFF, __ATOMIC_ACQ_REL);
        if (m == M_ON && g_recs) {
            void* p = g_recs;
            g_recs = NULL;
            g_cap = 0;
            munmap(p, BUF_BYTES);
        }
        return;
    }
    if (g_cfg_bt) {
        dl_iterate_phdr(phdr_cb, NULL);
        g_bt_on = 1;
    }
}

static int write_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void write_str(int fd, const char* s) {
    size_t n = 0;
    while (s[n])
        n++;
    write_all(fd, s, n);
}

static void make_path(const char* suffix) {
    size_t n = 0;
    for (const char* p = g_out; *p; p++)
        g_path[n++] = *p;
    if (g_cfg_pid) {
        char digits[24];
        int nd = 0;
        unsigned long v = (unsigned long)g_pid;
        do {
            digits[nd++] = (char)('0' + v % 10);
            v /= 10;
        } while (v);
        g_path[n++] = '.';
        while (nd)
            g_path[n++] = digits[--nd];
    }
    for (const char* p = suffix; *p; p++)
        g_path[n++] = *p;
    g_path[n] = 0;
}

__attribute__((destructor)) static void mtrace_dtor(void) {
    if (__atomic_load_n(&g_mode, __ATOMIC_ACQUIRE) != M_ON || !g_recs)
        return;
    /* A forked child that calls exit() must not overwrite the parent's trace. */
    if (getpid() != g_pid)
        return;
    __atomic_store_n(&g_mode, M_OFF, __ATOMIC_RELEASE);

    uint64_t total = __atomic_load_n(&g_idx, __ATOMIC_ACQUIRE);
    uint64_t count = total < g_cap ? total : g_cap;

    make_path(".bin");
    int fd = open(g_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd >= 0) {
        char hdr[16] = {'M', 'T', 'R', 'A', 'C', 'E', '0', '1'};
        for (int i = 0; i < 8; i++)
            hdr[8 + i] = (char)((count >> (8 * i)) & 0xff);
        write_all(fd, hdr, sizeof(hdr));
        write_all(fd, g_recs, (size_t)(count * sizeof(struct rec)));
        close(fd);
    }

    make_path(".maps");
    int in = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (in >= 0) {
        int out = open(g_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (out >= 0) {
            for (;;) {
                ssize_t r = read(in, g_iobuf, sizeof(g_iobuf));
                if (r < 0 && errno == EINTR)
                    continue;
                if (r <= 0)
                    break;
                if (write_all(out, g_iobuf, (size_t)r) < 0)
                    break;
            }
            close(out);
        }
        close(in);
    }

    if (total > count)
        write_str(2, "mtrace: record buffer full, later calls were dropped\n");
}
