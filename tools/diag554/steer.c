/*
 * steer.c - LD_PRELOAD shim that relocates the user-visible address of one
 * selected malloc'd object at a time without changing glibc's heap sequence.
 *
 * For a matching allocation the shim still performs the real glibc call and
 * keeps the resulting chunk allocated (the "shadow"), but hands the program a
 * pointer into a pre-mapped page range at a fixed address. free() of that
 * pointer frees the shadow. glibc therefore sees exactly the malloc/free
 * sequence the program issued.
 *
 * Environment (read in a constructor):
 *   STEER_SIZE=<decimal>  request size to steer (malloc size, or calloc
 *                         nmemb*size). Unset or 0 disables the shim.
 *   STEER_ADDR=<hex>      address handed out for the steered object.
 *   STEER_SKIP=<decimal>  number of matching calls to pass through first.
 *   STEER_REPORT=1        print counters to fd 2 at exit.
 *   STEER_PAD=<decimal>   leak one allocation of this size at startup, which
 *                         shifts later heap chunks without moving libraries.
 *
 * The shim never allocates, uses no TLS, and depends only on libc.
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE 4096UL

extern void* __libc_malloc(size_t);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
extern void __libc_free(void*);
extern char** environ;

/* All state is zero-initialised, which means pass-through until the
 * constructor has run (allocations made by ld.so before that are unaffected). */
static size_t g_size;         /* steered request size; 0 = disabled */
static unsigned char* g_addr; /* address returned for the steered object */
static void* g_shadow;        /* real glibc chunk backing the live slot */
static unsigned long g_skip;  /* matching calls to pass through first */
static unsigned long g_match; /* matching calls seen */
static unsigned long g_busy_hits;
static unsigned long g_steered;
static int g_busy; /* slot in use */
static int g_report;
static unsigned long g_frame; /* stack frame of the first steered malloc */

static void* steer(void* shadow) {
    if (__atomic_fetch_add(&g_match, 1, __ATOMIC_RELAXED) < g_skip)
        return shadow;
    if (__atomic_exchange_n(&g_busy, 1, __ATOMIC_ACQUIRE)) {
        __atomic_fetch_add(&g_busy_hits, 1, __ATOMIC_RELAXED);
        return shadow;
    }
    g_shadow = shadow;
    __atomic_fetch_add(&g_steered, 1, __ATOMIC_RELAXED);
    return g_addr;
}

void* malloc(size_t size) {
    void* p = __libc_malloc(size);
    if (p && g_size && size == g_size) {
        if (!g_frame)
            g_frame = (unsigned long)__builtin_frame_address(0);
        return steer(p);
    }
    return p;
}

void* calloc(size_t n, size_t size) {
    size_t total;
    void* p = __libc_calloc(n, size);
    if (p && g_size && !__builtin_mul_overflow(n, size, &total) && total == g_size) {
        void* q = steer(p);
        if (q != p)
            memset(q, 0, total);
        return q;
    }
    return p;
}

void free(void* p) {
    if (p == g_addr && __atomic_load_n(&g_busy, __ATOMIC_ACQUIRE)) {
        p = g_shadow;
        g_shadow = NULL;
        __atomic_store_n(&g_busy, 0, __ATOMIC_RELEASE);
    }
    __libc_free(p);
}

/*
 * realloc of the steered pointer: copy the live contents into the shadow and
 * realloc the shadow, so glibc sees the same realloc call as without the shim.
 * The result is an ordinary heap pointer and the slot is released. On failure
 * the steered object stays live and untouched.
 */
void* realloc(void* p, size_t n) {
    if (p != g_addr || !p || !__atomic_load_n(&g_busy, __ATOMIC_ACQUIRE))
        return __libc_realloc(p, n);

    void* shadow = g_shadow;
    memcpy(shadow, p, g_size < n ? g_size : n);
    void* r = __libc_realloc(shadow, n);
    if (r || n == 0) {
        g_shadow = NULL;
        __atomic_store_n(&g_busy, 0, __ATOMIC_RELEASE);
    }
    return r;
}

static const char* env_find(char** env, const char* key, size_t klen) {
    for (; env && *env; env++)
        if (memcmp(*env, key, klen) == 0)
            return *env + klen;
    return NULL;
}

static unsigned long parse(const char* s, unsigned base) {
    unsigned long v = 0;
    if (base == 16 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        s += 2;
    for (;; s++) {
        unsigned c = (unsigned char)*s, d;
        if (c >= '0' && c <= '9')
            d = c - '0';
        else if (base == 16 && c >= 'a' && c <= 'f')
            d = c - 'a' + 10;
        else if (base == 16 && c >= 'A' && c <= 'F')
            d = c - 'A' + 10;
        else
            return v;
        v = v * base + d;
    }
}

__attribute__((constructor)) static void steer_init(int argc, char** argv, char** envp) {
    (void)argc;
    (void)argv;
    char** env = envp ? envp : environ;
    const char* s;

    if ((s = env_find(env, "STEER_REPORT=", 13)))
        g_report = s[0] == '1';
    if ((s = env_find(env, "STEER_PAD=", 10))) {
        size_t pad = parse(s, 10);
        if (pad) {
            void* volatile leak = __libc_malloc(pad);
            (void)leak;
        }
    }
    if (!(s = env_find(env, "STEER_SIZE=", 11)))
        return;
    size_t size = parse(s, 10);
    if (!size || !(s = env_find(env, "STEER_ADDR=", 11)))
        return;
    unsigned long addr = parse(s, 16);
    if ((s = env_find(env, "STEER_SKIP=", 11)))
        g_skip = parse(s, 10);

    unsigned long lo = addr & ~(PAGE - 1);
    unsigned long hi = (addr + size + PAGE - 1) & ~(PAGE - 1);
    void* m = hi > lo && addr + size > addr
                  ? mmap((void*)lo, hi - lo, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0)
                  : MAP_FAILED;
    if (m != (void*)lo) {
        static const char msg[] = "steer: cannot map STEER_ADDR; pass-through\n";
        if (m != MAP_FAILED)
            munmap(m, hi - lo);
        (void)!write(2, msg, sizeof msg - 1);
        return;
    }
    memset(m, 0, hi - lo);
    g_addr = (unsigned char*)addr;
    g_size = size;
}

static char* put(char* o, const char* label, unsigned long v) {
    char t[20];
    int n = 0;
    while (*label)
        *o++ = *label++;
    do {
        t[n++] = '0' + v % 10;
        v /= 10;
    } while (v);
    while (n)
        *o++ = t[--n];
    return o;
}

__attribute__((destructor)) static void steer_fini(void) {
    if (!g_report)
        return;
    char buf[160];
    unsigned long skipped = g_match < g_skip ? g_match : g_skip;
    char* o = put(buf, "steer: steered=", g_steered);
    o = put(o, " busy=", g_busy_hits);
    o = put(o, " skipped=", skipped);
    o = put(o, " frame=", g_frame);
    *o++ = '\n';
    (void)!write(2, buf, o - buf);
}
