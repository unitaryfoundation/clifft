// bench_cpu.cpp -- CPU baseline for the clifft GPU microbenchmark.
//
// Two measurements:
//   (1) single-statevector per-op throughput, swept over active rank k
//       -> establishes the CPU cost curve each op must beat on GPU.
//   (2) batched-across-shots throughput at fixed small k
//       -> the CPU side of the central "batch many small states" hypothesis.
//
// Output: CSV lines on stdout (tag,...) plus a human header on stderr.
//
// Usage: bench_cpu [kmin kmax] [bk1,bk2,...] [layers]
//   defaults: kmin=10 kmax=26, batched k in {12,16,18,20}, layers=4
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "cpu_kernels.hpp"

using namespace mb;

static void init_state(cd* arr, uint64_t dim, Rng& rng) {
    double norm = 0.0;
    for (uint64_t i = 0; i < dim; ++i) {
        double re = rng.next_unit() * 2.0 - 1.0;
        double im = rng.next_unit() * 2.0 - 1.0;
        arr[i] = cd(re, im);
        norm += re * re + im * im;
    }
    double inv = 1.0 / std::sqrt(norm);
    for (uint64_t i = 0; i < dim; ++i) arr[i] *= inv;
}

// Adaptive timing: run fn() enough times to exceed ~50ms, return sec/call.
static double time_per_call(const std::function<void()>& fn) {
    fn();  // warmup
    long reps = 1;
    for (;;) {
        auto t0 = Clock::now();
        for (long i = 0; i < reps; ++i) fn();
        double el = seconds_since(t0);
        if (el > 0.05 || reps > (1L << 24)) return el / reps;
        reps *= 2;
    }
}

static const cd* u2_mat() {
    static const cd m[4] = {cd(kU2[0], kU2[1]), cd(kU2[2], kU2[3]), cd(kU2[4], kU2[5]),
                            cd(kU2[6], kU2[7])};
    return m;
}
static const cd* u4_mat() {
    static cd m[16];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 16; ++i) m[i] = cd(kU4[2 * i], kU4[2 * i + 1]);
        init = true;
    }
    return m;
}

static void apply_sched(cd* arr, unsigned k, const ScheduledOp& s, Rng& rng,
                        uint64_t* fw = nullptr) {
    switch (s.op) {
        case Op::H: op_h(arr, k, s.a); break;
        case Op::T: op_t(arr, k, s.a); break;
        case Op::CZ: op_cz(arr, k, s.a, s.b); break;
        case Op::CNOT: op_cnot(arr, k, s.a, s.b); break;
        case Op::U2: op_u2(arr, k, s.a, u2_mat()); break;
        case Op::U4: op_u4(arr, k, s.a, s.b, u4_mat()); break;
        // In the meas/real schedules, measurements run at rank k (-> k-1) and
        // the following EXPAND/EXPAND_T runs at rank k-1 (s.a carries the
        // current rank); the batch shape is restored every layer.
        case Op::EXPAND: op_expand(arr, s.a); break;
        case Op::EXPAND_T: op_expand_t(arr, s.a); break;
        case Op::MEAS_DIAG: op_meas_diag(arr, k, rng); break;
        case Op::MEAS_INTERFERE: op_meas_interfere(arr, k, rng); break;
        // One frame/bookkeeping bytecode op: a bit-op on the per-shot frame
        // word -- what clifft's frame ops cost, i.e. ~nothing on CPU. The
        // multiply-add (vs XOR) keeps the word from self-cancelling across
        // layers so validation can prove the ticks ran.
        case Op::FRAME_TICK:
            if (fw) *fw = *fw * 0x100000001B3ull + (uint64_t(s.a) + 1);
            break;
    }
}

static void bench_single(unsigned kmin, unsigned kmax) {
    fprintf(stderr, "\n=== Single-statevector per-op throughput (CPU) ===\n");
    fprintf(stderr, "%-4s %-15s %-12s %-12s %-10s\n", "k", "op", "dim", "ns/op", "Gamp/s");
    // headroom for EXPAND (writes 2^(k+1))
    uint64_t cap = uint64_t(1) << (kmax + 1);
    std::vector<cd> buf(cap);
    Rng rng(12345);
    for (unsigned k = kmin; k <= kmax; ++k) {
        uint64_t dim = uint64_t(1) << k;
        init_state(buf.data(), dim, rng);
        unsigned v = k / 2;            // representative axis
        unsigned c = 0, t = (k >= 2) ? 1u : 0u;
        struct Item { Op op; std::function<void()> fn; };
        std::vector<Item> items = {
            {Op::H, [&] { op_h(buf.data(), k, v); }},
            {Op::T, [&] { op_t(buf.data(), k, v); }},
            {Op::CZ, [&] { if (k >= 2) op_cz(buf.data(), k, c, t); }},
            {Op::CNOT, [&] { if (k >= 2) op_cnot(buf.data(), k, c, t); }},
            {Op::U2, [&] { op_u2(buf.data(), k, v, u2_mat()); }},
            {Op::U4, [&] { if (k >= 2) op_u4(buf.data(), k, c, t, u4_mat()); }},
            {Op::EXPAND, [&] { op_expand(buf.data(), k); }},
            {Op::EXPAND_T, [&] { op_expand_t(buf.data(), k); }},
            // force_bit=1 so the compact always runs -- same work as the GPU
            // row, which always compacts (see cpu_kernels.hpp note).
            {Op::MEAS_DIAG, [&] { op_meas_diag(buf.data(), k, rng, 1); }},
            {Op::MEAS_INTERFERE, [&] { op_meas_interfere(buf.data(), k, rng, 1); }},
        };
        for (auto& it : items) {
            double spc = time_per_call(it.fn);
            double gamps = (double)amps_touched(it.op, dim) / spc / 1e9;
            printf("single,%u,%s,%llu,%.1f,%.3f\n", k, op_name(it.op),
                   (unsigned long long)dim, spc * 1e9, gamps);
            fprintf(stderr, "%-4u %-15s %-12llu %-12.1f %-10.3f\n", k, op_name(it.op),
                    (unsigned long long)dim, spc * 1e9, gamps);
        }
    }
}

// Three batched workloads: gates-only (idealized ceiling), gate-heavy with a
// completed measurement per layer, and the census-calibrated real op mix
// (make_layer_schedule_real -- the schedule the go/no-go is quoted on).
enum class BatchMode { Gates, Meas, Real };

static void bench_batched(const std::vector<unsigned>& ks, unsigned layers, BatchMode mode) {
    const char* tag = mode == BatchMode::Gates ? "batched"
                      : mode == BatchMode::Meas ? "batchedmeas"
                                                : "batchedreal";
    fprintf(stderr, "\n=== Batched-across-shots throughput (CPU, %s) ===\n", tag);
    fprintf(stderr, "%-4s %-8s %-14s %-14s %-12s\n", "k", "B", "shots", "shots/s", "Mshot-lyr/s");
    std::vector<unsigned> Bs = {256, 1024, 4096, 16384};
    for (unsigned k : ks) {
        uint64_t dim = uint64_t(1) << k;
        auto sched = mode == BatchMode::Gates  ? make_layer_schedule(k, layers)
                     : mode == BatchMode::Meas ? make_layer_schedule_meas(k, layers)
                                               : make_layer_schedule_real(k, layers);
        for (unsigned B : Bs) {
            // cap memory at ~4 GB for the batch buffer
            if ((double)B * dim * sizeof(cd) > 4.0e9) continue;
            std::vector<cd> buf((size_t)B * dim);
            std::vector<uint64_t> frames(B, 0);  // per-shot frame words (FRAME_TICK)
            Rng init_rng(777 + k);
            for (unsigned b = 0; b < B; ++b) init_state(buf.data() + (size_t)b * dim, dim, init_rng);

            auto run = [&] {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int b = 0; b < (int)B; ++b) {
                    cd* slice = buf.data() + (size_t)b * dim;
                    Rng rng(0xABCDEF ^ (b * 2654435761u));
                    for (const auto& s : sched) apply_sched(slice, k, s, rng, &frames[b]);
                }
            };
            // time a few whole-batch runs, take median
            std::vector<double> samples;
            run();  // warmup
            for (int r = 0; r < 5; ++r) {
                auto t0 = Clock::now();
                run();
                samples.push_back(seconds_since(t0));
            }
            double sec = median(samples);
            double shots_per_s = B / sec;
            double mshot_layer = shots_per_s * layers / 1e6;
            printf("%s,%u,%u,%u,%.1f,%.3f\n", tag, k, B, B, shots_per_s, mshot_layer);
            fprintf(stderr, "%-4u %-8u %-14u %-14.1f %-12.3f\n", k, B, B, shots_per_s, mshot_layer);
        }
    }
}

int main(int argc, char** argv) {
    unsigned kmin = 10, kmax = 26, layers = 4;
    std::vector<unsigned> bks = {12, 16, 18, 20};
    if (argc >= 3) { kmin = atoi(argv[1]); kmax = atoi(argv[2]); }
    if (argc >= 4) {
        bks.clear();
        std::string s = argv[3];
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            bks.push_back((unsigned)atoi(s.substr(pos, comma - pos).c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    if (argc >= 5) layers = atoi(argv[4]);

#ifdef _OPENMP
    fprintf(stderr, "OpenMP: enabled, max_threads=%d\n", omp_get_max_threads());
#else
    fprintf(stderr, "OpenMP: DISABLED (single-threaded baseline)\n");
#endif
    printf("# tag,k,op_or_B,dim_or_shots,metric1,metric2\n");
    bench_single(kmin, kmax);
    bench_batched(bks, layers, BatchMode::Gates);
    bench_batched(bks, layers, BatchMode::Meas);
    bench_batched(bks, layers, BatchMode::Real);
    return 0;
}
