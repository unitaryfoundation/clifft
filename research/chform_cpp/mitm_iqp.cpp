// Exact meet-in-the-middle Born probabilities for T+CZ IQP circuits, in
// O(n * 2^{n/2}) time and O(2^{n/2}) memory -- the honest exact baseline for
// this circuit family (see findings.tex): amplitude(x) of H^n D H^n |0> is a
// Z8 Gauss sum, and splitting the register in two turns it into
//
//   <x|psi> = 2^{-n} sum_u f(u) (-1)^{x_L.u} * Ghat(x_R ^ w(u)),
//
// with f(u) = e^{i theta_L(u)} the left phase, w(u) the F2-linear cross-CZ
// coupling, and Ghat the Walsh-Hadamard transform of the right phase table --
// precomputed once, then 2^{nL} table lookups per target (Gray-code updates,
// threaded over targets). Ghat is complex<double> while it fits in memory
// (nR <= 28, i.e. n <= 57: exact to ~1e-15) and complex<float> above (8 B/entry,
// P(x) good to ~1e-6 relative -- still far below any delta we test).
//
// This makes every "past-clifft" IQP claim exactly checkable up to n ~ 60
// (2^30-entry table = 8.6 GB): the same 2^{n/2} scale as clifft's dense block
// on the *measured* dense-IQP program (peak_rank = n/2) -- neither clifft nor
// MitM reaches n = 66/72, which is what makes that regime interesting.
//
// Spec file (argv[1]) -- same format as bench_overlap.cpp (k/samples ignored):
//   n k samples
//   <n chars: dag pattern, '1' = T_dag>
//   ncz
//   a b           (ncz lines)
//   ntarget
//   x             (ntarget lines, basis-state integers, bit q of x = qubit q)
//
// Build:  clang++ -std=c++20 -O3 -o /tmp/chf_mitm mitm_iqp.cpp
// Output: "P <x> <prob>" per target (exact), plus timings.
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using cd = std::complex<double>;
using cf = std::complex<float>;

struct Spec {
    int n, nL, nR, nt;
    std::vector<int> sigma;
    std::vector<uint64_t> adjL, cross, adjR, targets;
};

template <typename CT>
static void run(const Spec& sp) {
    const int n = sp.n, nL = sp.nL, nR = sp.nR, nt = sp.nt;
    const double PI4 = 3.14159265358979323846 / 4.0;
    cd phase8[8];
    for (int e = 0; e < 8; ++e) phase8[e] = std::exp(cd(0, PI4 * e));

    auto t0 = std::chrono::steady_clock::now();

    // Right phase table g(v) = e^{i theta_R(v)}, exponent mod 8 by Gray code,
    // then in-place WHT -> Ghat.
    size_t NR = 1ULL << nR;
    std::vector<CT> G(NR);
    {
        uint64_t v = 0; int e8 = 0;
        G[0] = CT(1, 0);
        for (size_t i = 1; i < NR; ++i) {
            size_t gi = i ^ (i >> 1), gp = (i - 1) ^ ((i - 1) >> 1);
            int b = std::countr_zero(gi ^ gp);
            int setb = (gi >> b) & 1;
            int p = std::popcount(sp.adjR[b] & v) & 1;     // b not its own neighbor
            e8 = (((e8 + (setb ? sp.sigma[nL + b] : -sp.sigma[nL + b]) + 4 * p) % 8) + 8) % 8;
            v ^= 1ULL << b;
            G[gi] = CT(phase8[e8].real(), phase8[e8].imag());
        }
        for (int bit = 0; bit < nR; ++bit) {               // WHT, unnormalized
            size_t h = 1ULL << bit;
            for (size_t x = 0; x < NR; x += 2 * h)
                for (size_t j = x; j < x + h; ++j) {
                    CT a = G[j], b = G[j + h];
                    G[j] = a + b; G[j + h] = a - b;
                }
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    // Per target: Gray-code sweep over u in [0, 2^nL), threaded over targets.
    size_t NL = 1ULL << nL;
    uint64_t maskL = (1ULL << nL) - 1;
    uint64_t maskR = (nR == 0) ? 0 : ((1ULL << nR) - 1);
    std::vector<double> probs(nt);
    int nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    auto worker = [&](int tid) {
        for (int t = tid; t < nt; t += nthreads) {
            uint64_t xL = sp.targets[t] & maskL;
            uint64_t xR = (sp.targets[t] >> nL) & maskR;
            uint64_t u = 0, w = 0; int e8 = 0, sign = 0;
            cd acc = cd(G[xR].real(), G[xR].imag());       // u = 0 term
            for (size_t i = 1; i < NL; ++i) {
                size_t gi = i ^ (i >> 1), gp = (i - 1) ^ ((i - 1) >> 1);
                int a = std::countr_zero(gi ^ gp);
                int seta = (gi >> a) & 1;
                int p = std::popcount(sp.adjL[a] & u) & 1;
                e8 = (((e8 + (seta ? sp.sigma[a] : -sp.sigma[a]) + 4 * p) % 8) + 8) % 8;
                u ^= 1ULL << a;
                w ^= sp.cross[a];
                sign ^= (int)((xL >> a) & 1);
                CT gh = G[xR ^ w];
                cd term = phase8[e8] * cd(gh.real(), gh.imag());
                acc += sign ? -term : term;
            }
            double amp_scale = std::pow(0.5, n);           // 2^{-n}
            probs[t] = std::norm(acc * amp_scale);
        }
    };
    for (int tid = 0; tid < nthreads; ++tid) pool.emplace_back(worker, tid);
    for (auto& th : pool) th.join();
    auto t2 = std::chrono::steady_clock::now();

    printf("table_s %.6f\n", std::chrono::duration<double>(t1 - t0).count());
    printf("targets_s %.6f\n", std::chrono::duration<double>(t2 - t1).count());
    for (int t = 0; t < nt; ++t)
        printf("P %llu %.15e\n", (unsigned long long)sp.targets[t], probs[t]);
}

int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    Spec sp;
    int kdum, sdum;
    in >> sp.n >> kdum >> sdum;
    std::string dagstr; in >> dagstr;
    sp.sigma.resize(sp.n);                           // +1 for T, -1 for T_dag
    for (int q = 0; q < sp.n; ++q) sp.sigma[q] = dagstr[q] == '1' ? -1 : 1;
    int ncz; in >> ncz;
    std::vector<std::pair<int, int>> czs(ncz);
    for (int i = 0; i < ncz; ++i) in >> czs[i].first >> czs[i].second;
    in >> sp.nt;
    sp.targets.resize(sp.nt);
    for (int i = 0; i < sp.nt; ++i) in >> sp.targets[i];

    // Bipartition: L = [0, nL), R = [nL, n). Table lives on R.
    sp.nR = sp.n / 2; sp.nL = sp.n - sp.nR;
    if (sp.nL > 62 || sp.nR > 32) { fprintf(stderr, "n too large for MitM (nL=%d nR=%d)\n", sp.nL, sp.nR); return 1; }

    // Edge masks. adjL[a]: L-neighbors of a in L. cross[a]: R-neighbors of a
    // in L (bit b-nL). adjR[b]: R-neighbors within R.
    sp.adjL.assign(sp.nL, 0); sp.cross.assign(sp.nL, 0); sp.adjR.assign(sp.nR, 0);
    for (auto& e : czs) {
        int a = e.first, b = e.second;
        if (a > b) std::swap(a, b);
        if (b < sp.nL) { sp.adjL[a] |= 1ULL << b; sp.adjL[b] |= 1ULL << a; }
        else if (a >= sp.nL) { sp.adjR[a - sp.nL] |= 1ULL << (b - sp.nL); sp.adjR[b - sp.nL] |= 1ULL << (a - sp.nL); }
        else sp.cross[a] |= 1ULL << (b - sp.nL);
    }

    // Table precision: complex<double> when it fits (nR <= 28 -> 4.3 GB),
    // complex<float> above (rel error ~1e-6 -- still far below any tested
    // delta; that regime is n = 58..62 only).
    if (sp.nR <= 28) { printf("table double\n"); run<cd>(sp); }
    else { printf("table float\n"); run<cf>(sp); }
    return 0;
}
