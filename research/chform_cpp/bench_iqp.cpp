// C++ engine driver: low-rank sum over bit-packed CHForm terms + single-shot
// tensor-product magic sampling, on magic-sparse IQP -- the past-clifft
// benchmark in wall-clock. Validates exact (full branch enumeration == dense)
// at small n, then times single-shot at n=40/46/50 against the Python prototype.
//
// Build & run:
//   clang++ -std=c++20 -O3 -o /tmp/chf_iqp research/chform_cpp/bench_iqp.cpp
//   /tmp/chf_iqp
#include "chform.hpp"
#include "norm_est.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using chf::CHForm;
using chf::cd;

// T = alpha I + beta S  (extent-optimal {I,S} pair, |alpha|=|beta|).
static const cd ALPHA = cd(0.5, (std::sqrt(2.0) - 1) / 2);
static const cd BETA = cd(0.5, -(std::sqrt(2.0) - 1) / 2);
static const double ABSc = std::abs(ALPHA);  // = |beta|

struct IQP {
    int n;
    std::vector<uint8_t> dag;            // T or T_dag per qubit
    std::vector<std::pair<int, int>> czs;
};

static IQP make_iqp(int n, uint64_t seed) {
    std::mt19937_64 rg(seed);
    IQP c{n, std::vector<uint8_t>(n), {}};
    for (int q = 0; q < n; ++q) c.dag[q] = rg() & 1;
    for (int q = 0; q + 1 < n; ++q) if (rg() % 10 < 6) c.czs.push_back({q, q + 1});
    return c;
}

// Dense random IQP (CZ on each pair w.p. 1/2): the frontier family. On the
// nearest-neighbour chain above, the MEASURED program compiles to
// peak_rank ~ 1 (lightcone sweep) and exact P(x) is trivial for clifft; the
// dense ensemble is where the exact baselines genuinely pay 2^{n/2}.
static IQP make_dense_iqp(int n, uint64_t seed) {
    std::mt19937_64 rg(seed);
    IQP c{n, std::vector<uint8_t>(n), {}};
    for (int q = 0; q < n; ++q) c.dag[q] = rg() & 1;
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b) if (rg() & 1) c.czs.push_back({a, b});
    return c;
}

// base |+>^n
static CHForm base_plus(int n) { CHForm b(n); for (int q = 0; q < n; ++q) b.h(q); return b; }

static void finish(CHForm& t, const IQP& c) {  // CZ chain + H^n (post-magic Cliffords)
    for (auto& e : c.czs) t.cz(e.first, e.second);
    for (int q = 0; q < c.n; ++q) t.h(q);
}

// ---- exact: enumerate all 2^n branch strings (the full decomposition) ----
static std::vector<CHForm> build_exact(const IQP& c) {
    CHForm base = base_plus(c.n);
    std::vector<CHForm> terms;
    for (uint64_t b = 0; b < (1ULL << c.n); ++b) {
        CHForm t = base;
        cd coeff(1, 0);
        for (int q = 0; q < c.n; ++q) {
            if ((b >> q) & 1) {                 // S branch, coeff beta
                if (c.dag[q]) t.s_dag(q); else t.s_gate(q);
                coeff *= (c.dag[q] ? std::conj(BETA) : BETA);
            } else {
                coeff *= (c.dag[q] ? std::conj(ALPHA) : ALPHA);
            }
        }
        t.scale(coeff);
        finish(t, c);
        terms.push_back(std::move(t));
    }
    return terms;
}

// ---- single-shot: sample k branch strings (P(S)=|beta|/(|a|+|b|)=1/2) ----
static std::vector<CHForm> build_single_shot(const IQP& c, int k, std::mt19937_64& rng) {
    CHForm base = base_plus(c.n);
    double l1 = std::pow(2.0 * ABSc, c.n);      // (|alpha|+|beta|)^n
    std::vector<CHForm> terms; terms.reserve(k);
    for (int j = 0; j < k; ++j) {
        CHForm t = base;
        cd phase(1, 0);
        for (int q = 0; q < c.n; ++q) {
            bool sbranch = (rng() & 1);
            cd cf;
            if (sbranch) { if (c.dag[q]) t.s_dag(q); else t.s_gate(q); cf = c.dag[q] ? std::conj(BETA) : BETA; }
            else cf = c.dag[q] ? std::conj(ALPHA) : ALPHA;
            phase *= cf / std::abs(cf);
        }
        t.scale((l1 / k) * phase);
        finish(t, c);
        terms.push_back(std::move(t));
    }
    return terms;
}

static cd amp(const std::vector<CHForm>& terms, uint64_t x) {
    cd s(0, 0); for (auto& t : terms) s += t.amplitude(x); return s;
}

// dense IQP statevector for validation
static std::vector<cd> dense_iqp(const IQP& c) {
    int n = c.n; std::vector<cd> v((size_t)1 << n, cd(0, 0)); v[0] = 1;
    auto H = [&](int q) {
        for (uint64_t x = 0; x < (1ULL << n); ++x) if (!((x >> q) & 1)) {
            uint64_t x1 = x | (1ULL << q); cd a = v[x], b = v[x1];
            v[x] = chf::SQRT1_2 * (a + b); v[x1] = chf::SQRT1_2 * (a - b);
        }
    };
    for (int q = 0; q < n; ++q) H(q);
    cd tp = std::exp(cd(0, M_PI / 4));
    for (int q = 0; q < n; ++q) { cd ph = c.dag[q] ? std::conj(tp) : tp;
        for (uint64_t x = 0; x < (1ULL << n); ++x) if ((x >> q) & 1) v[x] *= ph; }
    for (auto& e : c.czs) for (uint64_t x = 0; x < (1ULL << n); ++x)
        if (((x >> e.first) & 1) && ((x >> e.second) & 1)) v[x] = -v[x];
    for (int q = 0; q < n; ++q) H(q);
    return v;
}

int main() {
    // 1) exact (full enumeration) == dense IQP
    double worst = 0;
    for (int n = 2; n <= 10; ++n) {
        IQP c = make_iqp(n, 100 + n);
        auto terms = build_exact(c);
        auto dv = dense_iqp(c);
        for (uint64_t x = 0; x < (1ULL << n); ++x) worst = std::max(worst, std::abs(amp(terms, x) - dv[x]));
    }
    printf("[%s] exact engine (full enumeration) == dense IQP, n<=10 (max abs err %.2e)\n",
           worst < 1e-9 ? "OK" : "FAIL", worst);

    // 2) single-shot wall-clock at the frontier (DENSE IQP): the CH-form
    //    backend holds chi terms of O(n^2) bits each; the accurate budget
    //    k ~ 2^{0.228 n}/delta^2 is reachable in seconds.
    // Honest exact-baseline column: on the MEASURED dense-IQP program clifft
    // compiles to peak_rank = n/2 (and exact meet-in-the-middle, mitm_iqp.cpp,
    // costs the same 2^{n/2}) -- so the frontier contrast is 16B * 2^{ceil(n/2)},
    // NOT 16B * 2^n (the unitary-path framing the review struck down).
    printf("  %3s %9s %9s %11s %10s %14s\n", "n", "budget k", "chi", "build (s)",
           "CH mem", "exact 2^{n/2}");
    std::mt19937_64 rng(7);
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"};
    auto human = [&](double b) { int u = 0; while (b >= 1024 && u < 7) { b /= 1024; ++u; }
                                 static char s[32]; snprintf(s, 32, "%.1f%s", b, units[u]); return std::string(s); };
    for (int n : {40, 46, 50, 54, 60, 66, 72}) {
        double delta = 0.4;
        int k = (int)std::max(2000.0, std::pow(2.0, 0.228 * n) / (delta * delta));
        IQP c = make_dense_iqp(n, 100 + n);
        auto t0 = std::chrono::steady_clock::now();
        auto terms = build_single_shot(c, k, rng);
        volatile double a0 = std::abs(amp(terms, 0)); (void)a0;
        auto t1 = std::chrono::steady_clock::now();
        int Wn = (n + 63) / 64;
        double mem = terms.size() * (double)(3 * n * Wn * 8 + 3 * n + 16);
        printf("  %3d %9d %9zu %11.2f %10s %14s\n", n, k, terms.size(),
               std::chrono::duration<double>(t1 - t0).count(),
               human(mem).c_str(), human(std::pow(2.0, (n + 1) / 2) * 16).c_str());
    }
    printf("  (Exact baselines -- clifft record_probabilities on the measured program\n");
    printf("   and meet-in-the-middle -- both scale as 2^{n/2}: feasible to n ~ 60-62\n");
    printf("   on a 36 GB workstation, infeasible at n = 66 [137 GB] and 72 [1.1 TB].)\n");

    // 3) full non-materialising pipeline: ||omega||^2 via norm estimation
    //    (~1 + delta^2 = the accuracy check, the part Python couldn't time).
    printf("  norm estimation (full pipeline; ||omega||^2 ~ 1+delta^2 confirms accuracy):\n");
    for (int n : {40, 46}) {
        double delta = 0.4;
        int k = (int)std::max(2000.0, std::pow(2.0, 0.228 * n) / (delta * delta));
        IQP c = make_dense_iqp(n, 100 + n);
        auto terms = build_single_shot(c, k, rng);
        auto t0 = std::chrono::steady_clock::now();
        double nm = chf::estimate_norm2(terms, n, 15, rng);
        auto t1 = std::chrono::steady_clock::now();
        printf("   n=%d: ||omega||^2 = %.3f (target ~%.2f)  norm-est %.1f s (L=15)\n",
               n, nm, 1.0 + delta * delta, std::chrono::duration<double>(t1 - t0).count());
    }
    return 0;
}
