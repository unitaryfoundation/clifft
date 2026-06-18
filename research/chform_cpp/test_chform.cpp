// Validate the bit-packed C++ CHForm against an in-C++ dense oracle (mirrors
// research/chform_backend/test_chform.py), then a wall-clock micro-benchmark.
//
// Build & run:
//   clang++ -std=c++20 -O3 -o /tmp/chf_test research/chform_cpp/test_chform.cpp
//   /tmp/chf_test
#include "chform.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using chf::CHForm;
using chf::cd;

// ----------------------------- dense oracle -----------------------------
static const cd H2[2][2] = {{0.70710678118654752, 0.70710678118654752},
                            {0.70710678118654752, -0.70710678118654752}};
static const cd Smat[2][2] = {{1, 0}, {0, cd(0, 1)}};
static const cd Sdg[2][2] = {{1, 0}, {0, cd(0, -1)}};
static const cd Xmat[2][2] = {{0, 1}, {1, 0}};
static const cd Ymat[2][2] = {{0, cd(0, -1)}, {cd(0, 1), 0}};
static const cd Zmat[2][2] = {{1, 0}, {0, -1}};
static const cd (*GATES[6])[2] = {H2, Smat, Sdg, Xmat, Ymat, Zmat};

static void d1q(std::vector<cd>& v, int n, int q, const cd U[2][2]) {
    for (uint64_t x = 0; x < (1ULL << n); ++x) {
        if ((x >> q) & 1) continue;
        uint64_t x1 = x | (1ULL << q);
        cd a = v[x], b = v[x1];
        v[x] = U[0][0] * a + U[0][1] * b;
        v[x1] = U[1][0] * a + U[1][1] * b;
    }
}
static void dcx(std::vector<cd>& v, int n, int c, int t) {
    std::vector<cd> o(v.size());
    for (uint64_t x = 0; x < (1ULL << n); ++x) o[x] = v[x ^ ((((x >> c) & 1)) << t)];
    v = o;
}
static void dcz(std::vector<cd>& v, int n, int c, int t) {
    for (uint64_t x = 0; x < (1ULL << n); ++x) if (((x >> c) & 1) && ((x >> t) & 1)) v[x] = -v[x];
}
static void dswap(std::vector<cd>& v, int n, int a, int b) {
    std::vector<cd> o(v.size());
    for (uint64_t x = 0; x < (1ULL << n); ++x) {
        uint64_t ba = (x >> a) & 1, bb = (x >> b) & 1;
        o[x] = v[x ^ ((ba ^ bb) << a) ^ ((ba ^ bb) << b)];
    }
    v = o;
}

static double max_abs_diff(const std::vector<cd>& a, const std::vector<cd>& b) {
    double w = 0; for (size_t i = 0; i < a.size(); ++i) w = std::max(w, std::abs(a[i] - b[i]));
    return w;
}

// ----------------------------- tests -----------------------------
int main() {
    std::mt19937_64 rng(12345);
    auto ri = [&](int lo, int hi) { return (int)(rng() % (hi - lo)) + lo; };

    // 1) full Clifford set vs dense
    double worst = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        int n = ri(1, 7);
        CHForm c(n);
        std::vector<cd> v((size_t)1 << n, cd(0, 0)); v[0] = 1;
        int depth = ri(2, 7);
        for (int d = 0; d < depth; ++d) {
            for (int q = 0; q < n; ++q) {
                int gname = ri(0, 6);
                c.clifford_1q(gname, q);
                d1q(v, n, q, GATES[gname]);
            }
            for (int q = 0; q + 1 < n; q += 2) {
                int kind = ri(0, 3);
                if (kind == 0) { c.cx(q, q + 1); dcx(v, n, q, q + 1); }
                else if (kind == 1) { c.cz(q, q + 1); dcz(v, n, q, q + 1); }
                else { c.swap(q, q + 1); dswap(v, n, q, q + 1); }
            }
        }
        worst = std::max(worst, max_abs_diff(c.statevector(), v));
    }
    printf("[%s] full Clifford set vs dense on 2000 circuits (max abs err %.2e)\n",
           worst < 1e-9 ? "OK" : "FAIL", worst);

    // 2) projection vs dense
    double pworst = 0; int none_ok = 0, nproj = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        int n = ri(2, 7);
        CHForm c(n);
        std::vector<cd> v((size_t)1 << n, cd(0, 0)); v[0] = 1;
        int depth = ri(2, 5);
        for (int d = 0; d < depth; ++d) {
            for (int q = 0; q < n; ++q) { int gn = ri(0, 6); c.clifford_1q(gn, q); d1q(v, n, q, GATES[gn]); }
            for (int q = 0; q + 1 < n; q += 2) { c.cz(q, q + 1); dcz(v, n, q, q + 1); }
        }
        int q = ri(0, n), bit = ri(0, 2);
        std::vector<cd> dv = v;
        for (uint64_t x = 0; x < (1ULL << n); ++x) if ((int)((x >> q) & 1) != bit) dv[x] = 0;
        double dn = 0; for (auto& z : dv) dn += std::norm(z);
        bool ok = c.project(q, bit);
        if (dn < 1e-12) { if (!ok) ++none_ok; else pworst = 1e9; }
        else { ++nproj; pworst = std::max(pworst, max_abs_diff(c.statevector(), dv)); }
    }
    printf("[%s] projection vs dense (max abs err %.2e, %d zero-support, %d projected)\n",
           pworst < 1e-9 ? "OK" : "FAIL", pworst, none_ok, nproj);

    // 3) wall-clock: Clifford gates on a single bit-packed term
    for (int n : {30, 60, 120}) {
        CHForm c(n);
        for (int q = 0; q < n; ++q) c.h(q);
        const int NG = 200000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < NG; ++i) {
            int q = ri(0, n);
            int g = ri(0, 6);
            c.clifford_1q(g, q);
            if (n >= 2) c.cx(ri(0, n), (q + 1) % n);
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        printf("  bench n=%3d: %d (1q+CX) gate-pairs in %.3f s  =>  %.2f M gates/s\n",
               n, NG, sec, (2.0 * NG / sec) / 1e6);
    }
    return 0;
}
