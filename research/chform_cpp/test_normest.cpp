// Validate C++ norm estimation (norm_est.hpp) vs brute force / dense.
// Build & run:
//   clang++ -std=c++20 -O3 -o /tmp/chf_ne research/chform_cpp/test_normest.cpp && /tmp/chf_ne
#include "chform.hpp"
#include "norm_est.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace chf;

static CHForm random_state(int n, std::mt19937_64& rng, int depth) {
    CHForm c(n);
    for (int d = 0; d < depth; ++d) {
        for (int q = 0; q < n; ++q) c.clifford_1q((int)(rng() % 6), q);
        for (int q = 0; q + 1 < n; ++q) if (rng() & 1) c.cx(q, q + 1); else if (n >= 2) c.cz(q, (q + 1) % n);
    }
    double ang = (rng() % 100000) / 100000.0 * 6.2831853;
    double mag = 0.3 + (rng() % 1000) / 1000.0 * 1.7;
    c.scale(std::polar(mag, ang));
    return c;
}

static std::vector<cd> equatorial_dense(const imat& A) {
    int n = (int)A.size();
    std::vector<cd> v((size_t)1 << n);
    for (uint64_t x = 0; x < (1ULL << n); ++x) {
        int e = 0;
        for (int a = 0; a < n; ++a) if ((x >> a) & 1) { e += A[a][a]; for (int b = a + 1; b < n; ++b) if ((x >> b) & 1) e += 2 * A[a][b]; }
        v[x] = ipow(e);
    }
    double s = std::pow(SQRT1_2, n);
    for (auto& z : v) z *= s;
    return v;
}

int main() {
    std::mt19937_64 rng(2024);

    // 1) Lemma 4 exp_sum vs brute force
    double w1 = 0;
    for (int t = 0; t < 3000; ++t) {
        int m = (int)(rng() % 7);
        imat B(m, std::vector<int>(m, 0));
        for (int a = 0; a < m; ++a) { B[a][a] = (int)(rng() & 3); for (int b = a + 1; b < m; ++b) { int v = (int)(rng() & 1); B[a][b] = v; B[b][a] = v; } }
        w1 = std::max(w1, std::abs(exp_sum(B) - exp_sum_direct(B)));
    }
    printf("[%s] Lemma 4 exp_sum vs brute force on 3000 matrices (max abs err %.2e)\n",
           w1 < 1e-9 ? "OK" : "FAIL", w1);

    // 2) Lemma 3 inner_equatorial vs dense <phi|phi_A>
    double w2 = 0;
    for (int t = 0; t < 2000; ++t) {
        int n = 1 + (int)(rng() % 5);
        CHForm phi = random_state(n, rng, 1 + (int)(rng() % 3));
        imat A = random_equatorial(n, rng);
        cd got = inner_equatorial(phi, A);
        auto pa = equatorial_dense(A);
        cd ref(0, 0);
        for (uint64_t x = 0; x < (1ULL << n); ++x) ref += std::conj(phi.amplitude(x)) * pa[x];
        w2 = std::max(w2, std::abs(got - ref));
    }
    printf("[%s] Lemma 3 <phi|phi_A> vs dense on 2000 states (max abs err %.2e)\n",
           w2 < 1e-9 ? "OK" : "FAIL", w2);

    // 3) Lemma 2 estimate_norm2 -> exact ||sum term||^2
    double worst_rel = 0;
    for (int t = 0; t < 20; ++t) {
        int n = 2 + (int)(rng() % 3);
        int k = 2 + (int)(rng() % 4);
        std::vector<CHForm> terms;
        for (int j = 0; j < k; ++j) terms.push_back(random_state(n, rng, 1 + (int)(rng() % 2)));
        std::vector<cd> psi((size_t)1 << n, cd(0, 0));
        for (auto& tm : terms) { auto sv = tm.statevector(); for (size_t x = 0; x < sv.size(); ++x) psi[x] += sv[x]; }
        double exact = 0; for (auto& z : psi) exact += std::norm(z);
        double est = estimate_norm2(terms, n, 6000, rng);
        worst_rel = std::max(worst_rel, std::abs(est - exact) / exact);
    }
    printf("[%s] Lemma 2 norm estimate vs exact on 20 low-rank states (worst rel err %.3f, L=6000)\n",
           worst_rel < 0.2 ? "OK" : "FAIL", worst_rel);
    return 0;
}
