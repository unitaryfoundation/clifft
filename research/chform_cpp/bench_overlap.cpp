// Overlapping-regime helper: read an IQP circuit spec + target basis states,
// build the single-shot CH-form state, and output approximate Born probabilities
// P(x) = |<x|psi>|^2 / ||psi||^2 (norm via estimation) with timings. Driven by
// bench_overlap.py, which compares against clifft's exact basis_probabilities.
//
// Spec file (argv[1]):
//   n k samples
//   <n chars: dag pattern, '1' = T_dag>
//   ncz
//   a b           (ncz lines)
//   ntarget
//   x             (ntarget lines, basis-state integers)
#include "chform.hpp"
#include "norm_est.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

using namespace chf;

static const cd ALPHA = cd(0.5, (std::sqrt(2.0) - 1) / 2);
static const cd BETA = cd(0.5, -(std::sqrt(2.0) - 1) / 2);

int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    int n, k, samples;
    in >> n >> k >> samples;
    std::string dagstr; in >> dagstr;
    std::vector<uint8_t> dag(n);
    for (int q = 0; q < n; ++q) dag[q] = (dagstr[q] == '1');
    int ncz; in >> ncz;
    std::vector<std::pair<int, int>> czs(ncz);
    for (int i = 0; i < ncz; ++i) in >> czs[i].first >> czs[i].second;
    int nt; in >> nt;
    std::vector<uint64_t> targets(nt);
    for (int i = 0; i < nt; ++i) in >> targets[i];

    uint64_t seed = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 12345;
    std::mt19937_64 rng(seed);
    auto t0 = std::chrono::steady_clock::now();
    // base |+>^n
    CHForm base(n);
    for (int q = 0; q < n; ++q) base.h(q);
    double l1 = std::pow(2.0 * std::abs(ALPHA), n);
    std::vector<CHForm> terms; terms.reserve(k);
    for (int j = 0; j < k; ++j) {
        CHForm t = base;
        cd phase(1, 0);
        for (int q = 0; q < n; ++q) {
            bool sbranch = (rng() & 1);
            cd cf;
            if (sbranch) { if (dag[q]) t.s_dag(q); else t.s_gate(q); cf = dag[q] ? std::conj(BETA) : BETA; }
            else cf = dag[q] ? std::conj(ALPHA) : ALPHA;
            phase *= cf / std::abs(cf);
        }
        t.scale((l1 / k) * phase);
        for (auto& e : czs) t.cz(e.first, e.second);
        for (int q = 0; q < n; ++q) t.h(q);
        terms.push_back(std::move(t));
    }
    auto t1 = std::chrono::steady_clock::now();
    // samples == 0: analytic normalization, valid for this benchmark's unitary
    // product-magic circuits only -- ||psi|| = 1 exactly and the single-shot
    // estimator has E||omega||^2 = 1 + (||c||_1^2 - 1)/k, so no norm estimation
    // is needed. Generic circuits (projections / non-product magic) must
    // estimate (samples > 0).
    double norm = (samples == 0) ? 1.0 + (l1 * l1 - 1.0) / k
                                 : estimate_norm2(terms, n, samples, rng);
    auto t2 = std::chrono::steady_clock::now();

    std::vector<double> probs(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        cd a(0, 0); for (auto& t : terms) a += t.amplitude(targets[i]);
        probs[i] = std::norm(a) / norm;
    }
    auto t3 = std::chrono::steady_clock::now();

    printf("build_s %.6f\n", std::chrono::duration<double>(t1 - t0).count());
    printf("norm_s %.6f\n", std::chrono::duration<double>(t2 - t1).count());
    printf("amps_s %.6f\n", std::chrono::duration<double>(t3 - t2).count());
    printf("total_s %.6f\n", std::chrono::duration<double>(t3 - t0).count());
    printf("chi %zu\n", terms.size());
    printf("norm2 %.10f\n", norm);
    for (size_t i = 0; i < targets.size(); ++i)
        printf("P %llu %.12e\n", (unsigned long long)targets[i], probs[i]);
    return 0;
}
