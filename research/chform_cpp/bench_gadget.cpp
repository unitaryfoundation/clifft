// General Clifford+T strong simulation via T-gadgetization -- the C++ scale
// arm of research/chform_backend/gadgetize.py (see its header for the math:
// P(x) = 2^t |<x,0^t| C_gadget (|0^n> (x) |T>^t)|^2, magic sparsified in one
// shot on the ancillas, analytic normalization, no norm estimation).
//
// Spec file (argv[1]):
//   n t k seed
//   <t chars: dagger pattern per T in program order, '1' = T_DAG>
//   nops
//   <nops lines>   H q | S q | SDG q | X q | Y q | Z q | CX a b | CZ a b | T q
//                  (the i-th T line consumes ancilla n+i)
//   ntargets
//   x              (ntargets lines, n-bit basis integers)
//
// Output: "P <x> <prob>" per target + build/amps timings (stdout CSV-ish).
// Build:  clang++ -std=c++20 -O3 -o /tmp/chf_gadget bench_gadget.cpp
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "chform.hpp"

using namespace chf;

static const cd ALPHA = cd(0.5, (std::sqrt(2.0) - 1) / 2);
static const cd BETA = cd(0.5, -(std::sqrt(2.0) - 1) / 2);

struct OpLine {
    std::string g;
    int a = 0, b = 0;
};

int main(int argc, char** argv) {
    std::ifstream in(argv[1]);
    int n, t, k;
    uint64_t seed;
    in >> n >> t >> k >> seed;
    std::string dagstr;
    in >> dagstr;
    int nops;
    in >> nops;
    std::vector<OpLine> ops(nops);
    for (auto& o : ops) {
        in >> o.g;
        if (o.g == "CX" || o.g == "CZ") in >> o.a >> o.b;
        else in >> o.a;
    }
    int nt;
    in >> nt;
    std::vector<uint64_t> targets(nt);
    for (auto& x : targets) in >> x;

    const int N = n + t;
    std::mt19937_64 rng(seed);
    auto t0 = std::chrono::steady_clock::now();

    // |0^n> (x) H^t on the ancillas, then one single-shot {I,S} draw per
    // ancilla per term (P(S) = 1/2 exactly since |alpha| = |beta|).
    CHForm base(N);
    for (int i = 0; i < t; ++i) base.h(n + i);
    double l1 = std::pow(2.0 * std::abs(ALPHA), t);
    double l1sq = l1 * l1;

    std::vector<CHForm> terms;
    terms.reserve(k);
    for (int j = 0; j < k; ++j) {
        CHForm tm = base;
        cd phase(1, 0);
        for (int i = 0; i < t; ++i) {
            bool dag = dagstr[i] == '1';
            bool sbranch = rng() & 1;
            cd cf;
            if (sbranch) {
                if (dag) tm.s_dag(n + i); else tm.s_gate(n + i);
                cf = dag ? std::conj(BETA) : BETA;
            } else {
                cf = dag ? std::conj(ALPHA) : ALPHA;
            }
            phase *= cf / std::abs(cf);
        }
        tm.scale((l1 / k) * phase);
        // stream the gadgetized circuit (all Clifford)
        int ti = 0;
        for (const auto& o : ops) {
            if (o.g == "T") tm.cx(o.a, n + ti++);
            else if (o.g == "CX") tm.cx(o.a, o.b);
            else if (o.g == "CZ") tm.cz(o.a, o.b);
            else if (o.g == "H") tm.h(o.a);
            else if (o.g == "S") tm.s_gate(o.a);
            else if (o.g == "SDG") tm.s_dag(o.a);
            else if (o.g == "X") tm.x(o.a);
            else if (o.g == "Y") tm.y(o.a);
            else if (o.g == "Z") tm.z(o.a);
        }
        terms.push_back(std::move(tm));
    }
    auto t1 = std::chrono::steady_clock::now();

    double norm = 1.0 + (l1sq - 1.0) / k;   // analytic E||omega||^2
    double two_t = std::pow(2.0, t);
    for (uint64_t x : targets) {
        cd a(0, 0);
        for (auto& tm : terms) a += tm.amplitude(x);  // ancilla bits zero
        printf("P %llu %.15e\n", (unsigned long long)x, two_t * std::norm(a) / norm);
    }
    auto t2 = std::chrono::steady_clock::now();
    printf("build_s %.6f\n", std::chrono::duration<double>(t1 - t0).count());
    printf("amps_s %.6f\n", std::chrono::duration<double>(t2 - t1).count());
    printf("chi %zu\n", terms.size());
    fprintf(stderr, "[gadget] n=%d t=%d N=%d k=%d  build %.2fs  amps %.2fs\n", n, t,
            N, k, std::chrono::duration<double>(t1 - t0).count(),
            std::chrono::duration<double>(t2 - t1).count());
    return 0;
}
