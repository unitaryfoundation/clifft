// End-to-end research comparison with identical noise and survivor output paths.
#pragma once
#include "fold_frame_bridge_native.h"
#include "fold_recognition.h"

namespace frame_bridge_probe {
using namespace fold_blocks;
struct FaultCase {
    unsigned stage;
    int x, z;
};
struct Region {
    unsigned stage, begin, core_begin, core_end, end;
};
inline void set(History& h, int b) {
    if (b >= 0)
        h[b / 64] ^= uint64_t(1) << (b % 64);
}
inline void check_terms(unsigned n, unsigned m, const std::array<Term, 2>& a,
                        const std::array<Term, 2>& b) {
    if (n != m)
        throw std::runtime_error("branch count mismatch");
    for (unsigned j = 0; j < n; ++j)
        if (std::abs(a[j].weight) != std::abs(b[j].weight) ||
            ((a[j].op.phase + 4 * (a[j].weight < 0)) & 7) !=
                ((b[j].op.phase + 4 * (b[j].weight < 0)) & 7) ||
            a[j].op.flips != b[j].op.flips || a[j].op.edges != b[j].op.edges ||
            a[j].op.linear != b[j].op.linear)
            throw std::runtime_error("coherent branch operator mismatch");
}
inline void check_branches(const Protocol& p, std::span<const fault_frame::Bridge* const> bridges,
                           unsigned stage, const History& h) {
    Executor base;
    std::array<Term, 2> a, b;
    const auto& fold = *p.stages[stage].fold;
    auto n = base.branches(fold, h, a);
    auto m = fault_frame::branches(*bridges[stage], fold, h, b);
    check_terms(n, m, a, b);
}
inline int run(const fold_recognition::FamilySource& source,
               std::span<const fault_frame::Bridge* const> bridges,
               std::span<const Fixture> fixtures, std::span<const FaultCase> faults,
               std::span<const Region> regions, std::string_view circuit_text, unsigned shots,
               unsigned trials) {
    const auto& p = *source.kernel;
    double max_error = 0;
    Executor base;
    fault_frame::Executor bridge(bridges);
    std::vector<const fault_frame::Bridge*> mixed(bridges.begin(), bridges.end());
    for (size_t j = 0; j < mixed.size(); ++j)
        if (p.stages[j].fold && p.stages[j].fold->geometry->width < 85)
            mixed[j] = nullptr;
    fault_frame::Executor fallback(mixed);
    for (const auto& f : fixtures) {
        auto a = base.evaluate(p, f.history), b = bridge.evaluate(p, f.history);
        auto c = fallback.evaluate(p, f.history);
        for (unsigned k = 0; k < 4; ++k)
            max_error = std::max({max_error, std::abs(a[k] - b[k]), std::abs(f.expected[k] - b[k]),
                                  std::abs(a[k] - c[k])});
    }
    if (max_error > 2e-12)
        throw std::runtime_error("full history mismatch");
    for (auto f : faults) {
        History h{};
        set(h, f.x);
        set(h, f.z);
        check_branches(p, bridges, f.stage, h);
    }
    std::mt19937_64 rng(47103);
    for (auto r : regions)
        for (unsigned j = 0; j < 256; ++j) {
            History h{};
            unsigned begin = j < 128 ? r.core_begin : r.begin;
            unsigned end = j < 128 ? r.core_end : r.end;
            for (unsigned b = begin; b < end; ++b)
                if (rng() & 1)
                    set(h, b);
            check_branches(p, bridges, r.stage, h);
        }
    for (auto r : regions)
        for (unsigned j = 0; j < 128; ++j) {
            const auto& br = *bridges[r.stage];
            const auto& fold = *p.stages[r.stage].fold;
            History h{};
            for (unsigned b = r.core_begin; b < r.end; ++b)
                if (rng() & 1)
                    set(h, b);
            fault_frame::Frame f;
            Mask x = 0, z = 0;
            for (unsigned q = 0; q < br.data.size(); ++q) {
                unsigned a = rng() & 1, b = rng() & 1;
                x |= Mask(a) << q;
                z |= Mask(b) << q;
                f.flips |= Mask(a) << br.data[q];
                f.linear[br.data[q]] = 4 * b;
            }
            auto incoming = pauli(x, z);
            std::array<Term, 2> a, b;
            auto n = base.branches(fold, h, a);
            for (unsigned k = 0; k < n; ++k)
                a[k].op = compose(a[k].op, incoming, *fold.geometry);
            fault_frame::evaluate(*br.frame, h, f);
            auto m = fault_frame::contract(br, fold, f, fold_blocks::apply(fold.records, h), b);
            check_terms(n, m, a, b);
        }
    auto circuit = clifft::parse(circuit_text);
    std::vector<uint8_t> postselection(circuit.num_detectors, 1);
    const std::array families{fold_recognition::Family(source)};
    auto decision =
        fold_recognition::recognize(circuit, {true, true, false, postselection, {}}, families);
    if (!decision.certificate)
        throw std::runtime_error(decision.reason);
    const auto& certificate = *decision.certificate;
    std::vector<double> base_us, bridge_us;
    std::vector<unsigned> passed;
    size_t measurements = 0, detectors = 0, probes = 0;
    double output_error = 0;
    using Result = clifft::sampling::SamplingSurvivorResult;
    auto timed = [&](auto& executor, unsigned seed, std::vector<double>& times) {
        auto begin = std::chrono::steady_clock::now();
        auto result =
            fold_recognition::sample_with_executor(certificate, shots, seed, true, executor);
        times.push_back(
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin)
                .count() /
            shots);
        return result;
    };
    for (unsigned trial = 0; trial < trials; ++trial) {
        Result a, b;
        unsigned seed = 19331 + trial;
        if (trial % 2) {
            b = timed(bridge, seed, bridge_us);
            a = timed(base, seed, base_us);
        } else {
            a = timed(base, seed, base_us);
            b = timed(bridge, seed, bridge_us);
        }
        if (a.passed_shots != b.passed_shots || a.measurements != b.measurements ||
            a.detectors != b.detectors || a.observables != b.observables ||
            a.exp_vals.size() != b.exp_vals.size())
            throw std::runtime_error("survivor output mismatch");
        for (size_t k = 0; k < a.exp_vals.size(); ++k)
            output_error = std::max(output_error, std::abs(a.exp_vals[k] - b.exp_vals[k]));
        if (output_error > 2e-12)
            throw std::runtime_error("survivor expectation mismatch");
        passed.push_back(a.passed_shots);
        measurements += a.measurements.size();
        detectors += a.detectors.size();
        probes += a.exp_vals.size();
    }
    auto array = [](auto values) {
        std::cout << '[';
        for (size_t k = 0; k < values.size(); ++k)
            std::cout << (k ? "," : "") << values[k];
        std::cout << ']';
    };
    std::cout << std::setprecision(12) << R"({"base_us_per_attempt":)";
    array(base_us);
    std::cout << R"(,"bridge_us_per_attempt":)";
    array(bridge_us);
    std::cout << R"(,"passed":)";
    array(passed);
    std::cout << R"(,"shots_per_trial":)" << shots << R"(,"fixtures":)" << fixtures.size()
              << R"(,"branch_histories":)" << faults.size() + 256 * regions.size()
              << R"(,"incoming_pauli_histories":)" << 128 * regions.size()
              << R"(,"max_fixture_error":)" << max_error << R"(,"max_output_error":)"
              << output_error << R"(,"measurement_values":)" << measurements
              << R"(,"detector_values":)" << detectors << R"(,"probe_values":)" << probes << '}'
              << std::endl;
    return 0;
}
}  // namespace frame_bridge_probe
