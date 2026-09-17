#pragma once

#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/executable_plan.h"
#include "clifft/sampling/planner.h"

#include "fold_recognition.h"

#include <fstream>
#include <iterator>

namespace fold_recognition {
template <class T>
void json_array(std::span<const T> values) {
    std::cout << '[';
    for (size_t k = 0; k < values.size(); ++k) {
        if (k)
            std::cout << ',';
        std::cout << +values[k];
    }
    std::cout << ']';
}

inline int run(std::span<const FamilySource> sources, int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument(
                "usage: recognizer CIRCUIT SHOTS SEED KEEP_RECORDS REPEATS REQUEST");
        auto number = [](const char* value) {
            size_t used;
            auto result = std::stoull(value, &used);
            if (used != std::string_view(value).size())
                throw std::invalid_argument("invalid integer");
            return result;
        };
        auto shots = number(argv[2]), seed = number(argv[3]), keep = number(argv[4]);
        auto repeats = number(argv[5]);
        std::string_view mode(argv[6]);
        if (shots > 1000000 || keep > 1 || !repeats || repeats > 1000000 ||
            (mode != "ordinary" && mode != "fixed" && mode != "records" && mode != "partial" &&
             mode != "nonzero" && mode != "fallback"))
            throw std::invalid_argument("unsupported experiment configuration");
        using Clock = std::chrono::steady_clock;
        auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        auto begin = Clock::now();
        std::vector<Family> families;
        families.reserve(sources.size());
        for (const auto& source : sources)
            families.emplace_back(source);
        auto catalog_end = Clock::now();
        std::ifstream input(argv[1]);
        if (!input)
            throw std::invalid_argument("cannot read input circuit");
        std::string text((std::istreambuf_iterator<char>(input)), {});
        auto parse_begin = Clock::now();
        auto circuit = clifft::parse(text);
        auto parse_end = Clock::now();
        std::vector<uint8_t> postselection(circuit.num_detectors, 1);
        std::vector<uint8_t> expected(circuit.num_detectors);
        if (mode == "partial" && !postselection.empty())
            postselection[0] = 0;
        if (mode == "nonzero" && !expected.empty())
            expected[0] = 1;
        Request request{mode != "records", bool(keep), mode == "fixed", postselection, expected};
        auto recognition_begin = Clock::now();
        Decision decision;
        for (size_t k = 0; k < repeats; ++k)
            decision = recognize(circuit, request, families);
        auto recognition_end = Clock::now();
        std::cout << std::setprecision(17) << "{\"catalog_ms\":" << milliseconds(begin, catalog_end)
                  << ",\"parse_ms\":" << milliseconds(parse_begin, parse_end)
                  << ",\"recognize_us\":"
                  << milliseconds(recognition_begin, recognition_end) * 1000 / repeats
                  << ",\"eligible\":" << bool(decision.certificate)
                  << ",\"reason\":" << std::quoted(decision.reason)
                  << ",\"source_line\":" << decision.source_line;
        if (decision.certificate) {
            const auto& certificate = *decision.certificate;
            std::cout << ",\"distance\":" << certificate.family->source.distance
                      << ",\"prefix_gates\":" << certificate.erased_prefix_gates
                      << ",\"suffix_gates\":" << certificate.suffix_gates
                      << ",\"measurements_per_row\":" << circuit.num_measurements
                      << ",\"detectors_per_row\":" << circuit.num_detectors << ",\"probes\":[";
            for (size_t k = 0; k < certificate.probes.size(); ++k) {
                if (k)
                    std::cout << ',';
                const auto& probe = certificate.probes[k];
                std::cout << '[' << probe.component << ',' << probe.sign << ']';
            }
            std::cout << "],\"probabilities\":";
            json_array<double>(certificate.probabilities);
            std::cout << ",\"qubits\":";
            json_array<uint32_t>(certificate.physical_qubits);
            auto sample_begin = Clock::now();
            auto result = sample(certificate, static_cast<unsigned>(shots), seed, bool(keep));
            auto sample_end = Clock::now();
            std::cout << ",\"sample_ms\":" << milliseconds(sample_begin, sample_end)
                      << ",\"total_shots\":" << result.total_shots
                      << ",\"passed_shots\":" << result.passed_shots
                      << ",\"measurement_values\":" << result.measurements.size()
                      << ",\"detector_values\":" << result.detectors.size()
                      << ",\"probe_values\":" << result.exp_vals.size();
            // Bounded diagnostics keep timing runs from emitting large result matrices.
            if (shots <= 1000) {
                std::cout << ",\"measurements\":";
                json_array<uint8_t>(result.measurements);
                std::cout << ",\"detectors\":";
                json_array<uint8_t>(result.detectors);
                std::cout << ",\"exp_vals\":";
                json_array<double>(result.exp_vals);
            }
        } else if (mode == "fallback") {
            auto fallback_begin = Clock::now();
            auto hir = clifft::trace(circuit);
            auto passes = clifft::default_hir_pass_manager();
            passes.run(hir);
            clifft::sampling::SamplingPlanOptions options;
            options.postselection_mask = postselection;
            auto plan = clifft::sampling::plan_sampling(hir, options);
            clifft::sampling::ExecutablePlan executable(plan);
            std::cout << ",\"fallback_compile_ms\":" << milliseconds(fallback_begin, Clock::now())
                      << ",\"fallback_peak_width\":" << plan.peak_active_width;
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
}  // namespace fold_recognition
