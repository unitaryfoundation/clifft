// Fixed-table coherent folded checks with a logical-state continuation.
#include "folded_sampling_kernel.h"

using namespace folded_study;

std::vector<Case> read_cases(const std::filesystem::path& directory, size_t leaves,
                             FoldedSampler& sampler) {
    std::ifstream input(directory / "inputs.txt");
    Reader reader{input};
    const auto count = reader.size();
    if (!count || count > 128)
        throw std::invalid_argument("invalid folded input case count");
    std::vector<Case> cases(count);
    for (auto& instance : cases) {
        double norm = 0;
        for (auto& choice : instance) {
            for (auto& terms : choice)
                for (auto& term : terms) {
                    const auto re = reader.number(), im = reader.number();
                    term.coefficient = {re, im};
                    if (std::abs(term.coefficient) > 1)
                        throw std::invalid_argument("invalid folded path coefficient");
                    term.local.resize(4 * leaves);
                    for (auto& value : term.local) {
                        const auto x = reader.number(), y = reader.number();
                        value = {x, y};
                        if (std::abs(std::norm(value) - 1) > 1e-9)
                            throw std::invalid_argument("folded local factor is not a phase");
                    }
                }
            norm += sampler.probability(choice);
        }
        if (std::abs(norm - 1) > 1e-9)
            throw std::invalid_argument("folded input is not normalized");
    }
    return cases;
}

int main(int argc, char** argv) {
    if (argc != 5)
        throw std::invalid_argument("usage: sample_folded_checks bundle shots traces seed");
    const std::filesystem::path directory(argv[1]);
    const auto shots = std::stoull(argv[2]), traces = std::stoull(argv[3]);
    if (!shots)
        throw std::invalid_argument("shots must be positive");
    std::ifstream dimensions(directory / "dimensions.txt");
    Reader reader{dimensions};
    const auto rank = reader.size(), leaves = reader.size(), characters = reader.size();
    FoldedSampler sampler(directory, std::stoull(argv[4]), rank, leaves, characters);
    const auto cases = read_cases(directory, leaves, sampler);
    std::cout << std::setprecision(17);
    for (size_t shot = 0; shot < traces; ++shot) {
        sampler.run(cases[shot % cases.size()]);
        std::cout << "{\"shot\":" << shot << ",\"root_outcomes\":" << sampler.root_outcomes
                  << ",\"syndrome\":" << sampler.syndrome
                  << ",\"log_probability\":" << sampler.log_probability << ",\"logical\":[";
        for (size_t i = 0; i < 2; ++i)
            std::cout << (i ? "," : "") << '[' << sampler.logical[i].real() << ','
                      << sampler.logical[i].imag() << ']';
        std::cout << "]}\n";
    }
    for (size_t i = 0; i < 16; ++i)
        sampler.run(cases[i % cases.size()]);
    std::array<double, 3> seconds;
    uint64_t checksum = 0;
    for (auto& elapsed : seconds) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t shot = 0; shot < shots; ++shot) {
            sampler.run(cases[shot % cases.size()]);
            checksum += sampler.syndrome + sampler.root_outcomes;
        }
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    std::cout << "{\"seconds\":[" << seconds[0] << ',' << seconds[1] << ',' << seconds[2]
              << "],\"shots_per_batch\":" << shots
              << ",\"max_normalization_error\":" << sampler.max_normalization_error
              << ",\"max_continuation_error\":" << sampler.max_continuation_error
              << ",\"checksum\":" << checksum << "}\n";
}
