// Fixed-table coherent folded checks with a logical-state continuation.
#include "folded_sampling_kernel.h"

using namespace folded_study;

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
    std::cout << std::setprecision(17);
    for (size_t shot = 0; shot < traces; ++shot) {
        sampler.run(shot);
        std::cout << "{\"shot\":" << shot << ",\"root_outcomes\":" << sampler.root_outcomes
                  << ",\"syndrome\":" << sampler.syndrome
                  << ",\"log_probability\":" << sampler.log_probability << ",\"logical\":[";
        for (size_t i = 0; i < 2; ++i)
            std::cout << (i ? "," : "") << '[' << sampler.logical[i].real() << ','
                      << sampler.logical[i].imag() << ']';
        std::cout << "]}\n";
    }
    for (size_t i = 0; i < 16; ++i)
        sampler.run(i);
    std::array<double, 3> seconds;
    uint64_t checksum = 0;
    for (auto& elapsed : seconds) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t shot = 0; shot < shots; ++shot) {
            sampler.run(shot);
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
