// Fixed-plan CSS syndrome contraction microbenchmark, not a full MSC executor.
#include "msc_factor_kernel.h"
using namespace msc_factor;
int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "usage: msc_factor_native PLAN factor|fourier|sample SHOTS SEED");
        std::string mode(argv[2]);
        require(mode == "factor" || mode == "fourier" || mode == "sample");
        Reader r(argv[1]);
        Worker worker(r, mode == "fourier");
        unsigned shots = std::stoul(argv[3]);
        require(shots > 0);
        worker.rng.seed(std::stoull(argv[4]));
        double checksum = 0;
        auto start = std::chrono::steady_clock::now();
        for (unsigned k = 0; k < shots; ++k) {
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            auto before = allocations;
#endif
            const auto& data = worker.cases[k % worker.cases.size()];
            Sample sample = mode == "fourier" ? worker.fourier(data) : worker.factor(data);
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            assert(allocations == before);
#endif
            checksum += sample.probability + sample.xs + sample.zs;
            if (mode == "sample")
                std::cout << std::setprecision(17) << "{\"case\":" << k % worker.cases.size()
                          << ",\"xs\":" << sample.xs << ",\"zs\":" << sample.zs
                          << ",\"probability\":" << sample.probability << "}\n";
        }
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (mode != "sample")
            std::cout << std::setprecision(17) << "{\"shots\":" << shots
                      << ",\"seconds\":" << seconds
                      << ",\"microseconds_per_sample\":" << seconds * 1e6 / shots
                      << ",\"coefficient_scratch_bytes\":"
                      << worker.scratch.size() * sizeof(C) + worker.tensor.size() * sizeof(C)
                      << ",\"probability_scratch_bytes\":" << worker.weights.size() * sizeof(double)
                      << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
