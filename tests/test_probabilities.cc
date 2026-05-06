#include "clifft/backend/backend.h"
#include "clifft/svm/svm.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace clifft;

TEST_CASE("Probabilities missing final tableau is rejected") {
    CompiledModule mod;
    mod.num_qubits = 1;
    mod.peak_rank = 0;
    mod.constant_pool.pauli_masks = PauliMaskArena(1, 0);
    mod.constant_pool.exp_val_masks = PauliMaskArena(1, 0);
    mod.constant_pool.noise_channel_masks = PauliMaskArena(1, 0);

    bool threw = false;
    try {
        std::vector<uint64_t> masks{0};
        (void)probabilities(mod, masks, 1, 1);
    } catch (const std::invalid_argument& ex) {
        threw = true;
        REQUIRE(std::string(ex.what()).find("final Clifford tableau") != std::string::npos);
    }

    REQUIRE(threw);
}
