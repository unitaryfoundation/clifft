#pragma once

#include "clifft/sampling/plan.h"

#include "stim.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling::internal {

using PlannerPauli = stim::PauliString<kStimWidth>;
using PlannerTableau = stim::Tableau<kStimWidth>;

// Tracks the cumulative relationship between physical HIR coordinates and the
// packed stabilizer coordinates selected while planning.
class CoordinateFrame {
  public:
    explicit CoordinateFrame(uint32_t num_qubits);

    [[nodiscard]] PlannerPauli to_current(const PlannerPauli& initial);
    [[nodiscard]] PlannerPauli to_initial(const PlannerPauli& current) const;

    void change_basis(const PlannerTableau& new_basis_in_old_coordinates);

    [[nodiscard]] bool has_cached_inverse_for_testing() const {
        return initial_to_current_.has_value();
    }

  private:
    // The selected generators are enough to recover either direction. The
    // inverse is cached only when repeated reverse lookups can amortize it.
    PlannerTableau current_to_initial_;
    std::optional<PlannerTableau> initial_to_current_;
    uint64_t direct_reverse_lookups_ = 0;
    std::vector<size_t> indices_;
};

// Tracks how sampled Boolean symbols contribute Pauli corrections in the
// initial physical coordinates. It uses O(num_qubits * num_symbols / 64) words
// and is discarded before executable lowering or hot execution.
class SymbolicPauliFrame {
  public:
    SymbolicPauliFrame(uint32_t num_qubits, uint32_t num_symbols);

    void apply(const PlannerPauli& correction, const AffineBool& condition);
    [[nodiscard]] AffineBool sign_for(const PlannerPauli& observable);

    // Reports the packed row and scratch storage owned by the frame, excluding
    // allocator and vector object overhead.
    [[nodiscard]] static size_t estimated_workspace_bytes(uint32_t num_qubits,
                                                          uint32_t num_symbols);

  private:
    [[nodiscard]] std::span<uint64_t> x_row(uint32_t q);
    [[nodiscard]] std::span<const uint64_t> x_row(uint32_t q) const;
    [[nodiscard]] std::span<uint64_t> z_row(uint32_t q);
    [[nodiscard]] std::span<const uint64_t> z_row(uint32_t q) const;

    static void xor_row(std::span<uint64_t> destination, std::span<const uint64_t> source);
    void xor_condition(std::span<uint64_t> row, uint8_t& constant,
                       const AffineBool& condition) const;

    uint32_t num_qubits_;
    uint32_t num_symbols_;
    size_t words_per_row_;
    std::vector<uint64_t> x_terms_;
    std::vector<uint64_t> z_terms_;
    std::vector<uint8_t> x_constants_;
    std::vector<uint8_t> z_constants_;
    std::vector<uint64_t> scratch_;
};

// Each tableau maps newly selected planner coordinates into the previous
// coordinate basis.
[[nodiscard]] PlannerTableau dormant_promotion_frame(const PlannerPauli& promoted,
                                                     uint32_t active_width, uint32_t dormant_pivot);
[[nodiscard]] PlannerTableau dormant_measurement_frame(const PlannerPauli& measured,
                                                       uint32_t dormant_pivot);
[[nodiscard]] PlannerTableau active_measurement_frame(const PlannerPauli& measured,
                                                      uint32_t active_width, uint32_t pivot);

}  // namespace clifft::sampling::internal
