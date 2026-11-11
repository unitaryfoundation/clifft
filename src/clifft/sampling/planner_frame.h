#pragma once

#include "clifft/sampling/plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace clifft::sampling::internal {

using PlannerPauli = PauliString;
using PlannerTableau = Tableau;

// Tracks the cumulative relationship between physical HIR coordinates and the
// packed stabilizer coordinates selected while planning. Maintains an unsigned
// transpose of current_to_initial_ so sparse physical Paulis map into current
// coordinates without scanning all 2n generator rows.
class CoordinateFrame {
  public:
    explicit CoordinateFrame(uint32_t num_qubits);

    [[nodiscard]] PlannerPauli to_current(MaskView initial_x, MaskView initial_z,
                                          bool negative = false);
    [[nodiscard]] PlannerPauli to_current(const PlannerPauli& initial);
    [[nodiscard]] PlannerPauli to_initial(const PlannerPauli& current) const;

    [[nodiscard]] const PlannerTableau& current_to_initial() const { return current_to_initial_; }

    void change_basis(const PlannerTableau& new_basis_in_old_coordinates);

    // Apply the planner's structured basis changes without constructing and
    // composing an identity-heavy intermediate tableau. Inputs use the current
    // coordinates, and the resulting coordinate order matches the generic
    // frame constructors below.
    void promote_dormant(const PlannerPauli& promoted, uint32_t active_width,
                         uint32_t dormant_pivot);
    void measure_dormant(const PlannerPauli& measured, uint32_t dormant_pivot);
    void measure_active(const PlannerPauli& measured, uint32_t active_width, uint32_t pivot);

  private:
    void rebuild_transpose();
    void xor_x_generator_delta(uint32_t q, MaskView dx, MaskView dz);
    void xor_z_generator_delta(uint32_t q, MaskView dx, MaskView dz);
    void set_x_generator(uint32_t q, PauliStringView next);
    void set_z_generator(uint32_t q, PauliStringView next);
    void multiply_x_generator(uint32_t q, PauliStringView rhs);
    void multiply_z_generator(uint32_t q, PauliStringView rhs);

    uint32_t num_qubits_;
    uint32_t num_words_;
    PlannerTableau current_to_initial_;
    // Row p holds bit q of z_output(q).z[p], x_output(q).z[p], z_output(q).x[p],
    // and x_output(q).x[p] respectively.
    std::vector<uint64_t> ix_to_cx_;
    std::vector<uint64_t> ix_to_cz_;
    std::vector<uint64_t> iz_to_cx_;
    std::vector<uint64_t> iz_to_cz_;
    std::vector<uint64_t> round_trip_z_;
};

// Tracks how sampled Boolean symbols contribute Pauli corrections in the
// initial physical coordinates using sparse sorted 256-symbol (4 x uint64_t)
// blocks per generator row so narrow frames vectorize across contiguous words
// while wide frames allocate only active symbol blocks. Discarded before
// executable lowering or hot execution.
class SymbolicPauliFrame {
  public:
    SymbolicPauliFrame(uint32_t num_qubits, uint32_t num_symbols);

    void apply(MaskView correction_x, MaskView correction_z, const AffineBool& condition);
    void apply(const PlannerPauli& correction, const AffineBool& condition);
    [[nodiscard]] AffineBool sign_for(MaskView observable_x, MaskView observable_z);
    [[nodiscard]] AffineBool sign_for(const PlannerPauli& observable);

  private:
    static constexpr uint32_t kBlockWords = 4;
    using WordBlock = std::array<uint64_t, kBlockWords>;

    struct BlockEntry {
        uint32_t block = 0;
        WordBlock words{};
    };

    void xor_packed_condition(std::vector<BlockEntry>& row, uint8_t& constant, bool cond_constant);

    uint32_t num_qubits_;
    uint32_t num_symbols_;
    std::vector<std::vector<BlockEntry>> x_rows_;
    std::vector<std::vector<BlockEntry>> z_rows_;
    std::vector<uint8_t> x_constants_;
    std::vector<uint8_t> z_constants_;
    std::vector<WordBlock> scratch_blocks_;
    std::vector<uint64_t> dirty_groups_;
    std::vector<const std::vector<BlockEntry>*> hit_rows_;
    std::vector<BlockEntry> cond_blocks_;
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
