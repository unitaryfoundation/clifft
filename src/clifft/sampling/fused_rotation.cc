#include "clifft/sampling/fused_rotation.h"

#include "clifft/sampling/direct_rotation_simd.h"
#include "clifft/sampling/indexing.h"
#include "clifft/sampling/kernels.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <complex>

namespace clifft::sampling {

namespace {

// A dense matrix requires more arithmetic per coefficient than one Pauli
// rotation. Three rotations remove enough full-state sweeps to amortize that
// extra work.
constexpr size_t kMinFusedRotationActions = 3;

// Z parities evaluated on an orbit representative select its precomposed
// matrix. Each independent parity condition doubles the table, so this cap
// bounds a descriptor at 32 matrices while covering the measured QV workloads.
constexpr uint32_t kMaxFusedRotationSelectors = 5;

// Sign variants grow the prepared matrix table exponentially, so keep both
// the sign rank and the minimum amount of eliminated work bounded.
constexpr uint32_t kMaxFusedRotationSigns = 2;
constexpr size_t kMinDynamicFusedRotationActions = 8;
constexpr uint32_t kMinDynamicFusedRotationPivot =
    static_cast<uint32_t>(std::countr_zero(kAvx512DoubleLanes));

// A row-reduced basis for a vector subspace of GF(2)^64, where each bit is one
// binary coordinate.
class BinaryBasis {
  public:
    [[nodiscard]] bool contains(uint64_t value) const {
        for (int pivot = 63; pivot >= 0; --pivot) {
            const uint64_t bit = uint64_t{1} << pivot;
            if ((value & bit) != 0 && rows_[pivot] != 0) {
                value ^= rows_[pivot];
            }
        }
        return value == 0;
    }

    // Extends the span with value when it is independent. A dependent value
    // succeeds without changing the basis; false means the rank cap prevented
    // adding an independent value.
    [[nodiscard]] bool insert(uint64_t value, uint32_t max_rank = 64) {
        uint64_t reduced = value;
        for (int pivot = 63; pivot >= 0; --pivot) {
            const uint64_t bit = uint64_t{1} << pivot;
            if ((reduced & bit) != 0 && rows_[pivot] != 0) {
                reduced ^= rows_[pivot];
            }
        }
        if (reduced == 0) {
            return true;
        }
        if (rank_ == max_rank) {
            return false;
        }
        const uint32_t pivot = 63U - static_cast<uint32_t>(std::countl_zero(reduced));
        const uint64_t bit = uint64_t{1} << pivot;
        for (uint64_t& row : rows_) {
            if ((row & bit) != 0) {
                row ^= reduced;
            }
        }
        rows_[pivot] = reduced;
        ++rank_;
        return true;
    }

    // Returns the dimension of the represented subspace.
    [[nodiscard]] uint32_t rank() const { return rank_; }

    // Returns the reduced basis vectors in ascending pivot-bit order.
    [[nodiscard]] std::vector<uint64_t> rows() const {
        std::vector<uint64_t> result;
        result.reserve(rank_);
        for (uint64_t row : rows_) {
            if (row != 0) {
                result.push_back(row);
            }
        }
        return result;
    }

  private:
    std::array<uint64_t, 64> rows_{};
    uint32_t rank_ = 0;
};

// Triangular basis for affine expressions with their constant terms removed.
// Coordinates let lowering precompute one matrix for each independent sign
// assignment while execution evaluates only the basis expressions.
class AffineBasis {
  public:
    [[nodiscard]] bool contains(const AffineBool& value) const {
        AffineBool reduced(false, value.terms());
        while (!reduced.terms().empty()) {
            const uint32_t pivot = index(reduced.terms().back());
            if (pivot >= rows_.size() || !rows_[pivot].has_value()) {
                return false;
            }
            reduced ^= rows_[pivot]->expression;
        }
        return true;
    }

    [[nodiscard]] bool insert(const AffineBool& value, uint32_t max_rank) {
        AffineBool reduced(false, value.terms());
        while (!reduced.terms().empty()) {
            const uint32_t pivot = index(reduced.terms().back());
            if (pivot < rows_.size() && rows_[pivot].has_value()) {
                reduced ^= rows_[pivot]->expression;
                continue;
            }
            if (expressions_.size() == max_rank) {
                return false;
            }
            if (pivot >= rows_.size()) {
                rows_.resize(static_cast<size_t>(pivot) + 1);
            }
            const uint32_t coordinate = static_cast<uint32_t>(expressions_.size());
            expressions_.push_back(reduced);
            rows_[pivot] = Row{std::move(reduced), coordinate};
            return true;
        }
        return true;
    }

    [[nodiscard]] uint32_t coordinates(const AffineBool& value) const {
        AffineBool reduced(false, value.terms());
        uint32_t result = 0;
        while (!reduced.terms().empty()) {
            const uint32_t pivot = index(reduced.terms().back());
            assert(pivot < rows_.size() && rows_[pivot].has_value() &&
                   "expression must belong to the prepared affine basis");
            result ^= uint32_t{1} << rows_[pivot]->coordinate;
            reduced ^= rows_[pivot]->expression;
        }
        return result;
    }

    [[nodiscard]] const std::vector<AffineBool>& expressions() const { return expressions_; }

  private:
    struct Row {
        AffineBool expression;
        uint32_t coordinate = 0;
    };

    std::vector<std::optional<Row>> rows_;
    std::vector<AffineBool> expressions_;
};

// Expresses value in the supplied reduced basis, with bit i of the result
// giving the coefficient of rows[i].
uint32_t basis_coordinates(uint64_t value, std::span<const uint64_t> rows) {
    uint32_t coordinates = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint64_t pivot = std::bit_floor(rows[i]);
        if ((value & pivot) != 0) {
            value ^= rows[i];
            coordinates |= uint32_t{1} << i;
        }
    }
    assert(value == 0 && "value must belong to the prepared binary span");
    return coordinates;
}

// Replaces matrix with left * matrix. Left multiplication preserves the plan's
// execution order as successive rotation matrices are composed.
void multiply_matrix_left(std::span<std::complex<double>> matrix,
                          std::span<const std::complex<double>> left, size_t dimension) {
    std::array<std::complex<double>, 16> result{};
    for (size_t row = 0; row < dimension; ++row) {
        for (size_t column = 0; column < dimension; ++column) {
            for (size_t inner = 0; inner < dimension; ++inner) {
                result[row * dimension + column] +=
                    left[row * dimension + inner] * matrix[inner * dimension + column];
            }
        }
    }
    std::copy_n(result.begin(), dimension * dimension, matrix.begin());
}

std::optional<PreparedFusedRotation> prepare_fused_rotation(std::span<const PlannedAction> actions,
                                                            const BinaryBasis& orbit_basis) {
    assert(actions.size() >= kMinFusedRotationActions &&
           "fused rotation requires at least three actions");
    const uint32_t active_width = actions.front().active_before;
    const std::vector<uint64_t> orbit_rows = orbit_basis.rows();

    // A representative is the unique state in an orbit whose basis pivot bits
    // are zero. Only the remaining Z bits affect its Pauli phase.
    uint64_t orbit_pivots = 0;
    for (uint64_t row : orbit_rows) {
        orbit_pivots |= std::bit_floor(row);
    }

    // Reduce those remaining Z masks to the independent parity conditions that
    // can select different matrices for different representatives.
    BinaryBasis selector_basis;
    for (const PlannedAction& planned : actions) {
        const auto& rotation = std::get<RotateActivePauli>(planned.action);
        assert(planned.active_before == active_width && planned.active_after == active_width &&
               rotation.sign.terms().empty() && !rotation.pauli.is_identity() &&
               "fused rotation input must be a constant-sign nonidentity run");
        if (!selector_basis.insert(rotation.pauli.z & ~orbit_pivots, kMaxFusedRotationSelectors)) {
            return std::nullopt;
        }
    }
    const std::vector<uint64_t> selector_rows = selector_basis.rows();

    PreparedFusedRotation result;
    result.active_width = active_width;
    result.orbit_rank = orbit_basis.rank();
    for (size_t i = 0; i < orbit_rows.size(); ++i) {
        result.orbit_masks[i] = orbit_rows[i];
        result.orbit_pivots[i] =
            static_cast<uint32_t>(std::countr_zero(std::bit_floor(orbit_rows[i])));
    }
    result.selector_masks = selector_rows;

    // Precompose one dense unitary for every possible assignment of the
    // independent representative parities.
    const size_t dimension = size_t{1} << result.orbit_rank;
    const size_t matrix_size = dimension * dimension;
    const size_t num_variants = size_t{1} << selector_rows.size();
    result.matrices.resize(num_variants * matrix_size);
    for (size_t variant = 0; variant < num_variants; ++variant) {
        std::span<std::complex<double>> matrix(result.matrices.data() + variant * matrix_size,
                                               matrix_size);
        for (size_t diagonal = 0; diagonal < dimension; ++diagonal) {
            matrix[diagonal * dimension + diagonal] = 1.0;
        }

        for (const PlannedAction& planned : actions) {
            const auto& rotation = std::get<RotateActivePauli>(planned.action);
            const PreparedRotation prepared =
                prepare_rotation(rotation.pauli, active_width, rotation.half_turns);
            const uint32_t x_coordinates = basis_coordinates(rotation.pauli.x, orbit_rows);
            const uint32_t selector_coordinates =
                basis_coordinates(rotation.pauli.z & ~orbit_pivots, selector_rows);

            // local_z records how the Pauli phase changes between members of
            // one orbit; representative_phase supplies the shared offset.
            uint32_t local_z = 0;
            for (size_t i = 0; i < orbit_rows.size(); ++i) {
                local_z |=
                    static_cast<uint32_t>(std::popcount(orbit_rows[i] & rotation.pauli.z) & 1U)
                    << i;
            }
            const bool representative_phase =
                (std::popcount(static_cast<uint32_t>(variant) & selector_coordinates) & 1U) != 0;
            const double sine = rotation.sign.constant() ? -prepared.sine : prepared.sine;

            std::array<std::complex<double>, 16> unitary{};
            for (size_t column = 0; column < dimension; ++column) {
                unitary[column * dimension + column] += prepared.cosine;
                const bool odd_phase =
                    representative_phase !=
                    ((std::popcount(static_cast<uint32_t>(column) & local_z) & 1U) != 0);
                const std::complex<double> phase =
                    odd_phase ? -prepared.pauli.even_phase : prepared.pauli.even_phase;
                unitary[(column ^ x_coordinates) * dimension + column] +=
                    std::complex<double>{0.0, -sine} * phase;
            }
            multiply_matrix_left(matrix,
                                 std::span<const std::complex<double>>(unitary).first(matrix_size),
                                 dimension);
        }
    }
    return result;
}

// Each orbit is a one-, two-, or four-dimensional coefficient subspace. Gather
// its coefficients, apply the selected dense unitary, and scatter them back.
template <size_t Dimension>
void apply_fused_rotation_orbits(State& state, const PreparedFusedRotation& rotation) noexcept {
    static_assert(Dimension == 1 || Dimension == 2 || Dimension == 4);
    const uint64_t orbit_count = state.size() / Dimension;
    const size_t matrix_size = Dimension * Dimension;
    assert(rotation.matrices.size() ==
               (size_t{1} << rotation.selector_masks.size()) * matrix_size &&
           "fused rotation matrix table must cover every selector value");

    double* const real = state.real_data();
    double* const imag = state.imag_data();
    for (uint64_t packed = 0; packed < orbit_count; ++packed) {
        // Reinserting zero pivot bits enumerates exactly one representative
        // from every orbit.
        uint64_t representative = packed;
        if constexpr (Dimension >= 2) {
            representative = insert_zero_bit(representative, rotation.orbit_pivots[0]);
        }
        if constexpr (Dimension == 4) {
            representative = insert_zero_bit(representative, rotation.orbit_pivots[1]);
        }

        const size_t selector = selector_index(representative, rotation.selector_masks);
        const std::complex<double>* const matrix =
            rotation.matrices.data() + selector * matrix_size;

        // XOR combinations of the orbit basis enumerate every coefficient in
        // this representative's subspace.
        std::array<uint64_t, Dimension> indices{};
        std::array<double, Dimension> input_real{};
        std::array<double, Dimension> input_imag{};
        for (size_t column = 0; column < Dimension; ++column) {
            uint64_t index = representative;
            if constexpr (Dimension >= 2) {
                if ((column & 1U) != 0) {
                    index ^= rotation.orbit_masks[0];
                }
            }
            if constexpr (Dimension == 4) {
                if ((column & 2U) != 0) {
                    index ^= rotation.orbit_masks[1];
                }
            }
            indices[column] = index;
            input_real[column] = real[index];
            input_imag[column] = imag[index];
        }

        for (size_t row = 0; row < Dimension; ++row) {
            double output_real = 0.0;
            double output_imag = 0.0;
            for (size_t column = 0; column < Dimension; ++column) {
                const std::complex<double> weight = matrix[row * Dimension + column];
                output_real +=
                    weight.real() * input_real[column] - weight.imag() * input_imag[column];
                output_imag +=
                    weight.real() * input_imag[column] + weight.imag() * input_real[column];
            }
            real[indices[row]] = output_real;
            imag[indices[row]] = output_imag;
        }
    }
}

}  // namespace

FusedRotationRun prepare_fused_rotation_run(std::span<const PlannedAction> actions) {
    FusedRotationRun result;
    if (actions.empty()) {
        return result;
    }
    const PlannedAction& first = actions.front();
    const auto* first_rotation = std::get_if<RotateActivePauli>(&first.action);

    // Dynamic signs require a per-shot choice, while identity rotations update
    // the global scalar. Neither can be folded into a fixed coefficient matrix.
    if (first_rotation == nullptr || !first_rotation->sign.terms().empty() ||
        first_rotation->pauli.is_identity()) {
        return result;
    }

    // Extend the maximal same-width prefix while its X masks span at most two
    // dimensions. Every additional dimension doubles the matrix side length.
    BinaryBasis orbit_basis;
    while (result.action_count < actions.size()) {
        const PlannedAction& candidate = actions[result.action_count];
        const auto* rotation = std::get_if<RotateActivePauli>(&candidate.action);
        if (rotation == nullptr || !rotation->sign.terms().empty() ||
            rotation->pauli.is_identity() || candidate.active_before != first.active_before) {
            break;
        }
        BinaryBasis next_basis = orbit_basis;
        if (!next_basis.insert(rotation->pauli.x, 2)) {
            break;
        }
        orbit_basis = next_basis;
        ++result.action_count;
    }

    // If selector growth rejects the descriptor, action_count still lets the
    // caller lower this entire eligible prefix as individual rotations.
    if (result.action_count >= kMinFusedRotationActions) {
        result.rotation = prepare_fused_rotation(actions.first(result.action_count), orbit_basis);
    }
    return result;
}

DynamicFusedRotationRun prepare_dynamic_fused_rotation_run(std::span<const PlannedAction> actions) {
    DynamicFusedRotationRun result;
    if (actions.size() < kMinDynamicFusedRotationActions) {
        return result;
    }
    const PlannedAction& first = actions.front();
    const auto* first_rotation = std::get_if<RotateActivePauli>(&first.action);
    if (first_rotation == nullptr || first_rotation->pauli.is_identity()) {
        return result;
    }

    // Reject ordinary high-rank runs without constructing affine expressions
    // or clearing the full binary bases. Any eligible run must keep the first
    // minimum-length prefix within the same two-dimensional X-mask span.
    std::array<uint64_t, 2> prefix_basis{};
    size_t prefix_rank = 0;
    for (const PlannedAction& candidate : actions.first(kMinDynamicFusedRotationActions)) {
        const auto* rotation = std::get_if<RotateActivePauli>(&candidate.action);
        if (rotation == nullptr || rotation->pauli.is_identity() ||
            candidate.active_before != first.active_before) {
            return result;
        }
        uint64_t reduced = rotation->pauli.x;
        for (size_t i = 0; i < prefix_rank; ++i) {
            reduced = std::min(reduced, reduced ^ prefix_basis[i]);
        }
        if (reduced != 0) {
            if (prefix_rank == prefix_basis.size()) {
                return result;
            }
            prefix_basis[prefix_rank++] = reduced;
        }
    }

    BinaryBasis orbit_basis;
    AffineBasis sign_basis;
    bool has_dynamic_sign = false;
    while (result.action_count < actions.size()) {
        const PlannedAction& candidate = actions[result.action_count];
        const auto* rotation = std::get_if<RotateActivePauli>(&candidate.action);
        if (rotation == nullptr || rotation->pauli.is_identity() ||
            candidate.active_before != first.active_before) {
            break;
        }
        if ((orbit_basis.rank() == 2 && !orbit_basis.contains(rotation->pauli.x)) ||
            (sign_basis.expressions().size() == kMaxFusedRotationSigns &&
             !sign_basis.contains(rotation->sign))) {
            break;
        }
        [[maybe_unused]] const bool orbit_inserted = orbit_basis.insert(rotation->pauli.x, 2);
        [[maybe_unused]] const bool sign_inserted =
            sign_basis.insert(rotation->sign, kMaxFusedRotationSigns);
        assert(orbit_inserted && sign_inserted && "prechecked bases must accept the rotation");
        has_dynamic_sign |= !rotation->sign.terms().empty();
        ++result.action_count;
    }

    const std::vector<uint64_t> orbit_rows = orbit_basis.rows();
    if (result.action_count < kMinDynamicFusedRotationActions || !has_dynamic_sign ||
        orbit_rows.size() != 2 ||
        std::countr_zero(std::bit_floor(orbit_rows[0])) < kMinDynamicFusedRotationPivot) {
        return result;
    }

    std::vector<uint32_t> sign_coordinates;
    sign_coordinates.reserve(result.action_count);
    for (size_t i = 0; i < result.action_count; ++i) {
        const auto& rotation = std::get<RotateActivePauli>(actions[i].action);
        sign_coordinates.push_back(sign_basis.coordinates(rotation.sign));
    }

    PreparedDynamicFusedRotation prepared;
    prepared.sign_basis = sign_basis.expressions();
    const size_t num_sign_variants = size_t{1} << prepared.sign_basis.size();
    prepared.variants.reserve(num_sign_variants);
    std::vector<PlannedAction> variant_actions(actions.begin(),
                                               actions.begin() + result.action_count);
    for (size_t variant = 0; variant < num_sign_variants; ++variant) {
        for (size_t i = 0; i < result.action_count; ++i) {
            const auto& source = std::get<RotateActivePauli>(actions[i].action);
            auto& destination = std::get<RotateActivePauli>(variant_actions[i].action);
            const bool sign =
                source.sign.constant() !=
                ((std::popcount(static_cast<uint32_t>(variant) & sign_coordinates[i]) & 1U) != 0);
            destination.sign = AffineBool(sign);
        }
        std::optional<PreparedFusedRotation> rotation =
            prepare_fused_rotation(variant_actions, orbit_basis);
        if (!rotation.has_value()) {
            return result;
        }
        prepared.variants.push_back(std::move(*rotation));
    }
    result.rotation = std::move(prepared);
    return result;
}

void apply_fused_rotation(State& state, const PreparedFusedRotation& rotation) noexcept {
    assert(state.active_width() == rotation.active_width &&
           "fused rotation width must match the active state");
    switch (rotation.orbit_rank) {
        case 0:
            apply_fused_rotation_orbits<1>(state, rotation);
            return;
        case 1:
            apply_fused_rotation_orbits<2>(state, rotation);
            return;
        case 2:
            apply_fused_rotation_orbits<4>(state, rotation);
            return;
        default:
            assert(false && "fused rotation orbit rank must be at most two");
            return;
    }
}

}  // namespace clifft::sampling
