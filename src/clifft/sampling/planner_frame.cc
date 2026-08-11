#include "clifft/sampling/planner_frame.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace clifft::sampling::internal {

namespace {

std::vector<size_t> identity_indices(size_t size) {
    std::vector<size_t> result(size);
    std::iota(result.begin(), result.end(), size_t{0});
    return result;
}

PlannerPauli single_x(uint32_t num_qubits, uint32_t q) {
    PlannerPauli result(num_qubits);
    result.xs[q] = true;
    return result;
}

PlannerPauli single_z(uint32_t num_qubits, uint32_t q) {
    PlannerPauli result(num_qubits);
    result.zs[q] = true;
    return result;
}

PlannerPauli positive_body_xor(const PlannerPauli& left, const PlannerPauli& right) {
    // Callers only combine canonical generator rows whose nonidentity supports
    // are disjoint. Overlapping supports could contribute a phase that this
    // body-only helper intentionally does not compute.
    PlannerPauli result(left);
    result.xs ^= right.xs;
    result.zs ^= right.zs;
    result.sign = false;
    return result;
}

bool anticommutes(const PlannerPauli& left, const PlannerPauli& right) {
    return !left.ref().commutes(right.ref());
}

bool has_x_below(const PlannerPauli& pauli, uint32_t end) {
    for (uint32_t q = 0; q < end; ++q) {
        if (pauli.xs[q]) {
            return true;
        }
    }
    return false;
}

void evaluate_generator_into(stim::PauliStringRef<kStimWidth> destination,
                             const stim::PauliStringRef<kStimWidth>& generator,
                             const PlannerTableau& map) {
    // Planner basis changes are sparse. Writing directly into tableau-owned
    // rows avoids the owning Pauli allocation that Stim's generic composition
    // performs for every X and Z generator.
    destination.xs.clear();
    destination.zs.clear();
    destination.sign = generator.sign;
    generator.for_each_active_pauli([&](size_t q) {
        const bool x = generator.xs[q];
        const bool z = generator.zs[q];
        if (x && z) {
            uint8_t log_i = 1;
            log_i += destination.inplace_right_mul_returning_log_i_scalar(map.xs[q]);
            log_i += destination.inplace_right_mul_returning_log_i_scalar(map.zs[q]);
            assert((log_i & 1) == 0);
            destination.sign ^= (log_i & 2) != 0;
        } else if (x) {
            destination *= map.xs[q];
        } else {
            destination *= map.zs[q];
        }
    });
}

size_t validated_words_per_row(uint32_t num_qubits, uint32_t num_symbols) {
    static_cast<void>(SymbolicPauliFrame::estimated_workspace_bytes(num_qubits, num_symbols));
    return (static_cast<size_t>(num_symbols) + 63) / 64;
}

}  // namespace

CoordinateFrame::CoordinateFrame(uint32_t num_qubits)
    : current_to_initial_(num_qubits), indices_(identity_indices(num_qubits)) {}

PlannerPauli CoordinateFrame::to_current(const PlannerPauli& initial) {
    if (initial_to_current_.has_value()) {
        return initial_to_current_->scatter_eval(initial.ref(), indices_);
    }

    // Inverting an n-qubit tableau evaluates 2n generator rows. Use the same
    // number of direct lookups as a conservative structural break-even before
    // materializing the inverse for a lookup-heavy basis interval.
    const uint64_t direct_lookup_limit =
        std::max<uint64_t>(1, 2 * static_cast<uint64_t>(initial.num_qubits));
    if (direct_reverse_lookups_ >= direct_lookup_limit) {
        initial_to_current_.emplace(current_to_initial_.inverse());
        return initial_to_current_->scatter_eval(initial.ref(), indices_);
    }
    ++direct_reverse_lookups_;

    PlannerPauli current(initial.num_qubits);
    for (uint32_t q = 0; q < initial.num_qubits; ++q) {
        // Coordinates in a symplectic basis are selected by commuting with its
        // opposite generator: Z generators select X and vice versa.
        current.xs[q] = !initial.ref().commutes(current_to_initial_.zs[q]);
        current.zs[q] = !initial.ref().commutes(current_to_initial_.xs[q]);
    }

    // Generator signs affect the sign, but not the symplectic coordinates. A
    // forward evaluation recovers that phase without materializing an inverse.
    const PlannerPauli round_trip = current_to_initial_.scatter_eval(current.ref(), indices_);
    assert(round_trip.xs == initial.xs);
    assert(round_trip.zs == initial.zs);
    current.sign = static_cast<bool>(round_trip.sign) ^ static_cast<bool>(initial.sign);
    return current;
}

PlannerPauli CoordinateFrame::to_initial(const PlannerPauli& current) const {
    return current_to_initial_.scatter_eval(current.ref(), indices_);
}

void CoordinateFrame::change_basis(const PlannerTableau& new_basis_in_old_coordinates) {
    invalidate_reverse_cache();
    PlannerTableau previous(std::move(current_to_initial_));
    assert(new_basis_in_old_coordinates.num_qubits == previous.num_qubits);
    current_to_initial_ = PlannerTableau(new_basis_in_old_coordinates.num_qubits);
    for (size_t q = 0; q < new_basis_in_old_coordinates.num_qubits; ++q) {
        evaluate_generator_into(current_to_initial_.xs[q], new_basis_in_old_coordinates.xs[q],
                                previous);
        evaluate_generator_into(current_to_initial_.zs[q], new_basis_in_old_coordinates.zs[q],
                                previous);
    }
    assert(current_to_initial_.satisfies_invariants());
}

// These transformations preserve the symplectic relations by construction.
// A full invariant scan is cubic in Debug builds, so exhaustive equivalence
// tests validate them instead of rescanning the tableau after every event.
void CoordinateFrame::promote_dormant(const PlannerPauli& promoted, uint32_t active_width,
                                      uint32_t dormant_pivot) {
    const uint32_t n = static_cast<uint32_t>(current_to_initial_.num_qubits);
    assert(promoted.num_qubits == n && active_width < n && dormant_pivot >= active_width &&
           dormant_pivot < n);

    PlannerPauli mapped_promoted = to_initial(promoted);
    PlannerPauli pivot_stabilizer(current_to_initial_.zs[dormant_pivot]);
    invalidate_reverse_cache();

    // The new promoted pair replaces the selected dormant pair. Every other
    // generator that anticommutes with the promoted Pauli acquires the old
    // pivot stabilizer, preserving the same symplectic basis as the generic
    // frame composition without materializing its identity rows.
    for (uint32_t q = 0; q < n; ++q) {
        if (q == dormant_pivot) {
            continue;
        }
        if (promoted.zs[q]) {
            current_to_initial_.xs[q] *= pivot_stabilizer;
        }
        if (promoted.xs[q]) {
            current_to_initial_.zs[q] *= pivot_stabilizer;
        }
    }

    for (uint32_t q = dormant_pivot; q > active_width; --q) {
        current_to_initial_.xs[q] = current_to_initial_.xs[q - 1];
        current_to_initial_.zs[q] = current_to_initial_.zs[q - 1];
    }
    current_to_initial_.xs[active_width] = mapped_promoted;
    current_to_initial_.zs[active_width] = pivot_stabilizer;
}

void CoordinateFrame::measure_dormant(const PlannerPauli& measured, uint32_t dormant_pivot) {
    const uint32_t n = static_cast<uint32_t>(current_to_initial_.num_qubits);
    assert(measured.num_qubits == n && dormant_pivot < n);

    PlannerPauli mapped_measured = to_initial(measured);
    PlannerPauli pivot_stabilizer(current_to_initial_.zs[dormant_pivot]);
    invalidate_reverse_cache();

    for (uint32_t q = 0; q < n; ++q) {
        if (q == dormant_pivot) {
            continue;
        }
        if (measured.zs[q]) {
            current_to_initial_.xs[q] *= pivot_stabilizer;
        }
        if (measured.xs[q]) {
            current_to_initial_.zs[q] *= pivot_stabilizer;
        }
    }
    current_to_initial_.xs[dormant_pivot] = pivot_stabilizer;
    current_to_initial_.zs[dormant_pivot] = mapped_measured;
}

void CoordinateFrame::measure_active(const PlannerPauli& measured, uint32_t active_width,
                                     uint32_t pivot) {
    assert(measured.num_qubits == current_to_initial_.num_qubits && active_width > 0 &&
           active_width <= current_to_initial_.num_qubits && pivot < active_width);

    const bool diagonal = !has_x_below(measured, active_width);
    PlannerPauli mapped_measured = to_initial(measured);
    PlannerPauli pivot_conjugate(diagonal ? current_to_initial_.xs[pivot]
                                          : current_to_initial_.zs[pivot]);
    invalidate_reverse_cache();

    for (uint32_t q = 0; q < active_width; ++q) {
        if (q == pivot) {
            continue;
        }
        if (diagonal) {
            if (measured.zs[q]) {
                current_to_initial_.xs[q] *= pivot_conjugate;
            }
        } else {
            if (measured.xs[q]) {
                current_to_initial_.zs[q] *= pivot_conjugate;
            }
            if (measured.zs[q]) {
                current_to_initial_.xs[q] *= pivot_conjugate;
            }
        }
    }

    for (uint32_t q = pivot; q + 1 < active_width; ++q) {
        current_to_initial_.xs[q] = current_to_initial_.xs[q + 1];
        current_to_initial_.zs[q] = current_to_initial_.zs[q + 1];
    }
    current_to_initial_.xs[active_width - 1] = pivot_conjugate;
    current_to_initial_.zs[active_width - 1] = mapped_measured;
}

void CoordinateFrame::invalidate_reverse_cache() {
    initial_to_current_.reset();
    direct_reverse_lookups_ = 0;
}

size_t SymbolicPauliFrame::estimated_workspace_bytes(uint32_t num_qubits, uint32_t num_symbols) {
    const size_t words_per_row = (static_cast<size_t>(num_symbols) + 63) / 64;
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    const size_t qubits = num_qubits;
    if (qubits > (kMax - 1) / 2) {
        throw std::length_error("sampling symbolic Pauli frame is too large");
    }
    const size_t packed_rows = 2 * qubits + 1;
    if (words_per_row != 0 && packed_rows > kMax / words_per_row) {
        throw std::length_error("sampling symbolic Pauli frame is too large");
    }
    const size_t packed_words = packed_rows * words_per_row;
    const size_t constant_bytes = 2 * qubits * sizeof(uint8_t);
    if (packed_words > (kMax - constant_bytes) / sizeof(uint64_t)) {
        throw std::length_error("sampling symbolic Pauli frame is too large");
    }
    return packed_words * sizeof(uint64_t) + constant_bytes;
}

SymbolicPauliFrame::SymbolicPauliFrame(uint32_t num_qubits, uint32_t num_symbols)
    : num_qubits_(num_qubits),
      num_symbols_(num_symbols),
      words_per_row_(validated_words_per_row(num_qubits, num_symbols)),
      x_constants_(num_qubits, 0),
      z_constants_(num_qubits, 0),
      scratch_(words_per_row_, 0) {
    const size_t storage_words = static_cast<size_t>(num_qubits) * words_per_row_;
    x_terms_.assign(storage_words, 0);
    z_terms_.assign(storage_words, 0);
}

void SymbolicPauliFrame::apply(const PlannerPauli& correction, const AffineBool& condition) {
    assert(correction.num_qubits == num_qubits_ &&
           "symbolic Pauli correction width must match the planner frame");
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (correction.xs[q]) {
            xor_condition(x_row(q), x_constants_[q], condition);
        }
        if (correction.zs[q]) {
            xor_condition(z_row(q), z_constants_[q], condition);
        }
    }
}

AffineBool SymbolicPauliFrame::sign_for(const PlannerPauli& observable) {
    assert(observable.num_qubits == num_qubits_ &&
           "symbolic Pauli observable width must match the planner frame");
    std::ranges::fill(scratch_, uint64_t{0});
    bool constant = false;
    for (uint32_t q = 0; q < num_qubits_; ++q) {
        if (observable.xs[q]) {
            xor_row(scratch_, z_row(q));
            constant ^= z_constants_[q] != 0;
        }
        if (observable.zs[q]) {
            xor_row(scratch_, x_row(q));
            constant ^= x_constants_[q] != 0;
        }
    }

    size_t term_count = 0;
    for (uint64_t word : scratch_) {
        term_count += std::popcount(word);
    }
    std::vector<SymbolId> terms;
    terms.reserve(term_count);
    for (size_t w = 0; w < scratch_.size(); ++w) {
        uint64_t word = scratch_[w];
        while (word != 0) {
            const uint32_t bit = std::countr_zero(word);
            const size_t symbol = 64 * w + bit;
            assert(symbol < num_symbols_ && "symbolic Pauli frame contains an out-of-range term");
            terms.push_back(SymbolId{static_cast<uint32_t>(symbol)});
            word &= word - 1;
        }
    }
    return AffineBool(constant, std::move(terms));
}

std::span<uint64_t> SymbolicPauliFrame::x_row(uint32_t q) {
    return std::span<uint64_t>{x_terms_}.subspan(static_cast<size_t>(q) * words_per_row_,
                                                 words_per_row_);
}

std::span<const uint64_t> SymbolicPauliFrame::x_row(uint32_t q) const {
    return std::span<const uint64_t>{x_terms_}.subspan(static_cast<size_t>(q) * words_per_row_,
                                                       words_per_row_);
}

std::span<uint64_t> SymbolicPauliFrame::z_row(uint32_t q) {
    return std::span<uint64_t>{z_terms_}.subspan(static_cast<size_t>(q) * words_per_row_,
                                                 words_per_row_);
}

std::span<const uint64_t> SymbolicPauliFrame::z_row(uint32_t q) const {
    return std::span<const uint64_t>{z_terms_}.subspan(static_cast<size_t>(q) * words_per_row_,
                                                       words_per_row_);
}

void SymbolicPauliFrame::xor_row(std::span<uint64_t> destination,
                                 std::span<const uint64_t> source) {
    assert(destination.size() == source.size());
    for (size_t w = 0; w < destination.size(); ++w) {
        destination[w] ^= source[w];
    }
}

void SymbolicPauliFrame::xor_condition(std::span<uint64_t> row, uint8_t& constant,
                                       const AffineBool& condition) const {
    constant ^= static_cast<uint8_t>(condition.constant());
    for (SymbolId term : condition.terms()) {
        const uint32_t symbol = index(term);
        if (symbol >= num_symbols_) {
            throw std::logic_error(
                "sampling symbolic Pauli condition contains an out-of-range symbol");
        }
        row[symbol / 64] ^= uint64_t{1} << (symbol % 64);
    }
}

PlannerTableau dormant_promotion_frame(const PlannerPauli& promoted, uint32_t active_width,
                                       uint32_t dormant_pivot) {
    const uint32_t n = static_cast<uint32_t>(promoted.num_qubits);
    PlannerTableau frame(n);
    const PlannerPauli old_stabilizer = single_z(n, dormant_pivot);

    // CoordinateFrame::promote_dormant is the in-place production twin. This
    // builder remains necessary for final-state maps and as its test oracle;
    // keep their pivot ordering and fixups in sync.
    //
    // The rows express the new coordinate generators in the old basis. Making
    // the promoted Pauli the next X generator turns its rotation into a
    // single-coordinate expansion; the stabilizer products keep every other
    // generator in a canonical symplectic pair.
    for (uint32_t q = 0; q < active_width; ++q) {
        PlannerPauli x = single_x(n, q);
        PlannerPauli z = single_z(n, q);
        if (anticommutes(x, promoted)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, promoted)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.xs[q] = x;
        frame.zs[q] = z;
    }

    frame.xs[active_width] = promoted;
    frame.zs[active_width] = old_stabilizer;

    uint32_t new_q = active_width + 1;
    for (uint32_t old_q = active_width; old_q < n; ++old_q) {
        if (old_q == dormant_pivot) {
            continue;
        }
        PlannerPauli x = single_x(n, old_q);
        PlannerPauli z = single_z(n, old_q);
        if (anticommutes(x, promoted)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, promoted)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.xs[new_q] = x;
        frame.zs[new_q] = z;
        ++new_q;
    }

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid promotion frame");
    }
    return frame;
}

PlannerTableau dormant_measurement_frame(const PlannerPauli& measured, uint32_t dormant_pivot) {
    const uint32_t n = static_cast<uint32_t>(measured.num_qubits);
    PlannerTableau frame(n);
    const PlannerPauli old_stabilizer = single_z(n, dormant_pivot);

    // CoordinateFrame::measure_dormant is the in-place production twin. This
    // builder is its independent test oracle; keep their pivot replacement and
    // fixups in sync.
    //
    // Replacing one dormant Z generator with the measured Pauli represents the
    // collapsed eigenspace without changing the dense coefficient state.
    for (uint32_t q = 0; q < n; ++q) {
        if (q == dormant_pivot) {
            continue;
        }
        PlannerPauli x = single_x(n, q);
        PlannerPauli z = single_z(n, q);
        if (anticommutes(x, measured)) {
            x = positive_body_xor(x, old_stabilizer);
        }
        if (anticommutes(z, measured)) {
            z = positive_body_xor(z, old_stabilizer);
        }
        frame.xs[q] = x;
        frame.zs[q] = z;
    }

    frame.xs[dormant_pivot] = old_stabilizer;
    frame.zs[dormant_pivot] = measured;

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid dormant measurement frame");
    }
    return frame;
}

PlannerTableau active_measurement_frame(const PlannerPauli& measured, uint32_t active_width,
                                        uint32_t pivot) {
    const uint32_t n = static_cast<uint32_t>(measured.num_qubits);
    PlannerTableau frame(n);
    const bool diagonal = !has_x_below(measured, active_width);
    const PlannerPauli pivot_x = single_x(n, pivot);
    const PlannerPauli pivot_z = single_z(n, pivot);

    // CoordinateFrame::measure_active is the in-place production twin. This
    // builder is its independent test oracle; keep their pivot compaction and
    // fixups in sync.
    //
    // Moving the measured Pauli to the last active Z generator gives the
    // direct measurement kernel one coordinate to remove. The cumulative
    // frame maps later operations into the remaining packed coordinates.
    frame.zs[active_width - 1] = measured;
    frame.xs[active_width - 1] = diagonal ? pivot_x : pivot_z;

    uint32_t new_q = 0;
    for (uint32_t old_q = 0; old_q < active_width; ++old_q) {
        if (old_q == pivot) {
            continue;
        }
        PlannerPauli x = single_x(n, old_q);
        PlannerPauli z = single_z(n, old_q);
        if (diagonal) {
            if (measured.zs[old_q]) {
                x = positive_body_xor(x, pivot_x);
            }
        } else {
            if (measured.xs[old_q]) {
                z = positive_body_xor(z, pivot_z);
            }
            if (measured.zs[old_q]) {
                x = positive_body_xor(x, pivot_z);
            }
        }
        frame.xs[new_q] = x;
        frame.zs[new_q] = z;
        ++new_q;
    }

    if (!frame.satisfies_invariants()) {
        throw std::logic_error("sampling planner produced an invalid active measurement frame");
    }
    return frame;
}

}  // namespace clifft::sampling::internal
