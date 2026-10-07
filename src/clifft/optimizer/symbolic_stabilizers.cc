#include "clifft/optimizer/symbolic_stabilizers.h"

#include "clifft/optimizer/pauli_axis.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

namespace clifft {
namespace {

uint32_t pivot_of(const PauliString& axis) {
    const uint32_t domain = axis.x().num_words() * 64;
    const uint32_t x = axis.x().lowest_bit();
    return x < domain ? x : domain + axis.z().lowest_bit();
}

bool body_bit(const PauliString& axis, uint32_t pivot) {
    const uint32_t domain = axis.x().num_words() * 64;
    return pivot < domain ? axis.x().bit_get(pivot) : axis.z().bit_get(pivot - domain);
}

bool identity(const PauliString& axis) {
    return axis.x().is_zero() && axis.z().is_zero();
}

using optimizer_detail::copy_axis;

}  // namespace

SymbolicStabilizers::SymbolicStabilizers(uint32_t num_qubits, SymbolicStabilizerOptions options)
    : options_(options) {
    for (uint32_t q = 0; q < num_qubits; ++q) {
        PauliString axis(num_qubits);
        axis.set_pauli(q, false, true);
        const auto pivot = pivot_of(axis);
        rows_.emplace(pivot, Row{std::move(axis), {}});
    }
}

bool SymbolicStabilizers::multiply(Row& left, const Row& right, size_t& products) const {
    if (products == options_.max_row_products) {
        return false;
    }
    ++products;
    left.sign ^= right.sign;
    if (left.sign.terms().size() > options_.max_expression_terms) {
        return false;
    }
    left.axis.right_multiply(right.axis.view());
    return true;
}

const KnownStabilizers& SymbolicStabilizers::fixed_constraints() const {
    if (fixed_valid_) {
        return fixed_;
    }
    fixed_.rows_.clear();
    // Preserve cheap constant facts even when symbolic elimination is capped.
    for (const auto& [pivot, row] : rows_) {
        if (row.sign.terms().empty()) {
            auto axis = row.axis;
            axis.set_sign(axis.sign() ^ row.sign.constant());
            fixed_.rows_.emplace(pivot, std::move(axis));
        }
    }
    std::map<sampling::SymbolId, Row> signs;
    size_t products = 0;
    for (const auto& [pivot, original] : rows_) {
        if (original.sign.terms().empty()) {
            continue;
        }
        Row row = original;
        bool capped = false;
        bool independent = false;
        while (!row.sign.terms().empty()) {
            const auto symbol = row.sign.terms().front();
            auto it = signs.find(symbol);
            if (it == signs.end()) {
                signs.emplace(symbol, std::move(row));
                independent = true;
                break;
            }
            if (!multiply(row, it->second, products)) {
                capped = true;
                break;
            }
        }
        if (capped) {
            break;
        }
        if (independent) {
            continue;
        }
        row.axis.set_sign(row.axis.sign() ^ row.sign.constant());
        for (auto it = fixed_.rows_.lower_bound(pivot_of(row.axis)); it != fixed_.rows_.end();
             ++it) {
            if (body_bit(row.axis, it->first)) {
                if (products == options_.max_row_products) {
                    capped = true;
                    break;
                }
                ++products;
                row.axis.right_multiply(it->second.view());
            }
        }
        if (capped) {
            break;
        }
        if (!identity(row.axis)) {
            fixed_.rows_.emplace(pivot_of(row.axis), std::move(row.axis));
        } else {
            assert(!row.axis.sign());
        }
    }
    fixed_valid_ = true;
    return fixed_;
}

std::optional<SymbolicStabilizers::AffineBool> SymbolicStabilizers::affine_eigenvalue(
    PauliString axis, bool& capped) const {
    Row reduced{std::move(axis), {}};
    size_t products = 0;
    for (auto it = rows_.lower_bound(pivot_of(reduced.axis)); it != rows_.end(); ++it) {
        if (body_bit(reduced.axis, it->first) && !multiply(reduced, it->second, products)) {
            capped = true;
            return std::nullopt;
        }
    }
    if (!identity(reduced.axis)) {
        return std::nullopt;
    }
    assert(reduced.axis.is_hermitian());
    reduced.sign ^= reduced.axis.sign();
    return std::move(reduced.sign);
}

bool SymbolicStabilizers::insert_row(Row row) {
    fixed_valid_ = false;
    assert(row.axis.is_hermitian());
    size_t products = 0;
    for (auto it = rows_.lower_bound(pivot_of(row.axis)); it != rows_.end(); ++it) {
        if (body_bit(row.axis, it->first) && !multiply(row, it->second, products)) {
            return false;
        }
    }
    if (identity(row.axis)) {
        assert(row.sign.terms().empty() && row.axis.sign() == row.sign.constant());
    } else {
        rows_.emplace(pivot_of(row.axis), std::move(row));
    }
    return true;
}

bool SymbolicStabilizers::intersect_rows(PauliStringView axis) {
    auto pivot = rows_.end();
    for (auto it = rows_.begin(); it != rows_.end(); ++it) {
        if (!axis.commutes(it->second.axis.view())) {
            pivot = it;
        }
    }
    if (pivot == rows_.end()) {
        return true;
    }
    fixed_valid_ = false;
    // The largest pivot preserves the earlier rows' leading bits.
    size_t products = 0;
    for (auto it = rows_.begin(); it != pivot; ++it) {
        if (!axis.commutes(it->second.axis.view()) &&
            !multiply(it->second, pivot->second, products)) {
            return false;
        }
    }
    rows_.erase(pivot);
    return true;
}

void SymbolicStabilizers::intersect(PauliStringView axis) {
    if (!intersect_rows(axis)) {
        forget();
    }
}

void SymbolicStabilizers::apply_pauli(PauliStringView axis, const AffineBool& condition) {
    if (condition.terms().empty() && !condition.constant()) {
        return;
    }
    for (auto& [pivot, row] : rows_) {
        if (!axis.commutes(row.axis.view())) {
            fixed_valid_ = false;
            row.sign ^= condition;
            if (row.sign.terms().size() > options_.max_expression_terms) {
                forget();
                return;
            }
        }
    }
}

std::optional<SymbolicStabilizers::AffineBool> SymbolicStabilizers::fresh_symbol() {
    if (options_.max_expression_terms == 0 || next_symbol_ > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return AffineBool::symbol(sampling::SymbolId{static_cast<uint32_t>(next_symbol_++)});
}

void SymbolicStabilizers::assign_record(uint32_t record, AffineBool value) {
    if (options_.max_record_entries == 0) {
        return;
    }
    if (records_.size() == options_.max_record_entries && !records_.contains(record)) {
        records_.erase(records_.begin());
    }
    records_.insert_or_assign(record, std::move(value));
}

void SymbolicStabilizers::forget() {
    rows_.clear();
    records_.clear();
    fixed_.rows_.clear();
    fixed_valid_ = false;
}

void SymbolicStabilizers::advance(const HirModule& hir, const HeisenbergOp& op) {
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            if (!rows_.empty()) {
                intersect(copy_axis(hir.mask_view(op), hir.num_qubits).view());
            }
            break;
        case OpType::MEASURE: {
            auto axis = copy_axis(hir.mask_view(op), hir.num_qubits);
            bool capped = false;
            auto outcome = affine_eigenvalue(axis, capped);
            if (capped) {
                forget();
            }
            if (!outcome) {
                outcome = fresh_symbol();
                intersect(axis.view());
                if (outcome && !insert_row(Row{std::move(axis), *outcome})) {
                    forget();
                }
            }
            if (outcome) {
                assign_record(static_cast<uint32_t>(op.meas_record_idx()), std::move(*outcome));
            }
            break;
        }
        case OpType::CONDITIONAL_PAULI: {
            if (rows_.empty()) {
                break;
            }
            const auto axis = copy_axis(hir.mask_view(op), hir.num_qubits);
            auto record = records_.find(static_cast<uint32_t>(op.controlling_meas()));
            if (record == records_.end()) {
                intersect(axis.view());
            } else {
                apply_pauli(axis.view(), record->second);
            }
            break;
        }
        case OpType::NOISE: {
            const auto& channels =
                hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
            for (const auto& channel : channels) {
                if (rows_.empty()) {
                    break;
                }
                if (channel.prob == 0) {
                    continue;
                }
                const auto axis =
                    copy_axis(hir.noise_channel_masks.at(channel.mask), hir.num_qubits);
                // Fault events are never feedback controls. Eliminate their
                // private signs now by retaining the commuting subgroup, whose
                // products preserve shared-fault correlations. Include the
                // no-fault path even at probability one for reference syndromes.
                intersect(axis.view());
            }
            break;
        }
        case OpType::READOUT_NOISE: {
            const auto& entry = hir.readout_noise[static_cast<uint32_t>(op.readout_noise_idx())];
            if (entry.prob_zero_to_one == 0 && entry.prob_one_to_zero == 0) {
                break;
            }
            auto record = records_.find(entry.meas_idx);
            if (record == records_.end()) {
                break;
            }
            const auto flip = fresh_symbol();
            if (!flip) {
                records_.erase(record);
                break;
            }
            // The flip can depend on the physical outcome. Only the reported
            // record changes; constraints retain the original collapse sign.
            record->second ^= *flip;
            if (record->second.terms().size() > options_.max_expression_terms) {
                records_.erase(record);
            }
            break;
        }
        case OpType::INSTRUMENT:
            forget();
            break;
        case OpType::EXP_VAL:
        case OpType::DETECTOR:
        case OpType::OBSERVABLE:
        case OpType::NUM_OP_TYPES:
            break;
    }
}

}  // namespace clifft
