#include "clifft/optimizer/symbolic_stabilizers.h"

#include "clifft/optimizer/pauli_axis.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <utility>

namespace clifft {
namespace {

using optimizer_detail::copy_axis;

}  // namespace

std::map<uint32_t, bool> collect_postselected_records(const HirModule& hir) {
    hir.postselection.validate_assumptions(hir.num_detectors);
    std::map<uint32_t, bool> records;
    if (std::ranges::none_of(hir.postselection.mask, [](uint8_t flag) { return flag != 0; })) {
        return records;
    }
    const size_t total_records =
        static_cast<size_t>(hir.num_measurements) + hir.num_hidden_measurements;
    std::vector<uint32_t> write_counts(total_records, 0);
    std::vector<bool> noisy_readout(total_records, false);
    for (const HeisenbergOp& op : hir.ops) {
        if (op.op_type() == OpType::MEASURE) {
            const uint32_t rec = static_cast<uint32_t>(op.meas_record_idx());
            if (rec < total_records) {
                ++write_counts[rec];
            }
        } else if (op.op_type() == OpType::READOUT_NOISE) {
            const uint32_t idx = static_cast<uint32_t>(op.readout_noise_idx());
            if (idx < hir.readout_noise.size()) {
                const ReadoutNoiseEntry& entry = hir.readout_noise[idx];
                if ((entry.prob_zero_to_one > 0.0 || entry.prob_one_to_zero > 0.0) &&
                    entry.meas_idx < total_records) {
                    noisy_readout[entry.meas_idx] = true;
                }
            }
        }
    }

    struct PostselectedDetector {
        std::vector<uint32_t> targets;
        bool expected = false;
    };
    std::vector<PostselectedDetector> detectors;
    uint32_t detector_index = 0;
    for (const HeisenbergOp& op : hir.ops) {
        if (op.op_type() != OpType::DETECTOR) {
            continue;
        }
        const uint32_t targets_index = static_cast<uint32_t>(op.detector_idx());
        const bool postselected = detector_index < hir.postselection.mask.size() &&
                                  hir.postselection.mask[detector_index] != 0;
        const bool expected = detector_index < hir.postselection.expected_detectors.size() &&
                              hir.postselection.expected_detectors[detector_index] != 0;
        ++detector_index;
        if (!postselected || targets_index >= hir.detector_targets.size()) {
            continue;
        }
        const auto& raw_targets = hir.detector_targets[targets_index];
        if (raw_targets.empty()) {
            continue;
        }
        std::vector<uint32_t> targets = raw_targets;
        std::ranges::sort(targets);
        std::vector<uint32_t> reduced;
        reduced.reserve(targets.size());
        for (uint32_t rec : targets) {
            if (!reduced.empty() && reduced.back() == rec) {
                reduced.pop_back();
            } else {
                reduced.push_back(rec);
            }
        }
        if (reduced.empty()) {
            continue;
        }
        if (std::ranges::any_of(reduced, [&](uint32_t rec) {
                return rec >= total_records || write_counts[rec] != 1 || noisy_readout[rec];
            })) {
            continue;
        }
        detectors.push_back(PostselectedDetector{std::move(reduced), expected});
    }

    std::vector<bool> conflicting(total_records, false);
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& det : detectors) {
            if (std::ranges::any_of(det.targets, [&](uint32_t r) { return conflicting[r]; })) {
                continue;
            }
            bool parity = det.expected;
            uint32_t unknown_count = 0;
            uint32_t unknown_rec = 0;
            for (uint32_t rec : det.targets) {
                const auto it = records.find(rec);
                if (it != records.end()) {
                    parity ^= it->second;
                } else {
                    ++unknown_count;
                    unknown_rec = rec;
                }
            }
            if (unknown_count == 0) {
                if (parity) {
                    for (uint32_t rec : det.targets) {
                        records.erase(rec);
                        conflicting[rec] = true;
                    }
                    changed = true;
                }
            } else if (unknown_count == 1) {
                records.emplace(unknown_rec, parity);
                changed = true;
            }
        }
    }
    return records;
}

SymbolicStabilizers::SymbolicStabilizers(const HirModule& hir, SymbolicStabilizerOptions options)
    : SymbolicStabilizers(hir, hir.ops.size(), options) {}

SymbolicStabilizers::SymbolicStabilizers(const HirModule& hir, size_t end,
                                         SymbolicStabilizerOptions options)
    : options_(options),
      num_qubits_(hir.num_qubits),
      domain_(((hir.num_qubits + 63) / 64) * 64),
      basis_(hir.num_qubits),
      postselected_records_(collect_postselected_records(hir)) {
    assert(end <= hir.ops.size());
    for (size_t i = 0; i < end; ++i) {
        if (hir.ops[i].op_type() == OpType::CONDITIONAL_PAULI) {
            ++remaining_uses_[static_cast<uint32_t>(hir.ops[i].controlling_meas())];
        }
    }
}

bool SymbolicStabilizers::multiply(Row& left, const Row& right, size_t& products) const {
    return basis_.multiply_row(left, right, products, options_.max_row_products,
                               options_.max_expression_terms);
}

bool SymbolicStabilizers::reduce_row(Row& target, size_t& products) const {
    return basis_.reduce(target, products, options_.max_row_products,
                         options_.max_expression_terms);
}

const KnownStabilizers& SymbolicStabilizers::fixed_constraints() {
    flush_feedback();
    if (fixed_valid_) {
        return fixed_;
    }
    fixed_.basis_.reset_unit_z_from(basis_);
    const auto& rows = basis_.rows();
    if (rows.empty()) {
        fixed_valid_ = true;
        return fixed_;
    }

    // Preserve known fixed-sign constraints even when symbolic elimination is capped.
    for (const Row& row : rows) {
        if (row.sign.terms().empty()) {
            auto axis = row.axis;
            axis.set_sign(axis.sign() ^ row.sign.constant());
            fixed_.basis_.insert_reduced(KnownStabilizers::Basis::Row{0, std::move(axis), {}});
        }
    }

    std::map<sampling::SymbolId, Row> signs;
    size_t products = 0;
    for (const Row& original : rows) {
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
            if (products == options_.max_row_products) {
                break;
            }
            // An oversized expression need not hide independent later constraints.
            continue;
        }
        if (independent) {
            continue;
        }
        row.axis.set_sign(row.axis.sign() ^ row.sign.constant());
        if (!fixed_.reduce_and_insert_bounded(std::move(row.axis), products,
                                              options_.max_row_products)) {
            break;
        }
    }
    fixed_valid_ = true;
    return fixed_;
}

std::optional<SymbolicStabilizers::AffineBool> SymbolicStabilizers::affine_eigenvalue(
    PauliString axis, bool& capped) const {
    Row reduced{0, std::move(axis), {}};
    size_t products = 0;
    if (!reduce_row(reduced, products)) {
        capped = true;
        return std::nullopt;
    }
    if (!Basis::identity_axis(reduced.axis)) {
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
    if (!reduce_row(row, products)) {
        return false;
    }
    if (Basis::identity_axis(row.axis)) {
        assert(row.sign.terms().empty() && row.axis.sign() == row.sign.constant());
        return true;
    }
    basis_.insert_reduced(std::move(row));
    return true;
}

void SymbolicStabilizers::intersect(MaskView ax, MaskView az) {
    bool modified = false;
    if (!basis_.intersect(ax, az, options_.max_row_products, options_.max_expression_terms,
                          &modified)) {
        forget();
        return;
    }
    if (modified) {
        fixed_valid_ = false;
    }
}

void SymbolicStabilizers::apply_pauli(MaskView ax, MaskView az, const AffineBool& condition) {
    if ((condition.terms().empty() && !condition.constant()) || empty_rows()) {
        return;
    }
    const PauliCommutation comm(ax, az);
    if (comm.begin_word() == comm.end_word()) {
        return;
    }

    struct DemotedUnitZ {
        uint32_t word;
        uint64_t bit;
        bool negative;
    };
    std::vector<DemotedUnitZ> demoted;
    auto& rows = basis_.mut_rows();
    size_t write = 0;
    for (size_t read = 0; read < rows.size(); ++read) {
        Row& row = rows[read];
        if (!row.misses(comm) && comm.anti_commutes(row.axis.x(), row.axis.z())) {
            fixed_valid_ = false;
            row.sign ^= condition;
            if (row.sign.terms().size() > options_.max_expression_terms) {
                forget();
                return;
            }
            if (row.sign.terms().empty() && row.pivot >= domain_) {
                const PauliCommutation range(row.axis.x(), row.axis.z());
                const uint32_t w = range.begin_word();
                if (range.end_word() == w + 1 && std::has_single_bit(row.axis.z().words[w])) {
                    demoted.push_back(DemotedUnitZ{w, row.axis.z().words[w],
                                                   row.axis.sign() != row.sign.constant()});
                    continue;
                }
            }
        }
        if (write != read) {
            rows[write] = std::move(row);
        }
        ++write;
    }
    rows.resize(write);

    if (basis_.unit_z_count() != 0) {
        if (condition.terms().empty()) {
            for (uint32_t w = comm.begin_word(); w < comm.end_word(); ++w) {
                if (basis_.flip_unit_z_signs(w, ax.words[w]) != 0) {
                    fixed_valid_ = false;
                }
            }
        } else {
            for (uint32_t w = comm.begin_word(); w < comm.end_word(); ++w) {
                if ((ax.words[w] & basis_.unit_z_mask()[w]) == 0) {
                    continue;
                }
                fixed_valid_ = false;
                if (condition.terms().size() > options_.max_expression_terms) {
                    forget();
                    return;
                }
                uint64_t signs = 0;
                uint64_t hits = basis_.take_unit_z_hits(w, ax.words[w], signs);
                while (hits != 0) {
                    const uint32_t b = static_cast<uint32_t>(std::countr_zero(hits));
                    const uint32_t q = w * 64U + b;
                    const bool s_q = ((signs >> b) & 1U) != 0;
                    PauliString axis(num_qubits_);
                    axis.set_pauli(q, false, true);
                    if (s_q) {
                        axis.set_sign(true);
                    }
                    basis_.insert_row_at_pivot(
                        Row{domain_ + q, std::move(axis), condition, w, w + 1});
                    hits &= hits - 1;
                }
            }
        }
    }

    for (const DemotedUnitZ& item : demoted) {
        basis_.set_unit_z(item.word, item.bit, item.negative);
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
        // Evict by measurement age: hidden reset slots follow visible slots
        // numerically, not temporally.
        auto oldest = std::ranges::min_element(records_, {},
                                               [](const auto& entry) { return entry.second.age; });
        records_.erase(oldest);
    }
    records_.insert_or_assign(record, Record{std::move(value), next_record_age_++});
}

void SymbolicStabilizers::forget() {
    basis_.clear();
    records_.clear();
    fixed_.clear();
    fixed_valid_ = false;
}

void SymbolicStabilizers::flush_feedback() {
    if (!feedback_) {
        return;
    }
    auto feedback = std::move(*feedback_);
    feedback_.reset();
    if (feedback.condition) {
        apply_pauli(feedback.axis.x(), feedback.axis.z(), *feedback.condition);
    } else {
        intersect(feedback.axis.x(), feedback.axis.z());
    }
}

void SymbolicStabilizers::advance(const HirModule& hir, const HeisenbergOp& op) {
    if (feedback_ && (op.op_type() != OpType::CONDITIONAL_PAULI ||
                      static_cast<uint32_t>(op.controlling_meas()) != feedback_->record)) {
        flush_feedback();
    }
    switch (op.op_type()) {
        case OpType::T_GATE:
        case OpType::PHASE_ROTATION:
            if (!empty_rows()) {
                const auto mask = hir.mask_view(op);
                intersect(mask.x(), mask.z());
            }
            break;
        case OpType::MEASURE: {
            const auto record = static_cast<uint32_t>(op.meas_record_idx());
            const auto postselected = !postselected_records_.empty()
                                          ? postselected_records_.find(record)
                                          : postselected_records_.end();
            if (!remaining_uses_.contains(record) && postselected == postselected_records_.end()) {
                // An outcome never used for feedback is a private sign, just
                // like a fault event. Keep only its commuting subgroup.
                if (!empty_rows()) {
                    const auto mask = hir.mask_view(op);
                    intersect(mask.x(), mask.z());
                }
                break;
            }
            auto axis = copy_axis(hir.mask_view(op), hir.num_qubits);
            bool capped = false;
            auto outcome = affine_eigenvalue(axis, capped);
            // Anticommutation proves this outcome is not in the known group.
            // Even if reduction hit a limit, intersection can retain spectators.
            // Only capped queries need this scan; deterministic queries stay cheap.
            if (capped && basis_.commutes(axis.x(), axis.z())) {
                forget();
            }
            if (!outcome) {
                outcome = postselected != postselected_records_.end()
                              ? std::optional{AffineBool(postselected->second)}
                              : fresh_symbol();
                intersect(axis.x(), axis.z());
                if (outcome && !insert_row(Row{0, std::move(axis), *outcome})) {
                    forget();
                }
            } else if (postselected != postselected_records_.end()) {
                outcome = AffineBool(postselected->second);
            }
            if (outcome && remaining_uses_.contains(record)) {
                assign_record(record, std::move(*outcome));
            }
            break;
        }
        case OpType::CONDITIONAL_PAULI: {
            const auto id = static_cast<uint32_t>(op.controlling_meas());
            if (!empty_rows()) {
                if (feedback_) {
                    // Consecutive Paulis with the same classical control act
                    // as their product, up to an irrelevant branch phase.
                    const auto mask = hir.mask_view(op);
                    feedback_->axis.mut_x().xor_with(mask.x());
                    feedback_->axis.mut_z().xor_with(mask.z());
                } else {
                    const auto record = records_.find(id);
                    feedback_ =
                        Feedback{id, copy_axis(hir.mask_view(op), hir.num_qubits),
                                 record == records_.end() ? std::nullopt
                                                          : std::optional{record->second.value}};
                }
            }
            auto use = remaining_uses_.find(id);
            if (use != remaining_uses_.end() && --use->second == 0) {
                records_.erase(id);
                remaining_uses_.erase(use);
            }
            break;
        }
        case OpType::NOISE: {
            const auto& channels =
                hir.noise_sites[static_cast<uint32_t>(op.noise_site_idx())].channels;
            for (const auto& channel : channels) {
                if (empty_rows()) {
                    break;
                }
                if (channel.prob == 0) {
                    continue;
                }
                const auto mask = hir.noise_channel_masks.at(channel.mask);
                // Fault events are never feedback controls. Eliminate their
                // private signs now by retaining the commuting subgroup, whose
                // products preserve shared-fault correlations. Include the
                // no-fault path even at probability one for reference syndromes.
                intersect(mask.x(), mask.z());
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
            record->second.value ^= *flip;
            if (record->second.value.terms().size() > options_.max_expression_terms) {
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
