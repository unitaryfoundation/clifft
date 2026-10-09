#pragma once

// Research host scheduling only; all checks precede native planning/execution.

#include "clifft/frontend/hir.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace squeeze_schedule_reuse {
using namespace clifft;

inline HirModule scheduling_snapshot(const HirModule& hir) {
    HirModule result;
    result.num_qubits = hir.num_qubits;
    result.ops = hir.ops;
    result.pauli_masks = hir.pauli_masks;
    result.num_measurements = hir.num_measurements;
    result.num_hidden_measurements = hir.num_hidden_measurements;
    result.num_detectors = hir.num_detectors;
    result.num_observables = hir.num_observables;
    result.num_exp_vals = hir.num_exp_vals;
    result.detector_targets = hir.detector_targets;
    result.observable_targets = hir.observable_targets;
    return result;
}

inline bool unchanged_scheduling_input(const HirModule& a, const HirModule& b) {
    // The final Clifford frame and provenance do not participate in can_swap.
    // Compare actual masks and payloads, never their diagnostic fingerprints.
    if (a.num_qubits != b.num_qubits || a.ops.size() != b.ops.size() ||
        a.num_measurements != b.num_measurements ||
        a.num_hidden_measurements != b.num_hidden_measurements ||
        a.num_detectors != b.num_detectors || a.num_observables != b.num_observables ||
        a.num_exp_vals != b.num_exp_vals || a.detector_targets != b.detector_targets ||
        a.observable_targets != b.observable_targets || !b.noise_sites.empty() ||
        !b.readout_noise.empty() || !b.instrument_sites.empty() || !b.logical_noise_prefix.empty())
        return false;
    for (size_t i = 0; i < a.ops.size(); ++i) {
        const auto& x = a.ops[i];
        const auto& y = b.ops[i];
        if (x.op_type() != y.op_type() || x.flags() != y.flags() || x.has_mask() != y.has_mask())
            return false;
        if (x.has_mask() && (a.destab_mask(x) != b.destab_mask(y) ||
                             a.stab_mask(x) != b.stab_mask(y) || a.sign(x) != b.sign(y)))
            return false;
        switch (x.op_type()) {
            case OpType::T_GATE:
                break;
            case OpType::PHASE_ROTATION:
                if (x.alpha() != y.alpha())
                    return false;
                break;
            case OpType::MEASURE:
                if (x.meas_record_idx() != y.meas_record_idx())
                    return false;
                break;
            case OpType::CONDITIONAL_PAULI:
                if (x.controlling_meas() != y.controlling_meas())
                    return false;
                break;
            case OpType::DETECTOR:
                if (x.detector_idx() != y.detector_idx())
                    return false;
                break;
            case OpType::OBSERVABLE:
                if (x.observable_idx() != y.observable_idx() ||
                    x.observable_target_list_idx() != y.observable_target_list_idx())
                    return false;
                break;
            case OpType::EXP_VAL:
                if (x.exp_val_idx() != y.exp_val_idx())
                    return false;
                break;
            default:
                return false;
        }
    }
    return true;
}

class Schedule {
  public:
    explicit Schedule(HirModule raw) {
        raw.source_map.clear();
        for (uint32_t i = 0; i < raw.ops.size(); ++i)
            raw.source_map.push_back({i + 1});
        StatevectorSqueezePass{}.run(raw);
        for (const auto& origins : raw.source_map) {
            if (origins.size() != 1 || origins[0] == 0 || origins[0] > raw.ops.size())
                throw std::runtime_error("Squeeze did not preserve operation identities");
            order_.push_back(origins[0] - 1);
        }
    }

    void apply(HirModule& hir) const {
        if (hir.ops.size() != order_.size() || !hir.logical_noise_prefix.empty())
            throw std::invalid_argument("Schedule applied outside its certified interface");
        auto ops = hir.ops;
        for (size_t i = 0; i < order_.size(); ++i)
            hir.ops[i] = ops[order_[i]];
        if (hir.source_map.size() == order_.size()) {
            auto origins = hir.source_map;
            for (size_t i = 0; i < order_.size(); ++i)
                hir.source_map[i] = std::move(origins[order_[i]]);
        }
    }

    size_t size() const { return order_.size(); }
    size_t moved() const {
        size_t result = 0;
        for (size_t i = 0; i < order_.size(); ++i)
            result += order_[i] != i;
        return result;
    }

  private:
    std::vector<size_t> order_;
};
}  // namespace squeeze_schedule_reuse
