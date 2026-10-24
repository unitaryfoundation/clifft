#include "zx_fresh_resets.h"

#include "clifft/circuit/gate_data.h"

#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zx_test {

using namespace clifft;
namespace {

class Relabeler {
  public:
    explicit Relabeler(const Circuit& source)
        : label_(source.num_qubits), used_(source.num_qubits, false), next_(source.num_qubits) {
        std::iota(label_.begin(), label_.end(), 0u);
    }

    [[nodiscard]] uint32_t current(uint32_t q) const { return label_[q]; }

    Target remap(Target t) {
        const uint32_t q = label_[t.value()];
        used_[q] = true;
        return Target{(t.bits & ~Target::kValueMask) | q};
    }

    // Point q at a qubit in |0>: the current one if nothing has touched it.
    uint32_t reset(uint32_t q) {
        if (used_[label_[q]]) {
            label_[q] = next_++;
            used_.push_back(false);
        }
        return label_[q];
    }

    void mark_used(uint32_t label) { used_[label] = true; }
    [[nodiscard]] uint32_t num_qubits() const { return next_; }

  private:
    std::vector<uint32_t> label_;
    std::vector<bool> used_;
    uint32_t next_;
};

AstNode single(GateType gate, uint32_t q, const AstNode& from) {
    return AstNode{.gate = gate,
                   .targets = {Target::qubit(q)},
                   .args = {},
                   .source_line = from.source_line,
                   .tag = {}};
}

}  // namespace

Circuit with_fresh_qubit_per_reset(const Circuit& circuit) {
    Circuit out = circuit.metadata_only_copy();
    out.nodes.reserve(circuit.nodes.size());
    Relabeler relabel(circuit);
    // Prepare the reset's basis state on a qubit in |0>.
    auto prepare = [&](GateType reset, uint32_t q, const AstNode& from) {
        const uint32_t fresh = relabel.reset(q);
        if (reset == GateType::RX || reset == GateType::RY) {
            out.nodes.push_back(single(GateType::H, fresh, from));
        }
        if (reset == GateType::RY) {
            out.nodes.push_back(single(GateType::S, fresh, from));
        }
        // A Z reset leaves the qubit in |0>, so a second reset can reuse it.
        if (reset != GateType::R) {
            relabel.mark_used(fresh);
        }
    };
    for (const AstNode& node : circuit.nodes) {
        // A lost qubit stays lost through a reset, so moving the reset onto a
        // fresh qubit would change the channel.
        if (is_noncomputational_annotation(node.gate)) {
            throw std::invalid_argument(
                "fresh qubits per reset do not apply to leakage and loss annotations");
        }
        switch (node.gate) {
            case GateType::R:
            case GateType::RX:
            case GateType::RY:
                for (Target t : node.targets) {
                    prepare(node.gate, t.value(), node);
                }
                continue;
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY: {
                const GateType measure = node.gate == GateType::MR    ? GateType::M
                                         : node.gate == GateType::MRX ? GateType::MX
                                                                      : GateType::MY;
                const GateType reset = node.gate == GateType::MR    ? GateType::R
                                       : node.gate == GateType::MRX ? GateType::RX
                                                                    : GateType::RY;
                for (Target t : node.targets) {
                    AstNode m = node;
                    m.gate = measure;
                    m.targets = {relabel.remap(t)};
                    out.nodes.push_back(std::move(m));
                    prepare(reset, t.value(), node);
                }
                continue;
            }
            case GateType::MPAD:
                // Its targets are record values, not qubits.
                out.nodes.push_back(node);
                continue;
            default:
                break;
        }
        AstNode copy = node;
        for (Target& t : copy.targets) {
            if (!t.is_rec()) {
                t = relabel.remap(t);
            }
        }
        out.nodes.push_back(std::move(copy));
    }
    out.num_qubits = relabel.num_qubits();
    return out;
}

}  // namespace zx_test
