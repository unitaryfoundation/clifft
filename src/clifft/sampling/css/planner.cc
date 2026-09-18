#include "clifft/sampling/css/planner.h"

#include "clifft/frontend/frontend.h"
#include "clifft/util/numeric.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace clifft::sampling {
namespace {
struct Decline : std::runtime_error {
    using std::runtime_error::runtime_error;
};
void need(bool condition, const char* message) {
    if (!condition)
        throw Decline(message);
}
bool annotation(GateType g) {
    return g == GateType::TICK;
}
bool noise(GateType g) {
    return g == GateType::DEPOLARIZE1 || g == GateType::X_ERROR || g == GateType::Y_ERROR ||
           g == GateType::Z_ERROR || g == GateType::PAULI_CHANNEL_1;
}
uint64_t support(const AstNode& node, char axis, unsigned width) {
    uint64_t result = 0;
    for (auto target : node.targets) {
        need(!target.is_rec() && !target.is_inverted() && target.value() < width &&
                 target.pauli_char() == axis,
             "unsupported Pauli product");
        auto bit = uint64_t{1} << target.value();
        need(!(result & bit), "repeated Pauli target");
        result |= bit;
    }
    need(result != 0, "empty Pauli product");
    return result;
}
PauliString pauli(unsigned n, uint64_t x, uint64_t z) {
    PauliString p(n);
    for (unsigned q = 0; q < n; ++q)
        p.set_pauli(q, x >> q & 1, z >> q & 1);
    p.set_sign(false);
    return p;
}
std::array<double, 3> initial_bloch(const Circuit& prefix, const css::Code& code) {
    auto hir = trace(prefix);
    need(hir.ops.size() <= 1 && (hir.ops.empty() || hir.ops[0].op_type() == OpType::T_GATE),
         "prefix requires an ideal Clifford circuit and at most one T");
    need(hir.final_tableau.has_value(), "prefix has no unitary coordinate map");
    auto inverse = hir.final_tableau->inverse();
    PauliString rotation(prefix.num_qubits);
    double angle = 0;
    if (!hir.ops.empty()) {
        const auto& op = hir.ops[0];
        for (unsigned q = 0; q < prefix.num_qubits; ++q)
            rotation.set_pauli(q, hir.destab_mask(op).bit_get(q), hir.stab_mask(op).bit_get(q));
        rotation.set_sign(hir.sign(op));
        angle = (op.is_dagger() ? -1 : 1) * std::numbers::pi / 4;
    }
    auto zero_expectation = [](const PauliString& p) {
        for (unsigned q = 0; q < p.num_qubits(); ++q)
            if (p.x().bit_get(q))
                return 0.;
        return p.sign() ? -1. : 1.;
    };
    auto expectation = [&](PauliString physical) {
        auto q = inverse.apply(physical.view());
        if (q.view().commutes(rotation.view()))
            return zero_expectation(q);
        auto pq = rotation;
        pq.right_multiply(q.view());
        pq.add_phase(1);
        return std::cos(angle) * zero_expectation(q) + std::sin(angle) * zero_expectation(pq);
    };
    for (auto check : code.checks())
        need(std::abs(expectation(pauli(prefix.num_qubits, check.x ? check.support : 0,
                                        check.x ? 0 : check.support)) -
                      1) < 1e-12,
             "prefix does not prepare the certified positive code space");
    auto all = (uint64_t{1} << prefix.num_qubits) - 1;
    std::array<double, 3> result{expectation(pauli(prefix.num_qubits, all, 0)),
                                 expectation(pauli(prefix.num_qubits, all, all)),
                                 expectation(pauli(prefix.num_qubits, 0, all))};
    if ((prefix.num_qubits - 1) / 2 % 2)
        result[1] = -result[1];
    need(
        std::abs(result[0] * result[0] + result[1] * result[1] + result[2] * result[2] - 1) < 1e-12,
        "prefix logical state is not pure");
    return result;
}
struct Builder {
    const Circuit& circuit;
    SamplingPlanOptions options;
    std::vector<AstNode> nodes;
    SamplingPlan plan;
    size_t cursor = 0;
    unsigned record = 0, detector = 0;
    unsigned n;
    uint64_t all;
    std::shared_ptr<const css::Code> code;
    std::vector<AffineBool> frame;
    std::vector<std::vector<RecordSlot>> observable_records;
    Builder(const Circuit& c, SamplingPlanOptions o)
        : circuit(c),
          options(o),
          n(c.num_qubits),
          all((uint64_t{1} << n) - 1),
          frame(2 * n),
          observable_records(c.num_observables) {
        for (const auto& op : c.nodes)
            if (!annotation(op.gate))
                nodes.push_back(op);
        plan.num_qubits = n;
        plan.initial_active_width = plan.peak_active_width = 1;
        plan.num_visible_records = c.num_measurements;
        plan.num_detectors = c.num_detectors;
        plan.num_observables = c.num_observables;
    }
    SymbolId symbol(SymbolKind kind) {
        auto id = SymbolId{unsigned(plan.symbols.size())};
        plan.symbols.push_back(kind);
        return id;
    }
    const AstNode& next() {
        need(cursor < nodes.size(), "incomplete CSS interval");
        return nodes[cursor];
    }
    AstNode measurement(unsigned slot) {
        auto op = next();
        need(op.gate == GateType::MPP, "expected Pauli product measurement");
        ++cursor;
        if (cursor < nodes.size() && nodes[cursor].gate == GateType::READOUT_NOISE) {
            const auto& readout = nodes[cursor++];
            need(readout.targets.size() == 1 && readout.targets[0] == Target::rec(slot) &&
                     (readout.args.size() == 1 || readout.args.size() == 2),
                 "readout must immediately target its own measurement");
            op.args = readout.args;
        }
        return op;
    }
    void layer(GateType gate) {
        uint64_t seen = 0;
        while (cursor < nodes.size() && nodes[cursor].gate == gate) {
            const auto& node = nodes[cursor++];
            for (auto target : node.targets) {
                need(!target.is_rec() && !target.has_pauli() && !target.is_inverted() &&
                         target.value() < n,
                     "invalid T layer target");
                uint64_t bit = uint64_t{1} << target.value();
                need(!(seen & bit), "repeated T in layer");
                seen |= bit;
            }
        }
        need(seen == all, "T layer does not cover the code block");
    }
    void faults(std::vector<AffineBool>& inputs, unsigned layer_index) {
        while (cursor < nodes.size() && noise(nodes[cursor].gate)) {
            const auto& node = nodes[cursor++];
            std::array<double, 3> probabilities{};
            if (node.gate == GateType::PAULI_CHANNEL_1) {
                need(node.args.size() == 3, "invalid Pauli channel");
                std::copy(node.args.begin(), node.args.end(), probabilities.begin());
            } else {
                need(node.args.size() == 1, "invalid noise probability");
                if (node.gate == GateType::DEPOLARIZE1)
                    probabilities.fill(node.args[0] / 3);
                else
                    probabilities[node.gate == GateType::X_ERROR   ? 0
                                  : node.gate == GateType::Y_ERROR ? 1
                                                                   : 2] = node.args[0];
            }
            for (auto target : node.targets) {
                need(!target.is_rec() && !target.has_pauli() && !target.is_inverted() &&
                         target.value() < n,
                     "invalid noise target");
                PresampledNoiseSite site;
                for (unsigned axis = 0; axis < 3; ++axis)
                    if (probabilities[axis] > 0) {
                        auto id = symbol(SymbolKind::Presampled);
                        site.outcomes.push_back({id, probabilities[axis]});
                        site.total_probability += probabilities[axis];
                        unsigned base = 2 * n * (1 + layer_index) + 2 * target.value();
                        if (axis != 2)
                            inputs[base] ^= AffineBool::symbol(id);
                        if (axis != 0)
                            inputs[base + 1] ^= AffineBool::symbol(id);
                    }
                plan.presampled_noise_sites.push_back(std::move(site));
            }
        }
    }
    void readout(const AstNode& op, SymbolId branch, RecordSlot slot, unsigned width) {
        if (op.args.empty())
            return;
        auto flip = symbol(SymbolKind::Readout);
        plan.actions.push_back(
            {width, width,
             ApplyReadoutNoise{flip, AffineBool::symbol(branch), slot, op.args[0],
                               op.args.size() == 2 ? op.args[1] : op.args[0]}});
    }
    void outputs(unsigned width) {
        while (cursor < nodes.size() && (nodes[cursor].gate == GateType::DETECTOR ||
                                         nodes[cursor].gate == GateType::OBSERVABLE_INCLUDE)) {
            const auto& op = nodes[cursor++];
            std::vector<RecordSlot> deps;
            for (auto target : op.targets) {
                need(target.is_rec() && target.value() < record, "output reads unavailable record");
                deps.push_back(RecordSlot{target.value()});
            }
            if (op.gate == GateType::DETECTOR) {
                bool expected = detector < options.expected_detectors.size() &&
                                options.expected_detectors[detector];
                bool post = detector < options.postselection_mask.size() &&
                            options.postselection_mask[detector];
                plan.actions.push_back({width, width,
                                        WriteDetector{RecordParity(expected, std::move(deps)),
                                                      DetectorSlot{detector++}, post}});
            } else {
                need(op.args.size() == 1 && op.args[0] >= 0 && op.args[0] < circuit.num_observables,
                     "invalid observable index");
                auto& target = observable_records[unsigned(op.args[0])];
                target.insert(target.end(), deps.begin(), deps.end());
            }
        }
    }
    void block() {
        ApplyCssBlock action;
        action.inputs.resize(10 * n);
        std::copy(frame.begin(), frame.end(), action.inputs.begin());
        faults(action.inputs, 0);
        layer(GateType::T);
        faults(action.inputs, 1);
        const auto gadget = measurement(record);
        need(gadget.gate == GateType::MPP && support(gadget, 'Y', n) == all,
             "expected all-data Y measurement");
        faults(action.inputs, 2);
        layer(GateType::T_DAG);
        faults(action.inputs, 3);
        std::vector<css::Check> checks;
        std::vector<AstNode> measurements{gadget};
        for (unsigned k = 0; k < n - 1; ++k) {
            auto op = measurement(record + k + 1);
            need(op.gate == GateType::MPP && !op.targets.empty(),
                 "missing complete CSS projection");
            char axis = op.targets[0].pauli_char();
            need(axis == 'X' || axis == 'Z', "non-CSS projection");
            checks.push_back({axis == 'X', support(op, axis, n)});
            measurements.push_back(std::move(op));
        }
        if (!code)
            code = std::make_shared<css::Code>(n, checks);
        need(code->checks() == checks, "code or check order changes between blocks");
        action.code = code;
        for (unsigned k = 0; k < n; ++k) {
            action.branches.push_back(symbol(SymbolKind::Branch));
            action.records.push_back(RecordSlot{record++});
        }
        frame.assign(2 * n, AffineBool{});
        for (unsigned k = 0; k < n - 1; ++k) {
            auto [x, z] = code->duals()[k];
            for (unsigned q = 0; q < n; ++q) {
                if (x >> q & 1)
                    frame[2 * q] ^= AffineBool::symbol(action.branches[k + 1]);
                if (z >> q & 1)
                    frame[2 * q + 1] ^= AffineBool::symbol(action.branches[k + 1]);
            }
        }
        plan.actions.push_back({1, 1, action});
        for (unsigned k = 0; k < n; ++k)
            readout(measurements[k], action.branches[k], action.records[k], 1);
        outputs(1);
    }
    SamplingPlan run() {
        size_t begin = 0;
        auto starts_layer = [&](size_t at) {
            uint64_t seen = 0;
            while (at < nodes.size() && nodes[at].gate == GateType::T) {
                for (auto target : nodes[at].targets) {
                    if (target.value() >= n)
                        return false;
                    seen |= uint64_t{1} << target.value();
                }
                ++at;
            }
            return seen == all;
        };
        while (begin < nodes.size() && !noise(nodes[begin].gate) && !starts_layer(begin))
            ++begin;
        need(begin < nodes.size(), "no supported T sandwich");
        Circuit prefix;
        prefix.num_qubits = n;
        prefix.nodes.assign(nodes.begin(), nodes.begin() + begin);
        cursor = begin;
        do {
            block();
        } while (cursor < nodes.size() &&
                 (noise(nodes[cursor].gate) || nodes[cursor].gate == GateType::T));
        auto bloch = initial_bloch(prefix, *code);
        std::vector<PlannedAction> initial;
        double theta = std::acos(std::clamp(bloch[2], -1., 1.)) / std::numbers::pi;
        double phi = std::atan2(bloch[1], bloch[0]) / std::numbers::pi;
        if (theta != 0)
            initial.push_back({1, 1, RotateActivePauli{{1, 1}, theta, {}}});
        if (phi != 0)
            initial.push_back({1, 1, RotateActivePauli{{0, 1}, phi, {}}});
        plan.actions.insert(plan.actions.begin(), initial.begin(), initial.end());
        auto op = measurement(record);
        need(op.gate == GateType::MPP && !op.targets.empty(),
             "expected terminal logical measurement");
        char axis = op.targets[0].pauli_char();
        need(support(op, axis, n) == all, "unsupported terminal measurement");
        need(axis == 'X' || axis == 'Y' || axis == 'Z', "invalid logical axis");
        bool sign = axis == 'Y' && (n - 1) / 2 % 2;
        auto branch = symbol(SymbolKind::Branch);
        RecordSlot slot{record++};
        auto outcome = AffineBool::symbol(branch) ^ sign;
        plan.actions.push_back(
            {1, 0,
             MeasureActivePauli{
                 {axis != 'Z' ? 1u : 0u, axis != 'X' ? 1u : 0u}, 0, branch, outcome, slot}});
        if (!op.args.empty()) {
            auto flip = symbol(SymbolKind::Readout);
            plan.actions.push_back(
                {0, 0,
                 ApplyReadoutNoise{flip, outcome, slot, op.args[0],
                                   op.args.size() == 2 ? op.args[1] : op.args[0]}});
        }
        outputs(0);
        need(cursor == nodes.size() && record == circuit.num_measurements &&
                 detector == circuit.num_detectors,
             "unsupported circuit suffix or record layout");
        for (unsigned k = 0; k < circuit.num_observables; ++k) {
            bool expected =
                k < options.expected_observables.size() && options.expected_observables[k];
            plan.actions.push_back(
                {0, 0,
                 WriteObservable{RecordParity(expected, std::move(observable_records[k])),
                                 ObservableSlot{k}}});
        }
        plan.validate();
        return std::move(plan);
    }
};
}  // namespace
bool prefer_css_blocks(const SamplingPlan& ordinary, const SamplingPlan& candidate) {
    if (ordinary.peak_active_width <= 10)
        return false;
    double dense_work = 0, factor_work = 0;
    for (const auto& action : ordinary.actions)
        dense_work +=
            std::ldexp(double(predicted_dense_passes(action.action)), action.active_before);
    for (const auto& action : candidate.actions)
        if (const auto* block = std::get_if<ApplyCssBlock>(&action.action)) {
            // Bra/ket cross terms and both logical labels reuse each prefix;
            // twelve traversals also bound the repeated unconditioned prefix.
            factor_work += 12. * block->code->gather_entries();
        }
    return factor_work > 0 && factor_work < dense_work;
}

CssPlanningResult try_plan_css_blocks(const Circuit& circuit, SamplingPlanOptions options,
                                      unsigned minimum_data_width) {
    if (circuit.num_qubits < minimum_data_width || circuit.num_qubits > css::kMaxData ||
        circuit.num_qubits % 2 == 0)
        return {std::nullopt, "CSS data width outside selected range"};
    if (circuit.num_exp_vals || options.retain_source_map)
        return {std::nullopt,
                "CSS blocks do not support expectation probes or retained source maps"};
    try {
        return {Builder(circuit, options).run(), {}};
    } catch (const std::invalid_argument& e) {
        return {std::nullopt, e.what()};
    } catch (const Decline& e) {
        return {std::nullopt, e.what()};
    }
}
}  // namespace clifft::sampling
