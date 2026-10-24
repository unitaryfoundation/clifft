#include "clifft/optimizer/global_reconstruction.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/optimizer/reconstruction_internal.h"
#include "clifft/tableau/tableau.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <map>
#include <numeric>
#include <utility>

namespace clifft {
namespace {

using namespace detail_reconstruction;

Bits copy(MaskView a) {
    return Bits(a.words.begin(), a.words.end());
}

bool intersects(const Bits& a, const Bits& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (a[i] & b[i])
            return true;
    return false;
}

PauliString axis(uint32_t n, const Bits& x, const Bits& z, bool sign = false) {
    PauliString p(n);
    each_bit(x, [&](uint32_t i) { p.mut_x().bit_set(i, true); });
    each_bit(z, [&](uint32_t i) { p.mut_z().bit_set(i, true); });
    p.set_sign(sign);
    return p;
}

void xor_sign(ReconstructionSign& a, const ReconstructionSign& b) {
    xor_into(a.words, b.words);
    a.constant ^= b.constant;
}

struct Output {
    Bits z;
    ReconstructionSign sign;
};

struct Rotation {
    uint32_t ancilla;
    double half_turns;
    ReconstructionSign sign;
};

struct Clifford {
    GateType gate;
    std::vector<uint32_t> targets;
};

class Dilation {
  public:
    explicit Dilation(const ReconstructionLimits& limits) : limits_(limits) {}

    void parse(const Circuit& circuit);
    GlobalReconstruction reconstruct(ReconstructionOutputs outputs);

  private:
    uint32_t fresh() {
        require_supported(n_ < limits_.max_dilation_qubits, "dilation qubit limit exceeded");
        fx_.emplace_back();
        fz_.emplace_back();
        return n_++;
    }
    uint32_t qubit(uint32_t source) {
        if (auto it = aliases_.find(source); it != aliases_.end()) {
            return it->second;
        }
        const uint32_t q = fresh();
        aliases_[source] = q;
        return q;
    }
    void gate(GateType type, std::vector<uint32_t> targets) {
        require_supported(body_.size() < limits_.max_operations * 8,
                          "dilation operation limit exceeded");
        body_.push_back({type, std::move(targets)});
    }
    uint32_t noise_bits(uint32_t count) {
        require_supported(count <= limits_.max_noise_bits - error_bits_,
                          "noise bit limit exceeded");
        const uint32_t start = error_bits_;
        error_bits_ += count;
        return start;
    }
    ReconstructionSign bit_error(double p) {
        if (p == 0) {
            return {};
        }
        const uint32_t i = noise_bits(1);
        channels_.push_back({{1 - p, {}}, {p, {i}}});
        return {unit(i), false};
    }
    void error(uint32_t q, char p, const ReconstructionSign& sign) {
        if (p == 'X' || p == 'Y') {
            xor_sign(fx_[q], sign);
        }
        if (p == 'Z' || p == 'Y') {
            xor_sign(fz_[q], sign);
        }
    }
    using Term = std::vector<std::pair<uint32_t, char>>;
    ReconstructionSign noise_sign(const Term& term) const {
        ReconstructionSign sign;
        for (auto [q, p] : term) {
            if (p == 'Z' || p == 'Y') {
                xor_sign(sign, fx_[q]);
            }
            if (p == 'X' || p == 'Y') {
                xor_sign(sign, fz_[q]);
            }
        }
        return sign;
    }
    void controlled(uint32_t control, const Term& term) {
        for (auto [q, p] : term) {
            gate(p == 'X' ? GateType::CX : p == 'Y' ? GateType::CY : GateType::CZ, {control, q});
        }
    }
    void measure(const Term& term, bool inverted) {
        const uint32_t a = fresh();
        gate(GateType::H, {a});
        controlled(a, term);
        gate(GateType::H, {a});
        auto sign = noise_sign(term);
        sign.constant ^= inverted;
        records_.push_back({unit(a), std::move(sign)});
    }
    Output parity(const std::vector<Target>& targets) const {
        Output result;
        for (Target t : targets) {
            require_supported(t.is_rec() && t.value() < records_.size(),
                              "unsupported output reference");
            xor_into(result.z, records_[t.value()].z);
            xor_sign(result.sign, records_[t.value()].sign);
        }
        return result;
    }
    void clifford(const AstNode& node);

    const ReconstructionLimits& limits_;
    uint32_t n_ = 0;
    uint32_t error_bits_ = 0;
    std::map<uint32_t, uint32_t> aliases_;
    std::vector<ReconstructionSign> fx_, fz_;
    std::vector<Clifford> body_;
    std::vector<Rotation> rotations_;
    std::vector<Output> records_, detectors_;
    std::vector<Output> observables_;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels_;
};

void Dilation::clifford(const AstNode& node) {
    require_supported(
        is_clifford(node.gate) && node.gate != GateType::SPP && node.gate != GateType::SPP_DAG,
        "unsupported source operation");
    const auto local = Tableau::from_named_gate(node.gate);
    const uint32_t arity = local.num_qubits();
    require_supported(node.targets.size() == arity, "unsupported Clifford operands");
    if (std::any_of(node.targets.begin(), node.targets.end(),
                    [](Target t) { return t.is_rec(); })) {
        require_supported(arity == 2 && node.targets[0].is_rec() && !node.targets[1].is_rec() &&
                              (node.gate == GateType::CX || node.gate == GateType::CY ||
                               node.gate == GateType::CZ),
                          "unsupported feedback");
        const auto& r = records_.at(node.targets[0].value());
        const uint32_t control = static_cast<uint32_t>(highest(r.z));
        const uint32_t q = qubit(node.targets[1].value());
        gate(node.gate, {control, q});
        error(q, node.gate == GateType::CX ? 'X' : node.gate == GateType::CY ? 'Y' : 'Z', r.sign);
        return;
    }
    std::vector<uint32_t> qs;
    for (Target t : node.targets) {
        qs.push_back(qubit(t.value()));
    }
    gate(node.gate, qs);
    std::vector<ReconstructionSign> xs(arity), zs(arity);
    for (uint32_t j = 0; j < arity; ++j) {
        for (uint32_t i = 0; i < arity; ++i) {
            if (local.x_output(j).x().bit_get(i))
                xor_sign(xs[i], fx_[qs[j]]);
            if (local.x_output(j).z().bit_get(i))
                xor_sign(zs[i], fx_[qs[j]]);
            if (local.z_output(j).x().bit_get(i))
                xor_sign(xs[i], fz_[qs[j]]);
            if (local.z_output(j).z().bit_get(i))
                xor_sign(zs[i], fz_[qs[j]]);
        }
    }
    for (uint32_t i = 0; i < arity; ++i) {
        fx_[qs[i]] = std::move(xs[i]);
        fz_[qs[i]] = std::move(zs[i]);
    }
}

void Dilation::parse(const Circuit& circuit) {
    require_supported(circuit.nodes.size() <= limits_.max_operations,
                      "source operation limit exceeded");
    require_supported(circuit.num_exp_vals == 0, "expectation values require the original program");
    require_supported(
        circuit.num_measurements <= limits_.max_outputs &&
            uint64_t{circuit.num_detectors} + circuit.num_observables <= limits_.max_outputs,
        "output limit exceeded");
    observables_.resize(circuit.num_observables);
    std::optional<size_t> chain;
    for (const auto& node : circuit.nodes) {
        const auto g = node.gate;
        if (g != GateType::CORRELATED_ERROR && g != GateType::ELSE_CORRELATED_ERROR)
            chain.reset();
        if (g == GateType::TICK)
            continue;
        if (g == GateType::DETECTOR) {
            detectors_.push_back(parity(node.targets));
        } else if (g == GateType::OBSERVABLE_INCLUDE) {
            const auto p = parity(node.targets);
            auto& output = observables_.at(static_cast<size_t>(node.args.at(0)));
            xor_into(output.z, p.z);
            xor_sign(output.sign, p.sign);
        } else if (g == GateType::T || g == GateType::T_DAG) {
            const uint32_t q = qubit(node.targets.at(0).value());
            const uint32_t a = fresh();
            rotations_.push_back({a, g == GateType::T ? 0.25 : -0.25, noise_sign({{q, 'Z'}})});
            gate(GateType::H, {a});
            controlled(a, {{q, 'Z'}});
            gate(GateType::H, {a});
        } else if (g == GateType::M || g == GateType::MX || g == GateType::MY ||
                   g == GateType::MR || g == GateType::MRX || g == GateType::MRY ||
                   g == GateType::R || g == GateType::RX || g == GateType::RY) {
            const Target t = node.targets.at(0);
            uint32_t q = qubit(t.value());
            const char p = (g == GateType::MX || g == GateType::MRX || g == GateType::RX)   ? 'X'
                           : (g == GateType::MY || g == GateType::MRY || g == GateType::RY) ? 'Y'
                                                                                            : 'Z';
            if (is_measurement(g))
                measure({{q, p}}, t.is_inverted());
            if (is_reset(g) || is_measure_reset(g)) {
                // The old wire remains an environment, including entanglement
                // with deferred records. Reusing it would change reset's law.
                q = fresh();
                aliases_[t.value()] = q;
                if (p != 'Z')
                    gate(GateType::H, {q});
                if (p == 'Y')
                    gate(GateType::S, {q});
            }
        } else if (g == GateType::MPP) {
            Term term;
            bool inverted = false;
            for (Target t : node.targets) {
                term.emplace_back(qubit(t.value()), t.pauli_char());
                inverted ^= t.is_inverted();
            }
            measure(term, inverted);
        } else if (g == GateType::MPAD) {
            records_.push_back({unit(fresh()), {{}, node.targets.at(0).value() != 0}});
        } else if (g == GateType::READOUT_NOISE) {
            require_supported(node.args.size() == 1 || node.args[0] == node.args[1],
                              "asymmetric readout requires the original program");
            for (Target t : node.targets)
                xor_sign(records_.at(t.value()).sign, bit_error(node.args.at(0)));
        } else if (g == GateType::X_ERROR || g == GateType::Y_ERROR || g == GateType::Z_ERROR) {
            const uint32_t q = qubit(node.targets.at(0).value());
            error(q,
                  g == GateType::X_ERROR   ? 'X'
                  : g == GateType::Y_ERROR ? 'Y'
                                           : 'Z',
                  bit_error(node.args.at(0)));
        } else if (g == GateType::DEPOLARIZE1 || g == GateType::DEPOLARIZE2 ||
                   g == GateType::PAULI_CHANNEL_1 || g == GateType::PAULI_CHANNEL_2) {
            const uint32_t arity = static_cast<uint32_t>(node.targets.size());
            const uint32_t count = uint32_t{1} << (2 * arity);
            const uint32_t start = noise_bits(2 * arity);
            const bool depol = g == GateType::DEPOLARIZE1 || g == GateType::DEPOLARIZE2;
            const double total =
                depol ? node.args[0] : std::accumulate(node.args.begin(), node.args.end(), 0.0);
            std::vector<ReconstructionNoiseOutcome> channel;
            for (uint32_t value = 0; value < count; ++value) {
                const double p = value == 0 ? std::max(0.0, 1 - total)
                                 : depol    ? total / (count - 1)
                                            : node.args[value - 1];
                if (p == 0)
                    continue;
                ReconstructionNoiseOutcome outcome{p, {}};
                for (uint32_t j = 0; j < arity; ++j) {
                    const uint32_t digit = (value >> (2 * (arity - j - 1))) & 3U;
                    if (digit == 1 || digit == 2)
                        outcome.bits.push_back(start + 2 * j);
                    if (digit == 2 || digit == 3)
                        outcome.bits.push_back(start + 2 * j + 1);
                }
                channel.push_back(std::move(outcome));
            }
            channels_.push_back(std::move(channel));
            for (uint32_t j = 0; j < arity; ++j) {
                const uint32_t q = qubit(node.targets[j].value());
                xor_sign(fx_[q], {unit(start + 2 * j), false});
                xor_sign(fz_[q], {unit(start + 2 * j + 1), false});
            }
        } else if (g == GateType::CORRELATED_ERROR || g == GateType::ELSE_CORRELATED_ERROR) {
            if (g == GateType::CORRELATED_ERROR) {
                chain = channels_.size();
                channels_.push_back({{1, {}}});
            }
            require_supported(chain.has_value(), "unpaired correlated error branch");
            const uint32_t i = noise_bits(1);
            auto& channel = channels_[*chain];
            const double p = channel.front().probability * node.args.at(0);
            channel.front().probability -= p;
            channel.push_back({p, {i}});
            for (Target t : node.targets)
                error(qubit(t.value()), t.pauli_char(), {unit(i), false});
        } else {
            clifford(node);
        }
    }
}

// Complete commuting stabilizers to a symplectic basis. This construction
// happens only during compilation; the executor receives no tableau work.
Tableau stabilizer_frame(const std::vector<PauliString>& generators, uint32_t n) {
    Basis columns;
    for (uint32_t i = 0; i < 2 * n; ++i) {
        Bits pattern;
        for (uint32_t j = 0; j < n; ++j) {
            if ((i < n ? generators[j].z() : generators[j].x()).bit_get(i % n))
                toggle(pattern, j);
        }
        columns.insert(std::move(pattern), unit(i));
    }
    require_supported(columns.size() == n, "dependent folded projections");
    Tableau frame(n);
    std::vector<PauliString> duals;
    for (uint32_t i = 0; i < n; ++i) {
        const auto solution = columns.solve(unit(i));
        assert(solution.has_value());
        PauliString p(n);
        each_bit(*solution, [&](uint32_t q) {
            if (q < n)
                p.mut_x().bit_set(q, true);
            else
                p.mut_z().bit_set(q - n, true);
        });
        p.set_sign(false);
        for (uint32_t j = 0; j < i; ++j) {
            if (!p.view().commutes(duals[j].view()))
                p.right_multiply(generators[j].view());
        }
        frame.set_x_output(i, p.view());
        frame.set_z_output(i, generators[i].view());
        duals.push_back(std::move(p));
    }
    return frame.inverse();
}

void schedule_reconstruction(GlobalReconstruction& result, const ReconstructionLimits& limits) {
    const uint32_t n = result.num_qubits;
    auto& ops = result.operations;
    require_supported(ops.size() + n <= limits.max_schedule_operations,
                      "schedule operation limit exceeded");
    Basis columns;
    std::vector<Bits> nullspace;
    for (uint32_t i = 0; i < 2 * n; ++i) {
        Bits column;
        for (uint32_t j = 0; j < ops.size(); ++j) {
            if ((i < n ? ops[j].axis.z() : ops[j].axis.x()).bit_get(i % n))
                toggle(column, j);
        }
        if (auto dependency = columns.insert(std::move(column), unit(i)))
            nullspace.push_back(std::move(*dependency));
    }
    auto weight = [](const Bits& value) {
        size_t count = 0;
        for (auto w : value)
            count += std::popcount(w);
        return count;
    };
    std::stable_sort(nullspace.begin(), nullspace.end(),
                     [&](const Bits& a, const Bits& b) { return weight(a) < weight(b); });
    Basis xparts;
    std::vector<ReconstructionOperation> hidden;
    for (const auto& value : nullspace) {
        PauliString p(n);
        each_bit(value, [&](uint32_t i) {
            if (i < n)
                p.mut_x().bit_set(i, true);
            else
                p.mut_z().bit_set(i - n, true);
        });
        p.set_sign(false);
        if (xparts.solve(copy(p.x())))
            continue;
        if (std::any_of(hidden.begin(), hidden.end(),
                        [&](const auto& h) { return !p.view().commutes(h.axis.view()); }))
            continue;
        xparts.insert(copy(p.x()), {});
        hidden.push_back({std::move(p), {}, 0, -1});
    }
    ops.insert(ops.begin(), std::make_move_iterator(hidden.begin()),
               std::make_move_iterator(hidden.end()));
    const uint32_t count = static_cast<uint32_t>(ops.size());
    std::vector<Bits> predecessors(count), ancestors(count);
    size_t steps = 0;
    auto budget = [&] {
        require_supported(steps++ < limits.max_planning_steps, "schedule work limit exceeded");
    };
    for (uint32_t j = 0; j < count; ++j) {
        for (uint32_t i = 0; i < j; ++i) {
            budget();
            if (!ops[i].axis.view().commutes(ops[j].axis.view()))
                toggle(predecessors[j], i);
        }
        ancestors[j] = predecessors[j];
        each_bit(predecessors[j], [&](uint32_t i) {
            ancestors[j].resize(std::max(ancestors[j].size(), ancestors[i].size()), 0);
            for (size_t w = 0; w < ancestors[i].size(); ++w)
                ancestors[j][w] |= ancestors[i][w];
        });
    }
    Bits pending;
    for (uint32_t i = 0; i < count; ++i)
        toggle(pending, i);
    DormantSubspace group(n);
    std::vector<uint32_t> order;
    auto apply = [&](DormantSubspace& state, uint32_t j) {
        budget();
        if (ops[j].half_turns != 0)
            state.apply_rotation(ops[j].axis);
        else
            state.apply_measurement(ops[j].axis);
    };
    auto run = [&](uint32_t j) {
        apply(group, j);
        toggle(pending, j);
        order.push_back(j);
        result.peak_active_width = std::max(result.peak_active_width, group.active_width());
    };
    while (highest(pending) >= 0) {
        bool progress;
        do {
            progress = false;
            const auto snapshot = pending;
            each_bit(snapshot, [&](uint32_t j) {
                if (!intersects(predecessors[j], pending) &&
                    (ops[j].half_turns == 0 || group.commutes_with_all(ops[j].axis))) {
                    run(j);
                    progress = true;
                }
            });
        } while (progress);
        uint32_t best = count;
        uint32_t lowest = n + 1;
        std::vector<uint32_t> best_chain;
        each_bit(pending, [&](uint32_t j) {
            if (ops[j].half_turns != 0)
                return;
            std::vector<uint32_t> chain;
            each_bit(ancestors[j], [&](uint32_t i) {
                if ((pending[i / 64] >> (i % 64)) & 1U)
                    chain.push_back(i);
            });
            auto trial = group;
            auto highest_width = trial.active_width();
            for (auto k : chain) {
                apply(trial, k);
                highest_width = std::max(highest_width, trial.active_width());
                if (highest_width >= lowest)
                    break;
            }
            if (highest_width < lowest) {
                lowest = highest_width;
                best = j;
                best_chain = std::move(chain);
            }
        });
        if (best == count) {
            const auto snapshot = pending;
            each_bit(snapshot, run);
        } else {
            for (auto j : best_chain)
                run(j);
            run(best);
        }
    }
    std::vector<ReconstructionOperation> scheduled;
    scheduled.reserve(count);
    for (auto j : order)
        scheduled.push_back(std::move(ops[j]));
    ops = std::move(scheduled);
}

GlobalReconstruction Dilation::reconstruct(ReconstructionOutputs outputs) {
    Tableau backward(n_);
    // Prepending inverses in forward order builds the inverse circuit while
    // touching only each gate's generator rows instead of every tableau row.
    for (const auto& gate : body_) {
        backward.prepend_local(Tableau::from_named_gate(gate.gate).inverse(), gate.targets);
    }
    std::vector<bool> magic(n_, false);
    for (const auto& r : rotations_)
        magic[r.ancilla] = true;
    std::vector<PauliString> initial, constraints;
    for (uint32_t i = 0; i < n_; ++i)
        initial.push_back(axis(n_, magic[i] ? unit(i) : Bits{}, magic[i] ? Bits{} : unit(i)));
    for (const auto& r : rotations_)
        constraints.emplace_back(backward.z_output(r.ancilla));
    auto pattern = [&](PauliStringView p) {
        Bits result;
        for (uint32_t j = 0; j < constraints.size(); ++j)
            if (!p.commutes(constraints[j].view()))
                toggle(result, j);
        return result;
    };
    std::vector<Bits> patterns;
    Basis allowed;
    for (uint32_t i = 0; i < n_; ++i) {
        patterns.push_back(pattern(initial[i].view()));
        if (!magic[i])
            allowed.insert(patterns.back(), unit(i));
    }
    GlobalReconstruction result;
    result.outputs = {
        outputs == ReconstructionOutputs::Records ? static_cast<uint32_t>(records_.size()) : 0,
        static_cast<uint32_t>(detectors_.size()), static_cast<uint32_t>(observables_.size())};
    result.dilation_qubits = n_;
    result.num_noise_bits = error_bits_;
    for (auto it = rotations_.rbegin(); it != rotations_.rend(); ++it) {
        auto p = axis(n_, {}, unit(it->ancilla));
        const auto dressing = allowed.solve(pattern(p.view()));
        require_supported(dressing.has_value(), "no causal stabilizer dressing");
        each_bit(*dressing, [&](uint32_t i) { p.right_multiply(initial[i].view()); });
        result.operations.push_back({std::move(p), it->sign, it->half_turns, -1});
        allowed.insert(patterns[it->ancilla], unit(it->ancilla));
    }
    std::reverse(result.operations.begin(), result.operations.end());
    Basis independent;
    std::vector<PauliString> generators = constraints;
    for (uint32_t i = 0; i < n_; ++i) {
        if (auto dependency = independent.insert(patterns[i], unit(i))) {
            PauliString p(n_);
            each_bit(*dependency, [&](uint32_t j) { p.right_multiply(initial[j].view()); });
            generators.push_back(std::move(p));
        }
    }
    // Full projection rank certifies a state-independent normalization factor.
    require_supported(independent.size() == constraints.size(), "nonuniform gadget projection");
    assert(generators.size() == n_);
    const auto frame = stabilizer_frame(generators, n_);
    std::vector<Output> requested;
    if (outputs == ReconstructionOutputs::Records)
        requested = records_;
    requested.insert(requested.end(), detectors_.begin(), detectors_.end());
    requested.insert(requested.end(), observables_.begin(), observables_.end());
    for (uint32_t i = 0; i < requested.size(); ++i) {
        const auto& out = requested[i];
        result.operations.push_back(
            {backward.apply(axis(n_, {}, out.z).view()), out.sign, 0, static_cast<int32_t>(i)});
    }
    Basis xb;
    std::vector<Bits> directions;
    for (auto& op : result.operations) {
        op.axis = frame.apply(op.axis.view());
        const Bits x = copy(op.axis.x());
        if (!xb.solve(x)) {
            xb.insert(x, unit(static_cast<uint32_t>(directions.size())));
            directions.push_back(x);
        }
    }
    result.num_qubits = static_cast<uint32_t>(directions.size());
    for (uint32_t i = 0; i < n_; ++i) {
        if (!xb.solve(unit(i))) {
            xb.insert(unit(i), unit(static_cast<uint32_t>(directions.size())));
            directions.push_back(unit(i));
        }
    }
    std::vector<Bits> inverse_rows(n_);
    for (uint32_t i = 0; i < n_; ++i)
        each_bit(*xb.solve(unit(i)), [&](uint32_t j) { toggle(inverse_rows[j], i); });
    Tableau linear(n_);
    for (uint32_t i = 0; i < n_; ++i) {
        linear.set_x_output(i, axis(n_, directions[i], {}).view());
        linear.set_z_output(i, axis(n_, {}, inverse_rows[i]).view());
    }
    linear = linear.inverse();
    for (auto& op : result.operations) {
        const auto p = linear.apply(op.axis.view());
        PauliString reduced(result.num_qubits);
        for (uint32_t i = 0; i < n_; ++i) {
            if (i < result.num_qubits)
                reduced.set_pauli(i, p.x().bit_get(i), p.z().bit_get(i));
            else
                assert(!p.x().bit_get(i));
        }
        reduced.set_sign(false);
        op.sign.constant ^= p.sign();
        op.axis = std::move(reduced);
    }
    result.noise_channels = std::move(channels_);
    schedule_reconstruction(result, limits_);
    return result;
}

}  // namespace

ReconstructionResult reconstruct_global_sampling(const Circuit& circuit,
                                                 ReconstructionOutputs outputs,
                                                 const ReconstructionLimits& limits) {
    try {
        Dilation dilation(limits);
        dilation.parse(circuit);
        return {dilation.reconstruct(outputs), {}};
    } catch (const Ineligible& e) {
        return {std::nullopt, e.reason};
    }
}

}  // namespace clifft
