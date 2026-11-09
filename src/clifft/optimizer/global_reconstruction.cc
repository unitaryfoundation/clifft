#include "clifft/optimizer/global_reconstruction.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/tableau/tableau.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <utility>

namespace clifft {
namespace {

using Bits = std::vector<uint64_t>;

// Each source operation expands into at most a small constant number of
// elementary Clifford dilation gates (e.g., H + controlled-Pauli + H).
constexpr size_t kMaxDilationGatesPerSourceOp = 8;
constexpr double kTHalfTurns = 0.25;

void toggle(Bits& a, uint32_t i) {
    a.resize(std::max(a.size(), size_t{i / 64 + 1}), 0);
    a[i / 64] ^= uint64_t{1} << (i % 64);
}

void xor_into(Bits& a, const Bits& b) {
    a.resize(std::max(a.size(), b.size()), 0);
    for (size_t i = 0; i < b.size(); ++i) {
        a[i] ^= b[i];
    }
}

std::optional<uint32_t> highest(const Bits& a) {
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i]) {
            return static_cast<uint32_t>(i * 64 + 63 - std::countl_zero(a[i]));
        }
    }
    return std::nullopt;
}

template <class F>
void each_bit(const Bits& a, F&& f) {
    for (size_t w = 0; w < a.size(); ++w) {
        for (uint64_t v = a[w]; v; v &= v - 1) {
            f(static_cast<uint32_t>(w * 64 + std::countr_zero(v)));
        }
    }
}

Bits unit(uint32_t i) {
    Bits a;
    toggle(a, i);
    return a;
}

Bits copy(MaskView a) {
    return Bits(a.words.begin(), a.words.end());
}

Bits canonical(Bits value) {
    while (!value.empty() && value.back() == 0) {
        value.pop_back();
    }
    return value;
}

bool intersects(const Bits& a, const Bits& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] & b[i]) {
            return true;
        }
    }
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

// Incremental GF(2) row-echelon basis keyed by highest set bit, tracking the
// input-row combination (`label`) that produces each basis vector.
class Basis {
  public:
    std::optional<Bits> insert(Bits value, Bits label) {
        reduce(value, label);
        const auto pivot = highest(value);
        if (!pivot) {
            return label;
        }
        rows_.emplace(*pivot, std::make_pair(std::move(value), std::move(label)));
        return std::nullopt;
    }

    std::optional<Bits> solve(Bits value) const {
        Bits label;
        reduce(value, label);
        return !highest(value) ? std::optional<Bits>{std::move(label)} : std::nullopt;
    }

    size_t size() const { return rows_.size(); }

  private:
    void reduce(Bits& value, Bits& label) const {
        for (;;) {
            const auto pivot = highest(value);
            if (!pivot) {
                return;
            }
            const auto it = rows_.find(*pivot);
            if (it == rows_.end()) {
                return;
            }
            xor_into(value, it->second.first);
            xor_into(label, it->second.second);
        }
    }
    std::map<uint32_t, std::pair<Bits, Bits>> rows_;
};

struct Ineligible {
    std::string reason;
};

void require_supported(bool condition, const char* reason) {
    if (!condition) {
        throw Ineligible{reason};
    }
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

// Purifies a noisy Clifford+T circuit by deferring measurements to ancillas,
// allocating fresh wires on reset, and propagating Pauli fault symbols forward.
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
        require_supported(body_.size() < limits_.max_operations * kMaxDilationGatesPerSourceOp,
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
                              (node.gate == GateType::CX || node.gate == GateType::CZ),
                          "unsupported feedback");
        const auto& r = records_.at(node.targets[0].value());
        const auto control = highest(r.z);
        assert(control.has_value());
        const uint32_t q = qubit(node.targets[1].value());
        gate(node.gate, {*control, q});
        error(q, node.gate == GateType::CX ? 'X' : 'Z', r.sign);
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
            rotations_.push_back(
                {a, g == GateType::T ? kTHalfTurns : -kTHalfTurns, noise_sign({{q, 'Z'}})});
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
                // Keep the discarded wire as an environment so entanglement
                // with deferred measurement ancillas is preserved.
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
            const Target t = node.targets.at(0);
            const bool sign = (t.value() != 0) ^ t.is_inverted();
            records_.push_back({unit(fresh()), {{}, sign}});
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

// Completes n independent commuting Pauli generators to a full symplectic
// frame by solving for dual X destabilizers and orthogonalizing them.
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

    // 1. Synthesize commuting hidden measurements from the nullspace of
    // operation commutation columns to shrink the initial active subspace.
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
        hidden.push_back({std::move(p), {}, 0, std::nullopt});
    }
    ops.insert(ops.begin(), std::make_move_iterator(hidden.begin()),
               std::make_move_iterator(hidden.end()));

    // 2. Build the DAG of non-commuting predecessors and transitive ancestors.
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

    // 3. Schedule ready operations that do not expand the active subspace, then
    // unlock the measurement chain with the lowest peak active width.
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
    while (highest(pending)) {
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
        std::optional<uint32_t> best;
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
        if (!best) {
            const auto snapshot = pending;
            each_bit(snapshot, run);
        } else {
            for (auto j : best_chain)
                run(j);
            run(*best);
        }
    }
    std::vector<ReconstructionOperation> scheduled;
    scheduled.reserve(count);
    for (auto j : order)
        scheduled.push_back(std::move(ops[j]));
    ops = std::move(scheduled);
}

GlobalReconstruction Dilation::reconstruct(ReconstructionOutputs outputs) {
    // Parsing has copied the fault signs needed by rotations and outputs.
    // Release the per-wire signs before allocating the reconstruction frames.
    fx_.clear();
    fz_.clear();

    // 1. Build the inverse Clifford frame by prepending local gate inverses in
    // forward order, touching only each gate's operand rows.
    Tableau backward(n_);
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

    // 2. Dress each T/T_DAG gadget axis with input stabilizers so it commutes
    // with all earlier gadget projections.
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
        result.operations.push_back({std::move(p), it->sign, it->half_turns, std::nullopt});
        allowed.insert(patterns[it->ancilla], unit(it->ancilla));
    }
    std::reverse(result.operations.begin(), result.operations.end());

    // 3. Complete the independent gadget projections to an n-qubit stabilizer
    // basis and fold them into the input state. Full projection rank certifies
    // a state-independent normalization factor.
    Basis independent;
    std::vector<PauliString> generators = constraints;
    for (uint32_t i = 0; i < n_; ++i) {
        if (auto dependency = independent.insert(patterns[i], unit(i))) {
            PauliString p(n_);
            each_bit(*dependency, [&](uint32_t j) { p.right_multiply(initial[j].view()); });
            generators.push_back(std::move(p));
        }
    }
    require_supported(independent.size() == constraints.size(), "nonuniform gadget projection");
    assert(generators.size() == n_);
    const auto frame = stabilizer_frame(generators, n_);

    // 4. Map requested outputs through the inverse frame and reduce all
    // operations to a basis of their X supports, eliminating fixed-|0> wires.
    std::vector<Output> requested;
    if (outputs == ReconstructionOutputs::Records)
        requested = std::move(records_);
    requested.insert(requested.end(), std::make_move_iterator(detectors_.begin()),
                     std::make_move_iterator(detectors_.end()));
    requested.insert(requested.end(), std::make_move_iterator(observables_.begin()),
                     std::make_move_iterator(observables_.end()));
    for (uint32_t i = 0; i < requested.size(); ++i) {
        auto& out = requested[i];
        result.operations.push_back(
            {backward.apply(axis(n_, {}, out.z).view()), std::move(out.sign), 0, i});
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

ReconstructionNoise compress_noise(const GlobalReconstruction& r,
                                   const ReconstructionLimits& limits) {
    // 1. Build a basis for the affine noise-bit masks referenced by operations,
    // prioritizing public outputs before hidden operations.
    Basis basis;
    std::vector<Bits> rows;
    std::vector<size_t> indices(r.operations.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return !r.operations[a].output.has_value() < !r.operations[b].output.has_value();
    });
    for (size_t i : indices) {
        const auto& terms = r.operations[i].sign.words;
        if (!basis.solve(terms)) {
            basis.insert(terms, unit(static_cast<uint32_t>(rows.size())));
            rows.push_back(terms);
        }
    }
    std::vector<Bits> columns(r.num_noise_bits);
    for (uint32_t j = 0; j < rows.size(); ++j) {
        each_bit(rows[j], [&](uint32_t i) { toggle(columns.at(i), j); });
    }

    // 2. Project each categorical channel onto the basis coordinates and
    // convolve channels that share the same image support.
    using Distribution = std::map<Bits, double>;
    std::map<std::vector<Bits>, Distribution> grouped;
    size_t products = 0;
    for (const auto& channel : r.noise_channels) {
        Distribution image;
        for (const auto& outcome : channel) {
            if (outcome.probability == 0)
                continue;
            Bits value;
            for (auto b : outcome.bits)
                xor_into(value, columns.at(b));
            image[canonical(std::move(value))] += outcome.probability;
        }
        if (image.size() == 1 && image.begin()->first.empty())
            continue;
        std::vector<Bits> key;
        for (const auto& [value, probability] : image)
            key.push_back(value);
        auto existing = grouped.find(key);
        if (existing != grouped.end()) {
            Distribution convolved;
            for (const auto& [a, p] : existing->second) {
                for (const auto& [b, q] : image) {
                    require_supported(products++ < limits.max_channel_products,
                                      "channel convolution work limit exceeded");
                    Bits sum = a;
                    xor_into(sum, b);
                    convolved[canonical(std::move(sum))] += p * q;
                    require_supported(convolved.size() <= limits.max_channel_support,
                                      "channel support limit exceeded");
                }
            }
            existing->second = std::move(convolved);
        } else {
            require_supported(image.size() <= limits.max_channel_support,
                              "channel support limit exceeded");
            grouped.emplace(std::move(key), std::move(image));
        }
    }
    ReconstructionNoise result;
    result.num_bits = static_cast<uint32_t>(rows.size());
    for (const auto& op : r.operations)
        result.signs.push_back({*basis.solve(op.sign.words), op.sign.constant});
    for (const auto& [key, distribution] : grouped) {
        auto& channel = result.channels.emplace_back();
        for (const auto& [value, probability] : distribution) {
            ReconstructionNoiseOutcome outcome{probability, {}};
            each_bit(value, [&](uint32_t i) { outcome.bits.push_back(i); });
            channel.push_back(std::move(outcome));
        }
    }
    return result;
}

HirModule lower_reconstruction(const GlobalReconstruction& r, const ReconstructionNoise& noise,
                               const PostselectionMetadata& postselection,
                               const ReconstructionLimits& limits) {
    // 1. Lay out quantum wires [0, num_qubits), auxiliary fault-bit wires
    // [num_qubits, num_qubits + num_bits), and one scratch wire (`carrier`) for
    // assembling visible record parities.
    const uint32_t visible = r.outputs.num_records;
    const uint32_t width = r.num_qubits + noise.num_bits + 1;
    const uint32_t carrier = width - 1;
    size_t masks = noise.num_bits + 1;
    for (const auto& sign : noise.signs) {
        size_t terms = 0;
        for (auto w : sign.words)
            terms += std::popcount(w);
        masks += 2 * (terms + 2) + 2;
    }
    size_t noise_masks = 0;
    for (const auto& channel : noise.channels)
        noise_masks += channel.size();
    const size_t bytes_per_mask = 2 * ((size_t{width} + 63) / 64) * sizeof(uint64_t) + 1;
    require_supported(masks + noise_masks <= limits.max_lowering_bytes / bytes_per_mask,
                      "lowering mask memory limit exceeded");
    HirModule hir(width, masks, noise_masks);
    hir.num_measurements = visible;
    hir.num_detectors = r.outputs.num_detectors;
    hir.num_observables = r.outputs.num_observables;
    hir.postselection.mask = postselection.mask;
    hir.postselection.expected_detectors = postselection.expected_detectors;
    hir.postselection.expected_observables = postselection.expected_observables;

    // 2. Emit compressed categorical noise channels as X flips on the auxiliary
    // fault-bit wires, then measure each fault wire into a hidden record.
    for (const auto& channel : noise.channels) {
        NoiseSite site;
        for (const auto& outcome : channel) {
            if (outcome.bits.empty() || outcome.probability == 0)
                continue;
            const auto handle = hir.claim_empty_noise_channel_mask();
            auto slot = hir.noise_channel_masks.mut_at(handle);
            for (uint32_t i : outcome.bits)
                slot.x().bit_set(r.num_qubits + i, true);
            site.channels.push_back({handle, outcome.probability});
            site.total_probability += outcome.probability;
        }
        // Clamp minor floating-point excess from convolution so the noise site
        // remains a valid probability distribution.
        if (site.total_probability > 1) {
            for (auto& c : site.channels)
                c.prob /= site.total_probability;
            site.total_probability = 1;
        }
        const auto index = static_cast<uint32_t>(hir.noise_sites.size());
        hir.noise_sites.push_back(std::move(site));
        hir.append_noise(NoiseSiteIdx{index});
    }
    auto hidden_measure = [&](auto&& fill) {
        const auto record = visible + hir.num_hidden_measurements++;
        hir.append_measure(MeasRecordIdx{record}, fill).set_hidden(true);
        return record;
    };
    std::vector<uint32_t> fault_records;
    for (uint32_t i = 0; i < noise.num_bits; ++i) {
        fault_records.push_back(
            hidden_measure([&](MutablePauliMaskView p) { p.z().bit_set(r.num_qubits + i, true); }));
    }
    const auto one = hidden_measure([](MutablePauliMaskView p) { p.set_sign(true); });

    // 3. Emit scheduled rotations and measurements.
    std::vector<std::vector<uint32_t>> output_parities(r.outputs.num_outputs());
    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
        std::vector<uint32_t> parity;
        each_bit(noise.signs[j].words, [&](uint32_t i) { parity.push_back(fault_records[i]); });
        auto fill_axis = [&](MutablePauliMaskView p) {
            for (uint32_t w = 0; w < op.axis.x().num_words(); ++w) {
                p.x().words[w] = op.axis.x().words[w];
                p.z().words[w] = op.axis.z().words[w];
            }
        };
        if (op.half_turns != 0) {
            const uint32_t pivot = std::min(op.axis.x().lowest_bit(), op.axis.z().lowest_bit());
            if (pivot >= r.num_qubits)
                continue;
            auto feedback = [&] {
                for (auto record : parity)
                    hir.append_conditional(ControllingMeasIdx{record}, [&](MutablePauliMaskView p) {
                        if (op.axis.x().bit_get(pivot))
                            p.z().bit_set(pivot, true);
                        else
                            p.x().bit_set(pivot, true);
                    });
            };
            feedback();
            hir.append_phase_rotation(op.half_turns, [&](MutablePauliMaskView p) {
                fill_axis(p);
                p.set_sign(noise.signs[j].constant);
            });
            feedback();
        } else {
            parity.push_back(hidden_measure(fill_axis));
            if (noise.signs[j].constant)
                parity.push_back(one);
            if (!op.output)
                continue;
            const uint32_t out_idx = *op.output;
            output_parities.at(out_idx) = parity;
            if (out_idx < visible) {
                // Materialize the affine record parity on the zero scratch wire
                // and uncompute the controls so the wire is reusable.
                auto copy_parity = [&] {
                    for (auto record : parity)
                        hir.append_conditional(
                            ControllingMeasIdx{record},
                            [&](MutablePauliMaskView p) { p.x().bit_set(carrier, true); });
                };
                copy_parity();
                hir.append_measure(MeasRecordIdx{out_idx},
                                   [&](MutablePauliMaskView p) { p.z().bit_set(carrier, true); });
                copy_parity();
            }
        }
    }

    // 4. Emit detectors and observables in source order.
    for (uint32_t i = 0; i < r.outputs.num_detectors; ++i) {
        hir.detector_targets.push_back(std::move(output_parities[visible + i]));
        hir.append_detector(DetectorIdx{i});
    }
    for (uint32_t i = 0; i < r.outputs.num_observables; ++i) {
        hir.observable_targets.push_back(
            std::move(output_parities[visible + r.outputs.num_detectors + i]));
        hir.append_observable(ObservableIdx{i}, i);
    }
    return hir;
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

ReconstructionNoiseResult compress_reconstruction_noise(const GlobalReconstruction& reconstruction,
                                                        const ReconstructionLimits& limits) {
    try {
        return {compress_noise(reconstruction, limits), {}};
    } catch (const Ineligible& e) {
        return {std::nullopt, e.reason};
    }
}

ReconstructionLoweringResult lower_global_reconstruction(const GlobalReconstruction& reconstruction,
                                                         const PostselectionMetadata& postselection,
                                                         const ReconstructionLimits& limits) {
    auto noise = compress_reconstruction_noise(reconstruction, limits);
    if (!noise.noise)
        return {std::nullopt, std::move(noise.fallback_reason)};
    try {
        return {lower_reconstruction(reconstruction, *noise.noise, postselection, limits), {}};
    } catch (const Ineligible& e) {
        return {std::nullopt, e.reason};
    }
}

}  // namespace clifft
