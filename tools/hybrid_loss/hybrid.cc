#include "hybrid.h"

#include "clifft/sampling/kernels.h"
#include "clifft/tableau/tableau.h"
#include "clifft/util/xoshiro.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace clifft::experimental {
namespace {
using namespace sampling;

// Rows map physical Paulis into current active/dormant coordinates. Physical
// Cliffords combine a few rows; measurement basis changes transform the rows.
class Frame {
  public:
    uint32_t n;
    std::vector<PauliString> rows;
    explicit Frame(uint32_t count) : n(count) {
        for (uint32_t i = 0; i < 2 * n; ++i)
            rows.emplace_back(n);
        reset();
    }
    void reset() {
        for (uint32_t i = 0; i < 2 * n; ++i) {
            auto& r = rows[i];
            r.mut_x().zero_out();
            r.mut_z().zero_out();
            r.set_phase(0);
            r.set_pauli(i % n, i < n, i >= n);
        }
    }
    PauliString axis(uint32_t q, char kind) const {
        if (kind == 'X')
            return rows[q];
        if (kind == 'Z')
            return rows[n + q];
        PauliString result(rows[q]);
        result.right_multiply(rows[n + q].view());
        result.add_phase(1);
        return result;
    }
    void physical(GateType gate, uint32_t a, uint32_t b = 0) {
        // Conjugate physical input by the inverse gate before the inverse frame.
        switch (gate) {
            case GateType::H:
                std::swap(rows[a], rows[n + a]);
                return;
            case GateType::S:
            case GateType::S_DAG:
                rows[a].right_multiply(rows[n + a].view());
                rows[a].add_phase(gate == GateType::S ? 3 : 1);
                return;
            case GateType::X:
                rows[n + a].negate();
                return;
            case GateType::Z:
                rows[a].negate();
                return;
            case GateType::Y:
                rows[a].negate();
                rows[n + a].negate();
                return;
            case GateType::CX:
                rows[a].right_multiply(rows[b].view());
                rows[n + b].right_multiply(rows[n + a].view());
                return;
            case GateType::CZ:
                rows[a].right_multiply(rows[n + b].view());
                rows[b].right_multiply(rows[n + a].view());
                return;
            case GateType::SWAP:
                std::swap(rows[a], rows[b]);
                std::swap(rows[n + a], rows[n + b]);
                return;
            default:
                break;
        }
        // The immutable local inverse is shared by every occurrence and shot.
        static const auto inverses = [] {
            std::map<GateType, Tableau> result;
            for (unsigned g = 0; g < static_cast<unsigned>(GateType::M); ++g) {
                const auto kind = static_cast<GateType>(g);
                if (is_clifford(kind) && kind != GateType::SPP && kind != GateType::SPP_DAG)
                    result.emplace(kind, Tableau::from_named_gate(kind).inverse());
            }
            return result;
        }();
        const auto& inverse = inverses.at(gate);
        const uint32_t count = inverse.num_qubits();
        const uint32_t targets[2] = {a, b};
        std::vector<PauliString> old;
        for (uint32_t i = 0; i < count; ++i)
            old.push_back(rows[targets[i]]);
        for (uint32_t i = 0; i < count; ++i)
            old.push_back(rows[n + targets[i]]);
        for (uint32_t i = 0; i < 2 * count; ++i) {
            const auto input = i < count ? inverse.x_output(i) : inverse.z_output(i - count);
            auto& output = rows[(i < count ? 0 : n) + targets[i % count]];
            output.mut_x().zero_out();
            output.mut_z().zero_out();
            output.set_phase(input.phase());
            for (uint32_t j = 0; j < count; ++j)
                if (input.x().bit_get(j))
                    output.right_multiply(old[j].view());
            for (uint32_t j = 0; j < count; ++j)
                if (input.z().bit_get(j))
                    output.right_multiply(old[count + j].view());
        }
    }
    void coordinate_h(uint32_t q) {
        for (auto& r : rows) {
            bool x = r.x().bit_get(q), z = r.z().bit_get(q);
            if (x && z)
                r.negate();
            r.set_pauli(q, z, x);
        }
    }
    void coordinate_x(uint32_t q) {
        for (auto& r : rows)
            if (r.z().bit_get(q))
                r.negate();
    }
    void coordinate_cx(uint32_t a, uint32_t b) {
        for (auto& r : rows) {
            bool xa = r.x().bit_get(a), za = r.z().bit_get(a);
            bool xb = r.x().bit_get(b), zb = r.z().bit_get(b);
            r.set_pauli(a, xa, za ^ zb);
            r.set_pauli(b, xb ^ xa, zb);
        }
    }
    void coordinate_swap(uint32_t a, uint32_t b) {
        if (a == b)
            return;
        for (auto& r : rows) {
            bool xa = r.x().bit_get(a), za = r.z().bit_get(a);
            r.set_pauli(a, r.x().bit_get(b), r.z().bit_get(b));
            r.set_pauli(b, xa, za);
        }
    }
    void compact(uint32_t pivot, uint32_t width) {
        // Dense measurement kernels remove the pivot while preserving other bit order.
        for (auto& r : rows) {
            bool xp = r.x().bit_get(pivot), zp = r.z().bit_get(pivot);
            for (uint32_t q = pivot; q + 1 < width; ++q)
                r.set_pauli(q, r.x().bit_get(q + 1), r.z().bit_get(q + 1));
            r.set_pauli(width - 1, xp, zp);
        }
    }
    void localize_x(const PauliString& p, uint32_t pivot) {
        // U = S_p^z controlled(Q), where P = (X or Y)_p Q. U fixes
        // states with dormant p=0. Apply U^-1 to every inverse-frame row.
        PauliString q(p);
        q.set_pauli(pivot, false, false);
        q.set_sign(false);
        const bool yp = p.z().bit_get(pivot);
        for (auto& r : rows) {
            const bool x = r.x().bit_get(pivot);
            if (yp && x) {
                r.set_pauli(pivot, true, !r.z().bit_get(pivot));
                r.add_phase(3);
            }
            const bool anti = !r.view().commutes(q.view());
            if (x)
                r.right_multiply(q.view());
            if (anti) {
                r.set_pauli(pivot, x, !r.z().bit_get(pivot));
                if (x)
                    r.negate();
            }
        }
    }
    void measure_dormant(const PauliString& p, uint32_t pivot, bool branch) {
        // Fuse U^-1, H, and the sampled X correction. The controlled-Pauli
        // commutator and product phase share the same packed-word traversal.
        PauliString q(p);
        q.set_pauli(pivot, false, false);
        q.set_sign(false);
        const bool yp = p.z().bit_get(pivot);
        for (auto& r : rows) {
            const bool x = r.x().bit_get(pivot);
            const bool z = r.z().bit_get(pivot);
            bool anti = false, crossing = false;
            auto rx = r.mut_x();
            auto rz = r.mut_z();
            for (uint32_t w = 0; w < rx.num_words(); ++w) {
                const uint64_t zx = rz.words[w] & q.x().words[w];
                anti ^= (std::popcount((rx.words[w] & q.z().words[w]) ^ zx) & 1) != 0;
                crossing ^= (std::popcount(zx) & 1) != 0;
                if (x) {
                    rx.words[w] ^= q.x().words[w];
                    rz.words[w] ^= q.z().words[w];
                }
            }
            const bool localized_z = z ^ (yp && x) ^ anti;
            unsigned phase = r.phase();
            if (x) {
                phase += (yp ? 3 : 0) + q.phase() + (crossing ? 2 : 0);
                phase += (anti ? 2 : 0) + (localized_z ? 2 : 0) + (branch ? 2 : 0);
            }
            r.set_pauli(pivot, localized_z, x);
            r.set_phase(phase);
        }
    }
};

class Quantum {
  public:
    Frame frame;
    State state;
    uint32_t peak = 0;
    explicit Quantum(uint32_t n, uint32_t capacity) : frame(n), state(capacity) {}
    void reset() {
        frame.reset();
        state.reset();
        peak = 0;
    }
    std::optional<uint32_t> dormant_x(const PauliString& p) const {
        for (uint32_t w = state.active_width() / 64; w < p.x().num_words(); ++w) {
            uint64_t bits = p.x().words[w];
            if (w == state.active_width() / 64)
                bits &= ~uint64_t{0} << (state.active_width() % 64);
            if (bits)
                return 64 * w + std::countr_zero(bits);
        }
        return std::nullopt;
    }
    ActivePauli active(const PauliString& p) const {
        const uint64_t mask = (uint64_t{1} << state.active_width()) - 1;
        return {p.x().words.empty() ? 0 : p.x().words[0] & mask,
                p.z().words.empty() ? 0 : p.z().words[0] & mask};
    }
    double expectation(const PauliString& p) const {
        if (dormant_x(p))
            return 0;
        const double value =
            expectation_value(state, prepare_pauli(active(p), state.active_width()));
        return p.sign() ? -value : value;
    }
    void promote(const PauliString& p, uint32_t pivot) {
        if (state.active_width() == state.max_active_width())
            throw std::runtime_error("hybrid active-width budget exceeded");
        frame.localize_x(p, pivot);
        frame.coordinate_swap(pivot, state.active_width());
        peak = std::max(peak, state.active_width() + 1);
    }
    void rotate(PauliString p, double half_turns) {
        const bool sign = p.sign();
        if (auto pivot = dormant_x(p)) {
            promote(p, *pivot);
            apply_promotion(state, prepare_promotion(half_turns), sign);
        } else {
            const auto a = active(p);
            if (!a.is_identity())
                apply_rotation(state, prepare_rotation(a, state.active_width(), half_turns), sign);
        }
    }
    void no_fire(PauliString p, double g, double e, double probability) {
        if (g == e)
            return;
        if (p.sign())
            std::swap(g, e);
        if (auto pivot = dormant_x(p)) {
            promote(p, *pivot);
            apply_new_x_instrument_no_fire(state, g, e, probability);
        } else {
            const auto a = active(p);
            if (!a.is_identity())
                apply_instrument_no_fire(state, prepare_pauli(a, state.active_width()), g, e,
                                         probability);
        }
    }
    bool measure(PauliString p, Xoshiro256PlusPlus& rng, std::optional<bool> forced = {}) {
        const bool sign = p.sign();
        if (auto pivot = dormant_x(p)) {
            const bool branch = forced ? (*forced ^ sign) : rng.next_double() < 0.5;
            frame.measure_dormant(p, *pivot, branch);
            return branch ^ sign;
        }
        const auto a = active(p);
        if (a.is_identity()) {
            if (forced && *forced != sign)
                throw std::runtime_error("forced impossible deterministic outcome");
            return sign;
        }
        uint32_t width = state.active_width();
        uint32_t pivot = a.x ? 63 - std::countl_zero(a.x) : std::countr_zero(a.z);
        const auto measurement = prepare_measurement(a, width, pivot);
        const auto probabilities = measurement_probabilities(state, measurement);
        const bool branch = forced ? (*forced ^ sign)
                                   : rng.next_double() * probabilities.total() < probabilities.one;
        if (probabilities.for_branch(branch) <= 0)
            throw std::runtime_error("forced impossible active outcome");
        collapse_measurement(state, measurement, branch, probabilities.for_branch(branch));
        PauliString body(frame.n);
        for (uint32_t q = 0; q < width; ++q)
            body.set_pauli(q, (a.x >> q) & 1, (a.z >> q) & 1);
        body.set_sign(false);
        if (a.x) {
            frame.localize_x(body, pivot);
            frame.coordinate_h(pivot);
        } else {
            for (uint32_t q = 0; q < width; ++q)
                if (q != pivot && ((a.z >> q) & 1))
                    frame.coordinate_cx(q, pivot);
        }
        if (branch)
            frame.coordinate_x(pivot);
        frame.compact(pivot, width);
        return branch ^ sign;
    }
};

class Runner {
  public:
    const ResolvedNonComputationalCircuit& resolved;
    const NonComputationalModel& model;
    Quantum quantum;
    Xoshiro256PlusPlus rng;
    std::vector<QubitStatus> status;
    std::vector<uint8_t> records, heralds, detectors, observables;
    size_t record = 0, detector = 0;
    bool error_chain = false;
    Runner(const PreparedNonComputationalRequest& request, uint32_t capacity, uint64_t seed)
        : resolved(request.resolved_circuit()),
          model(request.model()),
          quantum(resolved.circuit().num_qubits, capacity),
          rng(seed),
          status(resolved.circuit().num_qubits),
          records(resolved.circuit().num_measurements),
          heralds(records.size()),
          detectors(resolved.circuit().num_detectors),
          observables(resolved.circuit().num_observables) {
        for (const auto& node : resolved.circuit().nodes) {
            if (node.gate == GateType::MPP || node.gate == GateType::U3 ||
                node.gate == GateType::DEPOLARIZE3 || node.gate == GateType::PAULI_CHANNEL_3)
                throw std::invalid_argument("unsupported hybrid experiment instruction: " +
                                            std::string(gate_name(node.gate)));
        }
    }
    bool event(double p) { return p > 0 && rng.next_double() < p; }
    void pauli(uint32_t q, unsigned p) {
        if (p == 1)
            quantum.frame.physical(GateType::X, q);
        if (p == 2)
            quantum.frame.physical(GateType::Y, q);
        if (p == 3)
            quantum.frame.physical(GateType::Z, q);
    }
    void channel(uint32_t q, const std::array<double, 3>& probabilities) {
        double draw = rng.next_double();
        for (unsigned p = 0; p < 3; ++p) {
            draw -= probabilities[p];
            if (draw < 0) {
                pauli(q, p + 1);
                break;
            }
        }
    }
    void land(uint32_t q, Level dest, bool source) {
        // Vacated slots are an unobserved |0> auxiliary, isolated by the status walk.
        if (source != (dest == Level::E))
            quantum.frame.physical(GateType::X, q);
        status[q] = status_for(dest);
    }
    void transition(uint32_t q, const ResolvedTransition& transition) {
        if (const auto* id = std::get_if<TransitionId>(&transition)) {
            const auto& instrument = resolved.instrument(*id);
            if (!is_computational(status[q])) {
                const Level source = noncomp_level(status[q]);
                double draw = rng.next_double();
                for (Level dest : kAllLevels) {
                    draw -= instrument.prob(dest, source);
                    if (draw < 0) {
                        land(q, dest, false);
                        return;
                    }
                }
                return;
            }
            const auto axis = quantum.frame.axis(q, 'Z');
            const double one = std::clamp((1 - quantum.expectation(axis)) * 0.5, 0.0, 1.0);
            double draw = rng.next_double();
            for (unsigned s = 0; s < 2; ++s) {
                const double pop = s ? one : 1 - one;
                for (Level dest : kAllLevels) {
                    draw -= pop * instrument.prob(dest, static_cast<Level>(s));
                    if (draw < 0) {
                        quantum.measure(axis, rng, s != 0);
                        land(q, dest, s != 0);
                        return;
                    }
                }
            }
            const double pg = instrument.column_sum(Level::G), pe = instrument.column_sum(Level::E);
            if (model.policy().damping == DampingPolicy::Exact && pg != pe)
                quantum.no_fire(axis, std::sqrt(1 - pg), std::sqrt(1 - pe),
                                (1 - one) * (1 - pg) + one * (1 - pe));
            return;
        }
        const auto* loss = std::get_if<LossTransition>(&transition);
        if (is_lost(status[q]) || (!loss && !is_computational(status[q])))
            return;
        const double probability =
            loss ? loss->probability : std::get<LeakageTransition>(transition).probability;
        if (!event(probability))
            return;
        bool source = false;
        if (is_computational(status[q]))
            source = quantum.measure(quantum.frame.axis(q, 'Z'), rng);
        land(q, loss ? Level::Lost : source ? Level::LeakE : Level::LeakG, source);
    }
    void readout(const AstNode& node, Target target) {
        const uint32_t q = target.value();
        const bool live = is_computational(status[q]);
        const bool x =
            node.gate == GateType::MX || node.gate == GateType::MRX || node.gate == GateType::RX;
        const bool y =
            node.gate == GateType::MY || node.gate == GateType::MRY || node.gate == GateType::RY;
        const char basis = x ? 'X' : y ? 'Y' : 'Z';
        bool bit = false;
        if (live)
            bit = quantum.measure(quantum.frame.axis(q, basis), rng);
        if (!is_reset(node.gate)) {
            bool reported = bit ^ target.is_inverted();
            const auto channel = classified_readout_channel(model.classifier(), node.gate,
                                                            status[q], target.is_inverted());
            if (event(channel.herald)) {
                heralds[record] = 1;
                reported = event(0.5);
            } else if (event(reported ? channel.p10 : channel.p01))
                reported = !reported;
            records[record++] = reported;
        }
        if (is_reset(node.gate) || is_measure_reset(node.gate)) {
            if (is_lost(status[q]) && !model.policy().reset_restores_lost)
                return;
            if (live) {
                if (bit)
                    quantum.frame.physical(x || y ? GateType::Z : GateType::X, q);
            } else {
                if (x || y)
                    quantum.frame.physical(GateType::H, q);
                if (y)
                    quantum.frame.physical(GateType::S, q);
            }
            status[q] = QubitStatus::Computational;
        }
    }
    PauliString product(const AstNode& node) {
        PauliString p(quantum.frame.n);
        for (auto t : node.targets)
            p.right_multiply(quantum.frame.axis(t.value(), t.pauli_char()).view());
        return p;
    }
    void run() {
        quantum.reset();
        record = detector = 0;
        error_chain = false;
        std::fill(heralds.begin(), heralds.end(), 0);
        std::fill(observables.begin(), observables.end(), 0);
        for (uint32_t q = 0; q < status.size(); ++q) {
            double draw = rng.next_double();
            Level level = Level::Lost;
            for (Level candidate : kAllLevels) {
                draw -= model.initial_probability(candidate);
                if (draw < 0) {
                    level = candidate;
                    break;
                }
            }
            status[q] = status_for(level);
            if (level == Level::E)
                quantum.frame.physical(GateType::X, q);
        }
        for (uint32_t op = 0; op < resolved.circuit().nodes.size(); ++op) {
            const auto& node = resolved.circuit().nodes[op];
            const auto& annotation = resolved.operation(op).annotation;
            if (const auto* t = std::get_if<ResolvedTransition>(&annotation)) {
                for (auto target : node.targets)
                    transition(target.value(), *t);
                continue;
            }
            if (const auto* effect = std::get_if<ResolvedInteraction>(&annotation)) {
                if (effect->applies(status)) {
                    if (effect->effect.has_pauli())
                        channel(effect->partner, effect->effect.pauli);
                    if (effect->effect.spread_probability > 0)
                        transition(effect->partner,
                                   LeakageTransition{effect->effect.spread_probability});
                }
                continue;
            }
            if (const auto* probe = std::get_if<ResolvedStatusProbe>(&annotation)) {
                records[record++] = probe->source_status == InteractionSource::Lost
                                        ? is_lost(status[probe->qubit])
                                        : is_leaked(status[probe->qubit]);
                continue;
            }
            if (node.gate == GateType::TICK)
                continue;
            if (node.gate == GateType::MPAD) {
                for (auto t : node.targets)
                    records[record++] = t.value();
                continue;
            }
            if (node.gate == GateType::DETECTOR || node.gate == GateType::OBSERVABLE_INCLUDE) {
                uint8_t parity = 0;
                for (auto t : node.targets)
                    parity ^= records[t.value()];
                if (node.gate == GateType::DETECTOR)
                    detectors[detector++] = parity;
                else
                    observables.at(static_cast<size_t>(node.args[0])) ^= parity;
                continue;
            }
            if (node.gate == GateType::READOUT_NOISE) {
                for (auto t : node.targets) {
                    auto& bit = records[t.value()];
                    if (event(node.args.size() == 1 ? node.args[0] : node.args[bit]))
                        bit ^= 1;
                }
                continue;
            }
            if (node.gate == GateType::CORRELATED_ERROR ||
                node.gate == GateType::ELSE_CORRELATED_ERROR) {
                if (node.gate == GateType::CORRELATED_ERROR)
                    error_chain = false;
                if (!error_chain && event(node.args[0])) {
                    error_chain = true;
                    for (auto t : node.targets)
                        if (is_computational(status[t.value()]))
                            pauli(t.value(), t.pauli() >> Target::kPauliShift);
                }
                continue;
            }
            if (is_reset(node.gate) || is_physical_measurement(node.gate)) {
                for (auto t : node.targets)
                    readout(node, t);
                continue;
            }
            const auto arity = gate_arity(node.gate);
            const uint32_t step = arity == GateArity::PAIR    ? 2
                                  : arity == GateArity::MULTI ? node.targets.size()
                                                              : 1;
            for (uint32_t i = 0; i < node.targets.size(); i += step) {
                bool live = true;
                for (uint32_t j = i; j < i + step; ++j)
                    if (!node.targets[j].is_rec())
                        live &= is_computational(status[node.targets[j].value()]);
                if (!live)
                    continue;
                const uint32_t a = node.targets[i].value(),
                               b = step == 2 ? node.targets[i + 1].value() : 0;
                if (node.targets[i].is_rec()) {
                    if (records[a])
                        quantum.frame.physical(
                            node.gate == GateType::CX ? GateType::X : GateType::Z, b);
                } else if (is_clifford(node.gate) && node.gate != GateType::SPP &&
                           node.gate != GateType::SPP_DAG) {
                    quantum.frame.physical(node.gate, a, b);
                } else if (node.gate == GateType::T || node.gate == GateType::T_DAG) {
                    quantum.rotate(quantum.frame.axis(a, 'Z'),
                                   node.gate == GateType::T ? 0.25 : -0.25);
                } else if (node.gate == GateType::R_X || node.gate == GateType::R_Y ||
                           node.gate == GateType::R_Z) {
                    quantum.rotate(quantum.frame.axis(a, node.gate == GateType::R_X   ? 'X'
                                                         : node.gate == GateType::R_Y ? 'Y'
                                                                                      : 'Z'),
                                   node.args[0]);
                } else if (node.gate == GateType::R_XX || node.gate == GateType::R_YY ||
                           node.gate == GateType::R_ZZ) {
                    char axis = node.gate == GateType::R_XX   ? 'X'
                                : node.gate == GateType::R_YY ? 'Y'
                                                              : 'Z';
                    auto p = quantum.frame.axis(a, axis);
                    p.right_multiply(quantum.frame.axis(b, axis).view());
                    quantum.rotate(std::move(p), node.args[0]);
                } else if (node.gate == GateType::R_PAULI || node.gate == GateType::TPP ||
                           node.gate == GateType::TPP_DAG || node.gate == GateType::SPP ||
                           node.gate == GateType::SPP_DAG) {
                    const double angle = node.gate == GateType::R_PAULI   ? node.args[0]
                                         : node.gate == GateType::TPP     ? 0.25
                                         : node.gate == GateType::TPP_DAG ? -0.25
                                         : node.gate == GateType::SPP     ? 0.5
                                                                          : -0.5;
                    quantum.rotate(product(node), angle);
                } else if (node.gate == GateType::X_ERROR || node.gate == GateType::Y_ERROR ||
                           node.gate == GateType::Z_ERROR) {
                    if (event(node.args[0]))
                        pauli(a, node.gate == GateType::X_ERROR   ? 1
                                 : node.gate == GateType::Y_ERROR ? 2
                                                                  : 3);
                } else if (node.gate == GateType::DEPOLARIZE1 ||
                           node.gate == GateType::DEPOLARIZE2) {
                    if (event(node.args[0])) {
                        unsigned p =
                            1 + static_cast<unsigned>(rng.next_double() * (step == 1 ? 3 : 15));
                        if (step == 2) {
                            pauli(a, p / 4);
                            pauli(b, p % 4);
                        } else
                            pauli(a, p);
                    }
                } else if (node.gate == GateType::PAULI_CHANNEL_1 ||
                           node.gate == GateType::PAULI_CHANNEL_2) {
                    double draw = rng.next_double();
                    for (unsigned p = 1; p <= node.args.size(); ++p) {
                        draw -= node.args[p - 1];
                        if (draw < 0) {
                            if (step == 2) {
                                pauli(a, p / 4);
                                pauli(b, p % 4);
                            } else
                                pauli(a, p);
                            break;
                        }
                    }
                } else
                    throw std::invalid_argument("unsupported hybrid instruction: " +
                                                std::string(gate_name(node.gate)));
            }
        }
        if (record != records.size() || detector != detectors.size())
            throw std::logic_error("hybrid output layout mismatch");
    }
};
}  // namespace

NonComputationalSample sample_hybrid(const PreparedNonComputationalRequest& request, uint32_t shots,
                                     uint64_t seed, uint32_t capacity) {
    const auto& circuit = request.resolved_circuit().circuit();
    if (capacity > 24)
        throw std::invalid_argument("experiment coefficient budget must be at most 24");
    Runner runner(request, std::min(capacity, circuit.num_qubits), seed);
    NonComputationalSample output;
    output.shots = shots;
    output.num_qubits = circuit.num_qubits;
    output.num_measurements = circuit.num_measurements;
    output.num_detectors = circuit.num_detectors;
    output.num_observables = circuit.num_observables;
    output.measurements.resize(static_cast<size_t>(shots) * output.num_measurements);
    output.heralds.resize(output.measurements.size());
    output.detectors.resize(static_cast<size_t>(shots) * output.num_detectors);
    output.observables.resize(static_cast<size_t>(shots) * output.num_observables);
    if (request.requirements().final_status)
        output.final_status.resize(static_cast<size_t>(shots) * output.num_qubits);
    output.stats.emplace();
    output.stats->peak_active_width = 0;
    for (uint32_t shot = 0; shot < shots; ++shot) {
        runner.run();
        std::copy(
            runner.records.begin(), runner.records.end(),
            output.measurements.begin() + static_cast<size_t>(shot) * output.num_measurements);
        std::copy(runner.heralds.begin(), runner.heralds.end(),
                  output.heralds.begin() + static_cast<size_t>(shot) * output.num_measurements);
        std::copy(runner.detectors.begin(), runner.detectors.end(),
                  output.detectors.begin() + static_cast<size_t>(shot) * output.num_detectors);
        std::copy(runner.observables.begin(), runner.observables.end(),
                  output.observables.begin() + static_cast<size_t>(shot) * output.num_observables);
        if (request.requirements().final_status)
            std::copy(runner.status.begin(), runner.status.end(),
                      output.final_status.begin() + static_cast<size_t>(shot) * output.num_qubits);
        output.stats->peak_active_width =
            std::max(*output.stats->peak_active_width, runner.quantum.peak);
    }
    return output;
}
}  // namespace clifft::experimental
