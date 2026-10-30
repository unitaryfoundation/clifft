#include "clifft/noncomp/online.h"

#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
#include "noncomp_replay_hooks.h"
#endif

#include "clifft/noncomp/online_state.h"
#include "clifft/tableau/tableau.h"
#include "clifft/util/shot_parallel.h"
#include "clifft/util/shot_seed.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace clifft {
namespace noncomp_detail {

struct OnlineAction {
    uint32_t operation;
    uint32_t begin;
    uint32_t count;
};

struct OnlineOperation {
    OnlineRotation rotation;
    std::array<ClassifiedReadoutChannel, 8> readout{};
};

using CliffordTable = std::array<LocalClifford, static_cast<size_t>(GateType::UNKNOWN)>;

CliffordTable prepare_cliffords() {
    CliffordTable result{};
    for (size_t g = 0; g < result.size(); ++g) {
        const auto gate = static_cast<GateType>(g);
        if (!is_clifford(gate) || gate == GateType::SPP || gate == GateType::SPP_DAG)
            continue;
        const auto inverse = Tableau::from_named_gate(gate).inverse();
        auto& local = result[g];
        local.count = inverse.num_qubits();
        for (uint32_t i = 0; i < 2 * local.count; ++i) {
            const auto row =
                i < local.count ? inverse.x_output(i) : inverse.z_output(i - local.count);
            local.rows[i] = {static_cast<uint8_t>(row.x().words[0]),
                             static_cast<uint8_t>(row.z().words[0]), row.phase()};
        }
    }
    return result;
}

bool supported(GateType gate) {
    if (is_clifford(gate) || is_reset(gate) || is_identity_noop(gate) ||
        is_noncomputational_annotation(gate) || is_noncomputational_herald(gate) ||
        is_noncomputational_interaction(gate))
        return true;
    switch (gate) {
        case GateType::T:
        case GateType::T_DAG:
        case GateType::TPP:
        case GateType::TPP_DAG:
        case GateType::R_X:
        case GateType::R_Y:
        case GateType::R_Z:
        case GateType::R_XX:
        case GateType::R_YY:
        case GateType::R_ZZ:
        case GateType::R_PAULI:
        case GateType::M:
        case GateType::MX:
        case GateType::MY:
        case GateType::MR:
        case GateType::MRX:
        case GateType::MRY:
        case GateType::MPAD:
        case GateType::X_ERROR:
        case GateType::Y_ERROR:
        case GateType::Z_ERROR:
        case GateType::DEPOLARIZE1:
        case GateType::DEPOLARIZE2:
        case GateType::DEPOLARIZE3:
        case GateType::PAULI_CHANNEL_1:
        case GateType::PAULI_CHANNEL_2:
        case GateType::PAULI_CHANNEL_3:
        case GateType::CORRELATED_ERROR:
        case GateType::ELSE_CORRELATED_ERROR:
        case GateType::READOUT_NOISE:
        case GateType::DETECTOR:
        case GateType::OBSERVABLE_INCLUDE:
        case GateType::TICK:
            return true;
        default:
            return false;
    }
}

// Native callers can construct a Circuit without the parser. Check the storage
// and operand contracts here before noexcept dispatch can index any buffers.
void validate_circuit(const Circuit& circuit) {
    uint64_t records = 0, detectors = 0;
    for (size_t op = 0; op < circuit.nodes.size(); ++op) {
        const auto& node = circuit.nodes[op];
        auto require = [&](bool condition,
                           std::string_view reason = "invalid operands or arguments") {
            if (!condition)
                throw std::invalid_argument(
                    "invalid online circuit operation " + std::to_string(op) + " at circuit line " +
                    std::to_string(node.source_line) + ": " + std::string(reason));
        };
        require(static_cast<size_t>(node.gate) < static_cast<size_t>(GateType::UNKNOWN));
        const auto gate = node.gate;
        if (!supported(gate))
            throw std::invalid_argument("online execution does not support " +
                                        std::string(gate_name(gate)) + " at circuit line " +
                                        std::to_string(node.source_line) + " (expanded operation " +
                                        std::to_string(op) + ")");
        if (gate == GateType::TICK)
            require(node.targets.empty());
        const auto arity = gate_arity(gate);
        const size_t group = arity == GateArity::PAIR ? 2 : arity == GateArity::TRIPLE ? 3 : 1;
        const bool correlated =
            gate == GateType::CORRELATED_ERROR || gate == GateType::ELSE_CORRELATED_ERROR;
        // An identity branch still consumes probability and suppresses later links.
        require(arity == GateArity::ANNOTATION || correlated || !node.targets.empty(),
                "operation requires targets");
        require(node.targets.size() % group == 0);
        if (!is_noncomputational_annotation(gate) && arity != GateArity::ANNOTATION &&
            arity != GateArity::MULTI)
            require(node.targets.size() == group, "expected one target group per native operation");
        if (gate == GateType::ELSE_CORRELATED_ERROR)
            require(op > 0 && (circuit.nodes[op - 1].gate == GateType::CORRELATED_ERROR ||
                               circuit.nodes[op - 1].gate == GateType::ELSE_CORRELATED_ERROR),
                    "ELSE_CORRELATED_ERROR must immediately follow a correlated-error branch");
        for (double arg : node.args)
            require(is_finite_robust(arg));
        if (!is_noncomputational_annotation(gate)) {
            if (gate == GateType::DETECTOR) {
                ++detectors;
            } else if (gate == GateType::OBSERVABLE_INCLUDE) {
                require(node.args.size() == 1);
                require(node.args[0] >= 0 && node.args[0] < circuit.num_observables &&
                        std::floor(node.args[0]) == node.args[0]);
            } else if (gate == GateType::READOUT_NOISE) {
                require(node.args.size() == 1 || node.args.size() == 2);
            } else if (is_physical_measurement(gate) || gate == GateType::MPAD) {
                // The parser lowers inline record noise to READOUT_NOISE.
                require(node.args.empty() || (node.args.size() == 1 && node.args[0] == 0));
            } else {
                size_t args = 0;
                if (gate == GateType::PAULI_CHANNEL_1)
                    args = 3;
                else if (gate == GateType::PAULI_CHANNEL_2)
                    args = 15;
                else if (gate == GateType::PAULI_CHANNEL_3)
                    args = 63;
                else if (is_noise_gate(gate) || gate == GateType::R_X || gate == GateType::R_Y ||
                         gate == GateType::R_Z || gate == GateType::R_XX ||
                         gate == GateType::R_YY || gate == GateType::R_ZZ ||
                         gate == GateType::R_PAULI)
                    args = 1;
                require(node.args.size() == args);
            }
        }
        if (is_noise_gate(gate) || gate == GateType::READOUT_NOISE) {
            double total = 0;
            for (double arg : node.args) {
                require(arg >= 0 && arg <= 1);
                total += arg;
            }
            if (gate == GateType::PAULI_CHANNEL_1 || gate == GateType::PAULI_CHANNEL_2 ||
                gate == GateType::PAULI_CHANNEL_3)
                require(total <= 1 + 1e-15);
        }
        const bool record_targets = gate == GateType::DETECTOR ||
                                    gate == GateType::OBSERVABLE_INCLUDE ||
                                    gate == GateType::READOUT_NOISE;
        const bool product = arity == GateArity::MULTI;
        for (size_t i = 0; i < node.targets.size(); ++i) {
            const auto target = node.targets[i];
            if (record_targets || target.is_rec()) {
                require(target.is_rec() && !target.is_inverted() &&
                        target.pauli() == Target::kPauliNone && target.value() < records);
                require(record_targets ||
                        ((gate == GateType::CX || gate == GateType::CZ) && i % 2 == 0));
            } else if (gate == GateType::MPAD) {
                require(target.value() <= 1 && !target.is_inverted() &&
                        target.pauli() == Target::kPauliNone);
            } else {
                require(target.value() < circuit.num_qubits);
                require(product == (target.pauli() != Target::kPauliNone));
                require(!target.is_inverted() || is_physical_measurement(gate) ||
                            gate == GateType::SPP || gate == GateType::SPP_DAG ||
                            gate == GateType::TPP || gate == GateType::TPP_DAG,
                        "inverted target is not supported by this gate");
                for (size_t j = product ? 0 : i - i % group; j < i; ++j)
                    require(node.targets[j].is_rec() || node.targets[j].value() != target.value());
            }
        }
        if (is_measurement(gate))
            records += node.targets.size();
    }
    if (records != circuit.num_measurements || detectors != circuit.num_detectors)
        throw std::invalid_argument(
            "online circuit record or detector dimensions do not match operations");
}

std::optional<double> rotation_angle(const AstNode& node) {
    switch (node.gate) {
        case GateType::T:
        case GateType::TPP:
            return 0.25;
        case GateType::T_DAG:
        case GateType::TPP_DAG:
            return -0.25;
        case GateType::SPP:
            return 0.5;
        case GateType::SPP_DAG:
            return -0.5;
        case GateType::R_X:
        case GateType::R_Y:
        case GateType::R_Z:
        case GateType::R_XX:
        case GateType::R_YY:
        case GateType::R_ZZ:
        case GateType::R_PAULI:
            return node.args.at(0);
        default:
            return {};
    }
}

}  // namespace noncomp_detail

namespace detail {

struct OnlineNonComputationalData {
    PreparedNonComputationalRequest request;
    noncomp_detail::CliffordTable cliffords;
    std::vector<noncomp_detail::OnlineOperation> operations;
    std::vector<noncomp_detail::OnlineAction> actions;

    explicit OnlineNonComputationalData(PreparedNonComputationalRequest prepared)
        : request(std::move(prepared)), cliffords(noncomp_detail::prepare_cliffords()) {
        const auto& circuit = request.resolved_circuit().circuit();
        noncomp_detail::validate_circuit(circuit);
        operations.resize(circuit.nodes.size());
        for (uint32_t op = 0; op < circuit.nodes.size(); ++op) {
            const auto& node = circuit.nodes[op];
            auto& prepared_op = operations[op];
            if (auto angle = noncomp_detail::rotation_angle(node)) {
                prepared_op.rotation = {sampling::prepare_promotion(std::remainder(*angle, 4.0)),
                                        classify_clifford_rotation(*angle)};
            }
            if (is_physical_measurement(node.gate)) {
                for (uint32_t status = 0; status < 4; ++status) {
                    if (status != 0 && request.model().classifier() == nullptr)
                        continue;
                    for (uint32_t inverted = 0; inverted < 2; ++inverted)
                        prepared_op.readout[2 * status + inverted] = classified_readout_channel(
                            request.model().classifier(), node.gate,
                            static_cast<QubitStatus>(status), inverted != 0);
                }
            }
            const auto arity = gate_arity(node.gate);
            const uint32_t count = static_cast<uint32_t>(node.targets.size());
            const uint32_t stride = std::holds_alternative<ResolvedTransition>(
                                        request.resolved_circuit().operation(op).annotation)
                                        ? 1
                                    : arity == GateArity::PAIR   ? 2
                                    : arity == GateArity::TRIPLE ? 3
                                    : arity == GateArity::SINGLE ? 1
                                                                 : count;
            if (count == 0) {
                actions.push_back({op, 0, 0});
            } else {
                for (uint32_t begin = 0; begin < count; begin += stride)
                    actions.push_back({op, begin, stride});
            }
        }
    }
};

}  // namespace detail

namespace {
using namespace noncomp_detail;

class OnlineWorker {
  public:
    const detail::OnlineNonComputationalData& program;
    const ResolvedNonComputationalCircuit& resolved;
    const NonComputationalModel& model;
    OnlineState quantum;
    Xoshiro256PlusPlus rng{0};
    std::vector<QubitStatus> status;
    std::vector<uint8_t> records, heralds, detectors, observables;
    PauliString axis, factor;
    size_t record = 0, detector = 0, cursor = 0;
    bool error_chain = false;
    uint32_t peak_width = 0;
    uint64_t completed_shots = 0;
    uint64_t growths = 0;

    explicit OnlineWorker(const detail::OnlineNonComputationalData& data)
        : program(data),
          resolved(data.request.resolved_circuit()),
          model(data.request.model()),
          quantum(resolved.circuit().num_qubits),
          status(resolved.circuit().num_qubits),
          records(resolved.circuit().num_measurements),
          heralds(records.size()),
          detectors(resolved.circuit().num_detectors),
          observables(resolved.circuit().num_observables),
          axis(quantum.frame.n),
          factor(quantum.frame.n) {}

    bool event(double p) noexcept { return p > 0 && rng.next_double() < p; }

    void physical(GateType gate, uint32_t a, uint32_t b = 0) noexcept {
        quantum.frame.physical(gate, a, b, program.cliffords[static_cast<size_t>(gate)]);
    }

    PauliStringView mapped(uint32_t q, char basis) noexcept {
        quantum.frame.axis(q, basis, axis);
        return axis.view();
    }

    void pauli(uint32_t q, unsigned p) noexcept {
        if (p == 1)
            physical(GateType::X, q);
        if (p == 2)
            physical(GateType::Y, q);
        if (p == 3)
            physical(GateType::Z, q);
    }

    void channel(uint32_t q, const std::array<double, 3>& probabilities) noexcept {
        double draw = rng.next_double();
        for (unsigned p = 0; p < 3; ++p) {
            draw -= probabilities[p];
            if (draw < 0) {
                pauli(q, p + 1);
                break;
            }
        }
    }

    void land(uint32_t q, Level dest, bool source) noexcept {
        // Vacated slots hold an unobserved |0> factor until restoration.
        if (source != (dest == Level::E))
            physical(GateType::X, q);
        status[q] = status_for(dest);
    }

    Level destination(const TransitionInstrument& instrument, Level source) noexcept {
#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
        if (test::noncomp_replay) {
            std::array<double, 5> masses{};
            for (Level dest : kAllLevels)
                masses[static_cast<size_t>(dest)] = instrument.prob(dest, source);
            return static_cast<Level>(
                test::noncomp_replay->choose(test::ReplayChoice::Destination, masses));
        }
#endif
        double draw = rng.next_double() * instrument.column_sum(source);
        Level last = source;
        for (Level dest : kAllLevels) {
            const double p = instrument.prob(dest, source);
            if (p > 0)
                last = dest;
            draw -= p;
            if (draw < 0)
                return dest;
        }
        return last;
    }

    void transition(uint32_t q, const ResolvedTransition& transition) noexcept {
        if (const auto* id = std::get_if<TransitionId>(&transition)) {
            const auto& instrument = resolved.instrument(*id);
            if (!is_computational(status[q])) {
                // Ledger values 1..3 correspond to physical levels 2..4.
                const auto source = static_cast<Level>(static_cast<unsigned>(status[q]) + 1);
                if (event(instrument.column_sum(source)))
                    land(q, destination(instrument, source), false);
                return;
            }
            const auto p = mapped(q, 'Z');
            const auto populations = quantum.populations(p);
            const double pg = instrument.column_sum(Level::G);
            const double pe = instrument.column_sum(Level::E);
            const double dust = kMeasurementDustEpsilon * populations.total();
            const double fire_g = populations.zero <= dust ? 0 : pg * populations.zero;
            const double fire_e = populations.one <= dust ? 0 : pe * populations.one;
            const double draw = rng.next_double() * populations.total();
            unsigned selected = draw < fire_g ? 0 : draw < fire_g + fire_e ? 1 : 2;
#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
            if (test::noncomp_replay) {
                const double masses[] = {fire_g, fire_e, populations.total() - fire_g - fire_e};
                selected = test::noncomp_replay->choose(test::ReplayChoice::Source, masses);
            }
#endif
            if (selected < 2) {
                const bool source = selected != 0;
                quantum.measure(p, rng, source);
                land(q, destination(instrument, source ? Level::E : Level::G), source);
            } else if (model.policy().damping == DampingPolicy::Exact && pg != pe) {
                quantum.no_fire(p, std::sqrt(1 - pg), std::sqrt(1 - pe),
                                populations.zero * (1 - pg) + populations.one * (1 - pe));
            }
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
            source = quantum.measure(mapped(q, 'Z'), rng);
        land(q, loss ? Level::Lost : source ? Level::LeakE : Level::LeakG, source);
    }

    void readout(const AstNode& node, Target target, const OnlineOperation& prepared) noexcept {
        const uint32_t q = target.value();
        const bool live = is_computational(status[q]);
        const bool x =
            node.gate == GateType::MX || node.gate == GateType::MRX || node.gate == GateType::RX;
        const bool y =
            node.gate == GateType::MY || node.gate == GateType::MRY || node.gate == GateType::RY;
        bool bit = false;
        if (live) {
            const auto p = mapped(q, x ? 'X' : y ? 'Y' : 'Z');
            std::optional<bool> forced;
#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
            if (test::noncomp_replay) {
                const auto populations = quantum.populations(p);
                const double masses[] = {populations.zero, populations.one};
                forced = test::noncomp_replay->choose(test::ReplayChoice::Measurement, masses) != 0;
            }
#endif
            bit = quantum.measure(p, rng, forced);
        }
        if (!is_reset(node.gate)) {
            bool reported = bit ^ target.is_inverted();
            const auto channel =
                prepared.readout[2 * static_cast<unsigned>(status[q]) + target.is_inverted()];
            if (event(channel.herald)) {
                heralds[record] = 1;
                reported = event(0.5);
            } else if (event(reported ? channel.p10 : channel.p01)) {
                reported = !reported;
            }
            records[record++] = reported;
        }
        if (is_reset(node.gate) || is_measure_reset(node.gate)) {
            if (is_lost(status[q]) && !model.policy().reset_restores_lost)
                return;
            if (live) {
                if (bit)
                    physical(x || y ? GateType::Z : GateType::X, q);
            } else {
                if (x || y)
                    physical(GateType::H, q);
                if (y)
                    physical(GateType::S, q);
            }
            status[q] = QubitStatus::Computational;
        }
    }

    PauliStringView product(std::span<const Target> targets) noexcept {
        axis.mut_x().zero_out();
        axis.mut_z().zero_out();
        axis.set_phase(0);
        for (auto target : targets) {
            quantum.frame.axis(target.value(), target.pauli_char(), factor);
            axis.right_multiply(factor.view());
            if (target.is_inverted())
                axis.negate();
        }
        return axis.view();
    }

    void reset(const SeedRoot& root, uint32_t shot) noexcept {
        const auto words = derive_state(root, shot, kOnlineNonComputationalDomain);
        rng.seed_full(words[0], words[1], words[2], words[3]);
        quantum.reset();
        record = detector = cursor = 0;
        error_chain = false;
        std::fill(heralds.begin(), heralds.end(), 0);
        std::fill(observables.begin(), observables.end(), 0);
        for (uint32_t q = 0; q < status.size(); ++q) {
            const double draw = rng.next_double();
            double accumulated = 0;
            Level level = Level::G;
            for (Level candidate : kAllLevels) {
                const double p = model.initial_probability(candidate);
                if (p > 0)
                    level = candidate;
                accumulated += p;
                if (draw < accumulated)
                    break;
            }
            status[q] = status_for(level);
            if (level == Level::E)
                physical(GateType::X, q);
        }
    }

    void dispatch(const OnlineAction& action) noexcept {
        const auto& node = resolved.circuit().nodes[action.operation];
        const auto& annotation = resolved.operation(action.operation).annotation;
        const auto& prepared = program.operations[action.operation];
        const auto targets =
            std::span<const Target>(node.targets).subspan(action.begin, action.count);
        if (const auto* t = std::get_if<ResolvedTransition>(&annotation)) {
            assert(targets.size() == 1);
            transition(targets[0].value(), *t);
            return;
        }
        if (const auto* effect = std::get_if<ResolvedInteraction>(&annotation)) {
            if (effect->applies(status)) {
                if (effect->effect.has_pauli())
                    channel(effect->partner, effect->effect.pauli);
                if (effect->effect.spread_probability > 0)
                    transition(effect->partner,
                               LeakageTransition{effect->effect.spread_probability});
            }
            return;
        }
        if (const auto* probe = std::get_if<ResolvedStatusProbe>(&annotation)) {
            records[record++] = probe->source_status == InteractionSource::Lost
                                    ? is_lost(status[probe->qubit])
                                    : is_leaked(status[probe->qubit]);
            return;
        }
        if (node.gate == GateType::TICK || is_identity_noop(node.gate))
            return;
        if (node.gate == GateType::MPAD) {
            for (auto t : targets)
                records[record++] = t.value();
            return;
        }
        if (node.gate == GateType::DETECTOR || node.gate == GateType::OBSERVABLE_INCLUDE) {
            uint8_t parity = 0;
            for (auto t : targets)
                parity ^= records[t.value()];
            if (node.gate == GateType::DETECTOR)
                detectors[detector++] = parity;
            else
                observables[static_cast<size_t>(node.args[0])] ^= parity;
            return;
        }
        if (node.gate == GateType::READOUT_NOISE) {
            for (auto t : targets) {
                auto& bit = records[t.value()];
                if (event(node.args.size() == 1 ? node.args[0] : node.args[bit]))
                    bit ^= 1;
            }
            return;
        }
        if (node.gate == GateType::CORRELATED_ERROR ||
            node.gate == GateType::ELSE_CORRELATED_ERROR) {
            if (node.gate == GateType::CORRELATED_ERROR)
                error_chain = false;
            if (!error_chain && event(node.args[0])) {
                error_chain = true;
                for (auto t : targets)
                    if (is_computational(status[t.value()]))
                        pauli(t.value(), t.pauli() >> Target::kPauliShift);
            }
            return;
        }
        if (is_reset(node.gate) || is_physical_measurement(node.gate)) {
            for (auto t : targets)
                readout(node, t, prepared);
            return;
        }
        for (auto t : targets)
            if (!t.is_rec() && !is_computational(status[t.value()]))
                return;
        assert(!targets.empty());
        const uint32_t a = targets[0].value();
        const uint32_t b = targets.size() >= 2 ? targets[1].value() : 0;
        if (targets[0].is_rec()) {
            if (records[a])
                physical(node.gate == GateType::CX ? GateType::X : GateType::Z, b);
        } else if (is_clifford(node.gate) && node.gate != GateType::SPP &&
                   node.gate != GateType::SPP_DAG) {
            physical(node.gate, a, b);
        } else if (node.gate == GateType::T || node.gate == GateType::T_DAG ||
                   node.gate == GateType::R_X || node.gate == GateType::R_Y ||
                   node.gate == GateType::R_Z) {
            quantum.rotate(mapped(a, node.gate == GateType::R_X   ? 'X'
                                     : node.gate == GateType::R_Y ? 'Y'
                                                                  : 'Z'),
                           prepared.rotation);
        } else if (node.gate == GateType::R_XX || node.gate == GateType::R_YY ||
                   node.gate == GateType::R_ZZ) {
            const char basis = node.gate == GateType::R_XX   ? 'X'
                               : node.gate == GateType::R_YY ? 'Y'
                                                             : 'Z';
            quantum.frame.axis(a, basis, axis);
            quantum.frame.axis(b, basis, factor);
            axis.right_multiply(factor.view());
            quantum.rotate(axis.view(), prepared.rotation);
        } else if (node.gate == GateType::R_PAULI || node.gate == GateType::TPP ||
                   node.gate == GateType::TPP_DAG || node.gate == GateType::SPP ||
                   node.gate == GateType::SPP_DAG) {
            quantum.rotate(product(targets), prepared.rotation);
        } else if (node.gate == GateType::X_ERROR || node.gate == GateType::Y_ERROR ||
                   node.gate == GateType::Z_ERROR) {
            if (event(node.args[0]))
                pauli(a, node.gate == GateType::X_ERROR   ? 1
                         : node.gate == GateType::Y_ERROR ? 2
                                                          : 3);
        } else if (node.gate == GateType::DEPOLARIZE1 || node.gate == GateType::DEPOLARIZE2 ||
                   node.gate == GateType::DEPOLARIZE3 || node.gate == GateType::PAULI_CHANNEL_1 ||
                   node.gate == GateType::PAULI_CHANNEL_2 ||
                   node.gate == GateType::PAULI_CHANNEL_3) {
            unsigned paulis = 0;
            const unsigned outcomes = (1U << (2 * targets.size())) - 1;
            if (node.gate == GateType::DEPOLARIZE1 || node.gate == GateType::DEPOLARIZE2 ||
                node.gate == GateType::DEPOLARIZE3) {
                if (event(node.args[0]))
                    paulis = 1 + static_cast<unsigned>(rng.next_double() * outcomes);
            } else {
                double draw = rng.next_double();
                for (unsigned p = 1; p <= node.args.size(); ++p) {
                    draw -= node.args[p - 1];
                    if (draw < 0) {
                        paulis = p;
                        break;
                    }
                }
            }
            for (size_t i = targets.size(); i-- > 0;) {
                pauli(targets[i].value(), paulis & 3U);
                paulis >>= 2;
            }
        } else {
            assert(false && "online preparation admitted an unsupported gate");
        }
    }

    // An ordinary dispatch interval cannot allocate or throw. Activation is
    // completed by the caller only after leaving this interval.
    void dispatch_until_boundary() noexcept {
        while (cursor < program.actions.size()) {
            dispatch(program.actions[cursor++]);
            if (quantum.pending())
                return;
        }
        assert(record == records.size() && detector == detectors.size());
    }

    void run(const SeedRoot& root, uint32_t shot) {
        reset(root, shot);
        for (;;) {
            dispatch_until_boundary();
            if (!quantum.pending())
                break;
            const uint32_t required = quantum.state.active_width() + 1;
            const auto cap = program.request.requirements().max_active_width;
            const auto& node = resolved.circuit().nodes[program.actions[cursor - 1].operation];
            if ((cap && required > *cap) || required >= kDenseActiveWidthLimit) {
                throw std::invalid_argument(
                    "online execution active width " + std::to_string(required) +
                    " exceeds max_active_width or the dense storage limit at circuit line " +
                    std::to_string(node.source_line));
            }
            quantum.state.ensure_capacity(required);
            ++growths;
            quantum.finish_activation();
        }
        peak_width = std::max(peak_width, quantum.peak);
        ++completed_shots;
#ifdef CLIFFT_NONCOMP_REPLAY_TESTING
        if (test::noncomp_replay)
            test::noncomp_replay->inspect_online(quantum, status);
#endif
    }
};

}  // namespace

OnlineNonComputationalProgram::OnlineNonComputationalProgram(
    PreparedNonComputationalRequest request)
    : data_(std::make_shared<detail::OnlineNonComputationalData>(std::move(request))) {}

NonComputationalSample OnlineNonComputationalProgram::sample(uint32_t shots,
                                                             std::optional<uint64_t> seed,
                                                             uint32_t threads,
                                                             bool collect_stats) const {
    const auto& circuit = data_->request.resolved_circuit().circuit();
    NonComputationalSample output;
    output.shots = shots;
    output.num_qubits = circuit.num_qubits;
    output.num_measurements = circuit.num_measurements;
    output.num_detectors = circuit.num_detectors;
    output.num_observables = circuit.num_observables;
    if (collect_stats)
        output.stats.emplace();
    if (shots == 0)
        return output;
    const auto root = make_seed_root(shots, seed);
    output.measurements.resize(static_cast<size_t>(shots) * output.num_measurements);
    output.heralds.resize(output.measurements.size());
    output.detectors.resize(static_cast<size_t>(shots) * output.num_detectors);
    output.observables.resize(static_cast<size_t>(shots) * output.num_observables);
    if (data_->request.requirements().final_status)
        output.final_status.resize(static_cast<size_t>(shots) * output.num_qubits);
    const auto workers = run_shot_ranges(
        shots, threads, [&](uint32_t) { return std::make_unique<OnlineWorker>(*data_); },
        [&](const std::unique_ptr<OnlineWorker>& worker, ShotRange range) {
            for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                worker->run(root, shot);
                std::copy(worker->records.begin(), worker->records.end(),
                          output.measurements.begin() +
                              static_cast<size_t>(shot) * output.num_measurements);
                std::copy(
                    worker->heralds.begin(), worker->heralds.end(),
                    output.heralds.begin() + static_cast<size_t>(shot) * output.num_measurements);
                std::copy(
                    worker->detectors.begin(), worker->detectors.end(),
                    output.detectors.begin() + static_cast<size_t>(shot) * output.num_detectors);
                std::copy(worker->observables.begin(), worker->observables.end(),
                          output.observables.begin() +
                              static_cast<size_t>(shot) * output.num_observables);
                if (data_->request.requirements().final_status)
                    std::copy(worker->status.begin(), worker->status.end(),
                              output.final_status.begin() +
                                  static_cast<size_t>(shot) * output.num_qubits);
            }
        });
    if (output.stats) {
        auto& stats = *output.stats;
        stats.peak_active_width = 0;
        stats.workers = static_cast<uint32_t>(workers.size());
        for (const auto& worker : workers) {
            stats.online_shots += worker->completed_shots;
            stats.online_state_growths += worker->growths;
            stats.peak_active_width = std::max(*stats.peak_active_width, worker->peak_width);
        }
        stats.scalar_shots = stats.online_shots;
    }
    return output;
}

}  // namespace clifft
