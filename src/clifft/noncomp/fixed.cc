#include "clifft/noncomp/fixed.h"

#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/fixed_input_validation.h"
#include "clifft/noncomp/status_walk.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/planner.h"
#include "clifft/util/numeric.h"
#include "clifft/util/shot_parallel.h"
#include "clifft/util/shot_seed.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace clifft {
namespace {

// Runtime values contain one bit; analysis uses unions of reachable values.
// A leaked value deliberately carries no claim about the hidden G/E label.
constexpr uint8_t kComputational = 1;
constexpr uint8_t kLeaked = 2;
constexpr uint8_t kLost = 4;
constexpr std::array<QubitStatus, 3> kRepresentativeStatuses{QubitStatus::Computational,
                                                             QubitStatus::LeakG, QubitStatus::Lost};
constexpr std::array<uint8_t, 3> kStatusBits{kComputational, kLeaked, kLost};

struct TransitionInput {
    uint32_t qubit;
    double probability;
    bool loss;
};

struct ResetInput {
    uint32_t qubit;
    std::array<uint8_t, 3> destinations;
};

struct NoiseOutcome {
    uint32_t input;
    double cumulative_probability;
};

struct NoiseInput {
    std::vector<uint32_t> active_targets;
    std::optional<uint32_t> inactive_source;
    uint8_t source_status = 0;
    std::vector<NoiseOutcome> outcomes;
};

using ReadoutChannel = ClassifiedReadoutChannel;

struct ReadoutInput {
    uint32_t input;
    uint32_t record;
    std::optional<uint32_t> qubit;
    std::array<ReadoutChannel, 3> channels;
};

struct ProbeInput {
    uint32_t input;
    uint32_t qubit;
    uint8_t status;
};

using InputStep = std::variant<TransitionInput, ResetInput, NoiseInput, ReadoutInput, ProbeInput>;

bool full_depolarization(const PartnerEffect* effect) {
    return effect != nullptr && effect->spread_probability == 0.0 &&
           effect->pauli == std::array<double, 3>{0.25, 0.25, 0.25};
}

bool zero_instrument(const TransitionInstrument& instrument) {
    return std::ranges::all_of(kAllLevels,
                               [&](Level source) { return instrument.column_sum(source) == 0.0; });
}

uint8_t after_transition(uint8_t states, const TransitionInput& transition) {
    if (transition.probability == 0.0) {
        return states;
    }
    const uint8_t affected = transition.loss ? kComputational | kLeaked : kComputational;
    if ((states & affected) != 0) {
        if (transition.probability == 1.0) {
            states &= static_cast<uint8_t>(~affected);
        }
        states |= transition.loss ? kLost : kLeaked;
    }
    return states;
}

ResetInput reset_input(uint32_t qubit, GateType gate, const NonComputationalPolicy& policy) {
    ResetInput input{qubit, {}};
    for (size_t i = 0; i < kRepresentativeStatuses.size(); ++i) {
        const auto destination =
            normal_post_op_status(kRepresentativeStatuses[i], gate, OperandRole::Physical, policy);
        input.destinations[i] = is_computational(destination) ? kComputational
                                : is_leaked(destination)      ? kLeaked
                                                              : kLost;
    }
    return input;
}

uint8_t after_reset(uint8_t states, const ResetInput& input) {
    uint8_t result = 0;
    for (size_t i = 0; i < kStatusBits.size(); ++i) {
        if ((states & kStatusBits[i]) != 0) {
            result |= input.destinations[i];
        }
    }
    return result;
}

std::vector<QubitOperand> checked_operands(const AstNode& node, uint32_t num_qubits) {
    auto operands = qubit_operands(node);
    for (const auto& operand : operands) {
        if (operand.qubit >= num_qubits) {
            throw std::invalid_argument("fixed noncomputational operand is out of range");
        }
    }
    return operands;
}

}  // namespace

namespace detail {
struct FixedNonComputationalData {
    Circuit dimensions;
    bool final_status = false;
    bool dynamic_inputs = false;
    std::vector<InputStep> inputs;
    std::optional<sampling::ExecutablePlan> plan;
    FixedNonComputationalMetrics metrics;
};
}  // namespace detail

namespace {

class FixedBuilder {
  public:
    explicit FixedBuilder(const PreparedNonComputationalRequest& request)
        : request_(request),
          resolved_(request.resolved_circuit()),
          model_(request.model()),
          reachable_(resolved_.circuit().num_qubits, kComputational),
          lowered_(resolved_.circuit().metadata_only_copy()),
          data_(std::make_shared<detail::FixedNonComputationalData>()) {
        data_->dimensions = lowered_;
        data_->final_status = request.requirements().final_status;
    }

    std::variant<std::shared_ptr<const detail::FixedNonComputationalData>,
                 FixedNonComputationalRejection>
    build() {
        if (model_.initial_probability(Level::G) != 1.0) {
            return reject(FixedNonComputationalReason::InitialState);
        }
        for (operation_ = 0; operation_ < resolved_.circuit().nodes.size(); ++operation_) {
            if (const auto rejection = lower()) {
                return *rejection;
            }
        }
        if (data_->final_status && std::ranges::any_of(reachable_, [](uint8_t states) {
                return (states & kLeaked) != 0;
            })) {
            return reject(FixedNonComputationalReason::DetailedLeakedStatus);
        }

        // Bind status-conditioned sites before any probability-based rewrite
        // could combine or erase them. The planner still performs coordinate
        // planning and the executable lowering still fuses quantum actions.
        ++data_->metrics.preparation_compilations;
        TraceSiteOrigins site_origins;
        const HirModule hir = trace(lowered_, nullptr, &site_origins);
        sampling::SamplingPlan plan;
        try {
            plan = sampling::plan_sampling(hir);
        } catch (const std::overflow_error&) {
            // The fixed candidate can exceed the planner's width limit even
            // when optimized trajectory continuations fit within it.
            return reject(FixedNonComputationalReason::ActiveWidth);
        }
        if (request_.requirements().max_active_width.has_value() &&
            plan.peak_active_width > *request_.requirements().max_active_width) {
            return reject(FixedNonComputationalReason::ActiveWidth);
        }
        detail::validate_fixed_input_bindings(noise_nodes_, readout_sites_, site_origins, hir,
                                              plan);
        std::vector<uint32_t> symbol_to_input(plan.symbols.size(), 0);
        uint32_t input = 0;
        for (uint32_t symbol = 0; symbol < plan.symbols.size(); ++symbol) {
            if (plan.symbols[symbol] == sampling::SymbolKind::Presampled) {
                symbol_to_input[symbol] = input++;
            }
        }
        for (size_t site = 0; site < noise_steps_.size(); ++site) {
            auto& step = std::get<NoiseInput>(data_->inputs[noise_steps_[site]]);
            double cumulative = 0.0;
            for (const auto& outcome : plan.presampled_noise_sites[site].outcomes) {
                cumulative += outcome.probability;
                step.outcomes.push_back(
                    {symbol_to_input[sampling::index(outcome.symbol)], cumulative});
            }
            if (!step.outcomes.empty()) {
                // Match the ordinary executor's declared event probability
                // even when summing channel weights accumulates roundoff.
                step.outcomes.back().cumulative_probability =
                    plan.presampled_noise_sites[site].total_probability;
            }
        }
        if (!data_->dynamic_inputs) {
            // With no possible status change, bind ordinary readout channels
            // once and reuse the executor's sparse quantum-noise sampling.
            std::vector<ReadoutChannel> channels(readout_count_);
            for (const auto& step : data_->inputs) {
                if (const auto* readout = std::get_if<ReadoutInput>(&step)) {
                    channels[readout->input] = readout->channels[0];
                }
            }
            uint32_t site = 0;
            for (auto& action : plan.actions) {
                if (auto* readout = std::get_if<sampling::ApplyReadoutNoise>(&action.action)) {
                    readout->prob_zero_to_one = channels[site].p01;
                    readout->prob_one_to_zero = channels[site].p10;
                    ++site;
                }
            }
            assert(site == readout_count_);
        }
        data_->plan.emplace(plan);
        if (data_->plan->num_readout_noise_sites() != readout_count_) {
            throw std::logic_error("fixed noncomputational lowering lost a readout-site identity");
        }
        data_->metrics.peak_active_width = plan.peak_active_width;
        data_->metrics.quantum_noise_inputs = input;
        data_->metrics.readout_inputs = readout_count_;
        data_->metrics.classical_workspace_bytes =
            reachable_.size() + input + readout_count_ * sizeof(sampling::ReadoutNoiseInput) +
            lowered_.num_measurements;
        if (!data_->dynamic_inputs) {
            data_->inputs = std::vector<InputStep>{};
            data_->metrics.quantum_noise_inputs = 0;
            data_->metrics.readout_inputs = 0;
            data_->metrics.classical_workspace_bytes = 0;
        }
        return std::shared_ptr<const detail::FixedNonComputationalData>(std::move(data_));
    }

  private:
    FixedNonComputationalRejection reject(FixedNonComputationalReason reason) const {
        if (operation_ < resolved_.circuit().nodes.size()) {
            return {reason, operation_, resolved_.circuit().nodes[operation_].source_line};
        }
        return {reason, std::nullopt, 0};
    }

    uint32_t append_readout(uint32_t record, uint32_t source_line) {
        // A nonzero placeholder keeps the ordered record-dependent symbol.
        // Execution supplies the actual channel for each shot.
        readout_sites_.push_back({lowered_.nodes.size(), record});
        lowered_.nodes.push_back(
            {GateType::READOUT_NOISE, {Target::rec(record)}, {0.5, 0.5}, source_line});
        return readout_count_++;
    }

    void append_noise(AstNode node, NoiseInput input) {
        noise_nodes_.push_back(lowered_.nodes.size());
        lowered_.nodes.push_back(std::move(node));
        noise_steps_.push_back(data_->inputs.size());
        data_->inputs.emplace_back(std::move(input));
    }

    std::optional<FixedNonComputationalRejection> lower();

    const PreparedNonComputationalRequest& request_;
    const ResolvedNonComputationalCircuit& resolved_;
    const NonComputationalModel& model_;
    std::vector<uint8_t> reachable_;
    Circuit lowered_;
    std::shared_ptr<detail::FixedNonComputationalData> data_;
    std::vector<size_t> noise_steps_;
    std::vector<size_t> noise_nodes_;
    std::vector<detail::FixedReadoutSite> readout_sites_;
    uint32_t operation_ = std::numeric_limits<uint32_t>::max();
    uint32_t record_ = 0;
    uint32_t readout_count_ = 0;
};

std::optional<FixedNonComputationalRejection> FixedBuilder::lower() {
    const AstNode& node = resolved_.circuit().nodes[operation_];
    const auto& annotation = resolved_.operation(operation_).annotation;
    if (const auto* transition = std::get_if<ResolvedTransition>(&annotation)) {
        if (const auto* named = std::get_if<TransitionId>(transition)) {
            if (!zero_instrument(resolved_.instrument(*named))) {
                return reject(FixedNonComputationalReason::NamedTransition);
            }
            return std::nullopt;
        }
        const bool loss = std::holds_alternative<LossTransition>(*transition);
        const double probability = loss ? std::get<LossTransition>(*transition).probability
                                        : std::get<LeakageTransition>(*transition).probability;
        for (const Target& target : node.targets) {
            const TransitionInput input{target.value(), probability, loss};
            reachable_[input.qubit] = after_transition(reachable_[input.qubit], input);
            if (probability > 0.0) {
                data_->dynamic_inputs = true;
                data_->inputs.emplace_back(input);
            }
        }
        return std::nullopt;
    }
    if (const auto* interaction = std::get_if<ResolvedInteraction>(&annotation)) {
        const uint8_t source =
            interaction->source_status == InteractionSource::Leaked ? kLeaked : kLost;
        if ((reachable_[interaction->source] & source) == 0 ||
            (reachable_[interaction->partner] & kComputational) == 0) {
            return std::nullopt;
        }
        if (interaction->effect.spread_probability != 0.0) {
            return reject(FixedNonComputationalReason::Spreading);
        }
        if (interaction->effect.has_pauli()) {
            const auto& p = interaction->effect.pauli;
            append_noise({GateType::PAULI_CHANNEL_1,
                          {Target::qubit(interaction->partner)},
                          {p[0], p[1], p[2]},
                          node.source_line},
                         {{interaction->partner}, interaction->source, source, {}});
        }
        return std::nullopt;
    }
    if (const auto* probe = std::get_if<ResolvedStatusProbe>(&annotation)) {
        lowered_.nodes.push_back({GateType::MPAD, {Target::qubit(0)}, {}, node.source_line});
        const uint32_t input = append_readout(record_++, node.source_line);
        data_->inputs.emplace_back(
            ProbeInput{input, probe->qubit,
                       probe->source_status == InteractionSource::Leaked ? kLeaked : kLost});
        return std::nullopt;
    }

    const auto operands = checked_operands(node, resolved_.circuit().num_qubits);
    // Keep fixed eligibility aligned with future shared-policy rejections.
    // Dropped physical gates follow the retained-auxiliary proof below and
    // cannot be dropped here.
    for (const auto& operand : operands) {
        for (size_t i = 0; i < kStatusBits.size(); ++i) {
            if ((reachable_[operand.qubit] & kStatusBits[i]) != 0 &&
                operand_action(node.gate, kRepresentativeStatuses[i], model_.policy()) ==
                    OperandAction::Reject) {
                return reject(FixedNonComputationalReason::UnsupportedOperation);
            }
        }
    }
    if (node.gate == GateType::READOUT_NOISE) {
        if (node.args.size() != 1 && node.args.size() != 2) {
            throw std::invalid_argument(
                "fixed noncomputational readout requires one or two probabilities");
        }
        if (!std::ranges::all_of(node.args, [](double p) { return is_probability(p); })) {
            throw std::invalid_argument(
                "fixed noncomputational readout probabilities must lie in [0, 1]");
        }
        for (const Target& target : node.targets) {
            if (!target.is_rec() || target.is_inverted() || target.has_pauli()) {
                throw std::invalid_argument(
                    "fixed noncomputational readout requires plain record targets");
            }
            const ReadoutChannel channel{node.args[0],
                                         node.args.size() == 1 ? node.args[0] : node.args[1], 0.0};
            const uint32_t input = append_readout(target.value(), node.source_line);
            data_->inputs.emplace_back(
                ReadoutInput{input, target.value(), std::nullopt, {channel, channel, channel}});
        }
        return std::nullopt;
    }
    if (node.gate == GateType::CORRELATED_ERROR) {
        // Retain the entire mutually exclusive chain, including factors on
        // auxiliaries. A branch acting only on an auxiliary still suppresses
        // later ELSE branches on live sites.
        noise_nodes_.push_back(lowered_.nodes.size());
        noise_steps_.push_back(data_->inputs.size());
        data_->inputs.emplace_back(NoiseInput{});
        lowered_.nodes.push_back(node);
        while (operation_ + 1 < resolved_.circuit().nodes.size() &&
               resolved_.circuit().nodes[operation_ + 1].gate == GateType::ELSE_CORRELATED_ERROR) {
            const auto& branch = resolved_.circuit().nodes[++operation_];
            checked_operands(branch, resolved_.circuit().num_qubits);
            lowered_.nodes.push_back(branch);
        }
        return std::nullopt;
    }
    if (node.gate == GateType::ELSE_CORRELATED_ERROR) {
        throw std::invalid_argument("ELSE_CORRELATED_ERROR must follow CORRELATED_ERROR");
    }
    if (is_noise_gate(node.gate)) {
        size_t group;
        switch (gate_arity(node.gate)) {
            case GateArity::SINGLE:
                group = 1;
                break;
            case GateArity::PAIR:
                group = 2;
                break;
            case GateArity::TRIPLE:
                group = 3;
                break;
            default:
                return reject(FixedNonComputationalReason::UnsupportedOperation);
        }
        if (node.targets.size() % group != 0) {
            throw std::invalid_argument("fixed noncomputational noise has incomplete operands");
        }
        std::vector<uint32_t> active_targets;
        for (const auto& operand : operands) {
            active_targets.push_back(operand.qubit);
        }
        for (size_t begin = 0; begin < node.targets.size(); begin += group) {
            AstNode noise = node;
            noise.targets.assign(node.targets.begin() + begin,
                                 node.targets.begin() + begin + group);
            append_noise(std::move(noise), {active_targets, std::nullopt, 0, {}});
        }
        return std::nullopt;
    }
    if (is_physical_measurement(node.gate)) {
        if (node.gate == GateType::MPP || operands.size() != 1) {
            return reject(FixedNonComputationalReason::UnsupportedOperation);
        }
        const uint32_t qubit = operands[0].qubit;
        const auto* classifier = model_.classifier();
        if ((reachable_[qubit] & kLeaked) != 0) {
            assert(classifier != nullptr);
            for (uint8_t symbol = 0; symbol < classifier->num_symbols(); ++symbol) {
                if (classifier->prob(symbol, Level::LeakG) !=
                    classifier->prob(symbol, Level::LeakE)) {
                    return reject(FixedNonComputationalReason::LevelSensitiveReadout);
                }
            }
        }
        AstNode measurement = node;
        const bool inverted = node.targets.front().is_inverted();
        if (is_measure_reset(node.gate) && inverted) {
            // The front-end otherwise inserts an implicit readout site. Keep
            // all externally supplied sites explicit and ordered here.
            measurement.targets = {Target::qubit(qubit)};
        }
        lowered_.nodes.push_back(std::move(measurement));
        if (is_measure_reset(node.gate) && inverted) {
            const ReadoutChannel invert{1.0, 1.0, 0.0};
            const uint32_t input = append_readout(record_, node.source_line);
            data_->inputs.emplace_back(
                ReadoutInput{input, record_, std::nullopt, {invert, invert, invert}});
        }
        std::array<ReadoutChannel, 3> channels{};
        for (size_t category = 0; category < channels.size(); ++category) {
            if (classifier != nullptr || category == 0) {
                channels[category] = classified_readout_channel(
                    classifier, node.gate, kRepresentativeStatuses[category], inverted);
            }
        }
        const uint32_t input = append_readout(record_, node.source_line);
        data_->inputs.emplace_back(ReadoutInput{input, record_++, qubit, channels});
        if (is_measure_reset(node.gate)) {
            const auto reset = reset_input(qubit, node.gate, model_.policy());
            data_->inputs.emplace_back(reset);
            reachable_[qubit] = after_reset(reachable_[qubit], reset);
        }
        return std::nullopt;
    }
    if (is_reset(node.gate)) {
        // A lost operand drops the whole unsplit AST node in trajectories.
        // Independent resets would incorrectly restore its live partners.
        if (operands.size() > 1 && std::ranges::any_of(operands, [&](const auto& operand) {
                return (reachable_[operand.qubit] & kLost) != 0 &&
                       operand_action(node.gate, QubitStatus::Lost, model_.policy()) ==
                           OperandAction::Drop;
            })) {
            return reject(FixedNonComputationalReason::UnsupportedOperation);
        }
        lowered_.nodes.push_back(node);
        for (const auto& operand : operands) {
            const auto reset = reset_input(operand.qubit, node.gate, model_.policy());
            data_->inputs.emplace_back(reset);
            reachable_[operand.qubit] = after_reset(reachable_[operand.qubit], reset);
        }
        return std::nullopt;
    }
    if (node.gate == GateType::MPAD) {
        lowered_.nodes.push_back(node);
        ++record_;
        return std::nullopt;
    }
    if (is_unitary(node.gate)) {
        if (operands.size() > 1) {
            const bool all_live = std::ranges::all_of(operands, [&](const auto& operand) {
                return reachable_[operand.qubit] == kComputational;
            });
            if (!all_live) {
                // Products and grouped local gates have whole-operation drop
                // semantics; physical pairs can instead use partner channels.
                if (gate_arity(node.gate) != GateArity::PAIR) {
                    return reject(FixedNonComputationalReason::UnsupportedOperation);
                }
                if (operands.size() != 2 || !supports_partner_effect(node.gate)) {
                    return reject(FixedNonComputationalReason::PartnerChannel);
                }
                for (uint32_t source = 0; source < 2; ++source) {
                    if ((reachable_[operands[1 - source].qubit] & kComputational) == 0) {
                        continue;
                    }
                    for (const auto status : {InteractionSource::Leaked, InteractionSource::Lost}) {
                        const uint8_t bit = status == InteractionSource::Leaked ? kLeaked : kLost;
                        if ((reachable_[operands[source].qubit] & bit) == 0) {
                            continue;
                        }
                        const auto* effect = model_.partner_effect(node.gate, source, status);
                        if (effect != nullptr && effect->spread_probability != 0.0) {
                            return reject(FixedNonComputationalReason::Spreading);
                        }
                        if (!full_depolarization(effect)) {
                            return reject(FixedNonComputationalReason::PartnerChannel);
                        }
                    }
                }
            }
        }
        // For auxiliary A and live B, full D_B after any two-site U gives
        // Tr_A[D_B(U rho U*)] = I_B/2 tensor Tr_AB(rho), including an entangled
        // environment. Generated interactions supply D_B after the retained
        // gate. Local operations on auxiliaries disappear under partial trace.
        lowered_.nodes.push_back(node);
        return std::nullopt;
    }
    if (node.gate == GateType::DETECTOR || node.gate == GateType::OBSERVABLE_INCLUDE ||
        node.gate == GateType::TICK) {
        lowered_.nodes.push_back(node);
        return std::nullopt;
    }
    return reject(FixedNonComputationalReason::UnsupportedOperation);
}

struct FixedWorker {
    explicit FixedWorker(const detail::FixedNonComputationalData& data)
        : executor(*data.plan),
          status(data.dynamic_inputs ? data.dimensions.num_qubits : 0),
          noise(data.metrics.quantum_noise_inputs),
          readout(data.metrics.readout_inputs),
          heralds(data.dynamic_inputs ? data.dimensions.num_measurements : 0) {}

    sampling::Executor executor;
    std::vector<uint8_t> status;
    std::vector<uint8_t> noise;
    std::vector<sampling::ReadoutNoiseInput> readout;
    std::vector<uint8_t> heralds;

    void prepare_inputs(const detail::FixedNonComputationalData& data,
                        Xoshiro256PlusPlus& rng) noexcept {
        std::fill(status.begin(), status.end(), kComputational);
        std::fill(noise.begin(), noise.end(), 0);
        std::fill(heralds.begin(), heralds.end(), 0);
        for (const auto& input : data.inputs) {
            std::visit(
                [&](const auto& step) noexcept {
                    using T = std::decay_t<decltype(step)>;
                    if constexpr (std::is_same_v<T, TransitionInput>) {
                        const uint8_t before = status[step.qubit];
                        const bool affected =
                            step.loss ? before != kLost : before == kComputational;
                        if (affected && rng.next_double() < step.probability) {
                            status[step.qubit] = step.loss ? kLost : kLeaked;
                        }
                    } else if constexpr (std::is_same_v<T, ResetInput>) {
                        status[step.qubit] = after_reset(status[step.qubit], step);
                    } else if constexpr (std::is_same_v<T, NoiseInput>) {
                        if (step.inactive_source &&
                            status[*step.inactive_source] != step.source_status) {
                            return;
                        }
                        for (const uint32_t target : step.active_targets) {
                            if (status[target] != kComputational) {
                                return;
                            }
                        }
                        const double draw = rng.next_double();
                        for (const auto& outcome : step.outcomes) {
                            if (draw < outcome.cumulative_probability) {
                                noise[outcome.input] = 1;
                                break;
                            }
                        }
                    } else if constexpr (std::is_same_v<T, ReadoutInput>) {
                        const uint8_t current = step.qubit ? status[*step.qubit] : kComputational;
                        const size_t category = current == kComputational ? 0
                                                : current == kLeaked      ? 1
                                                                          : 2;
                        const auto& channel = step.channels[category];
                        const bool herald =
                            channel.herald > 0.0 && rng.next_double() < channel.herald;
                        if (herald) {
                            heralds[step.record] = 1;
                        }
                        const double draw = rng.next_double();
                        readout[step.input] = {draw < (herald ? 0.5 : channel.p01),
                                               draw < (herald ? 0.5 : channel.p10)};
                    } else if constexpr (std::is_same_v<T, ProbeInput>) {
                        const bool value = status[step.qubit] == step.status;
                        readout[step.input] = {value, value};
                    }
                },
                input);
        }
    }
};

}  // namespace

FixedNonComputationalProgram::FixedNonComputationalProgram(
    std::shared_ptr<const detail::FixedNonComputationalData> data)
    : data_(std::move(data)) {}

const FixedNonComputationalMetrics& FixedNonComputationalProgram::metrics() const {
    return data_->metrics;
}

bool FixedNonComputationalProgram::returns_final_status() const {
    return data_->final_status;
}

std::variant<FixedNonComputationalProgram, FixedNonComputationalRejection>
prepare_fixed_noncomputational(const PreparedNonComputationalRequest& request) {
    auto result = FixedBuilder(request).build();
    if (auto* rejection = std::get_if<FixedNonComputationalRejection>(&result)) {
        return *rejection;
    }
    return FixedNonComputationalProgram(
        std::get<std::shared_ptr<const detail::FixedNonComputationalData>>(std::move(result)));
}

NonComputationalSample FixedNonComputationalProgram::sample(uint32_t shots,
                                                            std::optional<uint64_t> seed,
                                                            uint32_t threads) const {
    const auto& data = *data_;
    const auto& dimensions = data.dimensions;
    NonComputationalSample result;
    result.shots = shots;
    result.num_qubits = dimensions.num_qubits;
    result.num_measurements = dimensions.num_measurements;
    result.num_detectors = dimensions.num_detectors;
    result.num_observables = dimensions.num_observables;
    const SeedRoot root = make_seed_root(shots, seed);
    if (shots == 0) {
        return result;
    }
    result.measurements.resize(static_cast<size_t>(shots) * dimensions.num_measurements);
    result.heralds.resize(result.measurements.size());
    result.detectors.resize(static_cast<size_t>(shots) * dimensions.num_detectors);
    result.observables.resize(static_cast<size_t>(shots) * dimensions.num_observables);
    if (data.final_status) {
        result.final_status.resize(static_cast<size_t>(shots) * dimensions.num_qubits);
    }
    run_shot_ranges(
        shots, threads, [&](uint32_t) { return std::make_unique<FixedWorker>(data); },
        [&](const std::unique_ptr<FixedWorker>& worker, ShotRange range) {
            for (uint32_t shot = range.begin; shot < range.end; ++shot) {
                if (data.dynamic_inputs) {
                    const auto input_seed =
                        derive_state(root, shot, kFixedNonComputationalInputDomain);
                    Xoshiro256PlusPlus rng(0);
                    rng.seed_full(input_seed[0], input_seed[1], input_seed[2], input_seed[3]);
                    worker->prepare_inputs(data, rng);
                }
                const auto quantum_seed =
                    derive_state(root, shot, kFixedNonComputationalExecutorDomain);
                worker->executor.reseed_full(quantum_seed[0], quantum_seed[1], quantum_seed[2],
                                             quantum_seed[3]);
                if (data.dynamic_inputs) {
                    worker->executor.run_shot(worker->noise, worker->readout);
                } else {
                    worker->executor.run_shot();
                }
                assert(!worker->executor.pending_trap() && !worker->executor.discarded());
                auto copy = [&](const auto& values, auto& destination) {
                    std::copy(values.begin(), values.end(),
                              destination.begin() + static_cast<size_t>(shot) * values.size());
                };
                copy(worker->executor.visible_records(), result.measurements);
                copy(worker->executor.detectors(), result.detectors);
                copy(worker->executor.observables(), result.observables);
                copy(worker->heralds, result.heralds);
                if (data.final_status && data.dynamic_inputs) {
                    for (uint32_t qubit = 0; qubit < dimensions.num_qubits; ++qubit) {
                        assert(worker->status[qubit] != kLeaked);
                        result.final_status[static_cast<size_t>(shot) * dimensions.num_qubits +
                                            qubit] = worker->status[qubit] == kLost
                                                         ? QubitStatus::Lost
                                                         : QubitStatus::Computational;
                    }
                }
            }
        });
    return result;
}

}  // namespace clifft
