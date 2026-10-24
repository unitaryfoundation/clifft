#include "clifft/noncomp/program.h"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace clifft {
namespace detail {

struct NonComputationalProgramData {
    std::variant<PreparedNonComputationalRequest, FixedNonComputationalProgram> implementation;
    NonComputationalExecution requested;
    NonComputationalRequirements requirements;
    std::optional<FixedNonComputationalRejection> rejection;
    uint64_t preparation_compilations = 0;
    double preparation_seconds = 0;
};

}  // namespace detail
namespace {

using Clock = std::chrono::steady_clock;

std::string describe_rejection(const FixedNonComputationalRejection& rejection) {
    std::ostringstream out;
    out << fixed_rejection_name(rejection.reason) << ": ";
    switch (rejection.reason) {
        case FixedNonComputationalReason::InitialState:
            out << "fixed execution requires ground-state initialization";
            break;
        case FixedNonComputationalReason::NamedTransition:
            out << "the invoked named transition is not supported by fixed execution";
            break;
        case FixedNonComputationalReason::Spreading:
            out << "leakage spreading is not supported by fixed execution";
            break;
        case FixedNonComputationalReason::LevelSensitiveReadout:
            out << "the classifier distinguishes unresolved leaked levels";
            break;
        case FixedNonComputationalReason::PartnerChannel:
            out << "a reachable inactive/live interaction requires full partner depolarization";
            break;
        case FixedNonComputationalReason::UnsupportedOperation:
            out << "the operation has no supported fixed lowering";
            break;
        case FixedNonComputationalReason::DetailedLeakedStatus:
            out << "requested final status distinguishes unresolved leaked levels";
            break;
        case FixedNonComputationalReason::ActiveWidth:
            out << "the fixed candidate exceeds the active-width limit";
            if (rejection.peak_active_width) {
                out << " (candidate width " << *rejection.peak_active_width << ')';
            }
            break;
    }
    if (rejection.source_line != 0) {
        out << " at circuit line " << rejection.source_line;
    }
    if (rejection.operation) {
        out << " (expanded operation " << *rejection.operation << ')';
    }
    return out.str();
}

std::shared_ptr<const detail::NonComputationalProgramData> prepare_program(
    const Circuit& circuit, const NonComputationalModel& model, NonComputationalExecution execution,
    NonComputationalRequirements requirements) {
    const auto start = Clock::now();
    auto data =
        std::make_shared<detail::NonComputationalProgramData>(detail::NonComputationalProgramData{
            .implementation = PreparedNonComputationalRequest(circuit, model, requirements),
            .requested = execution,
            .requirements = requirements,
            .rejection = std::nullopt});
    if (execution != NonComputationalExecution::Trajectory) {
        // Validation is shared and errors propagate. Only an explicit fixed-path
        // rejection permits automatic fallback to the validated trajectory request.
        auto candidate = prepare_fixed_noncomputational(
            std::get<PreparedNonComputationalRequest>(data->implementation));
        if (auto* fixed = std::get_if<FixedNonComputationalProgram>(&candidate)) {
            data->preparation_compilations = fixed->metrics().preparation_compilations;
            data->implementation = std::move(*fixed);
        } else {
            data->rejection = std::get<FixedNonComputationalRejection>(candidate);
            data->preparation_compilations = data->rejection->preparation_compilations;
            if (execution == NonComputationalExecution::Fixed) {
                throw std::invalid_argument("fixed execution unavailable: " +
                                            describe_rejection(*data->rejection));
            }
        }
    }
    data->preparation_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return data;
}

}  // namespace

std::string_view fixed_rejection_name(FixedNonComputationalReason reason) {
    switch (reason) {
        case FixedNonComputationalReason::InitialState:
            return "initial_state";
        case FixedNonComputationalReason::NamedTransition:
            return "named_transition";
        case FixedNonComputationalReason::Spreading:
            return "spreading";
        case FixedNonComputationalReason::LevelSensitiveReadout:
            return "level_sensitive_readout";
        case FixedNonComputationalReason::PartnerChannel:
            return "partner_channel";
        case FixedNonComputationalReason::UnsupportedOperation:
            return "unsupported_operation";
        case FixedNonComputationalReason::DetailedLeakedStatus:
            return "detailed_leaked_status";
        case FixedNonComputationalReason::ActiveWidth:
            return "active_width";
    }
    throw std::logic_error("unknown fixed noncomputational rejection");
}

NonComputationalProgram::NonComputationalProgram(const Circuit& circuit,
                                                 const NonComputationalModel& model,
                                                 NonComputationalExecution execution,
                                                 NonComputationalRequirements requirements)
    : data_(prepare_program(circuit, model, execution, requirements)) {}

NonComputationalExecution NonComputationalProgram::strategy() const {
    return std::holds_alternative<FixedNonComputationalProgram>(data_->implementation)
               ? NonComputationalExecution::Fixed
               : NonComputationalExecution::Trajectory;
}

bool NonComputationalProgram::returns_final_status() const {
    return data_->requirements.final_status;
}

const std::optional<FixedNonComputationalRejection>& NonComputationalProgram::rejection() const {
    return data_->rejection;
}

std::string NonComputationalProgram::explain() const {
    std::ostringstream out;
    out << "Execution: "
        << (strategy() == NonComputationalExecution::Fixed ? "fixed" : "trajectory") << '\n';
    out << "Final status: " << (returns_final_status() ? "requested" : "omitted") << '\n';
    if (data_->rejection) {
        out << "Fixed fallback reason: " << describe_rejection(*data_->rejection) << '\n';
    } else if (data_->requested == NonComputationalExecution::Trajectory) {
        out << "Trajectory execution was explicitly requested.\n";
    } else if (data_->requested == NonComputationalExecution::Fixed) {
        out << "Fixed execution was explicitly requested.\n";
    }
    if (data_->requirements.max_active_width) {
        out << "Active-width cap: " << *data_->requirements.max_active_width << '\n';
    }
    if (const auto* fixed = std::get_if<FixedNonComputationalProgram>(&data_->implementation)) {
        out << "Peak active width: " << fixed->metrics().peak_active_width << '\n';
        out << "Scalar input workspace: " << fixed->metrics().classical_workspace_bytes
            << " bytes per worker (excludes quantum executor and results).\n";
        out << "Batch capacity and packed worker storage are resolved by sample().\n";
    } else {
        out << "Continuations compile during sampling; width and storage depend on the path.\n";
    }
    out << "Preparation compilations: " << data_->preparation_compilations;
    return out.str();
}

NonComputationalSample NonComputationalProgram::sample(uint32_t shots, std::optional<uint64_t> seed,
                                                       uint32_t threads,
                                                       std::optional<uint32_t> batch_size,
                                                       bool collect_stats) const {
    if (batch_size && *batch_size == 0) {
        throw std::invalid_argument("batch_size must be a positive integer or 'auto'");
    }
    const auto start = collect_stats ? Clock::now() : Clock::time_point{};
    NonComputationalSample result;
    if (const auto* fixed = std::get_if<FixedNonComputationalProgram>(&data_->implementation)) {
        auto run = fixed->sample(shots, seed, threads, batch_size);
        result = std::move(run.output);
        if (collect_stats) {
            auto& stats = result.stats.emplace();
            stats.fixed_shots = run.metrics.scalar_shots + run.metrics.packed_shots;
            stats.scalar_shots = run.metrics.scalar_shots;
            stats.packed_shots = run.metrics.packed_shots;
            stats.packed_batches = run.metrics.packed_batches;
            stats.batch_size = run.metrics.batch_size;
            stats.workers = run.metrics.workers;
            stats.packed_worker_bytes = run.metrics.packed_worker_bytes;
            stats.peak_active_width = fixed->metrics().peak_active_width;
        }
    } else {
        if (batch_size.value_or(1) > 1) {
            throw std::invalid_argument(
                "batch_size greater than 1 requires fixed execution; selected strategy is "
                "trajectory");
        }
        result = std::get<PreparedNonComputationalRequest>(data_->implementation)
                     .sample(shots, seed, threads, collect_stats);
    }
    if (result.stats) {
        result.stats->preparation_compilations = data_->preparation_compilations;
        result.stats->preparation_seconds = data_->preparation_seconds;
        result.stats->sampling_seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
    }
    return result;
}

}  // namespace clifft
