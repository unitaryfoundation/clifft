#include "clifft/sampling/folded/compile.h"

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/sampling/executor.h"
#include "clifft/sampling/folded/region.h"

namespace clifft::sampling::folded {

std::optional<ExecutablePlan> compile_region(const std::string& source, const std::string& prefix,
                                             const std::string& block,
                                             const std::vector<std::string>& contractions,
                                             const std::vector<size_t>& dimensions,
                                             SamplingPlanOptions options, bool normalize_syndromes,
                                             HirPassManager* passes) {
    if (dimensions.size() != 3)
        throw std::invalid_argument("folded compilation requires three contraction dimensions");
    const auto circuit = parse(source);
    auto region = std::make_shared<RegionPlan>(block, contractions, dimensions[0], dimensions[1],
                                               dimensions[2]);
    auto hir = trace(parse(prefix));
    if (passes)
        passes->run(hir);
    auto plan = plan_sampling(hir);
    if (plan.num_visible_records != region->prefix_visible_ ||
        plan.num_hidden_records != region->prefix_hidden_ ||
        circuit.num_measurements != region->visible_)
        throw std::invalid_argument(
            "folded compiler record accounting differs from ordinary tracing");
    SampleFoldedRegion action;
    action.plan = region;
    std::vector<PlannedAction> retained;
    for (auto& planned : plan.actions) {
        if (const auto* probe = std::get_if<WriteExpectationValue>(&planned.action)) {
            if (planned.active_before != 1 || !probe->active)
                return std::nullopt;
            action.boundary.push_back(*probe->active);
        } else {
            if (std::holds_alternative<ApplyReadoutNoise>(planned.action) ||
                std::holds_alternative<ApplyInstrument>(planned.action))
                return std::nullopt;
            std::visit(
                [&](auto& typed) {
                    using T = std::decay_t<decltype(typed)>;
                    if constexpr (std::is_same_v<T, MeasureActivePauli> ||
                                  std::is_same_v<T, MeasureDormantRandom> ||
                                  std::is_same_v<T, RecordClassical>) {
                        if (index(typed.record) >= plan.num_visible_records)
                            typed.record =
                                RecordSlot{index(typed.record) + uint32_t(region->visible_) -
                                           plan.num_visible_records};
                    }
                },
                planned.action);
            retained.push_back(std::move(planned));
        }
    }
    try {
        region->certify(action.boundary);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
    plan.actions = std::move(retained);
    plan.num_visible_records = region->visible_;
    plan.num_hidden_records = region->hidden_;
    plan.num_detectors = circuit.num_detectors;
    plan.num_observables = circuit.num_observables;
    plan.num_exp_vals = 0;
    plan.final_tableau.reset();
    plan.source_map.reset();
    plan.specialization_note =
        "certified terminal folded MSC region; scalar sampling and postselection";
    for (const auto& site : region->sites_) {
        PresampledNoiseSite distribution;
        distribution.total_probability = site.probability;
        auto& symbols = action.faults.emplace_back();
        if (site.probability == 0) {
            plan.presampled_noise_sites.push_back(std::move(distribution));
            continue;
        }
        for (size_t i = 0; i < site.channels.size(); ++i) {
            SymbolId symbol{static_cast<uint32_t>(plan.symbols.size())};
            plan.symbols.push_back(SymbolKind::Presampled);
            symbols.push_back(symbol);
            distribution.outcomes.push_back({symbol, site.probability / site.channels.size()});
        }
        plan.presampled_noise_sites.push_back(std::move(distribution));
    }
    const auto check_option = [](std::span<const uint8_t> values, size_t count) {
        if ((!values.empty() && values.size() != count) ||
            std::ranges::any_of(values, [](auto v) { return v > 1; }))
            throw std::invalid_argument(
                "folded sampling options have incorrect size or non-Boolean values");
    };
    check_option(options.postselection_mask, plan.num_detectors);
    check_option(options.expected_detectors, plan.num_detectors);
    check_option(options.expected_observables, plan.num_observables);
    if (normalize_syndromes &&
        (!options.expected_detectors.empty() || !options.expected_observables.empty()))
        throw std::invalid_argument(
            "normalize_syndromes is mutually exclusive with explicit expected parities");
    std::vector<PlannedAction> after;
    std::vector<std::vector<RecordSlot>> observables(plan.num_observables);
    uint32_t detector = 0;
    for (const auto& node : circuit.nodes) {
        if (node.gate != GateType::DETECTOR && node.gate != GateType::OBSERVABLE_INCLUDE)
            continue;
        std::vector<RecordSlot> terms;
        for (auto target : node.targets) {
            if (!target.is_rec())
                throw std::invalid_argument("folded outputs require record targets");
            terms.push_back(RecordSlot{static_cast<uint32_t>(target.value())});
        }
        if (node.gate == GateType::OBSERVABLE_INCLUDE) {
            auto& out = observables.at(static_cast<size_t>(node.args[0]));
            out.insert(out.end(), terms.begin(), terms.end());
        } else {
            const bool before = std::ranges::all_of(
                terms, [&](auto r) { return index(r) < region->prefix_visible_; });
            const bool expected =
                !options.expected_detectors.empty() && options.expected_detectors[detector];
            const bool selected = !normalize_syndromes && !options.postselection_mask.empty() &&
                                  options.postselection_mask[detector];
            PlannedAction output{before ? 1U : 0U, before ? 1U : 0U,
                                 WriteDetector{RecordParity(expected, std::move(terms)),
                                               DetectorSlot{detector++}, selected}};
            (before ? plan.actions : after).push_back(std::move(output));
        }
    }
    plan.actions.push_back({1, 0, std::move(action)});
    plan.actions.insert(plan.actions.end(), std::make_move_iterator(after.begin()),
                        std::make_move_iterator(after.end()));
    for (uint32_t i = 0; i < plan.num_observables; ++i)
        plan.actions.push_back(
            {0, 0,
             WriteObservable{RecordParity(!options.expected_observables.empty() &&
                                              options.expected_observables[i],
                                          std::move(observables[i])),
                             ObservableSlot{i}}});
    if (normalize_syndromes) {
        ExecutablePlan reference(plan);
        Executor executor(reference, 0);
        std::vector<uint8_t> noise(reference.num_presampled_symbols());
        executor.run_shot(noise);
        for (auto& planned : plan.actions) {
            if (auto* output = std::get_if<WriteDetector>(&planned.action)) {
                output->outcome = RecordParity(executor.detectors()[index(output->detector)],
                                               output->outcome.records());
                output->postselected = !options.postselection_mask.empty() &&
                                       options.postselection_mask[index(output->detector)];
            } else if (auto* output = std::get_if<WriteObservable>(&planned.action)) {
                const auto& parity = std::get<RecordParity>(output->outcome);
                output->outcome = RecordParity(executor.observables()[index(output->observable)],
                                               parity.records());
            }
        }
    }
    return ExecutablePlan(plan);
}

}  // namespace clifft::sampling::folded
