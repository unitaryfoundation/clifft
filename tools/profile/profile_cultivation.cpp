// Source-mapped planning diagnostics for physical cultivation circuits.
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/sampling/planner.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    using namespace clifft;
    using namespace clifft::sampling;
    if (argc != 2)
        throw std::invalid_argument("usage: profile_cultivation circuit.stim");
    std::ifstream input(argv[1]);
    if (!input)
        throw std::invalid_argument("cannot read circuit");
    std::ostringstream text;
    text << input.rdbuf();
    auto circuit = parse(text.str());
    auto hir = trace(circuit);
    auto passes = default_hir_pass_manager();
    passes.run(hir);
    SamplingPlanOptions options;
    options.retain_source_map = true;
    auto plan = plan_sampling(hir, options);

    // Expand derived symbols and record snapshots so cancellation is retained.
    // Symmetric readout flips are state-independent, but are not necessarily
    // available at the start of the current production executor.
    std::vector<AffineBool> symbols;
    std::vector<bool> independent(plan.symbols.size());
    for (unsigned k = 0; k < plan.symbols.size(); ++k) {
        symbols.push_back(AffineBool::symbol(SymbolId{k}));
        independent[k] = plan.symbols[k] == SymbolKind::Presampled;
    }
    auto expand = [&](const AffineBool& value) {
        AffineBool result(value.constant());
        for (auto term : value.terms())
            result ^= symbols.at(index(term));
        return result;
    };
    std::vector<AffineBool> records(plan.num_visible_records + plan.num_hidden_records);
    std::cout << "action\tbefore\tafter\tdense_passes\tdetector\tnoise_only\tlines\tdescription\n";
    for (size_t k = 0; k < plan.actions.size(); ++k) {
        const auto& planned = plan.actions[k];
        const auto& action = planned.action;
        int detector = -1, noise_only = -1;
        if (const auto* op = std::get_if<DefineSymbol>(&action)) {
            symbols.at(index(op->symbol)) = expand(op->value);
        } else if (const auto* op = std::get_if<RecordClassical>(&action)) {
            records.at(index(op->record)) = expand(op->outcome);
        } else if (const auto* op = std::get_if<MeasureActivePauli>(&action)) {
            records.at(index(op->record)) = expand(op->outcome);
        } else if (const auto* op = std::get_if<MeasureDormantRandom>(&action)) {
            records.at(index(op->record)) = expand(op->outcome);
        } else if (const auto* op = std::get_if<ApplyReadoutNoise>(&action)) {
            independent.at(index(op->flip)) = op->prob_zero_to_one == op->prob_one_to_zero;
            records.at(index(op->record)) ^= symbols.at(index(op->flip));
        } else if (const auto* op = std::get_if<WriteDetector>(&action)) {
            detector = index(op->detector);
            AffineBool value(op->outcome.constant());
            for (auto record : op->outcome.records())
                value ^= records.at(index(record));
            noise_only = 1;
            for (auto term : value.terms())
                noise_only &= independent.at(index(term));
        } else if (std::holds_alternative<InstrumentBoundary>(action) ||
                   std::holds_alternative<ApplyInstrument>(action)) {
            throw std::invalid_argument("instrument analysis is unsupported");
        }
        std::cout << k << '\t' << planned.active_before << '\t' << planned.active_after << '\t'
                  << predicted_dense_passes(action) << '\t' << detector << '\t' << noise_only
                  << '\t';
        bool first = true;
        for (auto line : plan.source_map->lines_for(k)) {
            if (!first)
                std::cout << ',';
            std::cout << line;
            first = false;
        }
        std::cout << '\t' << plan.inspect_action_compact(k) << '\n';
    }
}
