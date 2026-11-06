#include "clifft/sampling/early_postselection.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace clifft::sampling {

std::vector<size_t> EarlyPostselection::schedule(const ExecutablePlan& plan, bool enabled) {
    std::vector<size_t> order(plan.actions_.size());
    std::iota(order.begin(), order.end(), size_t{0});
    if (!enabled || !plan.has_postselection()) {
        return order;
    }
    std::vector<size_t> ready = order;
    std::vector<size_t> record_ready(size_t{plan.num_visible_records_} + plan.num_hidden_records_);
    for (size_t i = 0; i < plan.actions_.size(); ++i) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                    record_ready[typed.record] = i + 1;
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDetector>) {
                    if (typed.postselected) {
                        ready[i] = 0;
                        for (uint32_t term = 0; term < typed.outcome.count; ++term) {
                            ready[i] = std::max(
                                ready[i],
                                record_ready[plan.record_parity_terms_[typed.outcome.begin +
                                                                       term]]);
                        }
                    }
                }
            },
            plan.actions_[i]);
    }
    // A check at position i runs before the original action i, after all
    // contributing writes. Other actions retain their original relative order.
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        if (ready[left] != ready[right]) {
            return ready[left] < ready[right];
        }
        const auto* detector = std::get_if<ExecutablePlan::ExecuteDetector>(&plan.actions_[left]);
        const auto* other = std::get_if<ExecutablePlan::ExecuteDetector>(&plan.actions_[right]);
        return (detector && detector->postselected) && !(other && other->postselected);
    });
    return order;
}

EarlyPostselection::EarlyPostselection(const ExecutablePlan& plan, bool enabled) : plan_(&plan) {
    if (plan.has_instruments()) {
        throw std::invalid_argument("entrance postselection requires a fixed sampling plan");
    }
    action_order_ = schedule(plan, enabled);
    if (!enabled || !plan.has_postselection()) {
        return;
    }
    std::vector<uint8_t> presampled(plan.num_symbols_);
    for (uint32_t symbol : plan.presampled_symbols_) {
        presampled[symbol] = 1;
    }
    const auto expressions = plan.affine_registers();

    // Retain an inverse map from noise effects to initial register parities.
    // Later checks can use the existing registers without extra noise columns.
    struct RegisterBasisRow {
        AffineBool noise;
        AffineBool registers;
    };
    std::map<uint32_t, RegisterBasisRow> register_basis;
    for (uint32_t reg = 0; reg < expressions.size(); ++reg) {
        AffineBool noise;
        for (SymbolId symbol : expressions[reg].terms()) {
            if (presampled[index(symbol)]) {
                noise ^= AffineBool::symbol(symbol);
            }
        }
        AffineBool registers(expressions[reg].constant(), {SymbolId{reg}});
        while (!noise.terms().empty()) {
            const uint32_t pivot = index(noise.terms().back());
            auto found = register_basis.find(pivot);
            if (found == register_basis.end()) {
                register_basis.emplace(pivot,
                                       RegisterBasisRow{std::move(noise), std::move(registers)});
                break;
            }
            noise ^= found->second.noise;
            registers ^= found->second.registers;
        }
    }

    std::vector<AffineBool> symbols;
    symbols.reserve(plan.num_symbols_);
    for (uint32_t symbol = 0; symbol < plan.num_symbols_; ++symbol) {
        symbols.push_back(AffineBool::symbol(SymbolId{symbol}));
    }
    std::vector<AffineBool> records(size_t{plan.num_visible_records_} + plan.num_hidden_records_);
    const auto expand = [&](ExecutablePlan::PreparedExpression expression) {
        const auto& original = expressions[expression.register_id];
        AffineBool result(original.constant());
        for (SymbolId symbol : original.terms()) {
            result ^= symbols[index(symbol)];
        }
        return result;
    };
    std::map<uint32_t, AffineBool> branch_basis;
    std::map<uint32_t, AffineBool> noise_basis;
    const auto append_check = [&](AffineBool noise) {
        AffineBool registers(noise.constant());
        while (!noise.terms().empty()) {
            const auto found = register_basis.find(index(noise.terms().back()));
            if (found == register_basis.end()) {
                throw std::logic_error(
                    "entrance constraint has no initial register representation");
            }
            noise ^= found->second.noise;
            registers ^= found->second.registers;
        }
        Check check;
        check.constant = registers.constant();
        for (SymbolId reg : registers.terms()) {
            check.registers.push_back(index(reg));
        }
        checks_.push_back(std::move(check));
    };

    // Expand record snapshots and derived symbols before eliminating unknown
    // measurement/readout outcomes from the selected detector equations.
    for (const auto& action : plan.actions_) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteActiveMeasurement> ||
                              std::is_same_v<T, ExecutablePlan::ExecuteDormantMeasurement>) {
                    records[typed.record] =
                        expand(typed.correction) ^ AffineBool::symbol(SymbolId{typed.branch});
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteClassicalRecord>) {
                    records[typed.record] = expand(typed.outcome);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteSymbolDefinition>) {
                    symbols[typed.symbol] = expand(typed.value);
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteReadoutNoise>) {
                    records[typed.record] ^= AffineBool::symbol(SymbolId{typed.flip});
                } else if constexpr (std::is_same_v<T, ExecutablePlan::ExecuteDetector>) {
                    if (!typed.postselected) {
                        return;
                    }
                    AffineBool equation(typed.outcome.constant);
                    for (uint32_t term = 0; term < typed.outcome.count; ++term) {
                        const auto record = plan.record_parity_terms_[typed.outcome.begin + term];
                        equation ^= records[record];
                    }
                    while (true) {
                        const auto pivot = std::find_if(
                            equation.terms().rbegin(), equation.terms().rend(),
                            [&](SymbolId symbol) { return !presampled[index(symbol)]; });
                        if (pivot == equation.terms().rend()) {
                            break;
                        }
                        const auto [row, inserted] = branch_basis.emplace(index(*pivot), equation);
                        if (inserted) {
                            return;
                        }
                        equation ^= row->second;
                    }
                    while (!equation.terms().empty()) {
                        const auto [row, inserted] =
                            noise_basis.emplace(index(equation.terms().back()), equation);
                        if (inserted) {
                            append_check(std::move(equation));
                            return;
                        }
                        equation ^= row->second;
                    }
                    if (equation.constant()) {
                        append_check(std::move(equation));
                    }
                }
            },
            action);
    }
}

}  // namespace clifft::sampling
