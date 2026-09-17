#pragma once

// Offline research preparation. All topology planning still happens before
// constructing a production executor; this is not a trajectory interpreter.

#include "clifft/circuit/circuit.h"
#include "clifft/util/fault_sampling.h"
#include "clifft/util/xoshiro.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace fault_study {
using namespace clifft;

struct Outcome {
    double probability;
    std::vector<Target> paulis;
};

struct Site {
    double probability = 0;
    std::vector<Outcome> outcomes;
    uint32_t source_line = 0;
};

struct Entry {
    AstNode node;
    std::optional<size_t> site;
};

struct Prepared {
    Circuit metadata;
    std::vector<Entry> entries;
    std::vector<Site> sites;

    explicit Prepared(const Circuit& circuit) : metadata(circuit.metadata_only_copy()) {
        for (size_t i = 0; i < circuit.nodes.size(); ++i) {
            const auto& node = circuit.nodes[i];
            if (node.gate == GateType::LEVEL_TRANSITION || node.gate == GateType::LEAKAGE ||
                node.gate == GateType::LOSS || node.gate == GateType::ELSE_CORRELATED_ERROR) {
                throw std::invalid_argument("unsupported specialization instruction");
            }
            if (!is_noise_gate(node.gate) || node.gate == GateType::READOUT_NOISE) {
                entries.push_back({node, std::nullopt});
                continue;
            }
            if (node.gate == GateType::CORRELATED_ERROR) {
                Site site;
                site.source_line = node.source_line;
                double remaining = 1;
                do {
                    const auto& link = circuit.nodes[i];
                    const double probability = remaining * link.args.at(0);
                    if (probability > 0 && !link.targets.empty()) {
                        site.outcomes.push_back({probability, link.targets});
                    }
                    remaining *= 1 - link.args.at(0);
                    ++i;
                } while (i < circuit.nodes.size() &&
                         circuit.nodes[i].gate == GateType::ELSE_CORRELATED_ERROR);
                --i;
                add_site(node, std::move(site));
                continue;
            }
            uint32_t arity = 0;
            bool depolarize = false;
            switch (node.gate) {
                case GateType::X_ERROR:
                case GateType::Y_ERROR:
                case GateType::Z_ERROR:
                case GateType::PAULI_CHANNEL_1:
                    arity = 1;
                    break;
                case GateType::PAULI_CHANNEL_2:
                    arity = 2;
                    break;
                case GateType::PAULI_CHANNEL_3:
                    arity = 3;
                    break;
                case GateType::DEPOLARIZE1:
                    arity = 1;
                    depolarize = true;
                    break;
                case GateType::DEPOLARIZE2:
                    arity = 2;
                    depolarize = true;
                    break;
                case GateType::DEPOLARIZE3:
                    arity = 3;
                    depolarize = true;
                    break;
                default:
                    throw std::invalid_argument("unsupported quantum noise channel");
            }
            const uint32_t count = (1u << (2 * arity)) - 1;
            for (size_t start = 0; start < node.targets.size(); start += arity) {
                Site site;
                site.source_line = node.source_line;
                for (uint32_t code = 1; code <= count; ++code) {
                    double probability;
                    if (depolarize) {
                        probability = node.args.at(0) / count;
                    } else if (node.gate == GateType::X_ERROR || node.gate == GateType::Y_ERROR ||
                               node.gate == GateType::Z_ERROR) {
                        const uint32_t selected = node.gate == GateType::X_ERROR   ? 1
                                                  : node.gate == GateType::Y_ERROR ? 2
                                                                                   : 3;
                        probability = code == selected ? node.args.at(0) : 0;
                    } else {
                        probability = node.args.at(code - 1);
                    }
                    if (probability == 0) {
                        continue;
                    }
                    Outcome outcome{probability, {}};
                    for (uint32_t q = 0; q < arity; ++q) {
                        const uint32_t digit = (code >> (2 * (arity - 1 - q))) & 3;
                        if (digit) {
                            outcome.paulis.push_back(Target::pauli(
                                node.targets.at(start + q).value(), digit << Target::kPauliShift));
                        }
                    }
                    site.outcomes.push_back(std::move(outcome));
                }
                add_site(node, std::move(site));
                if (depolarize) {
                    sites.back().probability = node.args.at(0) / count == 0 ? 0 : node.args.at(0);
                }
            }
        }
    }

    void add_site(const AstNode& node, Site site) {
        for (const auto& outcome : site.outcomes) {
            site.probability += outcome.probability;
        }
        entries.push_back({node, sites.size()});
        sites.push_back(std::move(site));
    }

    std::vector<double> probabilities() const {
        std::vector<double> result;
        for (const auto& site : sites) {
            result.push_back(site.probability);
        }
        return result;
    }

    std::vector<int> draw(Xoshiro256PlusPlus& rng, KFaultSampler* fixed = nullptr) const {
        std::vector<int> choices(sites.size(), -1);
        std::vector<bool> fired(sites.size());
        if (fixed) {
            for (auto i : fixed->sample([&] { return rng.next_double(); })) {
                fired[i] = true;
            }
        }
        for (size_t i = 0; i < sites.size(); ++i) {
            const auto& site = sites[i];
            if ((fixed && !fired[i]) || (!fixed && rng.next_double() >= site.probability)) {
                continue;
            }
            double draw = rng.next_double() * site.probability;
            for (size_t j = 0; j < site.outcomes.size(); ++j) {
                if (draw < site.outcomes[j].probability) {
                    choices[i] = static_cast<int>(j);
                    break;
                }
                draw -= site.outcomes[j].probability;
            }
            // Roundoff at the top of a conditioned categorical distribution
            // must not silently turn a selected fault site into identity.
            if (choices[i] < 0) {
                for (size_t j = site.outcomes.size(); j > 0; --j) {
                    if (site.outcomes[j - 1].probability > 0) {
                        choices[i] = static_cast<int>(j - 1);
                        break;
                    }
                }
            }
        }
        return choices;
    }

    Circuit materialize(const std::vector<int>& choices) const {
        if (choices.size() != sites.size()) {
            throw std::invalid_argument("one choice required per quantum noise site");
        }
        auto circuit = metadata;
        circuit.nodes.reserve(entries.size());
        for (const auto& entry : entries) {
            if (!entry.site) {
                circuit.nodes.push_back(entry.node);
                continue;
            }
            const auto& site = sites[*entry.site];
            const auto choice = choices[*entry.site];
            const Outcome* outcome = choice < 0 ? nullptr : &site.outcomes.at(choice);
            if (outcome) {
                for (auto pauli : outcome->paulis) {
                    auto gate = pauli.pauli() == Target::kPauliX   ? GateType::X
                                : pauli.pauli() == Target::kPauliY ? GateType::Y
                                                                   : GateType::Z;
                    circuit.nodes.push_back(
                        {gate, {Target::qubit(pauli.value())}, {}, site.source_line, {}});
                }
            }
        }
        return circuit;
    }
};
}  // namespace fault_study
