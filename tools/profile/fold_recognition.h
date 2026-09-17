#pragma once

#include "clifft/circuit/parser.h"
#include "clifft/sampling/results.h"
#include "clifft/tableau/tableau.h"

#include "fold_blocks_native.h"

#include <map>
#include <optional>
#include <unordered_map>

namespace fold_recognition {
using clifft::AstNode;
using clifft::Circuit;
using clifft::GateType;
using clifft::PauliString;
using clifft::Target;

struct FamilySource {
    unsigned distance, data_width;
    std::string_view circuit;
    std::span<const std::string_view> stabilizers;
    std::array<std::string_view, 3> logical;
    std::span<const std::array<unsigned, 6>> flag_records;
    const fold_blocks::Protocol* kernel;
};

struct Probe {
    // Component zero is the norm, one through three are unnormalized XYZ.
    unsigned component;
    double sign;
};

class Code {
    std::vector<PauliString> stabilizers_;
    std::array<PauliString, 3> logical_;
    std::map<unsigned, PauliString, std::greater<>> basis_;
    unsigned width_;

    static bool bit(const PauliString& p, unsigned index) {
        return index < p.num_qubits() ? p.x().bit_get(index)
                                      : p.z().bit_get(index - p.num_qubits());
    }

  public:
    explicit Code(const FamilySource& source)
        : logical_{PauliString::from_text(source.logical[0]),
                   PauliString::from_text(source.logical[1]),
                   PauliString::from_text(source.logical[2])},
          width_(source.data_width) {
        for (auto text : source.stabilizers) {
            auto p = PauliString::from_text(text);
            stabilizers_.push_back(p);
            for (const auto& [pivot, row] : basis_)
                if (bit(p, pivot))
                    p.right_multiply(row.view());
            bool inserted = false;
            for (unsigned k = 2 * width_; k-- > 0;)
                if (bit(p, k)) {
                    basis_.emplace(k, std::move(p));
                    inserted = true;
                    break;
                }
            if (!inserted)
                throw std::invalid_argument("dependent code certificate");
        }
    }

    Probe reduce(PauliString p) const {
        for (const auto& stabilizer : stabilizers_)
            if (!p.view().commutes(stabilizer.view()))
                return {0, 0};
        bool x = !p.view().commutes(logical_[2].view());
        bool z = !p.view().commutes(logical_[0].view());
        unsigned component = x ? (z ? 2 : 1) : (z ? 3 : 0);
        if (component)
            p.right_multiply(logical_[component - 1].view());
        for (const auto& [pivot, row] : basis_)
            if (bit(p, pivot))
                p.right_multiply(row.view());
        for (unsigned k = 0; k < 2 * width_; ++k)
            if (bit(p, k))
                throw std::invalid_argument("incomplete code certificate");
        if (!p.is_hermitian())
            throw std::invalid_argument("non-Hermitian logical reduction");
        return {component, p.sign() ? -1.0 : 1.0};
    }
};

struct Family {
    FamilySource source;
    Circuit body;
    Code code;
    explicit Family(const FamilySource& source)
        : source(source), body(clifft::parse(source.circuit)), code(source) {
        std::vector<bool> seen(body.num_qubits);
        size_t noise = 0;
        for (const auto& node : body.nodes) {
            if (clifft::gate_traits(node.gate).noise) {
                ++noise;
                continue;
            }
            for (auto target : node.targets)
                if (!target.is_rec() && !seen[target.value()]) {
                    if (node.gate != GateType::R)
                        throw std::invalid_argument("template reads a wire before its first reset");
                    seen[target.value()] = true;
                }
        }
        if (noise != source.kernel->noise.size())
            throw std::invalid_argument("template and kernel noise sites disagree");
    }
};

struct Request {
    bool survivors = true;
    bool keep_records = false;
    bool fixed_fault_weight = false;
    std::span<const uint8_t> postselection;
    std::span<const uint8_t> expected_detectors;
};

struct Certificate {
    const Family* family;
    std::vector<uint32_t> physical_qubits;
    std::vector<double> probabilities;
    std::vector<Probe> probes;
    unsigned erased_prefix_gates = 0, suffix_gates = 0;
};

struct Decision {
    std::optional<Certificate> certificate;
    std::string reason;
    uint32_t source_line = 0;
};

inline bool plain_clifford(const AstNode& node) {
    const auto& traits = clifft::gate_traits(node.gate);
    return traits.unitary && traits.clifford &&
           (traits.arity == clifft::GateArity::SINGLE || traits.arity == clifft::GateArity::PAIR) &&
           node.args.empty() && node.tag.empty() &&
           std::all_of(node.targets.begin(), node.targets.end(),
                       [](Target t) { return !t.is_rec() && !t.has_pauli() && !t.is_inverted(); });
}

inline Decision match(const Circuit& circuit, const Family& family,
                      std::span<const AstNode* const> nodes) {
    Certificate result{&family, std::vector<uint32_t>(family.body.num_qubits, UINT32_MAX), {}, {}};
    std::unordered_map<uint32_t, uint32_t> inverse;
    size_t pos = 0;
    while (pos < nodes.size() && plain_clifford(*nodes[pos]))
        ++pos;
    const auto prefix_end = pos;
    result.erased_prefix_gates = static_cast<unsigned>(pos);
    auto decline = [&](std::string reason) -> Decision {
        return {std::nullopt, std::move(reason), pos < nodes.size() ? nodes[pos]->source_line : 0};
    };
    for (const auto& expected : family.body.nodes) {
        bool noise = clifft::gate_traits(expected.gate).noise;
        if (noise) {
            // Adjacent optional channels can have different supports. Only bind
            // a site when its already-known physical targets match as well.
            bool present = pos < nodes.size() && nodes[pos]->gate == expected.gate &&
                           nodes[pos]->targets.size() == expected.targets.size();
            if (present)
                for (size_t j = 0; j < expected.targets.size(); ++j)
                    present &= nodes[pos]->targets[j].bits ==
                               result.physical_qubits[expected.targets[j].value()];
            if (!present) {
                result.probabilities.push_back(0);
                continue;
            }
        }
        if (pos == nodes.size())
            return decline("incomplete cultivation body");
        const auto& actual = *nodes[pos];
        if (actual.gate != expected.gate || actual.targets.size() != expected.targets.size() ||
            !actual.tag.empty())
            return decline("gate or physical noise location differs from certified family");
        if (noise) {
            if (actual.args.size() != 1 || !std::isfinite(actual.args[0]) || actual.args[0] < 0 ||
                actual.args[0] > 1)
                return decline("unsupported physical noise probability");
            result.probabilities.push_back(actual.args[0]);
        } else if (actual.args != expected.args) {
            return decline("gate arguments differ from certified family");
        }
        for (size_t j = 0; j < expected.targets.size(); ++j) {
            auto e = expected.targets[j], a = actual.targets[j];
            if (e.is_rec()) {
                if (e != a)
                    return decline("measurement or detector dependency differs");
            } else {
                if ((e.bits & ~Target::kValueMask) != (a.bits & ~Target::kValueMask))
                    return decline("target axes or inversions differ");
                auto& mapped = result.physical_qubits[e.value()];
                if (mapped == UINT32_MAX) {
                    if (inverse.contains(a.value()))
                        return decline("qubit relabeling is not injective");
                    mapped = a.value();
                    inverse.emplace(a.value(), e.value());
                } else if (mapped != a.value()) {
                    return decline("qubit wiring differs from certified family");
                }
            }
        }
        ++pos;
    }
    if (circuit.num_measurements != family.body.num_measurements ||
        circuit.num_detectors != family.body.num_detectors || circuit.num_observables)
        return decline("additional records or observables are unsupported");
    for (size_t i = 0; i < prefix_end; ++i)
        for (auto target : nodes[i]->targets)
            if (!inverse.contains(target.value()))
                return decline("prefix touches a wire outside the reset certificate");

    clifft::Tableau suffix(family.source.data_width);
    for (; pos < nodes.size(); ++pos) {
        const auto& node = *nodes[pos];
        if (plain_clifford(node)) {
            std::vector<uint32_t> targets;
            for (auto target : node.targets) {
                auto found = inverse.find(target.value());
                if (found == inverse.end() || found->second >= family.source.data_width)
                    return decline("suffix touches an ancilla or unrecognized wire");
                targets.push_back(found->second);
            }
            suffix.append_named_gate(node.gate, targets);
            ++result.suffix_gates;
        } else if (node.gate == GateType::EXP_VAL && node.args.empty() && node.tag.empty()) {
            PauliString observable(family.source.data_width);
            bool negative = false;
            for (auto target : node.targets) {
                auto found = inverse.find(target.value());
                if (target.is_rec() || !target.has_pauli() || found == inverse.end() ||
                    found->second >= family.source.data_width)
                    return decline("probe touches an ancilla or unrecognized wire");
                observable.set_pauli(found->second, target.pauli() != Target::kPauliZ,
                                     target.pauli() != Target::kPauliX);
                negative ^= target.is_inverted();
            }
            observable.set_sign(negative);
            result.probes.push_back(family.code.reduce(suffix.inverse().apply(observable.view())));
        } else {
            return decline("suffix needs live state execution or unsupported outputs");
        }
    }
    return {std::move(result), "eligible", 0};
}

inline Decision recognize(const Circuit& circuit, const Request& request,
                          std::span<const Family> families) {
    if (!request.survivors || request.fixed_fault_weight)
        return {std::nullopt, "only ordinary survivor sampling is supported"};
    if (request.postselection.size() != circuit.num_detectors ||
        std::any_of(request.postselection.begin(), request.postselection.end(),
                    [](uint8_t value) { return value != 1; }))
        return {std::nullopt, "every detector must be postselected"};
    if ((!request.expected_detectors.empty() &&
         request.expected_detectors.size() != circuit.num_detectors) ||
        std::any_of(request.expected_detectors.begin(), request.expected_detectors.end(),
                    [](uint8_t value) { return value != 0; }))
        return {std::nullopt, "only trivial expected detector values are supported"};
    std::vector<const AstNode*> nodes;
    nodes.reserve(circuit.nodes.size());
    for (const auto& node : circuit.nodes)
        if (node.gate != GateType::TICK)
            nodes.push_back(&node);
    Decision best{std::nullopt, "no certified family matched"};
    for (const auto& family : families) {
        auto result = match(circuit, family, nodes);
        if (result.certificate)
            return result;
        if (result.source_line >= best.source_line)
            best = std::move(result);
    }
    return best;
}

inline clifft::sampling::SamplingSurvivorResult sample(const Certificate& certificate,
                                                       unsigned shots, uint64_t seed,
                                                       bool keep_records) {
    const auto& family = *certificate.family;
    const auto& kernel = *family.source.kernel;
    clifft::sampling::SamplingSurvivorResult result;
    result.total_shots = shots;
    size_t records = family.body.num_measurements, detectors = family.body.num_detectors;
    size_t probes = certificate.probes.size();
    if (keep_records) {
        result.measurements.resize(size_t(shots) * records);
        result.detectors.resize(size_t(shots) * detectors);
        result.exp_vals.resize(size_t(shots) * probes);
    }
    fold_blocks::Executor executor;
    fold_blocks::History history;
    std::mt19937_64 rng(seed);
    std::mt19937_64 record_rng(seed ^ 0x94d049bb133111ebULL);
    for (unsigned shot = 0; shot < shots; ++shot) {
        fold_blocks::sample_sites(kernel, history, rng, certificate.probabilities);
        auto values = executor.evaluate(kernel, history);
        if ((rng() >> 11) * 0x1.0p-53 >= values[0])
            continue;
        size_t row = result.passed_shots++;
        if (keep_records) {
            for (const auto& group : family.source.flag_records) {
                uint8_t value = (record_rng() >> 11) * 0x1.0p-53 < .5;
                for (unsigned record : group)
                    result.measurements[row * records + record] = value;
            }
            for (size_t j = 0; j < probes; ++j) {
                const auto& probe = certificate.probes[j];
                result.exp_vals[row * probes + j] =
                    probe.sign * values[probe.component] / values[0];
            }
        }
    }
    // Only completed result storage shrinks; no executor capacity changes
    // inside dispatch or between shots.
    if (keep_records) {
        result.measurements.resize(size_t(result.passed_shots) * records);
        result.detectors.resize(size_t(result.passed_shots) * detectors);
        result.exp_vals.resize(size_t(result.passed_shots) * probes);
    }
    return result;
}
}  // namespace fold_recognition
