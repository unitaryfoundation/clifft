#include "clifft/optimizer/global_reconstruction.h"

#include "clifft/optimizer/active_width_analysis.h"
#include "clifft/tableau/tableau.h"
#include "clifft/util/numeric.h"
#include "clifft/util/symplectic.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <utility>

namespace clifft {
namespace {

using Bits = std::vector<uint64_t>;

// Per-source-operation multiplier for the total dilation-gate budget
// (`limits_.max_operations * kMaxDilationGatesPerSourceOp`). Multi-qubit
// Pauli-product rotations can exceed 16 gates individually while staying
// within the amortized circuit budget.
constexpr size_t kMaxDilationGatesPerSourceOp = 16;
constexpr double kTHalfTurns = 0.25;
constexpr double kSHalfTurns = 0.5;

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

void xor_sign(ReconstructionSign& a, const ReconstructionSign& b) {
    xor_into(a.words, b.words);
    a.constant ^= b.constant;
}

// Incremental GF(2) row-echelon basis keyed by highest set bit, tracking the
// input-row combination (`label`) that produces each basis vector.
class Basis {
  public:
    std::optional<Bits> insert(Bits value, Bits label) {
        if (reduce(value, &label)) {
            return label;
        }
        store_reduced(std::move(value), std::move(label));
        return std::nullopt;
    }

    std::optional<Bits> solve(Bits value) const {
        Bits label;
        if (!reduce(value, &label)) {
            return std::nullopt;
        }
        return label;
    }

    std::pair<Bits, bool> solve_or_insert(Bits value, uint32_t next_unit_index) {
        Bits label;
        if (reduce(value, &label)) {
            return {std::move(label), false};
        }
        toggle(label, next_unit_index);
        store_reduced(std::move(value), std::move(label));
        return {unit(next_unit_index), true};
    }

    size_t size() const { return size_; }

  private:
    struct Row {
        Bits value;
        Bits label;
    };

    void store_reduced(Bits value, Bits label) {
        value = canonical(std::move(value));
        assert(!value.empty());
        const uint32_t pivot =
            static_cast<uint32_t>((value.size() - 1) * 64 + 63 - std::countl_zero(value.back()));
        if (pivot >= rows_.size()) {
            rows_.resize(static_cast<size_t>(pivot) + 1);
        }
        rows_[pivot] = {std::move(value), canonical(std::move(label))};
        ++size_;
    }

    bool reduce(Bits& value, Bits* label) const {
        for (size_t w = value.size(); w-- > 0;) {
            while (value[w] != 0) {
                const uint32_t pivot =
                    static_cast<uint32_t>(w * 64 + 63 - std::countl_zero(value[w]));
                if (pivot >= rows_.size() || rows_[pivot].value.empty()) {
                    return false;
                }
                const auto& row = rows_[pivot];
                assert(row.value.size() == w + 1);
                for (size_t i = 0; i <= w; ++i) {
                    value[i] ^= row.value[i];
                }
                if (label != nullptr) {
                    xor_into(*label, row.label);
                }
            }
        }
        return true;
    }

    std::vector<Row> rows_;
    size_t size_ = 0;
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
    std::array<uint32_t, 2> targets{};
    uint8_t arity = 0;

    std::span<const uint32_t> span() const { return {targets.data(), arity}; }
};

struct PreparedOp {
    PauliString axis;
    ReconstructionSign sign;
    double half_turns = 0;
};

constexpr uint32_t kUnmappedQubit = UINT32_MAX;

// Purifies a noisy circuit by deferring measurements to ancillas, allocating
// fresh wires on reset, absorbing Clifford-angle rotations into the dilation
// frame, and propagating Pauli fault symbols forward.
class Dilation {
  public:
    explicit Dilation(const ReconstructionLimits& limits) : limits_(limits) {}

    void parse(const Circuit& circuit);

    uint32_t num_qubits() const { return n_; }
    uint32_t num_noise_bits() const { return error_bits_; }
    const std::vector<Clifford>& body() const { return body_; }
    const std::vector<Rotation>& rotations() const { return rotations_; }
    std::vector<Output>& records() { return records_; }
    std::vector<Output>& detectors() { return detectors_; }
    std::vector<Output>& observables() { return observables_; }
    std::vector<std::vector<ReconstructionNoiseOutcome>>& channels() { return channels_; }

  private:
    using Term = std::vector<std::pair<uint32_t, char>>;

    uint32_t plain_qubit_target(Target t) const {
        require_supported(!t.is_rec() && !t.is_inverted() && !t.has_pauli(),
                          "unsupported source operation");
        return t.value();
    }
    Term pauli_product_targets(const std::vector<Target>& targets, bool& inverted) const {
        require_supported(!targets.empty(), "unsupported source operation");
        Term term;
        term.reserve(targets.size());
        inverted = false;
        for (Target t : targets) {
            require_supported(t.has_pauli() && !t.is_rec(), "unsupported source operation");
            const uint32_t q = t.value();
            require_supported(std::none_of(term.begin(), term.end(),
                                           [q](const auto& entry) { return entry.first == q; }),
                              "unsupported source operation");
            term.emplace_back(q, t.pauli_char());
            inverted ^= t.is_inverted();
        }
        return term;
    }
    uint32_t fresh() {
        require_supported(n_ < limits_.max_dilation_qubits, "dilation qubit limit exceeded");
        fx_.emplace_back();
        fz_.emplace_back();
        return n_++;
    }
    uint32_t qubit(uint32_t source) {
        if (source < aliases_.size() && aliases_[source] != kUnmappedQubit) {
            return aliases_[source];
        }
        const uint32_t q = fresh();
        if (source >= aliases_.size()) {
            aliases_.resize(static_cast<size_t>(source) + 1, kUnmappedQubit);
        }
        aliases_[source] = q;
        return q;
    }
    void gate(GateType type, std::span<const uint32_t> targets) {
        require_supported(body_.size() < saturating_multiply_u64(limits_.max_operations,
                                                                 kMaxDilationGatesPerSourceOp),
                          "dilation operation limit exceeded");
        assert(targets.size() <= 2);
        Clifford op{type, {}, static_cast<uint8_t>(targets.size())};
        for (size_t i = 0; i < targets.size(); ++i) {
            op.targets[i] = targets[i];
        }
        body_.push_back(op);
    }
    void gate(GateType type, std::initializer_list<uint32_t> targets) {
        gate(type, std::span<const uint32_t>{targets.begin(), targets.size()});
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
    void apply_clifford(GateType type, std::span<const uint32_t> qs);
    void apply_clifford(GateType type, std::initializer_list<uint32_t> qs) {
        apply_clifford(type, std::span<const uint32_t>{qs.begin(), qs.size()});
    }
    void rotate(std::span<const std::pair<uint32_t, char>> source_term, double alpha,
                bool inverted = false);
    void clifford(const AstNode& node);

    const ReconstructionLimits& limits_;
    uint32_t n_ = 0;
    uint32_t error_bits_ = 0;
    std::vector<uint32_t> aliases_;
    std::vector<ReconstructionSign> fx_, fz_;
    std::vector<Clifford> body_;
    std::vector<Rotation> rotations_;
    std::vector<Output> records_, detectors_;
    std::vector<Output> observables_;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels_;
};

void Dilation::apply_clifford(GateType type, std::span<const uint32_t> qs) {
    gate(type, qs);
    switch (type) {
        case GateType::I:
        case GateType::II:
        case GateType::X:
        case GateType::Y:
        case GateType::Z:
            return;
        case GateType::H:
        case GateType::H_NXZ:
        case GateType::SQRT_Y:
        case GateType::SQRT_Y_DAG:
            std::swap(fx_[qs[0]], fz_[qs[0]]);
            return;
        case GateType::S:
        case GateType::S_DAG:
        case GateType::H_XY:
        case GateType::H_NXY:
            xor_sign(fz_[qs[0]], fx_[qs[0]]);
            return;
        case GateType::SQRT_X:
        case GateType::SQRT_X_DAG:
        case GateType::H_YZ:
        case GateType::H_NYZ:
            xor_sign(fx_[qs[0]], fz_[qs[0]]);
            return;
        case GateType::CX:
            assert(qs[0] != qs[1]);
            xor_sign(fx_[qs[1]], fx_[qs[0]]);
            xor_sign(fz_[qs[0]], fz_[qs[1]]);
            return;
        case GateType::CZ:
            assert(qs[0] != qs[1]);
            xor_sign(fz_[qs[0]], fx_[qs[1]]);
            xor_sign(fz_[qs[1]], fx_[qs[0]]);
            return;
        case GateType::SWAP:
            assert(qs[0] != qs[1]);
            std::swap(fx_[qs[0]], fx_[qs[1]]);
            std::swap(fz_[qs[0]], fz_[qs[1]]);
            return;
        default:
            break;
    }
    const auto& local = Tableau::from_named_gate(type);
    const uint32_t arity = static_cast<uint32_t>(qs.size());
    assert(local.num_qubits() == arity);
    assert(arity == 1 || qs[0] != qs[1]);
    std::array<ReconstructionSign, 2> xs, zs;
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

void Dilation::rotate(std::span<const std::pair<uint32_t, char>> source_term, double alpha,
                      bool inverted) {
    require_supported(!source_term.empty() && is_finite_robust(alpha),
                      "unsupported source operation");
    const auto clifford_rot = classify_clifford_rotation(alpha);
    if (clifford_rot == CliffordRotation::IDENTITY) {
        return;
    }

    Term term;
    term.reserve(source_term.size());
    for (auto [src_q, p] : source_term) {
        term.emplace_back(qubit(src_q), p);
    }

    if (clifford_rot == CliffordRotation::PAULI) {
        for (auto [q, p] : term) {
            apply_clifford(p == 'X' ? GateType::X : p == 'Y' ? GateType::Y : GateType::Z, {q});
        }
        return;
    }

    if (clifford_rot.has_value()) {
        const bool dagger = (*clifford_rot == CliffordRotation::SQRT_DAG) ^ inverted;
        if (term.size() == 1) {
            const auto [q, p] = term[0];
            apply_clifford(p == 'X'   ? (dagger ? GateType::SQRT_X_DAG : GateType::SQRT_X)
                           : p == 'Y' ? (dagger ? GateType::SQRT_Y_DAG : GateType::SQRT_Y)
                                      : (dagger ? GateType::S_DAG : GateType::S),
                           {q});
            return;
        }
        if (term.size() == 2 && term[0].second == term[1].second) {
            const char p = term[0].second;
            apply_clifford(p == 'X'   ? (dagger ? GateType::SQRT_XX_DAG : GateType::SQRT_XX)
                           : p == 'Y' ? (dagger ? GateType::SQRT_YY_DAG : GateType::SQRT_YY)
                                      : (dagger ? GateType::SQRT_ZZ_DAG : GateType::SQRT_ZZ),
                           {term[0].first, term[1].first});
            return;
        }
        auto conjugate_basis = [&] {
            for (auto [q, p] : term) {
                if (p == 'X')
                    apply_clifford(GateType::H, {q});
                else if (p == 'Y')
                    apply_clifford(GateType::H_YZ, {q});
            }
        };
        auto parity_ladder = [&] {
            for (size_t i = 1; i < term.size(); ++i) {
                apply_clifford(GateType::CX, {term[i].first, term[0].first});
            }
        };
        conjugate_basis();
        parity_ladder();
        apply_clifford(dagger ? GateType::S_DAG : GateType::S, {term[0].first});
        parity_ladder();
        conjugate_basis();
        return;
    }

    const uint32_t a = fresh();
    auto sign = noise_sign(term);
    sign.constant ^= inverted;
    rotations_.push_back({a, alpha, std::move(sign)});
    gate(GateType::H, {a});
    controlled(a, term);
    gate(GateType::H, {a});
}

void Dilation::clifford(const AstNode& node) {
    assert(is_clifford(node.gate) && node.gate != GateType::SPP && node.gate != GateType::SPP_DAG);
    const uint32_t arity = Tableau::from_named_gate(node.gate).num_qubits();
    require_supported(
        !node.targets.empty() && node.targets.size() % arity == 0 && node.args.empty(),
        "unsupported Clifford operands");
    if (arity == 1) {
        for (Target t : node.targets) {
            apply_clifford(node.gate, {qubit(plain_qubit_target(t))});
        }
        return;
    }
    for (size_t i = 0; i < node.targets.size(); i += 2) {
        const Target t0 = node.targets[i];
        const Target t1 = node.targets[i + 1];
        if (t0.is_rec() || t1.is_rec()) {
            require_supported(t0.is_rec() && !t0.is_inverted() && !t1.is_rec() &&
                                  (node.gate == GateType::CX || node.gate == GateType::CZ) &&
                                  t0.value() < records_.size(),
                              "unsupported feedback");
            const uint32_t q = qubit(plain_qubit_target(t1));
            const auto& r = records_[t0.value()];
            const auto control = highest(r.z);
            assert(control.has_value());
            gate(node.gate, {*control, q});
            error(q, node.gate == GateType::CX ? 'X' : 'Z', r.sign);
            continue;
        }
        const uint32_t q0 = plain_qubit_target(t0);
        const uint32_t q1 = plain_qubit_target(t1);
        require_supported(q0 != q1, "unsupported Clifford operands");
        apply_clifford(node.gate, {qubit(q0), qubit(q1)});
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
    aliases_.assign(std::min<size_t>(circuit.num_qubits, limits_.max_dilation_qubits),
                    kUnmappedQubit);
    observables_.resize(circuit.num_observables);
    std::optional<size_t> chain;
    for (const auto& node : circuit.nodes) {
        const auto g = node.gate;
        if (g != GateType::CORRELATED_ERROR && g != GateType::ELSE_CORRELATED_ERROR)
            chain.reset();
        require_supported(static_cast<uint32_t>(g) <= static_cast<uint32_t>(GateType::UNKNOWN),
                          "unsupported source operation");
        // Exhaustive switch (no `default:`) so `-Werror=switch` flags new GateType values.
        switch (g) {
            case GateType::TICK:
            case GateType::I:
            case GateType::II:
            case GateType::I_ERROR:
            case GateType::II_ERROR:
                break;
            case GateType::DETECTOR:
                detectors_.push_back(parity(node.targets));
                break;
            case GateType::OBSERVABLE_INCLUDE: {
                require_supported(!node.args.empty() && node.args[0] >= 0 &&
                                      static_cast<size_t>(node.args[0]) < observables_.size(),
                                  "unsupported output reference");
                const auto p = parity(node.targets);
                auto& output = observables_[static_cast<size_t>(node.args[0])];
                xor_into(output.z, p.z);
                xor_sign(output.sign, p.sign);
                break;
            }
            case GateType::T:
            case GateType::T_DAG: {
                require_supported(!node.targets.empty() && node.args.empty(),
                                  "unsupported source operation");
                const double alpha = g == GateType::T ? kTHalfTurns : -kTHalfTurns;
                for (Target t : node.targets) {
                    const std::pair<uint32_t, char> term{plain_qubit_target(t), 'Z'};
                    rotate({&term, 1}, alpha);
                }
                break;
            }
            case GateType::R_X:
            case GateType::R_Y:
            case GateType::R_Z: {
                require_supported(!node.targets.empty() && node.args.size() == 1,
                                  "unsupported source operation");
                const char axis = g == GateType::R_X ? 'X' : g == GateType::R_Y ? 'Y' : 'Z';
                for (Target t : node.targets) {
                    const std::pair<uint32_t, char> term{plain_qubit_target(t), axis};
                    rotate({&term, 1}, node.args[0]);
                }
                break;
            }
            case GateType::U3: {
                require_supported(!node.targets.empty() && node.args.size() == 3,
                                  "unsupported source operation");
                const double theta = node.args[0];
                const double phi = node.args[1];
                const double lambda = node.args[2];
                require_supported(
                    is_finite_robust(theta) && is_finite_robust(phi) && is_finite_robust(lambda),
                    "unsupported source operation");
                for (Target t : node.targets) {
                    const uint32_t q = plain_qubit_target(t);
                    const std::pair<uint32_t, char> z_term{q, 'Z'};
                    const std::pair<uint32_t, char> y_term{q, 'Y'};
                    rotate({&z_term, 1}, lambda);
                    rotate({&y_term, 1}, theta);
                    rotate({&z_term, 1}, phi);
                }
                break;
            }
            case GateType::R_XX:
            case GateType::R_YY:
            case GateType::R_ZZ: {
                require_supported(
                    !node.targets.empty() && node.targets.size() % 2 == 0 && node.args.size() == 1,
                    "unsupported source operation");
                const char axis = g == GateType::R_XX ? 'X' : g == GateType::R_YY ? 'Y' : 'Z';
                for (size_t i = 0; i < node.targets.size(); i += 2) {
                    const uint32_t q0 = plain_qubit_target(node.targets[i]);
                    const uint32_t q1 = plain_qubit_target(node.targets[i + 1]);
                    require_supported(q0 != q1, "unsupported source operation");
                    const std::array<std::pair<uint32_t, char>, 2> term{{{q0, axis}, {q1, axis}}};
                    rotate(term, node.args[0]);
                }
                break;
            }
            case GateType::SPP:
            case GateType::SPP_DAG:
            case GateType::TPP:
            case GateType::TPP_DAG:
            case GateType::R_PAULI: {
                require_supported(
                    g == GateType::R_PAULI ? node.args.size() == 1 : node.args.empty(),
                    "unsupported source operation");
                bool inverted = false;
                const Term term = pauli_product_targets(node.targets, inverted);
                const double alpha = g == GateType::SPP       ? kSHalfTurns
                                     : g == GateType::SPP_DAG ? -kSHalfTurns
                                     : g == GateType::TPP     ? kTHalfTurns
                                     : g == GateType::TPP_DAG ? -kTHalfTurns
                                                              : node.args[0];
                rotate(term, alpha, inverted);
                break;
            }
            case GateType::M:
            case GateType::MX:
            case GateType::MY:
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY:
            case GateType::R:
            case GateType::RX:
            case GateType::RY: {
                require_supported(!node.targets.empty(), "unsupported source operation");
                const char p = (g == GateType::MX || g == GateType::MRX || g == GateType::RX) ? 'X'
                               : (g == GateType::MY || g == GateType::MRY || g == GateType::RY)
                                   ? 'Y'
                                   : 'Z';
                for (Target t : node.targets) {
                    require_supported(
                        !t.is_rec() && !t.has_pauli() && (!is_reset(g) || !t.is_inverted()),
                        "unsupported source operation");
                    uint32_t q = qubit(t.value());
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
                }
                break;
            }
            case GateType::MPP: {
                bool inverted = false;
                const Term source_term = pauli_product_targets(node.targets, inverted);
                Term term;
                term.reserve(source_term.size());
                for (auto [src_q, p] : source_term) {
                    term.emplace_back(qubit(src_q), p);
                }
                measure(term, inverted);
                break;
            }
            case GateType::MPAD:
                require_supported(!node.targets.empty(), "unsupported source operation");
                for (Target t : node.targets) {
                    require_supported(!t.is_rec() && !t.has_pauli() && t.value() <= 1,
                                      "unsupported source operation");
                    const bool sign = (t.value() != 0) ^ t.is_inverted();
                    records_.push_back({unit(fresh()), {{}, sign}});
                }
                break;
            case GateType::READOUT_NOISE:
                require_supported(!node.targets.empty() &&
                                      (node.args.size() == 1 ||
                                       (node.args.size() == 2 && node.args[0] == node.args[1])) &&
                                      is_probability(node.args[0]),
                                  "asymmetric readout requires the original program");
                for (Target t : node.targets) {
                    require_supported(t.is_rec() && !t.is_inverted() && t.value() < records_.size(),
                                      "unsupported output reference");
                    xor_sign(records_[t.value()].sign, bit_error(node.args[0]));
                }
                break;
            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR: {
                require_supported(
                    !node.targets.empty() && node.args.size() == 1 && is_probability(node.args[0]),
                    "unsupported source operation");
                const char p = g == GateType::X_ERROR ? 'X' : g == GateType::Y_ERROR ? 'Y' : 'Z';
                for (Target t : node.targets) {
                    const uint32_t q = qubit(plain_qubit_target(t));
                    error(q, p, bit_error(node.args[0]));
                }
                break;
            }
            case GateType::DEPOLARIZE1:
            case GateType::DEPOLARIZE2:
            case GateType::DEPOLARIZE3:
            case GateType::PAULI_CHANNEL_1:
            case GateType::PAULI_CHANNEL_2:
            case GateType::PAULI_CHANNEL_3: {
                const uint32_t arity =
                    (g == GateType::DEPOLARIZE1 || g == GateType::PAULI_CHANNEL_1)   ? 1U
                    : (g == GateType::DEPOLARIZE2 || g == GateType::PAULI_CHANNEL_2) ? 2U
                                                                                     : 3U;
                const uint32_t count = uint32_t{1} << (2 * arity);
                const bool depol = g == GateType::DEPOLARIZE1 || g == GateType::DEPOLARIZE2 ||
                                   g == GateType::DEPOLARIZE3;
                require_supported(!node.targets.empty() && node.targets.size() % arity == 0 &&
                                      node.args.size() == (depol ? 1U : count - 1U),
                                  "unsupported source operation");
                for (double p : node.args) {
                    require_supported(is_probability(p), "unsupported source operation");
                }
                const double total =
                    depol ? node.args[0] : std::accumulate(node.args.begin(), node.args.end(), 0.0);
                require_supported(total <= 1.0 + kProbTolerance, "unsupported source operation");
                for (size_t offset = 0; offset < node.targets.size(); offset += arity) {
                    std::array<uint32_t, 3> src_qs{};
                    for (uint32_t j = 0; j < arity; ++j) {
                        src_qs[j] = plain_qubit_target(node.targets[offset + j]);
                        for (uint32_t prev = 0; prev < j; ++prev) {
                            require_supported(src_qs[j] != src_qs[prev],
                                              "unsupported source operation");
                        }
                    }
                    if (depol ? (total / (count - 1) == 0.0) : (total == 0.0)) {
                        continue;
                    }
                    const uint32_t start = noise_bits(2 * arity);
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
                        const uint32_t q = qubit(src_qs[j]);
                        xor_sign(fx_[q], {unit(start + 2 * j), false});
                        xor_sign(fz_[q], {unit(start + 2 * j + 1), false});
                    }
                }
                break;
            }
            case GateType::CORRELATED_ERROR:
            case GateType::ELSE_CORRELATED_ERROR: {
                require_supported(node.args.size() == 1 && is_probability(node.args[0]),
                                  "unsupported source operation");
                if (g == GateType::CORRELATED_ERROR) {
                    chain = channels_.size();
                    channels_.push_back({{1, {}}});
                }
                require_supported(chain.has_value(), "unpaired correlated error branch");
                const uint32_t i = noise_bits(1);
                auto& channel = channels_[*chain];
                const double p = channel.front().probability * node.args[0];
                channel.front().probability -= p;
                channel.push_back({p, {i}});
                for (Target t : node.targets) {
                    require_supported(t.has_pauli() && !t.is_rec() && !t.is_inverted(),
                                      "unsupported source operation");
                    error(qubit(t.value()), t.pauli_char(), {unit(i), false});
                }
                break;
            }
            case GateType::H:
            case GateType::S:
            case GateType::S_DAG:
            case GateType::X:
            case GateType::Y:
            case GateType::Z:
            case GateType::SQRT_X:
            case GateType::SQRT_X_DAG:
            case GateType::SQRT_Y:
            case GateType::SQRT_Y_DAG:
            case GateType::H_XY:
            case GateType::H_YZ:
            case GateType::H_NXY:
            case GateType::H_NXZ:
            case GateType::H_NYZ:
            case GateType::C_XYZ:
            case GateType::C_ZYX:
            case GateType::C_NXYZ:
            case GateType::C_NZYX:
            case GateType::C_XNYZ:
            case GateType::C_XYNZ:
            case GateType::C_ZNYX:
            case GateType::C_ZYNX:
            case GateType::CX:
            case GateType::CY:
            case GateType::CZ:
            case GateType::SWAP:
            case GateType::ISWAP:
            case GateType::ISWAP_DAG:
            case GateType::SQRT_XX:
            case GateType::SQRT_XX_DAG:
            case GateType::SQRT_YY:
            case GateType::SQRT_YY_DAG:
            case GateType::SQRT_ZZ:
            case GateType::SQRT_ZZ_DAG:
            case GateType::CXSWAP:
            case GateType::CZSWAP:
            case GateType::SWAPCX:
            case GateType::XCX:
            case GateType::XCY:
            case GateType::XCZ:
            case GateType::YCX:
            case GateType::YCY:
            case GateType::YCZ:
                clifford(node);
                break;
            case GateType::EXP_VAL:
                require_supported(false, "expectation values require the original program");
                break;
            case GateType::MXX:
            case GateType::MYY:
            case GateType::MZZ:
            case GateType::CH:
            case GateType::CCX:
            case GateType::CCZ:
            case GateType::LEVEL_TRANSITION:
            case GateType::LEAKAGE:
            case GateType::LOSS:
            case GateType::HERALD_LEAKAGE_EVENT:
            case GateType::HERALD_LOSS_EVENT:
            case GateType::LEAKAGE_INTERACTION:
            case GateType::LOSS_INTERACTION:
            case GateType::UNKNOWN:
                require_supported(false, "unsupported source operation");
                break;
        }
    }
}

// Completes n independent commuting Pauli generators to a full symplectic
// frame by solving for dual X destabilizers and orthogonalizing them.
Tableau stabilizer_frame(const std::vector<PauliString>& generators, uint32_t n) {
    std::vector<Bits> patterns(2 * n);
    for (uint32_t j = 0; j < n; ++j) {
        const auto z_words = generators[j].z().words;
        for (size_t w = 0; w < z_words.size(); ++w) {
            for (uint64_t v = z_words[w]; v != 0; v &= v - 1) {
                toggle(patterns[w * 64 + std::countr_zero(v)], j);
            }
        }
        const auto x_words = generators[j].x().words;
        for (size_t w = 0; w < x_words.size(); ++w) {
            for (uint64_t v = x_words[w]; v != 0; v &= v - 1) {
                toggle(patterns[n + w * 64 + std::countr_zero(v)], j);
            }
        }
    }
    Basis columns;
    for (uint32_t i = 0; i < 2 * n; ++i) {
        columns.insert(std::move(patterns[i]), unit(i));
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
        // Generator j anticommutes only with dual j, so multiplying p by it leaves
        // the other commutation bits of p unchanged and all corrections can be
        // collected before p changes.
        std::vector<uint32_t> corrections;
        {
            const PauliCommutation commutation(p.x(), p.z());
            for (uint32_t j = 0; j < i; ++j) {
                if (commutation.anti_commutes(duals[j].x(), duals[j].z()))
                    corrections.push_back(j);
            }
        }
        for (uint32_t j : corrections)
            p.right_multiply(generators[j].view());
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
    std::vector<Bits> column_patterns(2 * n);
    for (uint32_t j = 0; j < ops.size(); ++j) {
        const auto z_words = ops[j].axis.z().words;
        for (size_t w = 0; w < z_words.size(); ++w) {
            for (uint64_t v = z_words[w]; v != 0; v &= v - 1) {
                toggle(column_patterns[w * 64 + std::countr_zero(v)], j);
            }
        }
        const auto x_words = ops[j].axis.x().words;
        for (size_t w = 0; w < x_words.size(); ++w) {
            for (uint64_t v = x_words[w]; v != 0; v &= v - 1) {
                toggle(column_patterns[n + w * 64 + std::countr_zero(v)], j);
            }
        }
    }
    Basis columns;
    std::vector<Bits> nullspace;
    for (uint32_t i = 0; i < 2 * n; ++i) {
        if (auto dependency = columns.insert(std::move(column_patterns[i]), unit(i)))
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

struct PreparedDilation {
    ReconstructionLimits limits;
    uint32_t n = 0;
    uint32_t error_bits = 0;
    uint32_t num_records = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    std::vector<PreparedOp> dressed_rotations;
    std::vector<PreparedOp> mapped_checks;
    std::vector<Output> records;
    std::optional<std::vector<PreparedOp>> mapped_records;
    Tableau backward;
    Tableau frame;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels;

    PauliString map_output(const Bits& z) const {
        PauliString z_axis(n);
        each_bit(z, [&](uint32_t i) { z_axis.mut_z().bit_set(i, true); });
        return frame.apply(backward.apply(z_axis.view()).view());
    }
};

PreparedDilation prepare_dilation(const Circuit& circuit, const ReconstructionLimits& limits) {
    Dilation dilation(limits);
    dilation.parse(circuit);
    const uint32_t n = dilation.num_qubits();

    // 1. Build the inverse Clifford frame by prepending local gate inverses in
    // forward order, touching only each gate's operand rows.
    Tableau backward(n);
    for (const auto& gate : dilation.body()) {
        backward.prepend_named_gate(Tableau::inverse_named_gate(gate.gate), gate.span());
    }
    const auto& rotations = dilation.rotations();
    std::vector<bool> magic(n, false);
    for (const auto& r : rotations)
        magic[r.ancilla] = true;
    std::vector<PauliString> constraints;
    constraints.reserve(rotations.size());
    for (const auto& r : rotations)
        constraints.emplace_back(backward.z_output(r.ancilla));
    std::vector<Bits> patterns(n);
    for (uint32_t j = 0; j < constraints.size(); ++j) {
        const auto cx = constraints[j].x().words;
        const auto cz = constraints[j].z().words;
        for (size_t w = 0; w < cx.size(); ++w) {
            for (uint64_t v = cx[w]; v != 0; v &= v - 1) {
                const uint32_t i = static_cast<uint32_t>(w * 64 + std::countr_zero(v));
                if (!magic[i])
                    toggle(patterns[i], j);
            }
            for (uint64_t v = cz[w]; v != 0; v &= v - 1) {
                const uint32_t i = static_cast<uint32_t>(w * 64 + std::countr_zero(v));
                if (magic[i])
                    toggle(patterns[i], j);
            }
        }
    }
    Basis allowed;
    for (uint32_t i = 0; i < n; ++i) {
        if (!magic[i])
            allowed.insert(patterns[i], unit(i));
    }

    // 2. Dress each rotation gadget axis with input stabilizers so it commutes
    // with all earlier gadget projections.
    std::vector<PreparedOp> dressed_rotations;
    dressed_rotations.reserve(rotations.size());
    for (auto it = rotations.rbegin(); it != rotations.rend(); ++it) {
        Bits z_pattern;
        for (uint32_t j = 0; j < constraints.size(); ++j) {
            if (constraints[j].x().bit_get(it->ancilla))
                toggle(z_pattern, j);
        }
        const auto dressing = allowed.solve(std::move(z_pattern));
        require_supported(dressing.has_value(), "no causal stabilizer dressing");
        PauliString p(n);
        p.mut_z().bit_set(it->ancilla, true);
        each_bit(*dressing, [&](uint32_t i) {
            assert(i != it->ancilla);
            if (magic[i])
                p.mut_x().bit_set(i, true);
            else
                p.mut_z().bit_set(i, true);
        });
        p.set_sign(false);
        dressed_rotations.push_back({std::move(p), it->sign, it->half_turns});
        allowed.insert(patterns[it->ancilla], unit(it->ancilla));
    }
    std::reverse(dressed_rotations.begin(), dressed_rotations.end());

    // 3. Complete the independent gadget projections to an n-qubit stabilizer
    // basis and fold them into the input state. Full projection rank certifies
    // a state-independent normalization factor.
    Basis independent;
    std::vector<PauliString> generators = std::move(constraints);
    const size_t num_constraints = generators.size();
    generators.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (auto dependency = independent.insert(std::move(patterns[i]), unit(i))) {
            PauliString p(n);
            each_bit(*dependency, [&](uint32_t j) {
                if (magic[j])
                    p.mut_x().bit_set(j, true);
                else
                    p.mut_z().bit_set(j, true);
            });
            p.set_sign(false);
            generators.push_back(std::move(p));
        }
    }
    require_supported(independent.size() == num_constraints, "nonuniform gadget projection");
    assert(generators.size() == n);
    Tableau frame = stabilizer_frame(generators, n);

    for (auto& op : dressed_rotations)
        op.axis = frame.apply(op.axis.view());

    PreparedDilation prepared;
    prepared.limits = limits;
    prepared.n = n;
    prepared.error_bits = dilation.num_noise_bits();
    prepared.num_records = static_cast<uint32_t>(dilation.records().size());
    prepared.num_detectors = static_cast<uint32_t>(dilation.detectors().size());
    prepared.num_observables = static_cast<uint32_t>(dilation.observables().size());
    prepared.dressed_rotations = std::move(dressed_rotations);
    prepared.records = std::move(dilation.records());
    prepared.backward = std::move(backward);
    prepared.frame = std::move(frame);
    prepared.channels = std::move(dilation.channels());
    prepared.mapped_checks.reserve(prepared.num_detectors + prepared.num_observables);
    for (auto& out : dilation.detectors())
        prepared.mapped_checks.push_back({prepared.map_output(out.z), std::move(out.sign), 0});
    for (auto& out : dilation.observables())
        prepared.mapped_checks.push_back({prepared.map_output(out.z), std::move(out.sign), 0});
    return prepared;
}

GlobalReconstruction reconstruct_variant(PreparedDilation& prepared,
                                         ReconstructionOutputs outputs) {
    // 4. Map requested outputs through the inverse frame and reduce all
    // operations to a basis of their X supports, eliminating fixed-|0> wires.
    if (outputs == ReconstructionOutputs::Records && !prepared.mapped_records) {
        std::vector<PreparedOp> mapped;
        mapped.reserve(prepared.records.size());
        for (auto& out : prepared.records)
            mapped.push_back({prepared.map_output(out.z), std::move(out.sign), 0});
        prepared.mapped_records = std::move(mapped);
        prepared.records = {};
        prepared.backward = Tableau();
        prepared.frame = Tableau();
    }

    const bool keep_records = outputs == ReconstructionOutputs::Records;
    const size_t record_count = keep_records ? prepared.mapped_records->size() : 0;
    const size_t total_ops =
        prepared.dressed_rotations.size() + record_count + prepared.mapped_checks.size();

    Basis xb;
    std::vector<Bits> directions;
    std::vector<Bits> x_coords;
    x_coords.reserve(total_ops);
    auto record_x_support = [&](const PreparedOp& op) {
        Bits x = canonical(copy(op.axis.x()));
        auto [coords, inserted] = xb.solve_or_insert(x, static_cast<uint32_t>(directions.size()));
        if (inserted)
            directions.push_back(std::move(x));
        x_coords.push_back(std::move(coords));
    };
    for (const auto& op : prepared.dressed_rotations)
        record_x_support(op);
    if (keep_records) {
        for (const auto& op : *prepared.mapped_records)
            record_x_support(op);
    }
    for (const auto& op : prepared.mapped_checks)
        record_x_support(op);

    GlobalReconstruction result;
    result.outputs = {keep_records ? prepared.num_records : 0, prepared.num_detectors,
                      prepared.num_observables};
    result.dilation_qubits = prepared.n;
    result.num_noise_bits = prepared.error_bits;
    result.num_qubits = static_cast<uint32_t>(directions.size());
    result.operations.reserve(total_ops);

    size_t op_index = 0;
    auto append_reduced = [&](const PreparedOp& op, std::optional<uint32_t> output) {
        PauliString reduced(result.num_qubits);
        each_bit(x_coords[op_index++], [&](uint32_t q) { reduced.mut_x().bit_set(q, true); });
        const auto src_z = op.axis.z().words;
        for (uint32_t q = 0; q < result.num_qubits; ++q) {
            uint64_t acc = 0;
            const size_t limit = std::min(directions[q].size(), src_z.size());
            for (size_t w = 0; w < limit; ++w)
                acc ^= directions[q][w] & src_z[w];
            if ((std::popcount(acc) & 1) != 0)
                reduced.mut_z().bit_set(q, true);
        }
        reduced.set_sign(false);
        const uint8_t phase_delta = (op.axis.phase() - reduced.phase()) & 3U;
        assert((phase_delta & 1U) == 0U);
        ReconstructionSign sign = op.sign;
        sign.constant ^= (phase_delta == 2U);
        result.operations.push_back({std::move(reduced), std::move(sign), op.half_turns, output});
    };
    for (const auto& op : prepared.dressed_rotations)
        append_reduced(op, std::nullopt);
    uint32_t next_output = 0;
    if (keep_records) {
        for (const auto& op : *prepared.mapped_records)
            append_reduced(op, next_output++);
    }
    for (const auto& op : prepared.mapped_checks)
        append_reduced(op, next_output++);

    result.noise_channels = prepared.channels;
    schedule_reconstruction(result, prepared.limits);
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
    std::vector<ReconstructionSign> mapped_signs(r.operations.size());
    for (size_t i : indices) {
        const auto& terms = r.operations[i].sign.words;
        auto [coords, inserted] = basis.solve_or_insert(terms, static_cast<uint32_t>(rows.size()));
        if (inserted) {
            rows.push_back(terms);
        }
        mapped_signs[i] = {std::move(coords), r.operations[i].sign.constant};
    }
    std::vector<Bits> columns(r.num_noise_bits);
    for (uint32_t j = static_cast<uint32_t>(rows.size()); j-- > 0;) {
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
            auto [it, inserted] =
                image.try_emplace(canonical(std::move(value)), outcome.probability);
            if (!inserted) {
                it->second += outcome.probability;
            } else {
                require_supported(image.size() <= limits.max_channel_support,
                                  "channel support limit exceeded");
            }
        }
        if (image.empty() || (image.size() == 1 && image.begin()->first.empty()))
            continue;
        std::vector<Bits> key;
        key.reserve(image.size());
        for (const auto& [value, probability] : image)
            key.push_back(value);
        auto existing = grouped.find(key);
        if (existing != grouped.end()) {
            const size_t existing_size = existing->second.size();
            const size_t image_size = image.size();
            const size_t remaining_products = limits.max_channel_products - products;
            require_supported(existing_size <= remaining_products / image_size,
                              "channel convolution work limit exceeded");
            products += existing_size * image_size;

            Distribution convolved;
            for (const auto& [a, p] : existing->second) {
                for (const auto& [b, q] : image) {
                    Bits sum = a;
                    xor_into(sum, b);
                    auto [it, inserted] = convolved.try_emplace(canonical(std::move(sum)), p * q);
                    if (!inserted) {
                        it->second += p * q;
                    } else {
                        require_supported(convolved.size() <= limits.max_channel_support,
                                          "channel support limit exceeded");
                    }
                }
            }
            existing->second = std::move(convolved);
        } else {
            grouped.emplace(std::move(key), std::move(image));
        }
    }
    columns = {};

    ReconstructionNoise result;
    result.num_bits = static_cast<uint32_t>(rows.size());
    result.signs = std::move(mapped_signs);
    result.channels.reserve(grouped.size());
    for (auto& [key, distribution] : grouped) {
        auto& channel = result.channels.emplace_back();
        channel.reserve(distribution.size());
        for (const auto& [value, probability] : distribution) {
            ReconstructionNoiseOutcome outcome{probability, {}};
            each_bit(value, [&](uint32_t i) { outcome.bits.push_back(i); });
            channel.push_back(std::move(outcome));
        }
        distribution.clear();
    }
    return result;
}

HirModule lower_reconstruction(const GlobalReconstruction& r, const ReconstructionNoise& noise,
                               const PostselectionMetadata& postselection,
                               const ReconstructionLimits& limits) {
    // 1. Lay out quantum wires [0, num_qubits), auxiliary fault-bit wires
    // [num_qubits, num_qubits + num_bits), and an optional scratch wire
    // (`carrier`) when visible records must be materialized.
    const uint32_t visible = r.outputs.num_records;
    const bool need_carrier = visible != 0;
    const uint32_t width = r.num_qubits + noise.num_bits + (need_carrier ? 1U : 0U);
    const uint32_t carrier = width - 1;

    bool need_one = false;
    size_t masks = noise.num_bits;
    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
        size_t terms = 0;
        for (auto w : noise.signs[j].words)
            terms += std::popcount(w);
        if (op.half_turns != 0) {
            const uint32_t pivot = std::min(op.axis.x().lowest_bit(), op.axis.z().lowest_bit());
            if (pivot < r.num_qubits)
                masks += 2 * terms + 1;
        } else {
            const bool is_identity = op.axis.x().is_zero() && op.axis.z().is_zero();
            const bool constant = noise.signs[j].constant ^ op.axis.sign();
            if (!op.output) {
                if (!is_identity)
                    ++masks;
                continue;
            }
            if (constant)
                need_one = true;
            if (!is_identity)
                ++masks;
            if (*op.output < visible) {
                const size_t parity_len = terms + (!is_identity ? 1 : 0) + (constant ? 1 : 0);
                masks += 2 * parity_len + 1;
            }
        }
    }
    if (need_one)
        ++masks;

    size_t noise_masks = 0;
    for (const auto& channel : noise.channels) {
        for (const auto& outcome : channel) {
            if (!outcome.bits.empty() && outcome.probability != 0)
                ++noise_masks;
        }
    }
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
    fault_records.reserve(noise.num_bits);
    for (uint32_t i = 0; i < noise.num_bits; ++i) {
        fault_records.push_back(
            hidden_measure([&](MutablePauliMaskView p) { p.z().bit_set(r.num_qubits + i, true); }));
    }
    const uint32_t one =
        need_one ? hidden_measure([](MutablePauliMaskView p) { p.set_sign(true); }) : 0;

    // 3. Emit scheduled rotations and measurements. Identity-axis measurements
    // act trivially on the quantum state, so their parities reference fault
    // bits and the optional constant-1 record without a quantum measurement.
    std::vector<std::vector<uint32_t>> output_parities(r.outputs.num_outputs());
    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
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
            std::vector<uint32_t> parity;
            each_bit(noise.signs[j].words, [&](uint32_t i) { parity.push_back(fault_records[i]); });
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
                p.set_sign(noise.signs[j].constant ^ op.axis.sign());
            });
            feedback();
        } else {
            const bool is_identity = op.axis.x().is_zero() && op.axis.z().is_zero();
            const bool constant = noise.signs[j].constant ^ op.axis.sign();
            if (!op.output) {
                if (!is_identity)
                    hidden_measure(fill_axis);
                continue;
            }
            std::vector<uint32_t> parity;
            each_bit(noise.signs[j].words, [&](uint32_t i) { parity.push_back(fault_records[i]); });
            if (!is_identity)
                parity.push_back(hidden_measure(fill_axis));
            if (constant)
                parity.push_back(one);
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

struct GlobalSamplingReconstructor::Impl {
    std::optional<PreparedDilation> prepared;
    std::string fallback_reason;
};

GlobalSamplingReconstructor::GlobalSamplingReconstructor(const Circuit& circuit,
                                                         const ReconstructionLimits& limits)
    : impl_(std::make_unique<Impl>()) {
    try {
        impl_->prepared.emplace(prepare_dilation(circuit, limits));
    } catch (const Ineligible& e) {
        impl_->fallback_reason = e.reason;
    }
}

GlobalSamplingReconstructor::~GlobalSamplingReconstructor() = default;
GlobalSamplingReconstructor::GlobalSamplingReconstructor(GlobalSamplingReconstructor&&) noexcept =
    default;
GlobalSamplingReconstructor& GlobalSamplingReconstructor::operator=(
    GlobalSamplingReconstructor&&) noexcept = default;

ReconstructionResult GlobalSamplingReconstructor::reconstruct(ReconstructionOutputs outputs) {
    if (!impl_->prepared) {
        return {std::nullopt, impl_->fallback_reason};
    }
    try {
        return {reconstruct_variant(*impl_->prepared, outputs), {}};
    } catch (const Ineligible& e) {
        return {std::nullopt, e.reason};
    }
}

ReconstructionResult reconstruct_global_sampling(const Circuit& circuit,
                                                 ReconstructionOutputs outputs,
                                                 const ReconstructionLimits& limits) {
    return GlobalSamplingReconstructor(circuit, limits).reconstruct(outputs);
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
