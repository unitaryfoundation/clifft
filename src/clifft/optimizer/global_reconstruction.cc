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
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <utility>

namespace clifft {
namespace {

using Bits = std::vector<uint64_t>;
using SparseWords = std::vector<ReconstructionSignWord>;

// Per-source-operation multiplier for the total dilation-gate budget
// (`limits_.max_operations * kMaxDilationGatesPerSourceOp`). Multi-qubit
// Pauli-product rotations can exceed 16 gates individually while staying
// within the amortized circuit budget.
constexpr size_t kMaxDilationGatesPerSourceOp = 16;
constexpr double kTHalfTurns = 0.25;
constexpr double kSHalfTurns = 0.5;
constexpr uint32_t kUnmappedQubit = UINT32_MAX;
// Bound the dense liveness bitset on synthetic circuits with huge sparse wire IDs.
constexpr uint32_t kMaxTrackedLivenessQubits = 1U << 20;

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
void each_bit(std::span<const uint64_t> a, F&& f) {
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

// --- Sparse fault-word helpers ---

SparseWords sparse_unit(uint32_t bit) {
    return {{bit / 64, uint64_t{1} << (bit % 64)}};
}

void toggle_sparse_bit(SparseWords& words, uint32_t bit) {
    const uint32_t idx = bit / 64;
    const uint64_t m = uint64_t{1} << (bit % 64);
    const auto it = std::ranges::lower_bound(words, idx, {}, &ReconstructionSignWord::index);
    if (it != words.end() && it->index == idx) {
        it->mask ^= m;
        if (it->mask == 0) {
            words.erase(it);
        }
    } else {
        words.insert(it, {idx, m});
    }
}

void xor_sparse_words(SparseWords& dst, const SparseWords& src, SparseWords& scratch) {
    if (src.empty()) {
        return;
    }
    if (dst.empty()) {
        dst = src;
        return;
    }
    if (dst.size() == 1 && src.size() == 1) {
        if (dst[0].index == src[0].index) {
            dst[0].mask ^= src[0].mask;
            if (dst[0].mask == 0) {
                dst.clear();
            }
            return;
        }
        if (dst[0].index < src[0].index) {
            dst.push_back(src[0]);
            return;
        }
        dst.insert(dst.begin(), src[0]);
        return;
    }
    scratch.clear();
    scratch.reserve(dst.size() + src.size());
    size_t i = 0;
    size_t j = 0;
    while (i < dst.size() && j < src.size()) {
        if (dst[i].index < src[j].index) {
            scratch.push_back(dst[i++]);
        } else if (src[j].index < dst[i].index) {
            scratch.push_back(src[j++]);
        } else {
            const uint64_t merged = dst[i].mask ^ src[j].mask;
            if (merged != 0) {
                scratch.push_back({dst[i].index, merged});
            }
            ++i;
            ++j;
        }
    }
    while (i < dst.size()) {
        scratch.push_back(dst[i++]);
    }
    while (j < src.size()) {
        scratch.push_back(src[j++]);
    }
    dst.swap(scratch);
}

template <class F>
void each_sparse_bit(const SparseWords& words, F&& f) {
    for (const ReconstructionSignWord& w : words) {
        for (uint64_t v = w.mask; v != 0; v &= v - 1) {
            f(w.index * 64 + static_cast<uint32_t>(std::countr_zero(v)));
        }
    }
}

size_t sparse_popcount(const SparseWords& words) {
    size_t total = 0;
    for (const ReconstructionSignWord& w : words) {
        total += static_cast<size_t>(std::popcount(w.mask));
    }
    return total;
}

void xor_sign(ReconstructionSign& a, const ReconstructionSign& b, SparseWords& scratch) {
    xor_sparse_words(a.words, b.words, scratch);
    a.constant ^= b.constant;
}

// Incremental GF(2) row-echelon basis keyed by highest set bit on dense bit
// vectors, tracking the input-row combination (`label`) that produces each row.
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

// Sparse GF(2) row-echelon basis over large fault-bit domains. Stores only
// nonzero words for both reduced basis vectors and coordinate labels, with a
// flat pivot-to-row index table for O(1) pivot lookup.
class SparseBasis {
  public:
    explicit SparseBasis(uint32_t domain_bits) : pivot_to_row_(domain_bits, kUnmappedQubit) {}

    std::pair<SparseWords, bool> solve_or_insert(SparseWords value, uint32_t next_unit_index) {
        SparseWords label;
        while (!value.empty()) {
            const ReconstructionSignWord back = value.back();
            const uint32_t pivot =
                back.index * 64 + static_cast<uint32_t>(63 - std::countl_zero(back.mask));
            if (pivot >= pivot_to_row_.size() || pivot_to_row_[pivot] == kUnmappedQubit) {
                toggle_sparse_bit(label, next_unit_index);
                if (pivot >= pivot_to_row_.size()) {
                    pivot_to_row_.resize(static_cast<size_t>(pivot) + 1, kUnmappedQubit);
                }
                pivot_to_row_[pivot] = static_cast<uint32_t>(rows_.size());
                rows_.push_back({std::move(value), std::move(label)});
                return {sparse_unit(next_unit_index), true};
            }
            const Row& row = rows_[pivot_to_row_[pivot]];
            xor_sparse_words(value, row.value, scratch_);
            xor_sparse_words(label, row.label, scratch_);
        }
        return {std::move(label), false};
    }

  private:
    struct Row {
        SparseWords value;
        SparseWords label;
    };

    std::vector<uint32_t> pivot_to_row_;
    std::vector<Row> rows_;
    SparseWords scratch_;
};

struct Ineligible {
    std::string reason;
};

void require_supported(bool condition, const char* reason) {
    if (!condition) {
        throw Ineligible{reason};
    }
}

using Term = std::vector<std::pair<uint32_t, char>>;

uint32_t plain_qubit_target(Target t) {
    require_supported(!t.is_rec() && !t.is_inverted() && !t.has_pauli(),
                      "unsupported source operation");
    return t.value();
}

Term pauli_product_targets(const std::vector<Target>& targets, bool& inverted) {
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

// Decomposes a Clifford-angle Pauli-product rotation into elementary named
// Clifford gates. Returns false when `alpha` is genuinely non-Clifford.
template <class ApplyClifford>
bool try_emit_clifford_rotation(const Term& term, double alpha, bool inverted,
                                ApplyClifford&& apply_clifford) {
    const auto clifford_rot = classify_clifford_rotation(alpha);
    if (!clifford_rot.has_value()) {
        return false;
    }
    if (*clifford_rot == CliffordRotation::IDENTITY) {
        return true;
    }
    if (*clifford_rot == CliffordRotation::PAULI) {
        for (auto [q, p] : term) {
            const std::array<uint32_t, 1> single{q};
            apply_clifford(p == 'X'   ? GateType::X
                           : p == 'Y' ? GateType::Y
                                      : GateType::Z,
                           std::span<const uint32_t>{single});
        }
        return true;
    }
    const bool dagger = (*clifford_rot == CliffordRotation::SQRT_DAG) ^ inverted;
    if (term.size() == 1) {
        const auto [q, p] = term[0];
        const std::array<uint32_t, 1> single{q};
        apply_clifford(p == 'X'   ? (dagger ? GateType::SQRT_X_DAG : GateType::SQRT_X)
                       : p == 'Y' ? (dagger ? GateType::SQRT_Y_DAG : GateType::SQRT_Y)
                                  : (dagger ? GateType::S_DAG : GateType::S),
                       std::span<const uint32_t>{single});
        return true;
    }
    if (term.size() == 2 && term[0].second == term[1].second) {
        const char p = term[0].second;
        const std::array<uint32_t, 2> pair{term[0].first, term[1].first};
        apply_clifford(p == 'X'   ? (dagger ? GateType::SQRT_XX_DAG : GateType::SQRT_XX)
                       : p == 'Y' ? (dagger ? GateType::SQRT_YY_DAG : GateType::SQRT_YY)
                                  : (dagger ? GateType::SQRT_ZZ_DAG : GateType::SQRT_ZZ),
                       std::span<const uint32_t>{pair});
        return true;
    }
    auto conjugate_basis = [&] {
        for (auto [q, p] : term) {
            const std::array<uint32_t, 1> single{q};
            if (p == 'X') {
                apply_clifford(GateType::H, std::span<const uint32_t>{single});
            } else if (p == 'Y') {
                apply_clifford(GateType::H_YZ, std::span<const uint32_t>{single});
            }
        }
    };
    auto parity_ladder = [&] {
        for (size_t i = 1; i < term.size(); ++i) {
            const std::array<uint32_t, 2> pair{term[i].first, term[0].first};
            apply_clifford(GateType::CX, std::span<const uint32_t>{pair});
        }
    };
    conjugate_basis();
    parity_ladder();
    {
        const std::array<uint32_t, 1> pivot{term[0].first};
        apply_clifford(dagger ? GateType::S_DAG : GateType::S, std::span<const uint32_t>{pivot});
    }
    parity_ladder();
    conjugate_basis();
    return true;
}

// Pre-scan summary: split point after the last non-Clifford operation and a
// reverse-ordered stream of per-target CPTP liveness flags for Pauli noise ops.
struct SourceScan {
    size_t prefix_end = 0;
    std::vector<uint64_t> reversed_noise_live_bits;
    size_t noise_target_count = 0;

    bool pop_next_noise_target_live() {
        assert(noise_target_count > 0);
        --noise_target_count;
        return ((reversed_noise_live_bits[noise_target_count / 64] >> (noise_target_count % 64)) &
                1U) != 0;
    }
};

bool is_identity_angle(double alpha) {
    return is_finite_robust(alpha) &&
           classify_clifford_rotation(alpha) == CliffordRotation::IDENTITY;
}

bool is_identity_rotation_node(const AstNode& node) {
    switch (node.gate) {
        case GateType::R_X:
        case GateType::R_Y:
        case GateType::R_Z:
        case GateType::R_XX:
        case GateType::R_YY:
        case GateType::R_ZZ:
        case GateType::R_PAULI:
            return node.args.size() == 1 && is_identity_angle(node.args[0]);
        case GateType::U3:
            return node.args.size() == 3 && std::ranges::all_of(node.args, is_identity_angle);
        default:
            return false;
    }
}

bool node_is_nonclifford(const AstNode& node) {
    const GateType g = node.gate;
    if (static_cast<uint32_t>(g) > static_cast<uint32_t>(GateType::UNKNOWN)) {
        return false;
    }
    // Exhaustive switch so `-Werror=switch` flags new GateType values.
    switch (g) {
        case GateType::T:
        case GateType::T_DAG:
        case GateType::TPP:
        case GateType::TPP_DAG:
            return true;
        case GateType::R_X:
        case GateType::R_Y:
        case GateType::R_Z:
        case GateType::R_XX:
        case GateType::R_YY:
        case GateType::R_ZZ:
        case GateType::R_PAULI:
            return node.args.size() != 1 || !is_finite_robust(node.args[0]) ||
                   !classify_clifford_rotation(node.args[0]).has_value();
        case GateType::U3:
            return node.args.size() != 3 ||
                   std::any_of(node.args.begin(), node.args.end(), [](double a) {
                       return !is_finite_robust(a) || !classify_clifford_rotation(a).has_value();
                   });
        case GateType::TICK:
        case GateType::I:
        case GateType::II:
        case GateType::I_ERROR:
        case GateType::II_ERROR:
        case GateType::DETECTOR:
        case GateType::OBSERVABLE_INCLUDE:
        case GateType::SPP:
        case GateType::SPP_DAG:
        case GateType::M:
        case GateType::MX:
        case GateType::MY:
        case GateType::MR:
        case GateType::MRX:
        case GateType::MRY:
        case GateType::R:
        case GateType::RX:
        case GateType::RY:
        case GateType::MPP:
        case GateType::MPAD:
        case GateType::READOUT_NOISE:
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
        case GateType::EXP_VAL:
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
            return false;
    }
    return false;
}

SourceScan scan_source(const Circuit& circuit, const ReconstructionLimits& limits) {
    require_supported(circuit.nodes.size() <= limits.max_operations,
                      "source operation limit exceeded");
    require_supported(circuit.num_exp_vals == 0, "expectation values require the original program");
    require_supported(
        circuit.num_measurements <= limits.max_outputs &&
            uint64_t{circuit.num_detectors} + circuit.num_observables <= limits.max_outputs,
        "output limit exceeded");

    SourceScan scan;
    for (size_t i = 0; i < circuit.nodes.size(); ++i) {
        if (node_is_nonclifford(circuit.nodes[i])) {
            scan.prefix_end = i + 1;
        }
    }

    const uint32_t tracked_qubits = std::min(circuit.num_qubits, kMaxTrackedLivenessQubits);
    std::vector<uint64_t> live((tracked_qubits + 63) / 64, 0);
    auto get_live = [&](Target t) {
        if (t.is_rec()) {
            return false;
        }
        const uint32_t q = t.value();
        return q >= tracked_qubits || ((live[q / 64] >> (q % 64)) & 1U) != 0;
    };
    auto set_live = [&](Target t, bool value) {
        if (t.is_rec()) {
            return;
        }
        const uint32_t q = t.value();
        if (q < tracked_qubits) {
            if (value) {
                live[q / 64] |= uint64_t{1} << (q % 64);
            } else {
                live[q / 64] &= ~(uint64_t{1} << (q % 64));
            }
        }
    };
    auto push_noise_live = [&](bool value) {
        const size_t idx = scan.noise_target_count++;
        if (idx / 64 >= scan.reversed_noise_live_bits.size()) {
            scan.reversed_noise_live_bits.push_back(0);
        }
        if (value) {
            scan.reversed_noise_live_bits[idx / 64] |= uint64_t{1} << (idx % 64);
        }
    };

    // Walk backward to identify Pauli faults whose affected wires are discarded
    // by a reset or circuit termination before reaching any measurement.
    for (size_t idx = circuit.nodes.size(); idx-- > 0;) {
        const AstNode& node = circuit.nodes[idx];
        const GateType g = node.gate;
        if (static_cast<uint32_t>(g) > static_cast<uint32_t>(GateType::UNKNOWN)) {
            continue;
        }
        switch (g) {
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
                for (size_t k = node.targets.size(); k-- > 0;) {
                    push_noise_live(get_live(node.targets[k]));
                }
                break;
            case GateType::R:
            case GateType::RX:
            case GateType::RY:
                for (size_t k = node.targets.size(); k-- > 0;) {
                    set_live(node.targets[k], false);
                }
                break;
            case GateType::M:
            case GateType::MX:
            case GateType::MY:
            case GateType::MR:
            case GateType::MRX:
            case GateType::MRY:
            case GateType::MPP:
                for (size_t k = node.targets.size(); k-- > 0;) {
                    set_live(node.targets[k], true);
                }
                break;
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
            case GateType::R_XX:
            case GateType::R_YY:
            case GateType::R_ZZ:
                if (is_identity_rotation_node(node)) {
                    break;
                }
                for (size_t off = (node.targets.size() / 2) * 2; off >= 2; off -= 2) {
                    const Target t0 = node.targets[off - 2];
                    const Target t1 = node.targets[off - 1];
                    if (get_live(t0) || get_live(t1)) {
                        set_live(t0, true);
                        set_live(t1, true);
                    }
                }
                break;
            case GateType::SPP:
            case GateType::SPP_DAG:
            case GateType::TPP:
            case GateType::TPP_DAG:
            case GateType::R_PAULI:
                if (is_identity_rotation_node(node)) {
                    break;
                }
                if (std::any_of(node.targets.begin(), node.targets.end(), get_live)) {
                    for (Target t : node.targets) {
                        set_live(t, true);
                    }
                }
                break;
            case GateType::TICK:
            case GateType::I:
            case GateType::II:
            case GateType::I_ERROR:
            case GateType::II_ERROR:
            case GateType::DETECTOR:
            case GateType::OBSERVABLE_INCLUDE:
            case GateType::T:
            case GateType::T_DAG:
            case GateType::R_X:
            case GateType::R_Y:
            case GateType::R_Z:
            case GateType::U3:
            case GateType::MPAD:
            case GateType::READOUT_NOISE:
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
            case GateType::EXP_VAL:
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
                break;
        }
    }
    return scan;
}

// Shared categorical fault-channel allocator used across both the dilated
// non-Clifford prefix and the stabilizer-tracked pure-Clifford suffix.
class NoiseChannelBuilder {
  public:
    NoiseChannelBuilder(const ReconstructionLimits& limits, SourceScan& scan)
        : limits_(limits), scan_(scan) {}

    uint32_t num_noise_bits() const { return error_bits_; }
    std::vector<std::vector<ReconstructionNoiseOutcome>>& channels() { return channels_; }

    uint32_t allocate_bits(uint32_t count) {
        require_supported(count <= limits_.max_noise_bits - error_bits_,
                          "noise bit limit exceeded");
        const uint32_t start = error_bits_;
        error_bits_ += count;
        return start;
    }

    ReconstructionSign bernoulli(double p) {
        if (p == 0) {
            return {};
        }
        const uint32_t i = allocate_bits(1);
        channels_.push_back({{1 - p, {}}, {p, {i}}});
        return {sparse_unit(i), false};
    }

    template <class InjectError>
    void apply_single_axis_error(const AstNode& node, char p, InjectError&& inject) {
        require_supported(
            !node.targets.empty() && node.args.size() == 1 && is_probability(node.args[0]),
            "unsupported source operation");
        for (Target t : node.targets) {
            const uint32_t src_q = plain_qubit_target(t);
            if (!scan_.pop_next_noise_target_live()) {
                continue;
            }
            inject(src_q, p, bernoulli(node.args[0]));
        }
    }

    template <class InjectQubitFault>
    void apply_pauli_channel(const AstNode& node, InjectQubitFault&& inject_xy) {
        const GateType g = node.gate;
        const uint32_t arity = (g == GateType::DEPOLARIZE1 || g == GateType::PAULI_CHANNEL_1) ? 1U
                               : (g == GateType::DEPOLARIZE2 || g == GateType::PAULI_CHANNEL_2)
                                   ? 2U
                                   : 3U;
        const uint32_t count = uint32_t{1} << (2 * arity);
        const bool depol =
            g == GateType::DEPOLARIZE1 || g == GateType::DEPOLARIZE2 || g == GateType::DEPOLARIZE3;
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
            std::array<bool, 3> live_flags{};
            uint32_t live_arity = 0;
            for (uint32_t j = 0; j < arity; ++j) {
                src_qs[j] = plain_qubit_target(node.targets[offset + j]);
                live_flags[j] = scan_.pop_next_noise_target_live();
                if (live_flags[j]) {
                    ++live_arity;
                }
                for (uint32_t prev = 0; prev < j; ++prev) {
                    require_supported(src_qs[j] != src_qs[prev], "unsupported source operation");
                }
            }
            if (live_arity == 0 || (depol ? (total / (count - 1) == 0.0) : (total == 0.0))) {
                continue;
            }
            // Marginalize tensor-product Pauli outcomes over any discarded qubits
            // in the tuple so dead spectator wires allocate neither fault bits
            // nor dilation wires.
            const uint32_t live_count = uint32_t{1} << (2 * live_arity);
            std::array<double, 64> marginal{};
            for (uint32_t value = 0; value < count; ++value) {
                const double p = value == 0 ? std::max(0.0, 1 - total)
                                 : depol    ? total / (count - 1)
                                            : node.args[value - 1];
                if (p == 0) {
                    continue;
                }
                uint32_t compressed = 0;
                for (uint32_t j = 0; j < arity; ++j) {
                    if (!live_flags[j]) {
                        continue;
                    }
                    const uint32_t digit = (value >> (2 * (arity - j - 1))) & 3U;
                    compressed = (compressed << 2) | digit;
                }
                marginal[compressed] += p;
            }
            bool any_nontrivial = false;
            for (uint32_t c = 1; c < live_count; ++c) {
                if (marginal[c] != 0) {
                    any_nontrivial = true;
                    break;
                }
            }
            if (!any_nontrivial) {
                continue;
            }
            const uint32_t start = allocate_bits(2 * live_arity);
            std::vector<ReconstructionNoiseOutcome> channel;
            for (uint32_t c = 0; c < live_count; ++c) {
                const double p = marginal[c];
                if (p == 0) {
                    continue;
                }
                ReconstructionNoiseOutcome outcome{p, {}};
                for (uint32_t k = 0; k < live_arity; ++k) {
                    const uint32_t digit = (c >> (2 * (live_arity - k - 1))) & 3U;
                    if (digit == 1 || digit == 2) {
                        outcome.bits.push_back(start + 2 * k);
                    }
                    if (digit == 2 || digit == 3) {
                        outcome.bits.push_back(start + 2 * k + 1);
                    }
                }
                channel.push_back(std::move(outcome));
            }
            channels_.push_back(std::move(channel));
            uint32_t k = 0;
            for (uint32_t j = 0; j < arity; ++j) {
                if (live_flags[j]) {
                    inject_xy(src_qs[j], start + 2 * k, start + 2 * k + 1);
                    ++k;
                }
            }
        }
    }

    template <class InjectError>
    void apply_correlated_error(const AstNode& node, std::optional<size_t>& chain,
                                InjectError&& inject) {
        require_supported(node.args.size() == 1 && is_probability(node.args[0]),
                          "unsupported source operation");
        if (node.gate == GateType::CORRELATED_ERROR) {
            chain = channels_.size();
            channels_.push_back({{1, {}}});
        }
        require_supported(chain.has_value(), "unpaired correlated error branch");
        std::vector<std::pair<uint32_t, char>> live_targets;
        live_targets.reserve(node.targets.size());
        for (Target t : node.targets) {
            require_supported(t.has_pauli() && !t.is_rec() && !t.is_inverted(),
                              "unsupported source operation");
            if (scan_.pop_next_noise_target_live()) {
                live_targets.emplace_back(t.value(), t.pauli_char());
            }
        }
        if (live_targets.empty()) {
            // Still advance the correlated chain's remaining probability mass so
            // subsequent ELSE_CORRELATED_ERROR branches keep exact marginal odds.
            auto& channel = channels_[*chain];
            const double p = channel.front().probability * node.args[0];
            channel.front().probability -= p;
            channel.push_back({p, {}});
            return;
        }
        const uint32_t i = allocate_bits(1);
        auto& channel = channels_[*chain];
        const double p = channel.front().probability * node.args[0];
        channel.front().probability -= p;
        channel.push_back({p, {i}});
        const ReconstructionSign bit{sparse_unit(i), false};
        for (auto [src_q, pauli] : live_targets) {
            inject(src_q, pauli, bit);
        }
    }

  private:
    const ReconstructionLimits& limits_;
    SourceScan& scan_;
    uint32_t error_bits_ = 0;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels_;
};

struct PrefixOutput {
    std::optional<uint32_t> ancilla;
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

// Operator-valued parity `(-1)^constant * (-1)^(fault_bits) * axis`, where
// `axis` is an unsigned Hermitian Pauli (`axis.sign() == false`) belonging to
// an abelian subgroup on the folded prefix dilation wires.
struct PreparedOp {
    PauliString axis{0};
    ReconstructionSign sign;
    double half_turns = 0;
};

void xor_prepared_parity(PreparedOp& dst, const PreparedOp& src, SparseWords& scratch) {
    if (!src.axis.x().is_zero() || !src.axis.z().is_zero()) {
        if (dst.axis.num_qubits() == 0) {
            dst.axis = src.axis;
        } else {
            dst.axis.right_multiply(src.axis.view());
            assert(dst.axis.is_hermitian());
            dst.sign.constant ^= dst.axis.sign();
            dst.axis.set_sign(false);
        }
    }
    xor_sign(dst.sign, src.sign, scratch);
}

// Updates per-wire (fx, fz) parities forward across an elementary Clifford gate.
template <class Entry, class XorFn>
void advance_pauli_frame(GateType type, std::span<const uint32_t> qs, std::vector<Entry>& fx,
                         std::vector<Entry>& fz, XorFn&& xor_entry) {
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
            std::swap(fx[qs[0]], fz[qs[0]]);
            return;
        case GateType::S:
        case GateType::S_DAG:
        case GateType::H_XY:
        case GateType::H_NXY:
            xor_entry(fz[qs[0]], fx[qs[0]]);
            return;
        case GateType::SQRT_X:
        case GateType::SQRT_X_DAG:
        case GateType::H_YZ:
        case GateType::H_NYZ:
            xor_entry(fx[qs[0]], fz[qs[0]]);
            return;
        case GateType::CX:
            assert(qs[0] != qs[1]);
            xor_entry(fx[qs[1]], fx[qs[0]]);
            xor_entry(fz[qs[0]], fz[qs[1]]);
            return;
        case GateType::CZ:
            assert(qs[0] != qs[1]);
            xor_entry(fz[qs[0]], fx[qs[1]]);
            xor_entry(fz[qs[1]], fx[qs[0]]);
            return;
        case GateType::SWAP:
            assert(qs[0] != qs[1]);
            std::swap(fx[qs[0]], fx[qs[1]]);
            std::swap(fz[qs[0]], fz[qs[1]]);
            return;
        default:
            break;
    }
    const auto& local = Tableau::from_named_gate(type);
    const uint32_t arity = static_cast<uint32_t>(qs.size());
    assert(local.num_qubits() == arity);
    assert(arity == 1 || qs[0] != qs[1]);
    std::array<Entry, 2> xs{}, zs{};
    for (uint32_t j = 0; j < arity; ++j) {
        for (uint32_t i = 0; i < arity; ++i) {
            if (local.x_output(j).x().bit_get(i))
                xor_entry(xs[i], fx[qs[j]]);
            if (local.x_output(j).z().bit_get(i))
                xor_entry(zs[i], fx[qs[j]]);
            if (local.z_output(j).x().bit_get(i))
                xor_entry(xs[i], fz[qs[j]]);
            if (local.z_output(j).z().bit_get(i))
                xor_entry(zs[i], fz[qs[j]]);
        }
    }
    for (uint32_t i = 0; i < arity; ++i) {
        fx[qs[i]] = std::move(xs[i]);
        fz[qs[i]] = std::move(zs[i]);
    }
}

// Dispatches a half-open AST slice `[begin, end)` through a stage handler (`Dilation`
// for the non-Clifford prefix, `SuffixTracker` for the trailing Clifford suffix).
template <class Handler>
void dispatch_circuit_slice(const Circuit& circuit, size_t begin, size_t end,
                            NoiseChannelBuilder& noise, Handler& handler) {
    std::optional<size_t> chain;
    for (size_t idx = begin; idx < end; ++idx) {
        const AstNode& node = circuit.nodes[idx];
        const GateType g = node.gate;
        if (g != GateType::CORRELATED_ERROR && g != GateType::ELSE_CORRELATED_ERROR) {
            chain.reset();
        }
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
                handler.on_detector(node.targets);
                break;
            case GateType::OBSERVABLE_INCLUDE:
                require_supported(!node.args.empty() && node.args[0] >= 0 &&
                                      static_cast<size_t>(node.args[0]) < circuit.num_observables,
                                  "unsupported output reference");
                handler.on_observable(static_cast<size_t>(node.args[0]), node.targets);
                break;
            case GateType::T:
            case GateType::T_DAG: {
                require_supported(!node.targets.empty() && node.args.empty(),
                                  "unsupported source operation");
                const double alpha = g == GateType::T ? kTHalfTurns : -kTHalfTurns;
                for (Target t : node.targets) {
                    const std::pair<uint32_t, char> term{plain_qubit_target(t), 'Z'};
                    handler.on_rotate({&term, 1}, alpha, false);
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
                    handler.on_rotate({&term, 1}, node.args[0], false);
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
                    handler.on_rotate({&z_term, 1}, lambda, false);
                    handler.on_rotate({&y_term, 1}, theta, false);
                    handler.on_rotate({&z_term, 1}, phi, false);
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
                    handler.on_rotate(term, node.args[0], false);
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
                handler.on_rotate(term, alpha, inverted);
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
                    handler.on_single_instrument(g, t.value(), p, t.is_inverted());
                }
                break;
            }
            case GateType::MPP: {
                bool inverted = false;
                const Term source_term = pauli_product_targets(node.targets, inverted);
                handler.on_mpp(source_term, inverted);
                break;
            }
            case GateType::MPAD:
                require_supported(!node.targets.empty(), "unsupported source operation");
                for (Target t : node.targets) {
                    require_supported(!t.is_rec() && !t.has_pauli() && t.value() <= 1,
                                      "unsupported source operation");
                    handler.on_mpad((t.value() != 0) ^ t.is_inverted());
                }
                break;
            case GateType::READOUT_NOISE:
                require_supported(!node.targets.empty() &&
                                      (node.args.size() == 1 ||
                                       (node.args.size() == 2 && node.args[0] == node.args[1])) &&
                                      is_probability(node.args[0]),
                                  "asymmetric readout requires the original program");
                for (Target t : node.targets) {
                    require_supported(
                        t.is_rec() && !t.is_inverted() && t.value() < handler.num_records(),
                        "unsupported output reference");
                    handler.on_readout_noise(t.value(), noise.bernoulli(node.args[0]));
                }
                break;
            case GateType::X_ERROR:
            case GateType::Y_ERROR:
            case GateType::Z_ERROR: {
                const char p = g == GateType::X_ERROR ? 'X' : g == GateType::Y_ERROR ? 'Y' : 'Z';
                noise.apply_single_axis_error(
                    node, p, [&](uint32_t src_q, char axis, const ReconstructionSign& bit) {
                        handler.on_pauli_error(src_q, axis, bit);
                    });
                break;
            }
            case GateType::DEPOLARIZE1:
            case GateType::DEPOLARIZE2:
            case GateType::DEPOLARIZE3:
            case GateType::PAULI_CHANNEL_1:
            case GateType::PAULI_CHANNEL_2:
            case GateType::PAULI_CHANNEL_3:
                noise.apply_pauli_channel(node,
                                          [&](uint32_t src_q, uint32_t bit_x, uint32_t bit_z) {
                                              handler.on_pauli_channel_bits(src_q, bit_x, bit_z);
                                          });
                break;
            case GateType::CORRELATED_ERROR:
            case GateType::ELSE_CORRELATED_ERROR:
                noise.apply_correlated_error(
                    node, chain, [&](uint32_t src_q, char axis, const ReconstructionSign& bit) {
                        handler.on_pauli_error(src_q, axis, bit);
                    });
                break;
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
            case GateType::YCZ: {
                const uint32_t arity = Tableau::from_named_gate(g).num_qubits();
                require_supported(
                    !node.targets.empty() && node.targets.size() % arity == 0 && node.args.empty(),
                    "unsupported Clifford operands");
                if (arity == 1) {
                    for (Target t : node.targets) {
                        handler.on_clifford_1q(g, plain_qubit_target(t));
                    }
                    break;
                }
                for (size_t i = 0; i < node.targets.size(); i += 2) {
                    const Target t0 = node.targets[i];
                    const Target t1 = node.targets[i + 1];
                    if (t0.is_rec() || t1.is_rec()) {
                        require_supported(t0.is_rec() && !t0.is_inverted() && !t1.is_rec() &&
                                              (g == GateType::CX || g == GateType::CZ) &&
                                              t0.value() < handler.num_records(),
                                          "unsupported feedback");
                        handler.on_feedback(g, t0.value(), plain_qubit_target(t1));
                        continue;
                    }
                    const uint32_t q0 = plain_qubit_target(t0);
                    const uint32_t q1 = plain_qubit_target(t1);
                    require_supported(q0 != q1, "unsupported Clifford operands");
                    handler.on_clifford_2q(g, q0, q1);
                }
                break;
            }
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

// Purifies the circuit prefix `[0, prefix_end)` up to the last non-Clifford
// operation by deferring measurements to ancillas, allocating fresh wires on
// reset, absorbing Clifford-angle rotations into the dilation frame, and
// propagating Pauli fault symbols forward.
class Dilation {
  public:
    Dilation(const ReconstructionLimits& limits, NoiseChannelBuilder& noise)
        : limits_(limits), noise_(noise) {}

    void parse_prefix(const Circuit& circuit, size_t prefix_end) {
        aliases_.assign(std::min<size_t>(circuit.num_qubits, limits_.max_dilation_qubits),
                        kUnmappedQubit);
        observables_.resize(circuit.num_observables);
        dispatch_circuit_slice(circuit, 0, prefix_end, noise_, *this);
    }

    size_t num_records() const { return records_.size(); }
    void on_detector(const std::vector<Target>& targets) { detectors_.push_back(parity(targets)); }
    void on_observable(size_t index, const std::vector<Target>& targets) {
        const auto p = parity(targets);
        xor_into(observables_[index].first, p.first);
        xor_sign(observables_[index].second, p.second, scratch_);
    }
    void on_rotate(std::span<const std::pair<uint32_t, char>> source_term, double alpha,
                   bool inverted) {
        require_supported(!source_term.empty() && is_finite_robust(alpha),
                          "unsupported source operation");
        if (is_identity_angle(alpha)) {
            return;
        }
        Term term;
        term.reserve(source_term.size());
        for (auto [src_q, p] : source_term) {
            term.emplace_back(qubit(src_q), p);
        }
        if (try_emit_clifford_rotation(
                term, alpha, inverted,
                [&](GateType g, std::span<const uint32_t> qs) { apply_clifford(g, qs); })) {
            return;
        }
        require_supported(rotations_.size() < limits_.max_schedule_operations,
                          "schedule operation limit exceeded");
        const uint32_t a = fresh();
        auto sign = noise_sign(term);
        sign.constant ^= inverted;
        rotations_.push_back({a, alpha, std::move(sign)});
        gate(GateType::H, {a});
        controlled(a, term);
        gate(GateType::H, {a});
    }
    void on_single_instrument(GateType g, uint32_t src_q, char p, bool inverted) {
        uint32_t q = qubit(src_q);
        if (is_measurement(g)) {
            measure({{q, p}}, inverted);
        }
        if (is_reset(g) || is_measure_reset(g)) {
            // Keep the discarded wire as an environment so entanglement with
            // deferred measurement ancillas is preserved.
            q = fresh();
            aliases_[src_q] = q;
            if (p != 'Z')
                gate(GateType::H, {q});
            if (p == 'Y')
                gate(GateType::S, {q});
        }
    }
    void on_mpp(const Term& source_term, bool inverted) {
        Term term;
        term.reserve(source_term.size());
        for (auto [src_q, p] : source_term) {
            term.emplace_back(qubit(src_q), p);
        }
        measure(term, inverted);
    }
    void on_mpad(bool sign) { records_.push_back({std::nullopt, {{}, sign}}); }
    void on_readout_noise(uint32_t rec, const ReconstructionSign& flip) {
        xor_sign(records_[rec].sign, flip, scratch_);
    }
    void on_pauli_error(uint32_t src_q, char axis, const ReconstructionSign& bit) {
        error(qubit(src_q), axis, bit);
    }
    void on_pauli_channel_bits(uint32_t src_q, uint32_t bit_x, uint32_t bit_z) {
        const uint32_t q = qubit(src_q);
        toggle_sparse_bit(fx_[q].words, bit_x);
        toggle_sparse_bit(fz_[q].words, bit_z);
    }
    void on_clifford_1q(GateType g, uint32_t src_q) {
        const std::array<uint32_t, 1> single{qubit(src_q)};
        apply_clifford(g, single);
    }
    void on_clifford_2q(GateType g, uint32_t q0, uint32_t q1) {
        const std::array<uint32_t, 2> pair{qubit(q0), qubit(q1)};
        apply_clifford(g, pair);
    }
    void on_feedback(GateType g, uint32_t rec, uint32_t dst_q) {
        const uint32_t q = qubit(dst_q);
        const PrefixOutput& r = records_[rec];
        if (r.ancilla.has_value()) {
            gate(g, {*r.ancilla, q});
        }
        error(q, g == GateType::CX ? 'X' : 'Z', r.sign);
    }

    uint32_t num_qubits() const { return n_; }
    const std::vector<uint32_t>& aliases() const { return aliases_; }
    std::vector<Clifford>& body() { return body_; }
    const std::vector<Rotation>& rotations() const { return rotations_; }
    std::vector<PrefixOutput>& records() { return records_; }
    std::vector<std::pair<Bits, ReconstructionSign>>& detectors() { return detectors_; }
    std::vector<std::pair<Bits, ReconstructionSign>>& observables() { return observables_; }
    ReconstructionSign& fx(uint32_t wire) { return fx_[wire]; }
    ReconstructionSign& fz(uint32_t wire) { return fz_[wire]; }

  private:
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
    void error(uint32_t q, char p, const ReconstructionSign& sign) {
        if (p == 'X' || p == 'Y') {
            xor_sign(fx_[q], sign, scratch_);
        }
        if (p == 'Z' || p == 'Y') {
            xor_sign(fz_[q], sign, scratch_);
        }
    }
    ReconstructionSign noise_sign(const Term& term) {
        ReconstructionSign sign;
        for (auto [q, p] : term) {
            if (p == 'Z' || p == 'Y') {
                xor_sign(sign, fx_[q], scratch_);
            }
            if (p == 'X' || p == 'Y') {
                xor_sign(sign, fz_[q], scratch_);
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
        records_.push_back({a, std::move(sign)});
    }
    std::pair<Bits, ReconstructionSign> parity(const std::vector<Target>& targets) {
        std::pair<Bits, ReconstructionSign> result;
        for (Target t : targets) {
            require_supported(t.is_rec() && t.value() < records_.size(),
                              "unsupported output reference");
            const PrefixOutput& r = records_[t.value()];
            if (r.ancilla.has_value()) {
                toggle(result.first, *r.ancilla);
            }
            xor_sign(result.second, r.sign, scratch_);
        }
        return result;
    }
    void apply_clifford(GateType type, std::span<const uint32_t> qs) {
        gate(type, qs);
        advance_pauli_frame(type, qs, fx_, fz_,
                            [&](ReconstructionSign& dst, const ReconstructionSign& src) {
                                xor_sign(dst, src, scratch_);
                            });
    }

    const ReconstructionLimits& limits_;
    NoiseChannelBuilder& noise_;
    uint32_t n_ = 0;
    std::vector<uint32_t> aliases_;
    std::vector<ReconstructionSign> fx_, fz_;
    std::vector<Clifford> body_;
    std::vector<Rotation> rotations_;
    std::vector<PrefixOutput> records_;
    std::vector<std::pair<Bits, ReconstructionSign>> detectors_;
    std::vector<std::pair<Bits, ReconstructionSign>> observables_;
    SparseWords scratch_;
};

// Records which generators/operations anticommute with each single-qubit X_q
// (rows [0, n)) and Z_q (rows [n, 2*n)).
void record_commutation_columns(std::vector<Bits>& columns, uint32_t n, PauliStringView axis,
                                uint32_t op_index) {
    each_bit(axis.z().words, [&](uint32_t q) { toggle(columns[q], op_index); });
    each_bit(axis.x().words, [&](uint32_t q) { toggle(columns[n + q], op_index); });
}

PauliString pauli_from_symplectic_bits(std::span<const uint64_t> bits, uint32_t n) {
    PauliString p(n);
    each_bit(bits, [&](uint32_t q) {
        if (q < n) {
            p.mut_x().bit_set(q, true);
        } else {
            p.mut_z().bit_set(q - n, true);
        }
    });
    p.set_sign(false);
    return p;
}

// Completes n independent commuting Pauli generators to a full symplectic
// frame by solving for dual X destabilizers and orthogonalizing them.
Tableau stabilizer_frame(const std::vector<PauliString>& generators, uint32_t n) {
    if (n == 0) {
        return Tableau(0);
    }
    std::vector<Bits> patterns(2 * n);
    for (uint32_t j = 0; j < n; ++j) {
        record_commutation_columns(patterns, n, generators[j].view(), j);
    }
    Basis columns;
    for (uint32_t i = 0; i < 2 * n; ++i) {
        columns.insert(std::move(patterns[i]), unit(i));
    }
    require_supported(columns.size() == n, "dependent folded projections");
    Tableau frame(n);
    std::vector<PauliString> duals;
    duals.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto solution = columns.solve(unit(i));
        assert(solution.has_value());
        PauliString p = pauli_from_symplectic_bits(*solution, n);
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

// Evolves the trailing pure-Clifford suffix `[prefix_end, circuit.nodes.size())`
// without allocating dilation wires. Stabilizer checks and resets update an
// inverse tableau over only the touched physical qubits, allocating classical
// fair-coin bits for random syndrome projections and folding logical readouts
// directly onto the already-folded prefix dilation coordinates.
class SuffixTracker {
  public:
    SuffixTracker(uint32_t n_prefix, const ReconstructionLimits& limits, NoiseChannelBuilder& noise,
                  std::vector<PreparedOp>& records, std::vector<PreparedOp>& detectors,
                  std::vector<PreparedOp>& observables)
        : n_prefix_(n_prefix),
          limits_(limits),
          noise_(noise),
          records_(records),
          detectors_(detectors),
          observables_(observables) {}

    template <class LoadLiveWire>
    void initialize_qubits(const Circuit& circuit, size_t prefix_end,
                           const std::vector<uint32_t>& prefix_aliases,
                           LoadLiveWire&& load_live_wire) {
        for (size_t q = 0; q < prefix_aliases.size(); ++q) {
            if (prefix_aliases[q] != kUnmappedQubit) {
                ensure_mapped(static_cast<uint32_t>(q));
            }
        }
        for (size_t idx = prefix_end; idx < circuit.nodes.size(); ++idx) {
            const AstNode& node = circuit.nodes[idx];
            const GateType g = node.gate;
            // Noise ops only map live targets, and every live target reaches a
            // downstream measurement node that maps it below.
            if (static_cast<uint32_t>(g) <= static_cast<uint32_t>(GateType::UNKNOWN) &&
                (gate_arity(g) == GateArity::ANNOTATION || is_identity_noop(g) ||
                 is_noise_gate(g) || is_exp_val(g) || g == GateType::MPAD ||
                 is_identity_rotation_node(node))) {
                continue;
            }
            for (Target t : node.targets) {
                if (!t.is_rec()) {
                    ensure_mapped(t.value());
                }
            }
        }

        inv_ = Tableau(m_);
        row_scratch_ = PauliString(m_);
        is_stabilized_.assign(m_, true);
        sigma_.resize(m_);
        alpha_.resize(m_);
        beta_.resize(m_);
        logical_x_.resize(m_);
        logical_z_.resize(m_);
        fx_.resize(m_);
        fz_.resize(m_);

        for (uint32_t mode = 0; mode < m_; ++mode) {
            const uint32_t src_q = mode_to_source_[mode];
            if (src_q < prefix_aliases.size() && prefix_aliases[src_q] != kUnmappedQubit) {
                is_stabilized_[mode] = false;
                load_live_wire(prefix_aliases[src_q], logical_x_[mode], logical_z_[mode],
                               fx_[mode].sign, fz_[mode].sign);
            }
        }
    }

    void parse_suffix(const Circuit& circuit, size_t prefix_end) {
        dispatch_circuit_slice(circuit, prefix_end, circuit.nodes.size(), noise_, *this);
    }

    size_t num_records() const { return records_.size(); }
    void on_detector(const std::vector<Target>& targets) {
        detectors_.push_back(record_parity(targets));
    }
    void on_observable(size_t index, const std::vector<Target>& targets) {
        xor_prepared_parity(observables_[index], record_parity(targets), scratch_);
    }
    void on_rotate(std::span<const std::pair<uint32_t, char>> source_term, double alpha,
                   bool inverted) {
        require_supported(!source_term.empty() && is_finite_robust(alpha),
                          "unsupported source operation");
        if (is_identity_angle(alpha)) {
            return;
        }
        Term term;
        term.reserve(source_term.size());
        for (auto [src_q, p] : source_term) {
            term.emplace_back(mode_of(src_q), p);
        }
        const bool ok = try_emit_clifford_rotation(
            term, alpha, inverted,
            [&](GateType g, std::span<const uint32_t> qs) { apply_clifford(g, qs); });
        require_supported(ok, "unsupported source operation");
    }
    void on_single_instrument(GateType g, uint32_t src_q, char p, bool inverted) {
        const uint32_t q = mode_of(src_q);
        const Term single{{q, p}};
        const bool has_reset = is_reset(g) || is_measure_reset(g);
        CollapseResult collapsed = collapse_mode(
            single, has_reset ? std::optional<uint32_t>{q} : std::nullopt, is_measurement(g));
        if (is_measurement(g)) {
            PreparedOp rec = frame_parity(single);
            xor_prepared_parity(rec, collapsed.raw, scratch_);
            rec.sign.constant ^= inverted;
            records_.push_back(std::move(rec));
        }
        if (has_reset) {
            fx_[q] = {PauliString(n_prefix_), {}, 0};
            fz_[q] = {PauliString(n_prefix_), {}, 0};
            if (!collapsed.reset_clears_frame) {
                if (p == 'X') {
                    fz_[q] = std::move(collapsed.raw);
                } else {
                    fx_[q] = std::move(collapsed.raw);
                }
            }
        }
    }
    void on_mpp(const Term& source_term, bool inverted) {
        Term term;
        term.reserve(source_term.size());
        for (auto [src_q, p] : source_term) {
            term.emplace_back(mode_of(src_q), p);
        }
        CollapseResult collapsed = collapse_mode(term, std::nullopt, true);
        PreparedOp rec = frame_parity(term);
        xor_prepared_parity(rec, collapsed.raw, scratch_);
        rec.sign.constant ^= inverted;
        records_.push_back(std::move(rec));
    }
    void on_mpad(bool sign) { records_.push_back({PauliString(n_prefix_), {{}, sign}, 0}); }
    void on_readout_noise(uint32_t rec, const ReconstructionSign& flip) {
        xor_sign(records_[rec].sign, flip, scratch_);
    }
    void on_pauli_error(uint32_t src_q, char axis, const ReconstructionSign& bit) {
        if (const auto q = try_mode_of(src_q)) {
            inject_fault(*q, axis, bit);
        }
    }
    void on_pauli_channel_bits(uint32_t src_q, uint32_t bit_x, uint32_t bit_z) {
        if (const auto q = try_mode_of(src_q)) {
            toggle_sparse_bit(fx_[*q].sign.words, bit_x);
            toggle_sparse_bit(fz_[*q].sign.words, bit_z);
        }
    }
    void on_clifford_1q(GateType g, uint32_t src_q) {
        const std::array<uint32_t, 1> single{mode_of(src_q)};
        apply_clifford(g, single);
    }
    void on_clifford_2q(GateType g, uint32_t q0, uint32_t q1) {
        const std::array<uint32_t, 2> pair{mode_of(q0), mode_of(q1)};
        apply_clifford(g, pair);
    }
    void on_feedback(GateType g, uint32_t rec, uint32_t dst_q) {
        inject_parity(mode_of(dst_q), g == GateType::CX ? 'X' : 'Z', records_[rec]);
    }

  private:
    struct CollapseResult {
        PreparedOp raw;
        bool reset_clears_frame = false;
    };

    uint32_t ensure_mapped(uint32_t src_q) {
        if (src_q < source_to_mode_.size() && source_to_mode_[src_q] != kUnmappedQubit) {
            return source_to_mode_[src_q];
        }
        require_supported(m_ < limits_.max_dilation_qubits, "dilation qubit limit exceeded");
        if (src_q >= source_to_mode_.size()) {
            source_to_mode_.resize(static_cast<size_t>(src_q) + 1, kUnmappedQubit);
        }
        const uint32_t mode = m_++;
        source_to_mode_[src_q] = mode;
        mode_to_source_.push_back(src_q);
        return mode;
    }

    uint32_t mode_of(uint32_t src_q) const {
        assert(src_q < source_to_mode_.size() && source_to_mode_[src_q] != kUnmappedQubit);
        return source_to_mode_[src_q];
    }

    std::optional<uint32_t> try_mode_of(uint32_t src_q) const {
        if (src_q < source_to_mode_.size() && source_to_mode_[src_q] != kUnmappedQubit) {
            return source_to_mode_[src_q];
        }
        return std::nullopt;
    }

    void inject_fault(uint32_t mode, char p, const ReconstructionSign& sign) {
        if (p == 'X' || p == 'Y') {
            xor_sign(fx_[mode].sign, sign, scratch_);
        }
        if (p == 'Z' || p == 'Y') {
            xor_sign(fz_[mode].sign, sign, scratch_);
        }
    }

    void inject_parity(uint32_t mode, char p, const PreparedOp& op) {
        if (p == 'X' || p == 'Y') {
            xor_prepared_parity(fx_[mode], op, scratch_);
        }
        if (p == 'Z' || p == 'Y') {
            xor_prepared_parity(fz_[mode], op, scratch_);
        }
    }

    PreparedOp frame_parity(const Term& term) {
        PreparedOp out{PauliString(n_prefix_), {}, 0};
        for (auto [q, p] : term) {
            if (p == 'Z' || p == 'Y') {
                xor_prepared_parity(out, fx_[q], scratch_);
            }
            if (p == 'X' || p == 'Y') {
                xor_prepared_parity(out, fz_[q], scratch_);
            }
        }
        return out;
    }

    PreparedOp record_parity(const std::vector<Target>& targets) {
        PreparedOp out{PauliString(n_prefix_), {}, 0};
        for (Target t : targets) {
            require_supported(t.is_rec() && t.value() < records_.size(),
                              "unsupported output reference");
            xor_prepared_parity(out, records_[t.value()], scratch_);
        }
        return out;
    }

    void load_row_scratch(PauliStringView view) {
        row_scratch_.mut_x().zero_out();
        row_scratch_.mut_z().zero_out();
        row_scratch_.mut_x().xor_with(view.x());
        row_scratch_.mut_z().xor_with(view.z());
        row_scratch_.set_phase(view.phase());
    }

    void store_row_scratch(uint32_t q, bool z_row) {
        if (z_row) {
            inv_.set_z_output(q, row_scratch_.view());
        } else {
            inv_.set_x_output(q, row_scratch_.view());
        }
    }

    void swap_mode_xz(uint32_t p) {
        for (uint32_t q = 0; q < m_; ++q) {
            for (bool z_row : {false, true}) {
                const PauliStringView view = z_row ? inv_.z_output(q) : inv_.x_output(q);
                const bool a = view.x().bit_get(p);
                const bool b = view.z().bit_get(p);
                if (!a && !b) {
                    continue;
                }
                load_row_scratch(view);
                const bool old_sign = row_scratch_.sign();
                row_scratch_.set_pauli(p, b, a);
                row_scratch_.set_sign(old_sign ^ (a && b));
                store_row_scratch(q, z_row);
            }
        }
        assert(m_ > 16 || inv_.satisfies_invariants());
    }

    // Applies `W_M^{-1} = H_p * (S_dag_p)^zeta * C_{p_tilde}(p -> rest)` to
    // every row of `inv_` so that `M = p_tilde * sigma_p` maps to `+Z_p` while
    // every other `+Z_k` (`k != p`) outside `p_tilde`'s X-support stays `+Z_k`.
    void apply_mode_collapse_to_tableau(uint32_t p, bool zeta, const PauliString& p_tilde) {
        assert(!p_tilde.x().bit_get(p) && !p_tilde.z().bit_get(p));
        assert(!p_tilde.sign());
        const PauliCommutation tilde_comm(p_tilde.x(), p_tilde.z());
        for (uint32_t q = 0; q < m_; ++q) {
            for (bool z_row : {false, true}) {
                const PauliStringView view = z_row ? inv_.z_output(q) : inv_.x_output(q);
                const bool xp = view.x().bit_get(p);
                const bool zp = view.z().bit_get(p);
                const bool anti = tilde_comm.anti_commutes(view.x(), view.z());
                if (!xp && !zp && !anti) {
                    continue;
                }
                load_row_scratch(view);
                if (anti) {
                    // Right-multiplying Q by (-1)^(xp) * Z_p toggles Z_p:
                    // I_p -> Z_p, Z_p -> I_p, X_p*(-Z_p) = +i Y_p, (+i X_p Z_p)*(-Z_p) = -i X_p.
                    // Combined with p_tilde below, track exact Pauli multiplication.
                    PauliString z_p(m_);
                    z_p.set_pauli(p, false, true);
                    z_p.set_sign(xp);
                    row_scratch_.right_multiply(z_p.view());
                }
                if (xp) {
                    row_scratch_.right_multiply(p_tilde.view());
                }
                assert(row_scratch_.is_hermitian());
                const bool sign = row_scratch_.sign();
                const bool a = row_scratch_.x().bit_get(p);
                const bool b = row_scratch_.z().bit_get(p);
                if (!zeta) {
                    row_scratch_.set_pauli(p, b, a);
                    row_scratch_.set_sign(sign ^ (a && b));
                } else {
                    row_scratch_.set_pauli(p, a ^ b, a);
                    row_scratch_.set_sign(sign);
                }
                store_row_scratch(q, z_row);
            }
        }
        assert(m_ > 16 || inv_.satisfies_invariants());
    }

    // By symplectic orthogonality of `inv_`, `reset_q`'s anticommuting X and Z
    // rows are supported solely on `mode_p` if and only if every other qubit's
    // rows act as identity on `mode_p`.
    bool qubit_isolated_on_mode(uint32_t reset_q, uint32_t mode_p) const {
        const size_t target_word = mode_p / 64;
        const uint64_t allowed = uint64_t{1} << (mode_p % 64);
        const PauliStringView x_row = inv_.x_output(reset_q);
        const PauliStringView z_row = inv_.z_output(reset_q);
        const auto x_xw = x_row.x().words;
        const auto x_zw = x_row.z().words;
        const auto z_xw = z_row.x().words;
        const auto z_zw = z_row.z().words;
        for (size_t w = 0; w < x_xw.size(); ++w) {
            const uint64_t combined = x_xw[w] | x_zw[w] | z_xw[w] | z_zw[w];
            if ((combined & ~(w == target_word ? allowed : uint64_t{0})) != 0) {
                return false;
            }
        }
        return true;
    }

    CollapseResult collapse_mode(const Term& term, std::optional<uint32_t> reset_qubit,
                                 bool need_outcome) {
        PauliString mode_pauli(m_);
        for (auto [q, p] : term) {
            if (p == 'X') {
                mode_pauli.right_multiply(inv_.x_output(q));
            } else if (p == 'Y') {
                mode_pauli.right_multiply(inv_.y_output(q).view());
            } else {
                mode_pauli.right_multiply(inv_.z_output(q));
            }
        }
        assert(mode_pauli.is_hermitian());
        const bool s_m = mode_pauli.sign();
        mode_pauli.set_sign(false);

        // Case 1: Anticommuting stabilized mode -> uniform Bernoulli(1/2) outcome.
        std::optional<uint32_t> stab_pivot;
        each_bit(mode_pauli.x().words, [&](uint32_t k) {
            if (!stab_pivot.has_value() && is_stabilized_[k]) {
                stab_pivot = k;
            }
        });
        if (stab_pivot.has_value()) {
            const uint32_t p = *stab_pivot;
            const bool zeta = mode_pauli.z().bit_get(p);
            PauliString p_tilde = mode_pauli;
            p_tilde.set_pauli(p, false, false);
            p_tilde.set_sign(false);

            PreparedOp old_sigma_p = std::move(sigma_[p]);
            sigma_[p] = {PauliString(n_prefix_), {}, 0};
            each_bit(p_tilde.x().words, [&](uint32_t k) {
                if (is_stabilized_[k]) {
                    xor_prepared_parity(sigma_[k], old_sigma_p, scratch_);
                } else {
                    xor_prepared_parity(alpha_[k], old_sigma_p, scratch_);
                }
            });
            each_bit(p_tilde.z().words, [&](uint32_t k) {
                if (!is_stabilized_[k]) {
                    xor_prepared_parity(beta_[k], old_sigma_p, scratch_);
                }
            });

            apply_mode_collapse_to_tableau(p, zeta, p_tilde);

            const bool isolated_reset =
                reset_qubit.has_value() && qubit_isolated_on_mode(*reset_qubit, p);
            PreparedOp b_raw{PauliString(n_prefix_), {}, 0};
            if (!isolated_reset || need_outcome) {
                b_raw.sign = noise_.bernoulli(0.5);
            }
            if (isolated_reset) {
                sigma_[p] = {PauliString(n_prefix_), {{}, s_m}, 0};
                return {std::move(b_raw), true};
            }
            sigma_[p] = b_raw;
            sigma_[p].sign.constant ^= s_m;
            return {std::move(b_raw), false};
        }

        // Check whether any uncollapsed quantum mode participates.
        std::optional<uint32_t> quant_pivot;
        for (uint32_t q = 0; q < m_; ++q) {
            if (!is_stabilized_[q] && (mode_pauli.x().bit_get(q) || mode_pauli.z().bit_get(q))) {
                quant_pivot = q;
                break;
            }
        }

        // Case 2: Pure product of stabilized +Z_k generators -> deterministic parity.
        if (!quant_pivot.has_value()) {
            PreparedOp b_raw{PauliString(n_prefix_), {{}, s_m}, 0};
            uint32_t single_mode = kUnmappedQubit;
            uint32_t z_count = 0;
            each_bit(mode_pauli.z().words, [&](uint32_t k) {
                assert(is_stabilized_[k]);
                xor_prepared_parity(b_raw, sigma_[k], scratch_);
                single_mode = k;
                ++z_count;
            });
            if (reset_qubit.has_value() && z_count == 1 &&
                qubit_isolated_on_mode(*reset_qubit, single_mode)) {
                sigma_[single_mode] = {PauliString(n_prefix_), {{}, s_m}, 0};
                return {std::move(b_raw), true};
            }
            return {std::move(b_raw), false};
        }

        // Case 3: Projects a quantum logical observable onto the folded prefix wires.
        PauliString bar_p(n_prefix_);
        for (uint32_t q = 0; q < m_; ++q) {
            if (is_stabilized_[q]) {
                continue;
            }
            const bool mx = mode_pauli.x().bit_get(q);
            const bool mz = mode_pauli.z().bit_get(q);
            if (mx && !mz) {
                bar_p.right_multiply(logical_x_[q].view());
            } else if (!mx && mz) {
                bar_p.right_multiply(logical_z_[q].view());
            } else if (mx && mz) {
                PauliString ly = logical_x_[q];
                ly.right_multiply(logical_z_[q].view());
                ly.add_phase(1);
                assert(ly.is_hermitian());
                bar_p.right_multiply(ly.view());
            }
        }
        assert(bar_p.is_hermitian());
        const bool bar_sign = bar_p.sign();
        bar_p.set_sign(false);

        PreparedOp b_raw{std::move(bar_p), {{}, static_cast<bool>(s_m ^ bar_sign)}, 0};
        for (uint32_t k = 0; k < m_; ++k) {
            if (is_stabilized_[k]) {
                if (mode_pauli.z().bit_get(k)) {
                    xor_prepared_parity(b_raw, sigma_[k], scratch_);
                }
            } else {
                if (mode_pauli.z().bit_get(k)) {
                    xor_prepared_parity(b_raw, alpha_[k], scratch_);
                }
                if (mode_pauli.x().bit_get(k)) {
                    xor_prepared_parity(b_raw, beta_[k], scratch_);
                }
            }
        }

        const uint32_t p = *quant_pivot;
        if (!mode_pauli.x().bit_get(p)) {
            swap_mode_xz(p);
            std::swap(logical_x_[p], logical_z_[p]);
            std::swap(alpha_[p], beta_[p]);
            mode_pauli.set_pauli(p, true, false);
        }

        const bool zeta = mode_pauli.z().bit_get(p);
        PauliString p_tilde(m_);
        for (uint32_t q = 0; q < m_; ++q) {
            if (!is_stabilized_[q] && q != p) {
                p_tilde.set_pauli(q, mode_pauli.x().bit_get(q), mode_pauli.z().bit_get(q));
            }
        }
        p_tilde.set_sign(false);

        for (uint32_t q = 0; q < m_; ++q) {
            if (is_stabilized_[q] || q == p) {
                continue;
            }
            if (p_tilde.z().bit_get(q)) {
                logical_x_[q].right_multiply(logical_z_[p].view());
                assert(logical_x_[q].is_hermitian());
                xor_prepared_parity(beta_[q], alpha_[p], scratch_);
            }
            if (p_tilde.x().bit_get(q)) {
                logical_z_[q].right_multiply(logical_z_[p].view());
                assert(logical_z_[q].is_hermitian());
                xor_prepared_parity(alpha_[q], alpha_[p], scratch_);
            }
        }

        apply_mode_collapse_to_tableau(p, zeta, p_tilde);

        is_stabilized_[p] = true;
        logical_x_[p] = PauliString(0);
        logical_z_[p] = PauliString(0);
        alpha_[p] = PreparedOp{};
        beta_[p] = PreparedOp{};

        if (reset_qubit.has_value() && qubit_isolated_on_mode(*reset_qubit, p)) {
            sigma_[p] = {PauliString(n_prefix_), {{}, s_m}, 0};
            return {std::move(b_raw), true};
        }
        sigma_[p] = b_raw;
        sigma_[p].sign.constant ^= s_m;
        for (uint32_t k = 0; k < m_; ++k) {
            if (k != p && is_stabilized_[k] && mode_pauli.z().bit_get(k)) {
                xor_prepared_parity(sigma_[p], sigma_[k], scratch_);
            }
        }
        return {std::move(b_raw), false};
    }

    void apply_clifford(GateType type, std::span<const uint32_t> qs) {
        inv_.prepend_named_gate(Tableau::inverse_named_gate(type), qs);
        advance_pauli_frame(type, qs, fx_, fz_, [&](PreparedOp& dst, const PreparedOp& src) {
            xor_prepared_parity(dst, src, scratch_);
        });
    }

    uint32_t n_prefix_ = 0;
    uint32_t m_ = 0;
    const ReconstructionLimits& limits_;
    NoiseChannelBuilder& noise_;
    std::vector<PreparedOp>& records_;
    std::vector<PreparedOp>& detectors_;
    std::vector<PreparedOp>& observables_;

    std::vector<uint32_t> source_to_mode_;
    std::vector<uint32_t> mode_to_source_;
    Tableau inv_;
    PauliString row_scratch_;
    std::vector<bool> is_stabilized_;
    std::vector<PreparedOp> sigma_;
    std::vector<PreparedOp> alpha_;
    std::vector<PreparedOp> beta_;
    std::vector<PauliString> logical_x_;
    std::vector<PauliString> logical_z_;
    std::vector<PreparedOp> fx_;
    std::vector<PreparedOp> fz_;
    SparseWords scratch_;
};

void schedule_reconstruction(GlobalReconstruction& result, const ReconstructionLimits& limits) {
    const uint32_t n = result.num_qubits;

    // Separate trivial identity-axis outputs and duplicate-axis commuting
    // operations from the distinct quantum operations that govern active width.
    // Commuting same-axis rotations and duplicate-axis output measurements run
    // immediately after their representative without inflating the scheduler DAG.
    std::vector<ReconstructionOperation> trivial_outputs;
    std::vector<ReconstructionOperation> quantum_ops;
    std::vector<std::vector<ReconstructionOperation>> duplicate_followers;
    using AxisKey = std::pair<Bits, Bits>;
    std::map<AxisKey, size_t> rotation_representatives;
    std::map<AxisKey, size_t> measurement_representatives;

    for (auto& op : result.operations) {
        if (op.half_turns != 0) {
            AxisKey key{canonical(copy(op.axis.x())), canonical(copy(op.axis.z()))};
            if (const auto it = rotation_representatives.find(key);
                it != rotation_representatives.end()) {
                const size_t rep = it->second;
                bool commutes_between = true;
                for (size_t k = rep + 1; k < quantum_ops.size(); ++k) {
                    if (!op.axis.view().commutes(quantum_ops[k].axis.view())) {
                        commutes_between = false;
                        break;
                    }
                }
                if (commutes_between) {
                    duplicate_followers[rep].push_back(std::move(op));
                    continue;
                }
            }
            rotation_representatives[std::move(key)] = quantum_ops.size();
            quantum_ops.push_back(std::move(op));
            duplicate_followers.emplace_back();
            continue;
        }
        if (op.axis.x().is_zero() && op.axis.z().is_zero()) {
            if (op.output.has_value()) {
                trivial_outputs.push_back(std::move(op));
            }
            continue;
        }
        AxisKey key{canonical(copy(op.axis.x())), canonical(copy(op.axis.z()))};
        const auto [it, inserted] =
            measurement_representatives.try_emplace(std::move(key), quantum_ops.size());
        if (inserted) {
            quantum_ops.push_back(std::move(op));
            duplicate_followers.emplace_back();
        } else if (op.output.has_value()) {
            duplicate_followers[it->second].push_back(std::move(op));
        }
    }

    require_supported(quantum_ops.size() + n <= limits.max_schedule_operations,
                      "schedule operation limit exceeded");

    // 1. Synthesize commuting hidden measurements from the nullspace of
    // operation commutation columns to shrink the initial active subspace.
    // Insert Z columns first and rotation-touched X columns before
    // measurement-only X columns so single-X dressings stay minimal and
    // rotation-stabilizing candidates outrank measurement-only coordinates.
    std::vector<bool> rot_touches_x(n, false);
    for (const auto& op : quantum_ops) {
        if (op.half_turns != 0) {
            each_bit(op.axis.x().words, [&](uint32_t q) { rot_touches_x[q] = true; });
        }
    }
    std::vector<Bits> column_patterns(2 * n);
    for (uint32_t j = 0; j < quantum_ops.size(); ++j) {
        record_commutation_columns(column_patterns, n, quantum_ops[j].axis.view(), j);
    }
    std::vector<uint32_t> column_order;
    column_order.reserve(2 * n);
    for (uint32_t q = 0; q < n; ++q) {
        column_order.push_back(n + q);
    }
    for (uint32_t q = 0; q < n; ++q) {
        if (rot_touches_x[q]) {
            column_order.push_back(q);
        }
    }
    for (uint32_t q = 0; q < n; ++q) {
        if (!rot_touches_x[q]) {
            column_order.push_back(q);
        }
    }
    Basis columns;
    struct NullspaceCandidate {
        PauliString axis;
        size_t non_rot_x_weight = 0;
        size_t x_weight = 0;
        size_t total_weight = 0;
    };
    std::vector<NullspaceCandidate> candidates;
    for (uint32_t i : column_order) {
        auto dependency = columns.insert(std::move(column_patterns[i]), unit(i));
        if (!dependency) {
            continue;
        }
        PauliString p = pauli_from_symplectic_bits(*dependency, n);
        if (p.x().is_zero()) {
            continue;
        }
        size_t non_rot_x = 0;
        size_t x_w = 0;
        each_bit(p.x().words, [&](uint32_t q) {
            ++x_w;
            if (!rot_touches_x[q]) {
                ++non_rot_x;
            }
        });
        size_t z_w = 0;
        for (uint64_t w : p.z().words) {
            z_w += static_cast<size_t>(std::popcount(w));
        }
        candidates.push_back({std::move(p), non_rot_x, x_w, x_w + z_w});
    }
    std::ranges::stable_sort(candidates,
                             [](const NullspaceCandidate& a, const NullspaceCandidate& b) {
                                 return std::tie(a.non_rot_x_weight, a.x_weight, a.total_weight) <
                                        std::tie(b.non_rot_x_weight, b.x_weight, b.total_weight);
                             });
    Basis xparts;
    std::vector<ReconstructionOperation> hidden;
    for (auto& cand : candidates) {
        if (xparts.solve(copy(cand.axis.x())))
            continue;
        if (std::ranges::any_of(
                hidden, [&](const auto& h) { return !cand.axis.view().commutes(h.axis.view()); }))
            continue;
        xparts.insert(copy(cand.axis.x()), {});
        hidden.push_back({std::move(cand.axis), {}, 0, std::nullopt});
    }
    const size_t num_hidden = hidden.size();
    quantum_ops.insert(quantum_ops.begin(), std::make_move_iterator(hidden.begin()),
                       std::make_move_iterator(hidden.end()));

    // 2. Build the DAG of non-commuting predecessors and transitive ancestors.
    const uint32_t count = static_cast<uint32_t>(quantum_ops.size());
    std::vector<Bits> predecessors(count), ancestors(count);
    size_t steps = 0;
    auto budget = [&] {
        require_supported(steps++ < limits.max_planning_steps, "schedule work limit exceeded");
    };
    for (uint32_t j = 0; j < count; ++j) {
        for (uint32_t i = 0; i < j; ++i) {
            budget();
            if (!quantum_ops[i].axis.view().commutes(quantum_ops[j].axis.view()))
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
    order.reserve(count);
    std::vector<uint8_t> stabilized_rotation(count, 0);
    auto apply = [&](DormantSubspace& state, uint32_t j) {
        budget();
        if (quantum_ops[j].half_turns != 0)
            state.apply_rotation(quantum_ops[j].axis);
        else
            state.apply_measurement(quantum_ops[j].axis);
    };
    auto run = [&](uint32_t j) {
        if (quantum_ops[j].half_turns != 0 && group.contains(quantum_ops[j].axis)) {
            stabilized_rotation[j] = 1;
        }
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
                    (quantum_ops[j].half_turns == 0 ||
                     group.commutes_with_all(quantum_ops[j].axis))) {
                    run(j);
                    progress = true;
                }
            });
        } while (progress);
        std::optional<uint32_t> best;
        uint32_t lowest = n + 1;
        std::vector<uint32_t> best_chain;
        each_bit(pending, [&](uint32_t j) {
            if (quantum_ops[j].half_turns != 0)
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

    size_t total_scheduled = count + trivial_outputs.size();
    for (const auto& followers : duplicate_followers) {
        total_scheduled += followers.size();
    }
    std::vector<ReconstructionOperation> scheduled;
    scheduled.reserve(total_scheduled);
    for (uint32_t j : order) {
        if (stabilized_rotation[j]) {
            continue;
        }
        scheduled.push_back(std::move(quantum_ops[j]));
        if (j >= num_hidden) {
            auto& followers = duplicate_followers[j - num_hidden];
            for (auto& dup : followers) {
                scheduled.push_back(std::move(dup));
            }
        }
    }
    for (auto& op : trivial_outputs) {
        scheduled.push_back(std::move(op));
    }
    result.operations = std::move(scheduled);
}

struct PreparedDilation {
    ReconstructionLimits limits;
    uint32_t n = 0;
    uint32_t error_bits = 0;
    uint32_t num_records = 0;
    uint32_t num_detectors = 0;
    uint32_t num_observables = 0;
    std::vector<PreparedOp> dressed_rotations;
    std::vector<PreparedOp> mapped_records;
    std::vector<PreparedOp> mapped_checks;
    std::vector<std::vector<ReconstructionNoiseOutcome>> channels;
};

PreparedDilation prepare_dilation(const Circuit& circuit, const ReconstructionLimits& limits) {
    SourceScan scan = scan_source(circuit, limits);
    NoiseChannelBuilder noise(limits, scan);

    Dilation dilation(limits, noise);
    dilation.parse_prefix(circuit, scan.prefix_end);
    const uint32_t n = dilation.num_qubits();

    // 1. Build the inverse Clifford frame on the dilated prefix `[0, n)`.
    Tableau backward(n);
    for (const auto& gate : dilation.body()) {
        backward.prepend_named_gate(Tableau::inverse_named_gate(gate.gate), gate.span());
    }
    dilation.body() = {};

    const auto& rotations = dilation.rotations();
    std::vector<PauliString> constraints;
    constraints.reserve(rotations.size());
    for (const auto& r : rotations) {
        constraints.emplace_back(backward.z_output(r.ancilla));
    }

    // Restrict causal stabilizer dressing and O(k^3) symplectic completion to
    // the subset of prefix wires coupled to non-Clifford gadget constraints.
    std::vector<bool> is_coupled(n, false);
    for (const auto& r : rotations) {
        is_coupled[r.ancilla] = true;
    }
    for (const auto& c : constraints) {
        each_bit(c.x().words, [&](uint32_t i) { is_coupled[i] = true; });
        each_bit(c.z().words, [&](uint32_t i) { is_coupled[i] = true; });
    }
    std::vector<uint32_t> coupled_wires;
    std::vector<uint32_t> wire_to_compact(n, kUnmappedQubit);
    for (uint32_t i = 0; i < n; ++i) {
        if (is_coupled[i]) {
            wire_to_compact[i] = static_cast<uint32_t>(coupled_wires.size());
            coupled_wires.push_back(i);
        }
    }
    const uint32_t k_c = static_cast<uint32_t>(coupled_wires.size());

    std::vector<bool> magic_c(k_c, false);
    for (const auto& r : rotations) {
        magic_c[wire_to_compact[r.ancilla]] = true;
    }
    std::vector<PauliString> constraints_c;
    constraints_c.reserve(constraints.size());
    for (const auto& c : constraints) {
        PauliString cc(k_c);
        for (uint32_t idx = 0; idx < k_c; ++idx) {
            const uint32_t w = coupled_wires[idx];
            cc.set_pauli(idx, c.x().bit_get(w), c.z().bit_get(w));
        }
        cc.set_sign(c.sign());
        constraints_c.push_back(std::move(cc));
    }
    constraints = {};

    std::vector<Bits> patterns(k_c);
    for (uint32_t j = 0; j < constraints_c.size(); ++j) {
        each_bit(constraints_c[j].x().words, [&](uint32_t i) {
            if (!magic_c[i]) {
                toggle(patterns[i], j);
            }
        });
        each_bit(constraints_c[j].z().words, [&](uint32_t i) {
            if (magic_c[i]) {
                toggle(patterns[i], j);
            }
        });
    }
    Basis allowed;
    for (uint32_t i = 0; i < k_c; ++i) {
        if (!magic_c[i])
            allowed.insert(patterns[i], unit(i));
    }

    auto pauli_from_dressing = [&](const Bits& bits) {
        PauliString p(k_c);
        each_bit(bits, [&](uint32_t i) {
            if (magic_c[i]) {
                p.mut_x().bit_set(i, true);
            } else {
                p.mut_z().bit_set(i, true);
            }
        });
        p.set_sign(false);
        return p;
    };

    // 2. Dress each rotation gadget axis with input stabilizers so it commutes
    // with all earlier gadget projections.
    std::vector<PreparedOp> dressed_rotations;
    dressed_rotations.reserve(rotations.size());
    for (auto it = rotations.rbegin(); it != rotations.rend(); ++it) {
        const uint32_t anc_c = wire_to_compact[it->ancilla];
        Bits z_pattern;
        for (uint32_t j = 0; j < constraints_c.size(); ++j) {
            if (constraints_c[j].x().bit_get(anc_c))
                toggle(z_pattern, j);
        }
        const auto dressing = allowed.solve(std::move(z_pattern));
        require_supported(dressing.has_value(), "no causal stabilizer dressing");
        PauliString p_c = pauli_from_dressing(*dressing);
        assert(!p_c.x().bit_get(anc_c) && !p_c.z().bit_get(anc_c));
        p_c.mut_z().bit_set(anc_c, true);
        dressed_rotations.push_back({std::move(p_c), it->sign, it->half_turns});
        allowed.insert(patterns[anc_c], unit(anc_c));
    }
    std::reverse(dressed_rotations.begin(), dressed_rotations.end());

    // 3. Complete the independent gadget projections on the coupled support to a
    // k_c-qubit stabilizer basis and fold them into the input state. Uncoupled
    // wires remain identity (+Z_u) factors.
    Basis independent;
    std::vector<PauliString> generators_c = std::move(constraints_c);
    const size_t num_constraints = generators_c.size();
    generators_c.reserve(k_c);
    for (uint32_t i = 0; i < k_c; ++i) {
        if (auto dependency = independent.insert(std::move(patterns[i]), unit(i))) {
            generators_c.push_back(pauli_from_dressing(*dependency));
        }
    }
    require_supported(independent.size() == num_constraints, "nonuniform gadget projection");
    assert(generators_c.size() == k_c);
    Tableau frame_c = stabilizer_frame(generators_c, k_c);
    generators_c = {};

    auto embed_coupled = [&](const PauliString& mapped_c, PauliStringView base_full,
                             bool extra_sign) {
        PauliString out(n);
        out.mut_x().xor_with(base_full.x());
        out.mut_z().xor_with(base_full.z());
        for (uint32_t idx = 0; idx < k_c; ++idx) {
            out.set_pauli(coupled_wires[idx], mapped_c.x().bit_get(idx), mapped_c.z().bit_get(idx));
        }
        out.set_sign(extra_sign ^ mapped_c.sign());
        return out;
    };

    auto apply_coupled_frame = [&](PauliStringView full) {
        if (k_c == 0) {
            return PauliString(full);
        }
        PauliString sub(k_c);
        for (uint32_t idx = 0; idx < k_c; ++idx) {
            const uint32_t w = coupled_wires[idx];
            sub.set_pauli(idx, full.x().bit_get(w), full.z().bit_get(w));
        }
        sub.set_sign(false);
        const PauliString mapped_c = frame_c.apply(sub.view());
        return embed_coupled(mapped_c, full, full.sign());
    };

    const PauliString empty_full(n);
    for (auto& op : dressed_rotations) {
        op.axis = embed_coupled(frame_c.apply(op.axis.view()), empty_full.view(), false);
    }

    auto map_prefix_ancillas = [&](const Bits& z_bits, ReconstructionSign sign) -> PreparedOp {
        PauliString z_axis(n);
        each_bit(z_bits, [&](uint32_t i) { z_axis.mut_z().bit_set(i, true); });
        PauliString folded = apply_coupled_frame(backward.apply(z_axis.view()).view());
        sign.constant ^= folded.sign();
        folded.set_sign(false);
        return {std::move(folded), std::move(sign), 0};
    };

    std::vector<PreparedOp> records;
    records.reserve(circuit.num_measurements);
    for (auto& r : dilation.records()) {
        if (r.ancilla.has_value()) {
            PauliString folded = apply_coupled_frame(backward.z_output(*r.ancilla));
            r.sign.constant ^= folded.sign();
            folded.set_sign(false);
            records.push_back({std::move(folded), std::move(r.sign), 0});
        } else {
            records.push_back({PauliString(n), std::move(r.sign), 0});
        }
    }
    dilation.records() = {};

    std::vector<PreparedOp> detectors;
    detectors.reserve(circuit.num_detectors);
    for (auto& d : dilation.detectors()) {
        detectors.push_back(map_prefix_ancillas(d.first, std::move(d.second)));
    }
    dilation.detectors() = {};

    std::vector<PreparedOp> observables;
    observables.reserve(circuit.num_observables);
    for (auto& o : dilation.observables()) {
        observables.push_back(map_prefix_ancillas(o.first, std::move(o.second)));
    }
    dilation.observables() = {};

    // 4. Evolve the trailing pure-Clifford suffix (if any) without dilating its
    // syndrome measurements or resets, then release prefix tableaux immediately.
    if (scan.prefix_end < circuit.nodes.size()) {
        SuffixTracker suffix(n, limits, noise, records, detectors, observables);
        suffix.initialize_qubits(circuit, scan.prefix_end, dilation.aliases(),
                                 [&](uint32_t wire, PauliString& lx, PauliString& lz,
                                     ReconstructionSign& fx, ReconstructionSign& fz) {
                                     lx = apply_coupled_frame(backward.x_output(wire));
                                     lz = apply_coupled_frame(backward.z_output(wire));
                                     fx = std::move(dilation.fx(wire));
                                     fz = std::move(dilation.fz(wire));
                                 });
        backward = Tableau();
        frame_c = Tableau();
        suffix.parse_suffix(circuit, scan.prefix_end);
    } else {
        backward = Tableau();
        frame_c = Tableau();
    }
    assert(scan.noise_target_count == 0);

    std::vector<PreparedOp> mapped_checks;
    mapped_checks.reserve(detectors.size() + observables.size());
    mapped_checks.insert(mapped_checks.end(), std::make_move_iterator(detectors.begin()),
                         std::make_move_iterator(detectors.end()));
    mapped_checks.insert(mapped_checks.end(), std::make_move_iterator(observables.begin()),
                         std::make_move_iterator(observables.end()));

    PreparedDilation prepared;
    prepared.limits = limits;
    prepared.n = n;
    prepared.error_bits = noise.num_noise_bits();
    prepared.num_records = static_cast<uint32_t>(records.size());
    prepared.num_detectors = static_cast<uint32_t>(detectors.size());
    prepared.num_observables = static_cast<uint32_t>(observables.size());
    prepared.dressed_rotations = std::move(dressed_rotations);
    prepared.mapped_records = std::move(records);
    prepared.mapped_checks = std::move(mapped_checks);
    prepared.channels = std::move(noise.channels());
    return prepared;
}

// Fuse commuting same-axis rotations that share a noise-sign parity and fold
// any resulting Clifford rotations (I, Z, S, S_dag) into downstream operations.
// Lowered HIR wraps each rotation in conditional Pauli sign brackets that hide
// these cancellations from PeepholeFusionPass.
void fuse_reconstruction_rotations(std::vector<ReconstructionOperation>& operations) {
    const auto same_unsigned_axis = [](const PauliString& a, const PauliString& b) {
        return a.x() == b.x() && a.z() == b.z();
    };
    const auto signed_half_turns = [](const ReconstructionOperation& op) {
        return (op.sign.constant ^ op.axis.sign()) ? -op.half_turns : op.half_turns;
    };
    const auto normalize_mod2 = [](double alpha) { return alpha - 2.0 * std::floor(alpha / 2.0); };

    SparseWords scratch;
    bool changed = true;
    while (changed) {
        changed = false;
        const size_t n = operations.size();
        std::vector<uint8_t> deleted(n, 0);

        // 1. Merge commuting rotations with identical unsigned axis and sign mask.
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i] || operations[i].half_turns == 0) {
                continue;
            }
            for (size_t j = i + 1; j < n; ++j) {
                if (deleted[j]) {
                    continue;
                }
                if (operations[j].half_turns == 0) {
                    break;
                }
                if (same_unsigned_axis(operations[i].axis, operations[j].axis) &&
                    operations[i].sign.words == operations[j].sign.words) {
                    const double a_mod2 = normalize_mod2(signed_half_turns(operations[i]) +
                                                         signed_half_turns(operations[j]));
                    deleted[j] = 1;
                    changed = true;
                    if (classify_clifford_rotation(a_mod2) == CliffordRotation::IDENTITY) {
                        deleted[i] = 1;
                        break;
                    }
                    operations[i].half_turns = a_mod2 > 1.0 ? a_mod2 - 2.0 : a_mod2;
                    operations[i].sign.constant = false;
                    operations[i].axis.set_sign(false);
                    continue;
                }
                if (!operations[i].axis.view().commutes(operations[j].axis.view())) {
                    break;
                }
            }
        }

        // 2. Absorb Clifford rotations into downstream operations.
        for (size_t i = 0; i < n; ++i) {
            if (deleted[i] || operations[i].half_turns == 0) {
                continue;
            }
            if (operations[i].axis.x().is_zero() && operations[i].axis.z().is_zero()) {
                deleted[i] = 1;
                changed = true;
                continue;
            }
            const double a_mod2 = normalize_mod2(signed_half_turns(operations[i]));
            const auto clifford = classify_clifford_rotation(a_mod2);
            if (!clifford) {
                continue;
            }
            deleted[i] = 1;
            changed = true;
            const PauliString& p = operations[i].axis;
            switch (*clifford) {
                case CliffordRotation::IDENTITY:
                    break;
                case CliffordRotation::PAULI:
                    for (size_t k = i + 1; k < n; ++k) {
                        if (!deleted[k] && !p.view().commutes(operations[k].axis.view())) {
                            operations[k].sign.constant ^= true;
                        }
                    }
                    break;
                case CliffordRotation::SQRT:
                case CliffordRotation::SQRT_DAG: {
                    const uint8_t phase_step = (*clifford == CliffordRotation::SQRT_DAG) ? 3U : 1U;
                    for (size_t k = i + 1; k < n; ++k) {
                        if (deleted[k] || p.view().commutes(operations[k].axis.view())) {
                            continue;
                        }
                        PauliString prod(p.view());
                        prod.right_multiply(operations[k].axis.view());
                        prod.add_phase(phase_step);
                        assert(prod.is_hermitian());
                        operations[k].sign.constant ^= prod.sign();
                        prod.set_sign(false);
                        operations[k].axis = std::move(prod);
                        xor_sparse_words(operations[k].sign.words, operations[i].sign.words,
                                         scratch);
                    }
                    break;
                }
            }
        }

        if (changed) {
            size_t out = 0;
            for (size_t i = 0; i < n; ++i) {
                if (!deleted[i]) {
                    if (out != i) {
                        operations[out] = std::move(operations[i]);
                    }
                    ++out;
                }
            }
            operations.resize(out);
        }
    }
}

GlobalReconstruction reconstruct_variant(const PreparedDilation& prepared,
                                         ReconstructionOutputs outputs) {
    // Reduce all requested operations to a basis of their X supports,
    // eliminating fixed-|0> coordinates.
    const bool keep_records = outputs == ReconstructionOutputs::Records;
    const size_t record_count = keep_records ? prepared.mapped_records.size() : 0;
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
        for (const auto& op : prepared.mapped_records)
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
        for (const auto& op : prepared.mapped_records)
            append_reduced(op, next_output++);
    }
    for (const auto& op : prepared.mapped_checks)
        append_reduced(op, next_output++);

    result.noise_channels = prepared.channels;
    fuse_reconstruction_rotations(result.operations);
    schedule_reconstruction(result, prepared.limits);
    return result;
}

ReconstructionNoise compress_noise(const GlobalReconstruction& r,
                                   const ReconstructionLimits& limits) {
    // 1. Build a sparse GF(2) basis for the affine noise-bit masks referenced
    // by operations. Prioritize quantum rotations (2 * terms active-region
    // feedback ops), then visible records (2 * terms scratch-wire ops), then
    // detector/observable parities, and finally hidden measurements.
    auto sign_priority = [&](const ReconstructionOperation& op) -> int {
        if (op.half_turns != 0) {
            return 0;
        }
        if (op.output.has_value()) {
            return *op.output < r.outputs.num_records ? 1 : 2;
        }
        return 3;
    };
    SparseBasis basis(r.num_noise_bits);
    std::vector<SparseWords> rows;
    std::vector<size_t> indices(r.operations.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return sign_priority(r.operations[a]) < sign_priority(r.operations[b]);
    });
    std::vector<ReconstructionSign> mapped_signs(r.operations.size());
    for (size_t i : indices) {
        const SparseWords& terms = r.operations[i].sign.words;
        auto [coords, inserted] = basis.solve_or_insert(terms, static_cast<uint32_t>(rows.size()));
        if (inserted) {
            rows.push_back(terms);
        }
        mapped_signs[i] = {std::move(coords), r.operations[i].sign.constant};
    }

    // Reverse index from raw fault bit -> compressed coordinate mask. Append in
    // ascending coordinate order so each column builds canonically without binary search.
    std::vector<SparseWords> columns(r.num_noise_bits);
    for (uint32_t j = 0; j < rows.size(); ++j) {
        const uint32_t word_idx = j / 64;
        const uint64_t bit_mask = uint64_t{1} << (j % 64);
        each_sparse_bit(rows[j], [&](uint32_t b) {
            SparseWords& col = columns.at(b);
            if (!col.empty() && col.back().index == word_idx) {
                col.back().mask |= bit_mask;
            } else {
                col.push_back({word_idx, bit_mask});
            }
        });
    }
    const auto num_bits = static_cast<uint32_t>(rows.size());
    rows = {};

    // 2. Project each categorical channel onto the basis coordinates and
    // convolve channels that share the same image support.
    using OutcomeEntry = std::pair<SparseWords, double>;
    using Distribution = std::vector<OutcomeEntry>;

    std::map<std::vector<SparseWords>, Distribution> grouped;
    Distribution image;
    Distribution convolved;
    SparseWords scratch;
    size_t products = 0;

    auto sort_and_merge = [&](Distribution& dist) {
        std::ranges::sort(dist, {}, &OutcomeEntry::first);
        size_t out = 0;
        for (size_t i = 0; i < dist.size(); ++i) {
            if (out > 0 && dist[out - 1].first == dist[i].first) {
                dist[out - 1].second += dist[i].second;
            } else {
                if (out != i) {
                    dist[out] = std::move(dist[i]);
                }
                ++out;
                require_supported(out <= limits.max_channel_support,
                                  "channel support limit exceeded");
            }
        }
        dist.resize(out);
    };

    for (const auto& channel : r.noise_channels) {
        require_supported(channel.size() <= saturating_multiply_u64(limits.max_channel_support, 64),
                          "channel support limit exceeded");
        image.clear();
        for (const auto& outcome : channel) {
            if (outcome.probability == 0)
                continue;
            SparseWords value;
            for (uint32_t b : outcome.bits) {
                xor_sparse_words(value, columns.at(b), scratch);
            }
            image.emplace_back(std::move(value), outcome.probability);
        }
        sort_and_merge(image);
        if (image.empty() || (image.size() == 1 && image[0].first.empty()))
            continue;

        std::vector<SparseWords> key;
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

            // Check support growth incrementally when the Cartesian product is large.
            convolved.clear();
            if (existing_size * image_size <= limits.max_channel_support) {
                convolved.reserve(existing_size * image_size);
                for (const auto& [a, p] : existing->second) {
                    for (const auto& [b, q] : image) {
                        SparseWords sum = a;
                        xor_sparse_words(sum, b, scratch);
                        convolved.emplace_back(std::move(sum), p * q);
                    }
                }
                sort_and_merge(convolved);
            } else {
                std::map<SparseWords, double> merged;
                for (const auto& [a, p] : existing->second) {
                    for (const auto& [b, q] : image) {
                        SparseWords sum = a;
                        xor_sparse_words(sum, b, scratch);
                        auto [it, inserted] = merged.try_emplace(std::move(sum), p * q);
                        if (!inserted) {
                            it->second += p * q;
                        } else {
                            require_supported(merged.size() <= limits.max_channel_support,
                                              "channel support limit exceeded");
                        }
                    }
                }
                convolved.reserve(merged.size());
                for (auto& [val, prob] : merged) {
                    convolved.emplace_back(val, prob);
                }
            }
            existing->second.swap(convolved);
        } else {
            grouped.emplace(std::move(key), std::move(image));
        }
    }
    columns = {};

    ReconstructionNoise result;
    result.num_bits = num_bits;
    result.signs = std::move(mapped_signs);
    result.channels.reserve(grouped.size());
    for (auto& [key, distribution] : grouped) {
        auto& channel = result.channels.emplace_back();
        channel.reserve(distribution.size());
        for (const auto& [value, probability] : distribution) {
            ReconstructionNoiseOutcome outcome{probability, {}};
            each_sparse_bit(value, [&](uint32_t i) { outcome.bits.push_back(i); });
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

    // Bound both the sparse Pauli-mask arena and downstream HIR/planner frame
    // storage before allocating the lowered module.
    const auto axis_words = [](const PauliString& axis) {
        uint64_t words = 0;
        for (uint32_t w = 0; w < axis.x().num_words(); ++w) {
            words += (axis.x().words[w] | axis.z().words[w]) != 0;
        }
        return words;
    };
    const auto same_unsigned_axis = [](const PauliString& a, const PauliString& b) {
        return a.num_qubits() == b.num_qubits() && a.x() == b.x() && a.z() == b.z();
    };

    bool need_one = false;
    uint64_t masks = noise.num_bits;
    uint64_t mask_words = noise.num_bits;
    uint64_t hir_ops = noise.channels.size() + noise.num_bits + r.outputs.num_detectors +
                       r.outputs.num_observables;
    std::optional<PauliString> prev_meas_axis;

    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
        const uint64_t terms = sparse_popcount(noise.signs[j].words);
        if (op.half_turns != 0) {
            prev_meas_axis.reset();
            const uint32_t pivot = std::min(op.axis.x().lowest_bit(), op.axis.z().lowest_bit());
            if (pivot < r.num_qubits) {
                masks += 2 * terms + 1;
                mask_words += 2 * terms + axis_words(op.axis);
                hir_ops += 2 * terms + 1;
            }
        } else {
            const bool is_identity = op.axis.x().is_zero() && op.axis.z().is_zero();
            const bool constant = noise.signs[j].constant ^ op.axis.sign();
            if (!is_identity) {
                const bool reuse_prev =
                    prev_meas_axis.has_value() && same_unsigned_axis(*prev_meas_axis, op.axis);
                if (!reuse_prev) {
                    ++masks;
                    mask_words += axis_words(op.axis);
                    ++hir_ops;
                    prev_meas_axis = op.axis;
                    prev_meas_axis->set_sign(false);
                }
            }
            if (!op.output)
                continue;
            if (constant)
                need_one = true;
            if (*op.output < visible) {
                const uint64_t parity_len = terms + (!is_identity ? 1 : 0) + (constant ? 1 : 0);
                masks += 2 * parity_len + 1;
                mask_words += 2 * parity_len + 1;
                hir_ops += 2 * parity_len + 1;
            }
        }
    }
    if (need_one) {
        ++masks;
        ++hir_ops;
    }

    for (const auto& channel : noise.channels) {
        for (const auto& outcome : channel) {
            if (!outcome.bits.empty() && outcome.probability != 0) {
                ++masks;
                mask_words += outcome.bits.size();
            }
        }
    }
    const uint64_t arena_bytes = PauliMaskArena::storage_bytes(masks, mask_words);
    // Account for lowered HIR operations plus the dense width x width bit
    // matrices allocated during active-width analysis (`DormantSubspace`: 4)
    // and sampling-plan construction (`CoordinateFrame`: 8).
    constexpr uint64_t kPlannerBitMatrices = 8 + 4;
    const uint64_t width_words = (uint64_t{width} + 63) / 64;
    const uint64_t dense_frame_bytes =
        saturating_multiply_u64(kPlannerBitMatrices * width, width_words * sizeof(uint64_t));
    const uint64_t op_bytes = saturating_multiply_u64(hir_ops, sizeof(HeisenbergOp));
    const uint64_t total_bytes =
        saturating_add_u64(arena_bytes, saturating_add_u64(dense_frame_bytes, op_bytes));
    require_supported(total_bytes <= limits.max_lowering_bytes, "lowering memory limit exceeded");

    HirModule hir(width);
    hir.ops.reserve(static_cast<size_t>(hir_ops));
    hir.num_measurements = visible;
    hir.num_detectors = r.outputs.num_detectors;
    hir.num_observables = r.outputs.num_observables;
    hir.postselection.mask = postselection.mask;
    hir.postselection.expected_detectors = postselection.expected_detectors;
    hir.postselection.expected_observables = postselection.expected_observables;

    PauliMask mask;
    auto single = [&](uint32_t qubit, bool x) -> PauliMaskView {
        mask.clear();
        mask.toggle(qubit, x, !x);
        return mask;
    };

    // 2. Partition compressed categorical noise channels into early channels
    // touching fault bits read by quantum rotations or postselected detectors,
    // and late channels touching only unpostselected output parities. Keeping
    // postselected fault records ahead of the quantum section lets
    // `EarlyPostselection::schedule` reject shots before quantum execution,
    // while deferring unpostselected output-only channels avoids analyzing
    // thousands of memory-tail syndrome channels during HIR rotation passes.
    std::vector<uint8_t> early_fault(noise.num_bits, 0);
    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
        bool need_early = op.half_turns != 0;
        if (!need_early && op.output.has_value() && *op.output >= visible &&
            *op.output < visible + r.outputs.num_detectors) {
            const uint32_t det = *op.output - visible;
            need_early = det < postselection.mask.size() && postselection.mask[det] != 0;
        }
        if (need_early) {
            each_sparse_bit(noise.signs[j].words, [&](uint32_t i) { early_fault[i] = 1; });
        }
    }
    auto is_early_channel = [&](const std::vector<ReconstructionNoiseOutcome>& ch) {
        for (const auto& outcome : ch) {
            if (outcome.probability == 0) {
                continue;
            }
            for (uint32_t b : outcome.bits) {
                if (early_fault[b] != 0) {
                    return true;
                }
            }
        }
        return false;
    };
    auto emit_noise_channel = [&](const std::vector<ReconstructionNoiseOutcome>& channel) {
        NoiseSite site;
        for (const auto& outcome : channel) {
            if (outcome.bits.empty() || outcome.probability == 0)
                continue;
            mask.clear();
            for (uint32_t i : outcome.bits)
                mask.toggle(r.num_qubits + i, true, false);
            site.channels.push_back({hir.noise_channel_masks.append(mask), outcome.probability});
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
    };
    auto hidden_measure = [&](PauliMaskView body) {
        const auto record = visible + hir.num_hidden_measurements++;
        hir.append_measure(MeasRecordIdx{record}, body).set_hidden(true);
        return record;
    };

    for (const auto& channel : noise.channels) {
        if (is_early_channel(channel)) {
            emit_noise_channel(channel);
        }
    }
    std::vector<uint32_t> fault_records(noise.num_bits, kUnmappedQubit);
    for (uint32_t i = 0; i < noise.num_bits; ++i) {
        if (early_fault[i] != 0) {
            fault_records[i] = hidden_measure(single(r.num_qubits + i, false));
        }
    }
    const uint32_t one = need_one ? hidden_measure(PauliMaskView{{}, true}) : 0;

    // 3. Emit scheduled quantum rotations and hidden quantum measurements.
    struct PendingOutputParity {
        size_t op_index = 0;
        std::optional<uint32_t> quantum_record;
        bool constant = false;
    };
    std::vector<PendingOutputParity> pending_outputs;
    pending_outputs.reserve(r.outputs.num_outputs());
    std::optional<std::pair<PauliString, uint32_t>> cached_meas;

    for (size_t j = 0; j < r.operations.size(); ++j) {
        const auto& op = r.operations[j];
        auto axis_body = [&]() -> PauliMaskView {
            mask.assign(op.axis.x(), op.axis.z(), false);
            return mask;
        };
        if (op.half_turns != 0) {
            cached_meas.reset();
            const uint32_t pivot = std::min(op.axis.x().lowest_bit(), op.axis.z().lowest_bit());
            if (pivot >= r.num_qubits)
                continue;
            std::vector<uint32_t> parity;
            each_sparse_bit(noise.signs[j].words, [&](uint32_t i) {
                assert(early_fault[i] != 0);
                parity.push_back(fault_records[i]);
            });
            auto feedback = [&] {
                for (auto record : parity)
                    hir.append_conditional(ControllingMeasIdx{record},
                                           single(pivot, !op.axis.x().bit_get(pivot)));
            };
            feedback();
            mask.assign(op.axis.x(), op.axis.z(), noise.signs[j].constant ^ op.axis.sign());
            hir.append_phase_rotation(op.half_turns, mask);
            feedback();
        } else {
            const bool is_identity = op.axis.x().is_zero() && op.axis.z().is_zero();
            const bool constant = noise.signs[j].constant ^ op.axis.sign();
            std::optional<uint32_t> quantum_record;
            if (!is_identity) {
                if (cached_meas.has_value() && same_unsigned_axis(cached_meas->first, op.axis)) {
                    quantum_record = cached_meas->second;
                } else {
                    quantum_record = hidden_measure(axis_body());
                    PauliString key = op.axis;
                    key.set_sign(false);
                    cached_meas.emplace(std::move(key), *quantum_record);
                }
            }
            if (!op.output) {
                continue;
            }
            pending_outputs.push_back({j, quantum_record, constant});
        }
    }

    // 4. Emit remaining output-only noise channels and fault-wire measurements,
    // then materialize visible records on the reusable zero scratch wire.
    for (const auto& channel : noise.channels) {
        if (!is_early_channel(channel)) {
            emit_noise_channel(channel);
        }
    }
    for (uint32_t i = 0; i < noise.num_bits; ++i) {
        if (early_fault[i] == 0) {
            fault_records[i] = hidden_measure(single(r.num_qubits + i, false));
        }
    }

    std::vector<std::vector<uint32_t>> output_parities(r.outputs.num_outputs());
    for (const PendingOutputParity& item : pending_outputs) {
        const auto& op = r.operations[item.op_index];
        std::vector<uint32_t> parity;
        each_sparse_bit(noise.signs[item.op_index].words,
                        [&](uint32_t i) { parity.push_back(fault_records[i]); });
        if (item.quantum_record.has_value()) {
            parity.push_back(*item.quantum_record);
        }
        if (item.constant) {
            parity.push_back(one);
        }
        output_parities.at(*op.output) = std::move(parity);
    }
    for (uint32_t out_idx = 0; out_idx < visible; ++out_idx) {
        const auto& parity = output_parities[out_idx];
        auto copy_parity = [&] {
            for (uint32_t record : parity)
                hir.append_conditional(ControllingMeasIdx{record}, single(carrier, true));
        };
        copy_parity();
        hir.append_measure(MeasRecordIdx{out_idx}, single(carrier, false));
        copy_parity();
    }

    // 5. Emit detectors and observables in source order.
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
