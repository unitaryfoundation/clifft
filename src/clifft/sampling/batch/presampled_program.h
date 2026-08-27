#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace clifft::sampling {

class BatchExecutor;
class ExecutablePlan;
class ExecutablePlanBuilder;
class SamplingPlan;

// Immutable packed-only program for evaluating presampled affine inputs.
// Categorical outcomes set XOR combinations of compact carrier columns based
// on their downstream effects. For effects A={r0,r1} and B={r0,r2}, A can set
// c0 while B sets c0 and c1, where c1={r1,r2}. The expression tape also reuses
// shared prefixes: r1=a^b^d can copy r0=a^b and then XOR d. Absence from an
// ExecutablePlan selects ordinary per-symbol propagation instead; rejected
// candidates are destroyed before the plan retains any of their storage.
class BatchPresampledProgram {
  public:
    [[nodiscard]] uint32_t num_carriers() const noexcept { return num_carriers_; }

  private:
    friend class BatchExecutor;
    friend class ExecutablePlan;
    friend class ExecutablePlanBuilder;

    struct OutcomeAssignments {
        uint32_t begin = 0;
        uint32_t count = 0;
    };

    struct InitializeExpression {
        uint32_t destination = 0;
        uint32_t parent = 0;
        bool invert_parent = false;
    };

    struct XorCarrierIntoExpression {
        uint32_t carrier = 0;
        uint32_t destination = 0;
    };

    struct CopyExpression {
        uint32_t source = 0;
        uint32_t destination = 0;
    };

    [[nodiscard]] static std::optional<BatchPresampledProgram> build(
        const ExecutablePlan& executable, const SamplingPlan& source,
        std::span<const uint32_t> expression_terms,
        std::span<const uint32_t> expression_term_begins,
        std::span<const uint8_t> bound_presampled_symbols);
    void validate(size_t num_noise_outcomes, size_t num_expression_registers) const noexcept;

    uint32_t num_carriers_ = 0;
    std::vector<OutcomeAssignments> outcome_assignments_;
    std::vector<uint32_t> assigned_carriers_;
    std::vector<uint32_t> initialization_level_offsets_;
    std::vector<InitializeExpression> initializations_;
    std::vector<uint32_t> carrier_xor_level_offsets_;
    std::vector<XorCarrierIntoExpression> carrier_xors_;
    std::vector<CopyExpression> copies_;
};

}  // namespace clifft::sampling
