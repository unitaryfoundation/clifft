#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>

namespace clifft {

struct PhasePolynomialOptions {
    // Bounds the cubic phase algebra. Collection starts at min 32 and this cap,
    // trying a complete region up to 64 before an optional multiword attempt.
    // Zero disables the pass. Synthesis retains a 64-variable core limit.
    uint32_t max_variables = 64;
    // Shared by all optional attempts in one pass, including both bases.
    size_t max_wide_work = 50000000;
    size_t max_wide_terms = 200000;
};

// Reduces commuting T rotations using Pauli constraints proved from the complete
// circuit's |0> input and recovered through measurement and feedback.
// Separates a Clifford correction and synthesizes the
// non-Clifford core with TOHPE.
// Run before squeezing and scheduling to expose simpler rotation structure.
// Pauli measurements, expectation probes and classically controlled Pauli gates
// can cross a phase prefix only if their conjugates remain Paulis and preserve
// each used constraint.
// No relation is inferred from postselection.
class PhasePolynomialPass : public HirPass {
  public:
    explicit PhasePolynomialPass(PhasePolynomialOptions options = {});
    void run(HirModule& hir) override;

    size_t blocks_reduced() const { return blocks_reduced_; }
    // Search effort includes rejected candidates and any narrow fallback rerun.
    size_t blocks_examined() const { return blocks_examined_; }
    size_t blocks_capped() const { return blocks_capped_; }
    size_t expansion_attempts() const { return expansion_attempts_; }
    size_t blocks_expanded() const { return blocks_expanded_; }
    size_t pauli_pullbacks() const { return pauli_pullbacks_; }
    size_t input_t_count() const { return input_t_count_; }
    size_t output_t_count() const { return output_t_count_; }
    size_t wide_attempts() const { return wide_attempts_; }
    size_t wide_rewrites() const { return wide_rewrites_; }
    size_t wide_work() const { return wide_work_; }
    bool wide_budget_exhausted() const { return wide_budget_exhausted_; }
    bool applied() const { return blocks_reduced_ != 0; }

  private:
    PhasePolynomialOptions options_;
    size_t wide_attempts_ = 0;
    size_t wide_rewrites_ = 0;
    size_t wide_work_ = 0;
    bool wide_budget_exhausted_ = false;
    size_t blocks_reduced_ = 0;
    size_t blocks_examined_ = 0;
    size_t blocks_capped_ = 0;
    size_t expansion_attempts_ = 0;
    size_t blocks_expanded_ = 0;
    size_t pauli_pullbacks_ = 0;
    size_t input_t_count_ = 0;
    size_t output_t_count_ = 0;
};

}  // namespace clifft
