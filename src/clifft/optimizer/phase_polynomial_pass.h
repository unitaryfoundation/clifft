#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>

namespace clifft {

struct PhasePolynomialOptions {
    // Bounds the cubic phase algebra. Zero disables the pass; at most 64
    // independent commuting axes can be represented in one region.
    uint32_t max_variables = 32;
};

// Opt-in, state-independent reduction of commuting T rotations modulo Clifford
// factors. Measurements and feedback cross a phase prefix only when its exact
// Pauli conjugate exists. No entry-state or postselection relations are assumed.
class PhasePolynomialPass : public HirPass {
  public:
    explicit PhasePolynomialPass(PhasePolynomialOptions options = {});
    void run(HirModule& hir) override;

    size_t blocks_reduced() const { return blocks_reduced_; }
    size_t pauli_pullbacks() const { return pauli_pullbacks_; }
    size_t input_t_count() const { return input_t_count_; }
    size_t output_t_count() const { return output_t_count_; }
    bool applied() const { return blocks_reduced_ != 0; }

  private:
    PhasePolynomialOptions options_;
    size_t blocks_reduced_ = 0;
    size_t pauli_pullbacks_ = 0;
    size_t input_t_count_ = 0;
    size_t output_t_count_ = 0;
};

}  // namespace clifft
