#pragma once

#include "clifft/optimizer/hir_pass.h"

#include <cstddef>
#include <cstdint>

namespace clifft {

struct PhasePolynomialOptions {
    // The representation is cubic in the independent commuting eigenbits.
    uint32_t max_variables = 32;
};

/// Reduce commuting T rotations modulo proven entry stabilizers and a fixed
/// Clifford. Pauli noise may be crossed only when every channel commutes with
/// the whole block. Measurements and feedback can be pulled before the block
/// when conjugation through its phase prefix is a Pauli on the entry code.
/// Record order and noise distributions are preserved.
class PhasePolynomialPass : public HirPass {
  public:
    explicit PhasePolynomialPass(PhasePolynomialOptions options = {});
    void run(HirModule& hir) override;

    size_t blocks_examined() const { return blocks_examined_; }
    size_t blocks_reduced() const { return blocks_reduced_; }
    size_t pauli_pullbacks() const { return pauli_pullbacks_; }
    size_t oversized_blocks() const { return oversized_blocks_; }
    size_t expansion_rejections() const { return expansion_rejections_; }
    size_t input_t_count() const { return input_t_count_; }
    size_t output_t_count() const { return output_t_count_; }
    uint32_t incumbent_peak() const { return incumbent_peak_; }
    uint32_t result_peak() const { return result_peak_; }
    bool applied() const { return applied_; }

  private:
    PhasePolynomialOptions options_;
    size_t blocks_examined_ = 0;
    size_t blocks_reduced_ = 0;
    size_t pauli_pullbacks_ = 0;
    size_t oversized_blocks_ = 0;
    size_t expansion_rejections_ = 0;
    size_t input_t_count_ = 0;
    size_t output_t_count_ = 0;
    uint32_t incumbent_peak_ = 0;
    uint32_t result_peak_ = 0;
    bool applied_ = false;
};

}  // namespace clifft
