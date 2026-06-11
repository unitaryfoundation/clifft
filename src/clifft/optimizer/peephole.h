#pragma once

#include "clifft/optimizer/hir_pass.h"
#include "clifft/util/mask_view.h"
#include "clifft/util/stim_mask.h"

#include <complex>
#include <cstddef>

namespace clifft {

namespace internal {

/// Compose U_C' = U_C * S_P into `tab` for the S/S_dag rotation on the
/// virtual Pauli (x_v, z_v, sign_v), updating only the symplectic action.
void apply_s_to_tableau(stim::Tableau<kStimWidth>& tab, MaskView x_v, MaskView z_v, bool sign_v,
                        bool is_dagger);

/// Phase factor relating stim's canonical unitaries across that update:
///   canonical(original) * R == phase * canonical(updated)
/// where R = Pi_+ + (+-i) Pi_- is the exact projector-form rotation on the
/// virtual Pauli. Multiply global_weight by the result when replacing the
/// physical rotation with the tableau update.
[[nodiscard]] std::complex<double> s_absorption_phase(const stim::Tableau<kStimWidth>& original,
                                                      const stim::Tableau<kStimWidth>& updated,
                                                      MaskView x_v, MaskView z_v, bool sign_v,
                                                      bool is_dagger);

}  // namespace internal

/// Peephole fusion pass: scans the HIR to algebraically cancel or fuse
/// T/T_dag gates acting on the same virtual Pauli axis using the
/// symplectic inner product as a commutation check.
class PeepholeFusionPass : public HirPass {
  public:
    void run(HirModule& hir) override;

    /// Statistics from the last run.
    size_t cancellations() const { return cancellations_; }
    size_t fusions() const { return fusions_; }

  private:
    size_t cancellations_ = 0;
    size_t fusions_ = 0;
};

}  // namespace clifft
