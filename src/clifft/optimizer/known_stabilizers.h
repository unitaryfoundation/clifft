#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/tableau/pauli_string.h"

#include <map>
#include <optional>

namespace clifft {

// Signed Pauli constraints with eigenvalue +1 on every reachable trajectory,
// including the noiseless reference used for syndrome normalization.
// This differs from dormant-width analysis, which also retains bodies whose
// eigenvalues depend on sampled outcomes. An empty group means no knowledge.
class KnownStabilizers {
  public:
    KnownStabilizers() = default;
    explicit KnownStabilizers(uint32_t num_qubits);

    [[nodiscard]] PauliString reduce_body(PauliString axis) const;
    // A missing value cannot justify a fixed rewrite; true denotes eigenvalue -1.
    [[nodiscard]] std::optional<bool> eigenvalue(PauliString axis) const;
    [[nodiscard]] bool commutes(PauliStringView axis) const;

    // The caller must prove that the inserted signed axis has eigenvalue +1.
    void insert(PauliString axis);
    void intersect(PauliStringView axis);
    void advance(const HirModule& hir, const HeisenbergOp& op);

  private:
    std::map<uint32_t, PauliString> rows_;
};

}  // namespace clifft
