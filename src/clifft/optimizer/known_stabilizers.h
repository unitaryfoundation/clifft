#pragma once

#include "clifft/frontend/hir.h"
#include "clifft/tableau/pauli_string.h"

#include <map>
#include <optional>

namespace clifft {

// Tracks signed Pauli constraints independent of noise and measurement outcomes
// to justify compile-time rewrites shared by all trajectories. Each operator
// stabilizes every reachable state and the noiseless reference used for syndrome
// normalization. Ordinary factored-state bookkeeping can also retain stabilizers
// whose signs depend on the sampled trajectory. An empty group means no knowledge.
class KnownStabilizers {
  public:
    KnownStabilizers() = default;
    explicit KnownStabilizers(uint32_t num_qubits);

    [[nodiscard]] bool empty() const { return rows_.empty(); }

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
