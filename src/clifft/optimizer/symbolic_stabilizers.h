#pragma once

#include "clifft/optimizer/known_stabilizers.h"
#include "clifft/sampling/plan.h"

#include <cstddef>
#include <map>
#include <optional>

namespace clifft {

struct SymbolicStabilizerOptions {
    // Rank is at most physical width. Bound additional affine storage and work.
    size_t max_expression_terms = 64;
    size_t max_record_entries = 4096;
    size_t max_row_products = 4096;
};

// Forward facts have affine eigenvalue signs. Only their fixed subgroup can
// justify rewrites shared by every trajectory and the noiseless reference.
// Atoms denote physical outcomes and faults, with no assumed independence.
// They never escape into a sampling plan, whose branch symbols use a different
// coordinate-dependent outcome convention.
class SymbolicStabilizers {
  public:
    explicit SymbolicStabilizers(const HirModule& hir, SymbolicStabilizerOptions options = {});
    SymbolicStabilizers(const HirModule& hir, size_t end, SymbolicStabilizerOptions options = {});

    // The reference remains valid until advance; callers modifying a lookahead
    // group take a copy. Fixed groups have no records or inference budgets.
    [[nodiscard]] const KnownStabilizers& fixed_constraints();
    void advance(const HirModule& hir, const HeisenbergOp& op);

  private:
    using AffineBool = sampling::AffineBool;
    struct Row {
        PauliString axis;
        AffineBool sign;
    };
    struct Record {
        AffineBool value;
        size_t age;
    };
    struct Feedback {
        uint32_t record;
        PauliString axis;
        std::optional<AffineBool> condition;
    };

    [[nodiscard]] std::optional<AffineBool> affine_eigenvalue(PauliString axis, bool& capped) const;
    [[nodiscard]] std::optional<AffineBool> fresh_symbol();
    bool multiply(Row& left, const Row& right, size_t& products) const;
    bool insert_row(Row row);
    bool intersect_rows(PauliStringView axis);
    void intersect(PauliStringView axis);
    void apply_pauli(PauliStringView axis, const AffineBool& condition);
    void assign_record(uint32_t record, AffineBool value);
    void flush_feedback();
    void forget();

    SymbolicStabilizerOptions options_;
    uint64_t next_symbol_ = 0;
    size_t next_record_age_ = 0;
    std::map<uint32_t, Row> rows_;
    std::map<uint32_t, Record> records_;
    // Uses follow emitted classical operations even when rotations are moved.
    // This schedule survives discarded facts and requires O(feedback records) storage.
    std::map<uint32_t, size_t> remaining_uses_;
    std::optional<Feedback> feedback_;
    KnownStabilizers fixed_;
    bool fixed_valid_ = false;
};

}  // namespace clifft
