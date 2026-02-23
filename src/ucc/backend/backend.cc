#include "ucc/backend/backend.h"

#include <bit>
#include <stdexcept>

namespace ucc {

namespace {

// Compute base_phase_idx incorporating Y-count from Pauli observable.
// The observable P = ±(i^y_count) X^destab Z^stab has phases:
//   sign=false, y_count=0 -> +1 -> idx 0
//   sign=false, y_count=1 -> +i -> idx 1
//   sign=false, y_count=2 -> -1 -> idx 2
//   sign=false, y_count=3 -> -i -> idx 3
//   sign=true adds 2 (multiply by -1)
inline uint8_t compute_base_phase_idx(stim::bitword<kStimWidth> destab,
                                      stim::bitword<kStimWidth> stab, bool sign) {
    // Y = iXZ, so Y-count is popcount(destab & stab)
    uint32_t y_count = (destab & stab).popcount();
    return static_cast<uint8_t>(((sign ? 2 : 0) + y_count) & 3);
}

}  // namespace

// =============================================================================
// GF(2) Basis Implementation
// =============================================================================

std::optional<uint32_t> GF2Basis::find_in_span(stim::bitword<kStimWidth> beta) const {
    if (beta == stim::bitword<kStimWidth>(0)) {
        return 0;  // Zero vector is trivially in span (empty combination)
    }

    uint32_t x_mask = 0;
    uint64_t remaining = static_cast<uint64_t>(beta);

    while (remaining != 0) {
        int lead_bit = 63 - std::countl_zero(remaining);
        if (echelon_basis_[lead_bit] == 0) {
            return std::nullopt;  // No basis vector with this leading bit
        }
        remaining ^= echelon_basis_[lead_bit];
        x_mask ^= echelon_x_mask_[lead_bit];
    }

    return x_mask;
}

uint32_t GF2Basis::add(stim::bitword<kStimWidth> destab) {
    if (basis_.size() >= kMaxRank) {
        throw std::runtime_error(
            "GF(2) rank limit exceeded: circuit requires >32 dimensions. "
            "This would need >68 GB RAM per shot.");
    }
    uint32_t idx = static_cast<uint32_t>(basis_.size());
    basis_.push_back(destab);
    add_to_echelon(destab, 1u << idx);
    return idx;
}

void GF2Basis::remove(uint32_t bit_index) {
    if (bit_index < basis_.size()) {
        basis_.erase(basis_.begin() + static_cast<ptrdiff_t>(bit_index));
        rebuild_echelon();
    }
}

uint32_t GF2Basis::rank() const {
    return static_cast<uint32_t>(basis_.size());
}

const std::vector<stim::bitword<kStimWidth>>& GF2Basis::vectors() const {
    return basis_;
}

void GF2Basis::add_to_echelon(stim::bitword<kStimWidth> beta, uint32_t initial_x_mask) {
    uint64_t remaining = static_cast<uint64_t>(beta);
    uint32_t x_mask = initial_x_mask;

    while (remaining != 0) {
        int lead_bit = 63 - std::countl_zero(remaining);
        if (echelon_basis_[lead_bit] == 0) {
            // Found empty slot - insert here
            echelon_basis_[lead_bit] = remaining;
            echelon_x_mask_[lead_bit] = x_mask;
            return;
        }
        // Eliminate this leading bit using existing row
        remaining ^= echelon_basis_[lead_bit];
        x_mask ^= echelon_x_mask_[lead_bit];
    }
}

void GF2Basis::rebuild_echelon() {
    std::fill(std::begin(echelon_basis_), std::end(echelon_basis_), 0);
    std::fill(std::begin(echelon_x_mask_), std::end(echelon_x_mask_), 0);
    for (size_t i = 0; i < basis_.size(); ++i) {
        add_to_echelon(basis_[i], 1u << i);
    }
}

namespace {

// =============================================================================
// Helper Functions
// Compute commutation_mask: which basis vectors anti-commute with this Pauli.
// Two Paulis anti-commute when (X1 & Z2) ^ (Z1 & X2) has odd popcount.
// For basis vector i with X-bits only (destab), anti-commutes with stab_mask
// of the instruction when their AND has odd popcount.
uint32_t compute_commutation_mask(const std::vector<stim::bitword<kStimWidth>>& basis,
                                  [[maybe_unused]] stim::bitword<kStimWidth> destab_mask,
                                  stim::bitword<kStimWidth> stab_mask) {
    uint32_t mask = 0;
    for (size_t i = 0; i < basis.size(); ++i) {
        // Basis vectors are pure X (destab only, no Z component).
        // Anti-commutation: (basis[i] & stab_mask) XOR (0 & destab_mask)
        // = (basis[i] & stab_mask) has odd popcount
        uint64_t overlap = static_cast<uint64_t>(basis[i]) & static_cast<uint64_t>(stab_mask);
        if (std::popcount(overlap) & 1) {
            mask |= (1u << i);
        }
    }
    return mask;
}

}  // namespace

CompiledModule lower(const HirModule& hir) {
    CompiledModule result;
    result.num_measurements = hir.num_measurements;

    GF2Basis basis;
    uint32_t peak_rank = 0;

    // Copy AG matrices from HIR to ConstantPool
    result.constant_pool.ag_matrices = hir.ag_matrices;

    for (const auto& op : hir.ops) {
        switch (op.op_type()) {
            case OpType::T_GATE: {
                stim::bitword<kStimWidth> destab = op.destab_mask();
                stim::bitword<kStimWidth> stab = op.stab_mask();
                bool sign = op.sign();
                bool is_dagger = op.is_dagger();

                // The spatial shift β is the X-component (destab_mask)
                stim::bitword<kStimWidth> beta = destab;

                Instruction instr{};
                instr.is_dagger = is_dagger;
                // base_phase_idx encodes sign and Y-count: i^y_count * (-1)^sign
                instr.base_phase_idx = compute_base_phase_idx(destab, stab, sign);
                instr.branch.destab_mask = static_cast<uint64_t>(destab);
                instr.branch.stab_mask = static_cast<uint64_t>(stab);

                if (beta == stim::bitword<kStimWidth>(0)) {
                    // Diagonal: no spatial shift, but Z components can still
                    // anti-commute with active basis vectors in V.
                    instr.opcode = Opcode::OP_SCALAR_PHASE;
                    instr.branch.x_mask = 0;
                    instr.branch.bit_index = 0;
                    // Must compute commutation even for diagonal ops!
                    instr.commutation_mask =
                        compute_commutation_mask(basis.vectors(), destab, stab);
                } else if (auto x_mask_opt = basis.find_in_span(beta)) {
                    // In span: butterfly operation
                    instr.opcode = Opcode::OP_COLLIDE;
                    instr.branch.x_mask = *x_mask_opt;
                    instr.branch.bit_index = 0;  // Not used for COLLIDE
                    instr.commutation_mask =
                        compute_commutation_mask(basis.vectors(), destab, stab);
                } else {
                    // New dimension: branch
                    uint32_t new_idx = basis.add(destab);
                    instr.opcode = Opcode::OP_BRANCH;
                    instr.branch.x_mask = 1u << new_idx;  // Just this new vector
                    instr.branch.bit_index = new_idx;
                    instr.commutation_mask =
                        compute_commutation_mask(basis.vectors(), destab, stab);
                    peak_rank = std::max(peak_rank, basis.rank());
                }

                result.bytecode.push_back(instr);
                break;
            }

            case OpType::MEASURE: {
                stim::bitword<kStimWidth> destab = op.destab_mask();
                stim::bitword<kStimWidth> stab = op.stab_mask();
                bool sign = op.sign();
                AgMatrixIdx ag_idx = op.ag_matrix_idx();
                uint8_t ag_ref = op.ag_ref_outcome();
                bool emitted_merge = false;

                // For measurements, β is the destab_mask (X-component of rewound observable)
                stim::bitword<kStimWidth> beta = destab;

                Instruction instr{};
                // base_phase_idx encodes sign and Y-count: i^y_count * (-1)^sign
                instr.base_phase_idx = compute_base_phase_idx(destab, stab, sign);
                instr.ag_ref_outcome = ag_ref;
                instr.branch.destab_mask = static_cast<uint64_t>(destab);
                instr.branch.stab_mask = static_cast<uint64_t>(stab);

                // Compute commutation_mask for all paths (needed for FILTER)
                instr.commutation_mask = compute_commutation_mask(basis.vectors(), destab, stab);

                // Measurement opcode routing (from Python prototype aot_compiler.py:427):
                //   β ≠ 0 and β ∈ span(V) → MERGE (array halves, reclaim dimension)
                //   β ≠ 0 and β ∉ span(V) → AG_PIVOT only (no dimension change)
                //   β = 0 and comm_mask ≠ 0 → FILTER (zero half array)
                //   β = 0 and comm_mask = 0 → DETERMINISTIC

                if (beta == stim::bitword<kStimWidth>(0)) {
                    // β = 0: measurement is diagonal in GF(2) space
                    if (instr.commutation_mask == 0) {
                        // Fully deterministic: no basis vectors to interfere
                        instr.opcode = Opcode::OP_MEASURE_DETERMINISTIC;
                        instr.branch.x_mask = 0;
                        instr.branch.bit_index = 0;
                    } else {
                        // FILTER: zero half the array based on commutation
                        instr.opcode = Opcode::OP_MEASURE_FILTER;
                        instr.branch.x_mask = 0;
                        // bit_index = highest set bit in commutation_mask
                        instr.branch.bit_index = 31 - std::countl_zero(instr.commutation_mask);
                    }
                    result.bytecode.push_back(instr);
                } else if (auto x_mask_opt = basis.find_in_span(beta)) {
                    // β ∈ span(V): MERGE - collapse an existing dimension
                    instr.opcode = Opcode::OP_MEASURE_MERGE;
                    instr.branch.x_mask = *x_mask_opt;
                    uint32_t bit_idx = 31 - std::countl_zero(*x_mask_opt);
                    instr.branch.bit_index = bit_idx;

                    result.bytecode.push_back(instr);
                    emitted_merge = true;

                    // Reclaim dimension to mirror VM array compaction.
                    // The VM halves v_size and shifts indices; we must match.
                    basis.remove(bit_idx);
                } else {
                    // β ∉ span(V): measurement introduces new randomness but
                    // does NOT add a dimension to the GF(2) basis. The AG_PIVOT
                    // (emitted below) handles the basis change. No MEASURE_*
                    // instruction is emitted here - just the AG_PIVOT.
                }

                // Emit AG_PIVOT if the Front-End provided one
                if (ag_idx != AgMatrixIdx::None) {
                    Instruction pivot_instr{};
                    pivot_instr.opcode = Opcode::OP_AG_PIVOT;
                    pivot_instr.ag_ref_outcome = ag_ref;
                    pivot_instr.meta.payload_idx = static_cast<uint32_t>(ag_idx);

                    // If we just emitted MEASURE_MERGE, signal the SVM to reuse
                    // that outcome (don't sample fresh).
                    pivot_instr.reuse_outcome = emitted_merge;

                    // ag_stab_slot: stabilizer row where measured observable landed.
                    // Computed in frontend by scanning post-collapse tableau.
                    pivot_instr.meta.ag_stab_slot = op.ag_stab_slot();

                    result.bytecode.push_back(pivot_instr);
                }
                break;
            }

            case OpType::CONDITIONAL_PAULI: {
                stim::bitword<kStimWidth> destab = op.destab_mask();
                stim::bitword<kStimWidth> stab = op.stab_mask();
                bool sign = op.sign();
                ControllingMeasIdx ctrl = op.controlling_meas();

                Instruction instr{};
                instr.opcode = Opcode::OP_CONDITIONAL;
                // base_phase_idx encodes sign and Y-count: i^y_count * (-1)^sign
                instr.base_phase_idx = compute_base_phase_idx(destab, stab, sign);
                instr.meta.controlling_meas = static_cast<uint32_t>(ctrl);
                instr.meta.destab_mask = static_cast<uint64_t>(destab);
                instr.meta.stab_mask = static_cast<uint64_t>(stab);
                instr.commutation_mask = compute_commutation_mask(basis.vectors(), destab, stab);

                result.bytecode.push_back(instr);
                break;
            }

            default:
                throw std::runtime_error("Unsupported OpType in Back-End");
        }
    }

    // Store final GF(2) basis for statevector expansion
    result.constant_pool.gf2_basis = basis.vectors();
    result.peak_rank = peak_rank;

    // Copy forward tableau and global weight for statevector expansion
    result.constant_pool.final_tableau = hir.final_tableau;
    result.constant_pool.global_weight = hir.global_weight;

    return result;
}

}  // namespace ucc
