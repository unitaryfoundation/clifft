#pragma once

// CompilerContext: Internal state for the AOT Back-End compiler.
//
// Exposed in a header (rather than anonymous namespace) so that
// test_backend.cc can directly instantiate contexts, feed them
// PauliStrings, and verify the resulting V_cum and bytecode.

#include "ucc/backend/backend.h"
#include "ucc/frontend/hir.h"

#include "stim.h"

#include <cstdint>
#include <vector>

namespace ucc {
namespace internal {

// =============================================================================
// VirtualRegisterManager
// =============================================================================
//
// Tracks the Active/Dormant partition of virtual axes using a simple
// contiguous split: axes 0..k-1 are Active, axes k..n-1 are Dormant.
//
// The compiler does NOT maintain a software dictionary mapping virtual
// qubits to array axes. Instead, V_cum itself tracks the permutation
// via explicit SWAP instructions emitted by the compiler. This means
// the axis indices in opcodes are always literal -- no translation needed.
//
// All n axes start Dormant (k=0). activate() increments k, deactivate()
// decrements k. The caller must SWAP a qubit to the boundary before
// changing its partition.

class VirtualRegisterManager {
  public:
    explicit VirtualRegisterManager(uint32_t num_qubits) : num_qubits_(num_qubits), active_k_(0) {}

    [[nodiscard]] uint32_t num_qubits() const { return num_qubits_; }
    [[nodiscard]] uint32_t active_k() const { return active_k_; }

    [[nodiscard]] bool is_active(uint16_t axis) const { return axis < active_k_; }
    [[nodiscard]] bool is_dormant(uint16_t axis) const { return axis >= active_k_; }

    // Promote the next dormant axis (k) to active. The caller must have
    // already SWAPped the target qubit to axis k before calling this.
    void activate() {
        assert(active_k_ < num_qubits_ && "All qubits already active");
        ++active_k_;
        if (active_k_ > peak_k_) {
            peak_k_ = active_k_;
        }
    }

    // Demote axis k-1 from active to dormant. The caller must have
    // already SWAPped the target qubit to axis k-1 before calling this.
    void deactivate() {
        assert(active_k_ > 0 && "No active qubits to deactivate");
        --active_k_;
    }

    [[nodiscard]] uint32_t peak_k() const { return peak_k_; }

  private:
    uint32_t num_qubits_;
    uint32_t active_k_;
    uint32_t peak_k_ = 0;
};

// =============================================================================
// CompilerContext
// =============================================================================
//
// Bundle of mutable state threaded through the Back-End compilation.
// Holds V_cum, the register manager, and the output bytecode.

struct CompilerContext {
    stim::Tableau<kStimWidth> v_cum;
    VirtualRegisterManager reg_manager;
    std::vector<Instruction> bytecode;
    ConstantPool constant_pool;
    double noise_hazards_accum = 0.0;

    explicit CompilerContext(uint32_t num_qubits) : v_cum(num_qubits), reg_manager(num_qubits) {}
};

// =============================================================================
// Result of compress_pauli: identifies what the Pauli was compressed to.
// =============================================================================

enum class CompressedBasis : uint8_t {
    Z_BASIS,  // Compressed to Z_v
    X_BASIS,  // Compressed to X_v
};

struct CompressionResult {
    uint16_t pivot;         // The virtual qubit the Pauli was compressed onto
    CompressedBasis basis;  // Whether it ended up as X_v or Z_v
    bool sign;              // Accumulated sign (true = negative)
};

// =============================================================================
// Core Compression Algorithm
// =============================================================================

/// Compress an arbitrary multi-qubit Pauli string onto a single virtual qubit.
///
/// Given a PauliString P, computes a sequence V of virtual Clifford gates
/// such that V P V^dag = (+/-)P_v where P_v in {X_v, Z_v}.
///
/// Side effects:
///   - Appends virtual gates to ctx.v_cum
///   - Emits corresponding RISC opcodes to ctx.bytecode
///   - Does NOT modify ctx.reg_manager (activation is the caller's job)
///
/// The input PauliString must be non-identity (at least one qubit set).
CompressionResult compress_pauli(CompilerContext& ctx, const stim::PauliString<kStimWidth>& pauli);

}  // namespace internal
}  // namespace ucc
