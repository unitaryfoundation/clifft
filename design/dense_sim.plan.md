# UCC Implementation Plan: Dual-Engine Dense Statevector Simulator

## Context & Goal

UCC’s current Virtual Machine (`svm.cc`) is a state-of-the-art **streaming RISC engine**. It uses an Array of Structures (AoS) memory layout (`std::complex<double>`), executes instructions strictly sequentially, and uses branchless bit-weaving (`_pdep_u64`) to iterate over sparse virtual axes. This is mathematically optimal for Clifford+T algorithms where operations map to low-rank $\mathcal{O}(N)$ updates.

However, for dense $SU(N)$ heuristic circuits (like Quantum Volume), this architecture is bottlenecked entirely by **Memory Bandwidth**. Every single array instruction triggers a full memory sweep, repeatedly flushing the CPU's L3 cache.

**Goal:** Implement a **Dual-Engine Architecture**. We will retain the pristine RISC VM for sparse QEC circuits, but introduce a parallel **Dense Engine (`svm_dense.cc`)** to compete with `qsim`. This engine will optimize for **Arithmetic Intensity** using a Structure of Arrays (SoA) memory layout, OpenMP threading, explicit SIMD auto-vectorization, and a new L1 cache-blocked execution loop (`OP_LOCAL_BLOCK`). The AOT compiler will automatically route execution to the correct engine based on the circuit graph.

---

## Phase 1: Structure of Arrays (SoA) Data Layout (C++)

To allow the compiler to emit dense AVX2/AVX-512 `vfmadd` instructions natively, we must abandon `std::complex<double>` in the dense engine and use SoA.

**1. Create `src/ucc/svm/svm_dense.h**`
Define the `DenseSchrodingerState`. Note that it does *not* track a Pauli Frame (`p_x`, `p_z`). The dense engine operates strictly on physical amplitudes.

```cpp
#pragma once

#include "ucc/backend/backend.h"
#include "ucc/svm/svm.h" // For Xoshiro256PlusPlus

#include <cstdint>
#include <vector>
#include <optional>

namespace ucc {

// L1 Cache Block size (e.g., 10 qubits = 1024 complex doubles = 16KB SoA)
// This size ensures the block fits comfortably in the L1 Data Cache.
constexpr uint32_t kDenseCacheBlockBits = 10;
constexpr uint64_t kDenseCacheBlockSize = 1ULL << kDenseCacheBlockBits;

class DenseSchrodingerState {
  public:
    explicit DenseSchrodingerState(uint32_t num_qubits, uint32_t num_measurements,
                                   std::optional<uint64_t> seed = std::nullopt);
    ~DenseSchrodingerState();

    // Non-copyable
    DenseSchrodingerState(const DenseSchrodingerState&) = delete;
    DenseSchrodingerState& operator=(const DenseSchrodingerState&) = delete;

    void reset();

    // Flat SoA arrays (64-byte aligned for AVX-512)
    double* __restrict real_ = nullptr;
    double* __restrict imag_ = nullptr;

    uint64_t array_size_ = 0;
    uint32_t active_k = 0; // In dense mode, usually equal to total qubits

    std::vector<uint8_t> meas_record;
    Xoshiro256PlusPlus rng_;

    // Gap-based noise sampling components identical to SchrodingerState
    uint32_t next_noise_idx = 0;
    void draw_next_noise(const std::vector<double>& hazards);
    double random_double() { return static_cast<double>(rng_() >> 11) * 0x1.0p-53; }
};

void execute_dense(const CompiledModule& program, DenseSchrodingerState& state);

} // namespace ucc

```

**2. Implement Memory Allocation in `src/ucc/svm/svm_dense.cc**`
Use the existing portable aligned allocator to guarantee cache-line alignment.

```cpp
#include "ucc/svm/svm_dense.h"

#include <cstring>
#include <stdexcept>

// Ensure aligned_alloc_portable is accessible (may need to move to a shared util header)

namespace ucc {

DenseSchrodingerState::DenseSchrodingerState(uint32_t num_qubits, uint32_t num_measurements,
                                             std::optional<uint64_t> seed) {
    active_k = num_qubits;
    array_size_ = 1ULL << active_k;
    size_t bytes = array_size_ * sizeof(double);
    size_t aligned_bytes = (bytes + 63) & ~63ULL;

    real_ = static_cast<double*>(aligned_alloc_portable(64, aligned_bytes));
    imag_ = static_cast<double*>(aligned_alloc_portable(64, aligned_bytes));

    if (!real_ || !imag_) {
        throw std::bad_alloc();
    }

    if (seed.has_value()) rng_.seed(*seed);
    else rng_.seed_from_entropy();

    meas_record.resize(num_measurements, 0);
    reset();
}

DenseSchrodingerState::~DenseSchrodingerState() {
    aligned_free_portable(real_);
    aligned_free_portable(imag_);
}

void DenseSchrodingerState::reset() {
    size_t bytes = array_size_ * sizeof(double);
    size_t aligned_bytes = (bytes + 63) & ~63ULL;
    std::memset(real_, 0, aligned_bytes);
    std::memset(imag_, 0, aligned_bytes);
    real_[0] = 1.0; // Initialize to |0...0>
    std::fill(meas_record.begin(), meas_record.end(), 0);
    next_noise_idx = 0;
}

} // namespace ucc

```

---

## Phase 2: "Thick RISC" Opcodes & Constant Pool

The Dense Engine needs instructions that execute batches of gates on a subset of qubits that perfectly fits into the CPU's L1 Cache. It also needs an instruction to "flush" the sparse engine's Pauli frame onto the statevector before dense execution begins.

**1. Update `src/ucc/backend/backend.h**`
Extend the `Opcode` enum and the `ConstantPool`.

```cpp
enum class Opcode : uint8_t {
    // ... existing opcodes ...

    // --- Dense Engine Opcodes ---
    OP_FLUSH_FRAME,     // Physically applies p_x and p_z to the array, resetting frame to I
    OP_LOCAL_BLOCK,     // Execute a MicroProgram entirely within L1 cache

    NUM_OPCODES
};

// Represents a fused 2x2, 4x4, or 8x8 complex matrix for dense gate application
struct DenseMatrix {
    uint8_t num_targets;     // 1, 2, or 3
    std::vector<double> real; // Size: 4, 16, or 64
    std::vector<double> imag; // Size: 4, 16, or 64
};

// A single step within a cache-blocked micro-program
struct MicroInstruction {
    uint8_t num_targets;
    uint8_t targets[3];  // The relative bit positions inside the cache block (0-9)
    uint32_t matrix_idx; // Index into ConstantPool::dense_matrices
};

// A sequence of instructions applied to a 16KB L1 cache block
struct MicroProgram {
    uint8_t cache_bits; // The number of lowest-order bits defining the block (e.g., 10)
    std::vector<MicroInstruction> ops;
};

struct ConstantPool {
    // ... existing fields ...
    std::vector<DenseMatrix> dense_matrices;
    std::vector<MicroProgram> micro_programs;
};

struct CompiledModule {
    // ... existing fields ...
    bool is_dense = false; // Flag to indicate which engine to run
};

```

**2. Update `Instruction` struct union**
Add the block payload variant:

```cpp
        // Variant E: Cache Block
        struct {
            uint32_t micro_program_idx; // Offset 8
            uint8_t _pad_e[20];         // Offset 12
        } block;

```

---

## Phase 3: The Cache-Blocked Execution Loop (C++)

This is the computational heart of the dense engine. It sweeps the macroscopic statevector using OpenMP, but isolates the heavy mathematical lifting inside the L1 Cache.

**1. Implement `execute_dense` in `src/ucc/svm/svm_dense.cc**`

```cpp
// Executes a MicroProgram on a specific 16KB sub-block of the statevector.
// Because 'base_idx' aligns with the cache line and 'cache_bits' bounds
// the memory stride, this never triggers an L2/L3 miss.
static inline void exec_micro_program(
    double* __restrict real, double* __restrict imag,
    uint64_t base_idx,
    const MicroProgram& prog,
    const ConstantPool& pool)
{
    uint64_t half = 1ULL << (prog.cache_bits - 1);

    for (const auto& instr : prog.ops) {
        const auto& mat = pool.dense_matrices[instr.matrix_idx];

        // Example: Apply 1-qubit matrix (2x2).
        if (instr.num_targets == 1) {
            uint64_t t_bit = 1ULL << instr.targets[0];
            uint64_t pdep_mask = ~t_bit;

            const double* __restrict mr = mat.real.data();
            const double* __restrict mi = mat.imag.data();

            // #pragma omp simd ensures the compiler vectorizes this inner loop
            // using AVX2/AVX512, completely unimpeded by memory latency.
            #pragma omp simd
            for (uint64_t i = 0; i < half; ++i) {
                // Assuming x86 _pdep_u64 or the insert_zero_bit fallback
                uint64_t local_idx0 = scatter_bits_1(i, pdep_mask, instr.targets[0]);
                uint64_t local_idx1 = local_idx0 | t_bit;

                uint64_t idx0 = base_idx + local_idx0;
                uint64_t idx1 = base_idx + local_idx1;

                double r0 = real[idx0], i0 = imag[idx0];
                double r1 = real[idx1], i1 = imag[idx1];

                real[idx0] = (mr[0]*r0 - mi[0]*i0) + (mr[1]*r1 - mi[1]*i1);
                imag[idx0] = (mr[0]*i0 + mi[0]*r0) + (mr[1]*i1 + mi[1]*r1);

                real[idx1] = (mr[2]*r0 - mi[2]*i0) + (mr[3]*r1 - mi[3]*i1);
                imag[idx1] = (mr[2]*i0 + mi[2]*r0) + (mr[3]*i1 + mi[3]*r1);
            }
        }
        // TODO: Implement 2-target (4x4) and 3-target (8x8) explicitly unrolled loops
    }
}

void execute_dense(const CompiledModule& program, DenseSchrodingerState& state) {
    const Instruction* pc = program.bytecode.data();
    const Instruction* end = pc + program.bytecode.size();

    while (pc != end) {
        if (pc->opcode == Opcode::OP_LOCAL_BLOCK) {
            const auto& mprog = program.constant_pool.micro_programs[pc->block.micro_program_idx];
            uint64_t cache_size = 1ULL << mprog.cache_bits;
            uint64_t num_blocks = state.array_size_ / cache_size;

            // --- THE MULTI-THREADED CACHE SWEEP ---
            // Distribute blocks across CPU cores. Each core pulls a block
            // into its L1 cache, runs multiple gates on it, and writes it back.
            #pragma omp parallel for schedule(static)
            for (int64_t b = 0; b < static_cast<int64_t>(num_blocks); ++b) {
                uint64_t base_idx = b * cache_size;
                exec_micro_program(state.real_, state.imag_, base_idx, mprog, program.constant_pool);
            }
        }
        else if (pc->opcode == Opcode::OP_FLUSH_FRAME) {
            // Apply pending sparse p_x, p_z to the dense array before starting the dense simulation
        }
        else if (pc->opcode == Opcode::OP_ARRAY_SWAP) {
            // Global array swap to route qubits in/out of the L1 cache window (axes 0-9)
        }
        // Handle measurements...

        ++pc;
    }
}

```

---

## Phase 4: Compiler Routing Passes

The compiler must automatically partition the bytecode into `MicroPrograms` and route execution to the appropriate engine.

**1. Create `src/ucc/optimizer/dense_block_pass.h` & `.cc**`
This pass analyzes the unoptimized bytecode. If invoked, it:

1. Emits an `OP_EXPAND` block to force `active_k` up to the full circuit width instantly.
2. Prepends `OP_FLUSH_FRAME` to resolve the sparse `p_x`/`p_z` state into the raw physical amplitudes.
3. Groups operations acting on axes `0` to `kDenseCacheBlockBits - 1` into a `MicroProgram`.
4. Converts these local operations into dense matrices.
5. If an operation targets an axis outside the cache window, emits an `OP_ARRAY_SWAP` to bring it into the window, runs a local block, and swaps it back.
6. Replaces the sparse operations with `OP_LOCAL_BLOCK` instructions.
7. Sets `module.is_dense = true`.

---

## Phase 5: Python Bindings & Dual-Engine Routing

Expose the new engine safely in Python, retaining the default sparse engine for standard workloads.

**1. Update `src/python/bindings.cc**`
Update the `compile` binding to accept an `engine` parameter to allow users to override the compiler's heuristic.

```cpp
    m.def(
        "compile",
        [](const std::string& stim_text,
           std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors,
           std::vector<uint8_t> expected_observables,
           bool normalize_syndromes,
           std::string target_engine, // "auto", "sparse", or "dense"
           ucc::HirPassManager* hir_passes,
           ucc::BytecodePassManager* bytecode_passes) {

            // ... Parse, trace, and auto-normalize ...

            auto program = ucc::lower(hir, postselection_mask, expected_detectors, expected_observables);

            // Heuristic routing: Use dense engine if memory is large and circuit is dense/deep
            bool use_dense = (target_engine == "dense") ||
                             (target_engine == "auto" && program.peak_rank >= 12 && hir.num_t_gates() > 100);

            if (use_dense) {
                program.is_dense = true;
                ucc::BytecodePassManager dense_bpm;
                dense_bpm.add_pass(std::make_unique<ucc::ExpandTPass>());
                // Add the new dense cache-blocking fusion pass
                // dense_bpm.add_pass(std::make_unique<ucc::DenseBlockPass>());
                dense_bpm.run(program);
            } else {
                if (bytecode_passes) bytecode_passes->run(program);
            }

            return program;
        },
        // ... kwargs ...
        nb::arg("target_engine") = "auto"
    );

```

**2. Update `ucc::sample**`
Instantiate the correct C++ state based on `program.is_dense`.

```cpp
    m.def(
        "sample",
        [](const ucc::CompiledModule& program, uint32_t shots, std::optional<uint64_t> seed) {
            if (program.is_dense) {
                ucc::SampleResult result;
                // ... setup result arrays ...

                ucc::DenseSchrodingerState state(program.num_qubits, program.total_meas_slots, seed);
                for (uint32_t shot = 0; shot < shots; ++shot) {
                    if (shot > 0) state.reset();
                    // (Handle gap noise sampling setup)
                    ucc::execute_dense(program, state);
                    // (Copy state.meas_record to result)
                }
                return result;
            } else {
                // Existing sparse SchrodingerState execution
                return ucc::sample(program, shots, seed);
            }
        },
        // ... kwargs ...
    );

```

*(Note: Also implement a `ucc.get_dense_statevector` or update `get_statevector` to zip the SoA layout back into NumPy `complex128` arrays for the dense engine).*

---

## Phase 6: Testing Strategy

Because both engines simulate identical exact physical states, testing is highly deterministic via equivalence fuzzing.

**1. Cross-Backend Equivalence Testing (`tests/python/test_dual_engine.py`)**
Generate random Quantum Volume $SU(4)$ circuits, compile them for both engines, and assert that the statevectors match exactly.

```python
import numpy as np
import ucc

def test_engine_equivalence():
    # A textbook Quantum Volume style block
    circuit = """
        H 0 1 2 3
        U3(0.1, 0.2, 0.3) 0
        U3(0.4, 0.5, 0.6) 1
        CZ 0 1
        CNOT 1 2
        U3(0.7, 0.8, 0.9) 2
        CZ 2 3
    """

    # 1. Compile and execute in Sparse RISC engine
    prog_sparse = ucc.compile(circuit, target_engine="sparse")
    state_sparse = ucc.State(prog_sparse.peak_rank, 0)
    ucc.execute(prog_sparse, state_sparse)
    sv_sparse = ucc.get_statevector(prog_sparse, state_sparse)

    # 2. Compile and execute in Dense SoA engine
    prog_dense = ucc.compile(circuit, target_engine="dense")
    state_dense = ucc.DenseState(prog_dense.peak_rank, 0)
    ucc.execute_dense(prog_dense, state_dense)
    sv_dense = ucc.get_dense_statevector(prog_dense, state_dense)

    # 3. Assert mathematically identical exact states
    np.testing.assert_allclose(sv_sparse, sv_dense, atol=1e-10)

```

**2. C++ Unit Tests (`tests/test_svm_dense.cc`)**

* **Vectorization Verification:** Create a `MicroProgram` with a single $H$ gate matrix. Assert that `execute_dense` produces the exact same SoA statevector as the mathematical ideal.
* **Cache-Block Boundaries:** Test `OP_LOCAL_BLOCK` with `cache_bits = 2` on a 4-qubit array. Ensure the OpenMP loop bounds properly chunk the array without overflowing or mixing indices between chunks.
* **Frame Flushing:** Initialize a `DenseSchrodingerState`. Manually simulate a sparse state with `p_x = 1` on axis 0. Execute `OP_FLUSH_FRAME`. Assert that `real_[0] == 0.0`, `real_[1] == 1.0`, and the abstract tracking frame is conceptually cleared.
