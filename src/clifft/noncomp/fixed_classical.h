#pragma once

#include "clifft/noncomp/classifier.h"
#include "clifft/sampling/batch/external_inputs.h"
#include "clifft/sampling/batch/random_bits.h"
#include "clifft/util/shot_seed.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace clifft {
struct NonComputationalSample;

namespace detail {

// Runtime categories deliberately carry no claim about the hidden G/E label.
inline constexpr uint8_t kComputational = 1;
inline constexpr uint8_t kLeaked = 2;
inline constexpr uint8_t kLost = 4;
inline constexpr std::array<uint8_t, 3> kStatusBits{kComputational, kLeaked, kLost};

struct TransitionInput {
    uint32_t qubit;
    double probability;
    bool loss;
};
struct ResetInput {
    uint32_t qubit;
    std::array<uint8_t, 3> destinations;
};
struct NoiseOutcome {
    uint32_t input;
    double cumulative_probability;
};
struct NoiseInput {
    std::vector<uint32_t> active_targets;
    std::optional<uint32_t> inactive_source;
    uint8_t source_status = 0;
    std::vector<NoiseOutcome> outcomes;
};
using ReadoutChannel = ClassifiedReadoutChannel;
struct ReadoutInput {
    uint32_t input;
    uint32_t record;
    std::optional<uint32_t> qubit;
    std::array<ReadoutChannel, 3> channels;
};
struct ProbeInput {
    uint32_t input;
    uint32_t qubit;
    uint8_t status;
};
using InputStep = std::variant<TransitionInput, ResetInput, NoiseInput, ReadoutInput, ProbeInput>;

class FixedClassicalWorkspace {
  public:
    FixedClassicalWorkspace(uint32_t qubits, uint32_t capacity);
    [[nodiscard]] static uint64_t bit_columns(uint32_t qubits) noexcept {
        return uint64_t{2} * qubits + 5;
    }

  private:
    friend class FixedClassicalProgram;
    [[maybe_unused]] uint32_t qubits_;
    [[maybe_unused]] uint32_t capacity_;
    sampling::PackedBitColumns bits_;
};

// Compiled from the same ordered inputs consumed by the scalar reference.
// Owns all operands, probability tables, and quantum-plan input bindings.
// Status-independent operations can share a prepared condition until a status
// write invalidates it. Runtime never examines the circuit or resolves bindings.
class FixedClassicalProgram {
  public:
    FixedClassicalProgram(std::span<const InputStep> inputs, uint32_t qubits, uint32_t noise_inputs,
                          uint32_t readout_inputs, uint32_t records);
    // Output rows are fresh, zero-initialized slices owned by the sample call.
    // Only heralded slots and lost final statuses need writes.
    void prepare(const SeedRoot& root, uint32_t first, uint32_t count,
                 FixedClassicalWorkspace& workspace, sampling::BatchExternalInputs& inputs,
                 NonComputationalSample& output) const noexcept;

  private:
    struct Transition {
        uint32_t qubit;
        sampling::PreparedBernoulli event;
        bool loss;
    };
    struct Condition {
        uint32_t begin;
        uint32_t count;
        std::optional<uint32_t> source;
        uint8_t source_status;
    };
    struct Noise {
        uint32_t begin;
        uint32_t count;
        sampling::PreparedBernoulli event;
    };
    struct Readout {
        uint32_t input;
        uint32_t record;
        std::optional<uint32_t> qubit;
        uint8_t categories;
        bool high_is_zero;
        sampling::PreparedBernoulli herald;
        sampling::PreparedBernoulli high;
        sampling::PreparedBernoulli both_given_high;
    };
    using Operation = std::variant<Transition, ResetInput, Condition, Noise, Readout, ProbeInput>;
    uint32_t qubits_;
    std::vector<Operation> operations_;
    std::vector<uint32_t> operands_;
    std::vector<NoiseOutcome> outcomes_;
    sampling::PreparedBernoulli fair_{0.5};
};

}  // namespace detail
}  // namespace clifft
