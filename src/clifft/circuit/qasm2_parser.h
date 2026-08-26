#pragma once

// Native unitary OpenQASM 2.0 importer.

#include "clifft/circuit/circuit.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace clifft {

// The imported AST stays phase-free so ordinary sampling compilation does not
// acquire scalar bookkeeping. Phase-sensitive consumers can apply
// exp(i * pi * global_phase_turns) at their separate compilation boundary.
struct Qasm2Import {
    Circuit circuit;
    double global_phase_turns = 0.0;
};

// Parse and lower the supported unitary OpenQASM 2.0 subset.
[[nodiscard]] Qasm2Import parse_qasm2(std::string_view text);

// Parse with an explicit limit on the number of lowered AST nodes.
[[nodiscard]] Qasm2Import parse_qasm2(std::string_view text, size_t max_ops);

// Parse an OpenQASM 2.0 file.
[[nodiscard]] Qasm2Import parse_qasm2_file(const std::string& path);

// Parse a file with an explicit lowered-node limit.
[[nodiscard]] Qasm2Import parse_qasm2_file(const std::string& path, size_t max_ops);

}  // namespace clifft
