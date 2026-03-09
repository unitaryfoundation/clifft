#pragma once

// Circuit parser for .stim-superset format.
//
// Parses text input into a Circuit AST. Supports:
// - Clifford gates: H, S, S_DAG, X, Y, Z, CX, CY, CZ
// - Non-Clifford gates: T, T_DAG
// - Measurements: M, MX, MY, MR, MRX, MPP
// - Classical feedback: CX rec[-k] q, CZ rec[-k] q
// - Annotations: TICK
//
// Parser transformations:
// - R q  -> M q; CX rec[-1] q  (reset decomposition)
// - RX q -> MX q; CZ rec[-1] q (reset decomposition)
// - MPP X0*Z1 X2 -> two separate AstNodes (unrolling)
//
// REPEAT handling:
// - REPEAT N { ... } blocks are unrolled at parse time via text-level replay.
//   The body text is re-parsed N times against the same Circuit, so rec[-k]
//   references naturally resolve to correct absolute measurement indices.
//   Nested REPEAT is supported. A safety limit (kMaxUnrolledOps) prevents OOM.
//
// Errors:
// - Unknown gates: raises error
// - Malformed syntax: raises error with line number

#include "ucc/circuit/circuit.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ucc {

// Parse exception with line information.
class ParseError : public std::runtime_error {
  public:
    ParseError(const std::string& msg, uint32_t line)
        : std::runtime_error("Line " + std::to_string(line) + ": " + msg), line_(line) {}

    uint32_t line() const { return line_; }

  private:
    uint32_t line_;
};

// Parse a circuit from text.
// Uses kMaxUnrolledOps as the safety limit on total AST nodes.
[[nodiscard]] Circuit parse(std::string_view text);

// Parse a circuit from text with an explicit AST node limit.
// Used by tests to exercise the limit at smaller values.
[[nodiscard]] Circuit parse(std::string_view text, size_t max_ops);

// Parse a circuit from a file.
// Throws ParseError on syntax errors, std::runtime_error on file errors.
[[nodiscard]] Circuit parse_file(const std::string& path);

// Parse a circuit from a file with an explicit AST node limit.
[[nodiscard]] Circuit parse_file(const std::string& path, size_t max_ops);

}  // namespace ucc
