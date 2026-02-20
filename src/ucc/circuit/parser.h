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
// Errors:
// - REPEAT blocks: not supported in MVP, raises error
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
    ParseError(const std::string& msg, int line)
        : std::runtime_error("Line " + std::to_string(line) + ": " + msg), line_(line) {}

    int line() const { return line_; }

  private:
    int line_;
};

// Parse a circuit from text.
// Throws ParseError on syntax errors or unsupported features.
Circuit parse(std::string_view text);

// Parse a circuit from a file.
// Throws ParseError on syntax errors, std::runtime_error on file errors.
Circuit parse_file(const std::string& path);

}  // namespace ucc
