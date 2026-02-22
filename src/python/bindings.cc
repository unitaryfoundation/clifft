#include "ucc/backend/backend.h"
#include "ucc/circuit/circuit.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/svm/svm.h"
#include "ucc/util/config.h"
#include "ucc/util/version.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_ucc_core, m) {
    m.doc() = "UCC core C++ extension module";

    // Register ParseError as a Python exception.
    // This creates ucc.ParseError in Python (re-exported from __init__.py).
    nb::exception<ucc::ParseError>(m, "ParseError");

    // Version info (from generated header, source of truth is pyproject.toml)
    m.def("version", []() { return ucc::kVersion; }, "Return the UCC version string");

    // Expose compile-time constants
    m.def(
        "max_sim_qubits", []() { return ucc::kMaxInlineQubits; },
        "Return the maximum number of qubits supported by the simulator");

    // GateType enum
    nb::enum_<ucc::GateType>(m, "GateType", "Quantum gate types")
        .value("H", ucc::GateType::H)
        .value("S", ucc::GateType::S)
        .value("S_DAG", ucc::GateType::S_DAG)
        .value("X", ucc::GateType::X)
        .value("Y", ucc::GateType::Y)
        .value("Z", ucc::GateType::Z)
        .value("T", ucc::GateType::T)
        .value("T_DAG", ucc::GateType::T_DAG)
        .value("CX", ucc::GateType::CX)
        .value("CY", ucc::GateType::CY)
        .value("CZ", ucc::GateType::CZ)
        .value("M", ucc::GateType::M)
        .value("MX", ucc::GateType::MX)
        .value("MY", ucc::GateType::MY)
        .value("MR", ucc::GateType::MR)
        .value("MRX", ucc::GateType::MRX)
        .value("MPP", ucc::GateType::MPP)
        .value("TICK", ucc::GateType::TICK)
        .value("UNKNOWN", ucc::GateType::UNKNOWN);

    // Target class (32-bit encoded quantum target)
    // Note: Use lambdas to wrap constexpr methods for nanobind compatibility.
    nb::class_<ucc::Target>(m, "Target", "Encoded quantum target (qubit, rec, or Pauli)")
        .def_prop_ro(
            "value", [](const ucc::Target& t) { return t.value(); },
            "Get the target value (qubit index or rec index)")
        .def_prop_ro(
            "is_rec", [](const ucc::Target& t) { return t.is_rec(); },
            "Check if this is a measurement record reference")
        .def_prop_ro(
            "is_inverted", [](const ucc::Target& t) { return t.is_inverted(); },
            "Check if this target has an inversion flag")
        .def_prop_ro(
            "has_pauli", [](const ucc::Target& t) { return t.has_pauli(); },
            "Check if this target has a Pauli tag")
        .def_prop_ro(
            "pauli", [](const ucc::Target& t) { return t.pauli(); },
            "Get the Pauli tag (0=none, 1=X, 2=Y, 3=Z)")
        .def_prop_ro(
            "pauli_char", [](const ucc::Target& t) { return std::string(1, t.pauli_char()); },
            "Get the Pauli tag as a character (I/X/Y/Z)")
        .def("__repr__", [](const ucc::Target& t) {
            std::string result;
            if (t.is_inverted())
                result += "!";
            if (t.is_rec()) {
                result += "rec[" + std::to_string(t.value()) + "]";
            } else if (t.has_pauli()) {
                result += t.pauli_char();
                result += std::to_string(t.value());
            } else {
                result += std::to_string(t.value());
            }
            return result;
        });

    // AstNode class (single circuit operation)
    nb::class_<ucc::AstNode>(m, "AstNode", "A single circuit operation")
        .def_ro("gate", &ucc::AstNode::gate, "Gate type")
        .def_ro("targets", &ucc::AstNode::targets, "Target qubits/records")
        .def_ro("arg", &ucc::AstNode::arg, "Optional gate argument")
        .def("__repr__", [](const ucc::AstNode& n) {
            std::string result = std::string(ucc::gate_name(n.gate));
            for (const auto& t : n.targets) {
                result += " ";
                if (t.is_inverted())
                    result += "!";
                if (t.is_rec()) {
                    result += "rec[" + std::to_string(t.value()) + "]";
                } else if (t.has_pauli()) {
                    result += t.pauli_char();
                    result += std::to_string(t.value());
                } else {
                    result += std::to_string(t.value());
                }
            }
            return result;
        });

    // Circuit class (parsed circuit)
    nb::class_<ucc::Circuit>(m, "Circuit", "A parsed quantum circuit")
        .def_ro("nodes", &ucc::Circuit::nodes, "List of circuit operations")
        .def_ro("num_qubits", &ucc::Circuit::num_qubits, "Number of qubits")
        .def_ro("num_measurements", &ucc::Circuit::num_measurements, "Number of measurements")
        .def("__len__", [](const ucc::Circuit& c) { return c.nodes.size(); })
        .def("__repr__", [](const ucc::Circuit& c) {
            return "Circuit(" + std::to_string(c.nodes.size()) + " ops, " +
                   std::to_string(c.num_qubits) + " qubits, " + std::to_string(c.num_measurements) +
                   " measurements)";
        });

    // Circuit parsing
    m.def("parse", &ucc::parse, nb::arg("text"),
          "Parse a quantum circuit from a string.\n\n"
          "Args:\n"
          "    text: Circuit description in .stim format\n\n"
          "Returns:\n"
          "    Circuit object\n\n"
          "Raises:\n"
          "    ParseError: If the circuit syntax is invalid");

    m.def("parse_file", &ucc::parse_file, nb::arg("path"),
          "Parse a quantum circuit from a file.\n\n"
          "Args:\n"
          "    path: Path to the circuit file\n\n"
          "Returns:\n"
          "    Circuit object\n\n"
          "Raises:\n"
          "    ParseError: If the circuit syntax is invalid\n"
          "    RuntimeError: If the file cannot be read");

    // =========================================================================
    // Compiled Program and Sampling
    // =========================================================================

    // CompiledModule wrapper (opaque to Python, just holds the bytecode)
    nb::class_<ucc::CompiledModule>(m, "Program", "A compiled quantum program")
        .def_prop_ro(
            "peak_rank", [](const ucc::CompiledModule& p) { return p.peak_rank; },
            "Maximum GF(2) dimension (determines memory usage)")
        .def_prop_ro(
            "num_measurements", [](const ucc::CompiledModule& p) { return p.num_measurements; },
            "Number of measurements in the circuit")
        .def_prop_ro(
            "num_instructions", [](const ucc::CompiledModule& p) { return p.bytecode.size(); },
            "Number of bytecode instructions")
        .def("__repr__", [](const ucc::CompiledModule& p) {
            return "Program(" + std::to_string(p.bytecode.size()) +
                   " instructions, peak_rank=" + std::to_string(p.peak_rank) + ", " +
                   std::to_string(p.num_measurements) + " measurements)";
        });

    // Compile: stim text -> Program
    m.def(
        "compile",
        [](const std::string& stim_text) {
            ucc::Circuit circuit = ucc::parse(stim_text);
            ucc::HirModule hir = ucc::trace(circuit);
            return ucc::lower(hir);
        },
        nb::arg("stim_text"),
        "Compile a quantum circuit to executable bytecode.\n\n"
        "Args:\n"
        "    stim_text: Circuit description in .stim format\n\n"
        "Returns:\n"
        "    Program object ready for sampling\n\n"
        "Raises:\n"
        "    ParseError: If the circuit syntax is invalid\n"
        "    RuntimeError: If compilation fails (e.g., >32 T-gate dimensions)");

    // Sample: Program + shots -> list of measurement results
    m.def(
        "sample",
        [](const ucc::CompiledModule& program, uint32_t shots, uint64_t seed) {
            std::vector<uint8_t> results = ucc::sample(program, shots, seed);
            // Return as 2D list [shots][num_measurements]
            size_t num_meas = program.num_measurements;
            std::vector<std::vector<uint8_t>> output(shots);
            for (uint32_t shot = 0; shot < shots; ++shot) {
                output[shot].resize(num_meas);
                for (size_t m = 0; m < num_meas; ++m) {
                    output[shot][m] = results[shot * num_meas + m];
                }
            }
            return output;
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = 0,
        "Run a compiled program and return measurement results.\n\n"
        "Args:\n"
        "    program: Compiled Program object\n"
        "    shots: Number of shots to run\n"
        "    seed: Random seed for reproducibility (default: 0)\n\n"
        "Returns:\n"
        "    List of lists with shape [shots][num_measurements], values are 0 or 1");
}
