#include "ucc/backend/backend.h"
#include "ucc/circuit/circuit.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/svm/svm.h"
#include "ucc/util/config.h"
#include "ucc/util/version.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
            "Number of visible measurements in the circuit")
        .def_prop_ro(
            "num_detectors", [](const ucc::CompiledModule& p) { return p.num_detectors; },
            "Number of detectors in the circuit")
        .def_prop_ro(
            "num_observables", [](const ucc::CompiledModule& p) { return p.num_observables; },
            "Number of observables in the circuit")
        .def_prop_ro(
            "num_instructions", [](const ucc::CompiledModule& p) { return p.bytecode.size(); },
            "Number of bytecode instructions")
        .def("__repr__", [](const ucc::CompiledModule& p) {
            return "Program(" + std::to_string(p.bytecode.size()) +
                   " instructions, peak_rank=" + std::to_string(p.peak_rank) + ", " +
                   std::to_string(p.num_measurements) + " measurements, " +
                   std::to_string(p.num_detectors) + " detectors, " +
                   std::to_string(p.num_observables) + " observables)";
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

    // Helper to create a numpy array from a vector with given shape
    auto make_numpy_array = [](const std::vector<uint8_t>& vec, size_t rows, size_t cols) {
        auto* data = new uint8_t[vec.size()];
        std::copy(vec.begin(), vec.end(), data);
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
        return nb::ndarray<nb::numpy, uint8_t, nb::c_contig>(data, {rows, cols}, owner);
    };

    // Sample: Program + shots -> tuple of (measurements, detectors, observables)
    m.def(
        "sample",
        [make_numpy_array](const ucc::CompiledModule& program, uint32_t shots, uint64_t seed) {
            ucc::SampleResult result = ucc::sample(program, shots, seed);

            auto meas_arr = make_numpy_array(result.measurements, shots, program.num_measurements);
            auto det_arr = make_numpy_array(result.detectors, shots, program.num_detectors);
            auto obs_arr = make_numpy_array(result.observables, shots, program.num_observables);

            return nb::make_tuple(meas_arr, det_arr, obs_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = 0,
        "Run a compiled program and return all result records.\n\n"
        "Args:\n"
        "    program: Compiled Program object\n"
        "    shots: Number of shots to run\n"
        "    seed: Random seed for reproducibility (default: 0)\n\n"
        "Returns:\n"
        "    Tuple of three numpy arrays:\n"
        "    - measurements: shape (shots, num_measurements), dtype=uint8\n"
        "    - detectors: shape (shots, num_detectors), dtype=uint8\n"
        "    - observables: shape (shots, num_observables), dtype=uint8");

    // =========================================================================
    // Statevector API
    // =========================================================================

    // SchrodingerState wrapper for manual execution
    nb::class_<ucc::SchrodingerState>(m, "State", "Schr\u00f6dinger VM execution state")
        .def(nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>(), nb::arg("peak_rank"),
             nb::arg("num_measurements"), nb::arg("num_detectors") = 0,
             nb::arg("num_observables") = 0, nb::arg("seed") = 0,
             "Create a new execution state.\n\n"
             "Args:\n"
             "    peak_rank: Maximum GF(2) dimension (from Program.peak_rank)\n"
             "    num_measurements: Number of measurements (from Program.num_measurements)\n"
             "    num_detectors: Number of detectors (from Program.num_detectors)\n"
             "    num_observables: Number of observables (from Program.num_observables)\n"
             "    seed: Random seed for reproducibility")
        .def("reset", &ucc::SchrodingerState::reset, nb::arg("seed"),
             "Reset state to |0...0\u27e9 for a new shot")
        .def_prop_ro(
            "meas_record",
            [](const ucc::SchrodingerState& s) { return std::vector<uint8_t>(s.meas_record); },
            "Copy of measurement record after execution")
        .def_prop_ro(
            "det_record",
            [](const ucc::SchrodingerState& s) { return std::vector<uint8_t>(s.det_record); },
            "Copy of detector record after execution")
        .def_prop_ro(
            "obs_record",
            [](const ucc::SchrodingerState& s) { return std::vector<uint8_t>(s.obs_record); },
            "Copy of observable record after execution")
        .def("__repr__", [](const ucc::SchrodingerState& s) {
            return "State(array_size=" + std::to_string(s.array_size()) + ")";
        });

    // Execute: Program + State -> mutates state
    m.def(
        "execute",
        [](const ucc::CompiledModule& program, ucc::SchrodingerState& state) {
            ucc::execute(program, state);
        },
        nb::arg("program"), nb::arg("state"),
        "Execute a compiled program, updating the state in-place.\n\n"
        "Args:\n"
        "    program: Compiled Program object\n"
        "    state: State object to execute into\n\n"
        "After execution, state.meas_record contains the measurement outcomes.");

    // Get statevector: Program + State -> numpy array
    m.def(
        "get_statevector",
        [](const ucc::CompiledModule& program, const ucc::SchrodingerState& state) {
            if (!program.constant_pool.final_tableau.has_value()) {
                throw std::runtime_error(
                    "Program has no final_tableau - cannot expand statevector");
            }

            auto sv = ucc::get_statevector(state, program.constant_pool.gf2_basis,
                                           program.constant_pool.final_tableau.value(),
                                           program.constant_pool.global_weight);

            // Allocate owned data
            size_t n = sv.size();
            auto* data = new std::complex<double>[n];
            std::copy(sv.begin(), sv.end(), data);

            nb::capsule owner(
                data, [](void* p) noexcept { delete[] static_cast<std::complex<double>*>(p); });

            return nb::ndarray<nb::numpy, std::complex<double>, nb::c_contig>(data, {n}, owner);
        },
        nb::arg("program"), nb::arg("state"),
        "Expand the SVM state into a dense statevector.\n\n"
        "Args:\n"
        "    program: Compiled Program object\n"
        "    state: State object after execution\n\n"
        "Returns:\n"
        "    numpy.ndarray of complex128 with shape (2^num_qubits,)\n\n"
        "Note: For circuits with measurements, the statevector represents\n"
        "the post-measurement state for one particular shot.");
}
