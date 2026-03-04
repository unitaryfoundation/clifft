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
    nb::exception<ucc::ParseError>(m, "ParseError");

    // Version info
    m.def("version", []() { return ucc::kVersion; }, "Return the UCC version string");

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

    // Target class
    nb::class_<ucc::Target>(m, "Target", "Encoded quantum target")
        .def_prop_ro("value", [](const ucc::Target& t) { return t.value(); })
        .def_prop_ro("is_rec", [](const ucc::Target& t) { return t.is_rec(); })
        .def_prop_ro("is_inverted", [](const ucc::Target& t) { return t.is_inverted(); })
        .def_prop_ro("has_pauli", [](const ucc::Target& t) { return t.has_pauli(); })
        .def_prop_ro("pauli", [](const ucc::Target& t) { return t.pauli(); })
        .def_prop_ro("pauli_char",
                     [](const ucc::Target& t) { return std::string(1, t.pauli_char()); })
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

    // AstNode class
    nb::class_<ucc::AstNode>(m, "AstNode", "A single circuit operation")
        .def_ro("gate", &ucc::AstNode::gate)
        .def_ro("targets", &ucc::AstNode::targets)
        .def_ro("arg", &ucc::AstNode::arg)
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

    // Circuit class
    nb::class_<ucc::Circuit>(m, "Circuit", "A parsed quantum circuit")
        .def_ro("nodes", &ucc::Circuit::nodes)
        .def_ro("num_qubits", &ucc::Circuit::num_qubits)
        .def_ro("num_measurements", &ucc::Circuit::num_measurements)
        .def("__len__", [](const ucc::Circuit& c) { return c.nodes.size(); })
        .def("__repr__", [](const ucc::Circuit& c) {
            return "Circuit(" + std::to_string(c.nodes.size()) + " ops, " +
                   std::to_string(c.num_qubits) + " qubits, " + std::to_string(c.num_measurements) +
                   " measurements)";
        });

    // Circuit parsing
    m.def("parse", &ucc::parse, nb::arg("text"), "Parse a quantum circuit from a string.");
    m.def("parse_file", &ucc::parse_file, nb::arg("path"), "Parse a quantum circuit from a file.");

    // =========================================================================
    // Compiled Program and Sampling
    // =========================================================================

    nb::class_<ucc::CompiledModule>(m, "Program", "A compiled quantum program")
        .def_prop_ro("peak_rank", [](const ucc::CompiledModule& p) { return p.peak_rank; })
        .def_prop_ro("num_measurements",
                     [](const ucc::CompiledModule& p) { return p.num_measurements; })
        .def_prop_ro("num_detectors", [](const ucc::CompiledModule& p) { return p.num_detectors; })
        .def_prop_ro("num_observables",
                     [](const ucc::CompiledModule& p) { return p.num_observables; })
        .def_prop_ro("num_instructions",
                     [](const ucc::CompiledModule& p) { return p.bytecode.size(); })
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
        nb::arg("stim_text"), "Compile a quantum circuit to executable bytecode.");

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
        "Run a compiled program and return all result records.");

    // =========================================================================
    // Statevector API
    // =========================================================================

    nb::class_<ucc::SchrodingerState>(m, "State", "Schrodinger VM execution state")
        .def(nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint64_t>(), nb::arg("peak_rank"),
             nb::arg("num_measurements"), nb::arg("num_detectors") = 0,
             nb::arg("num_observables") = 0, nb::arg("seed") = 0)
        .def("reset", &ucc::SchrodingerState::reset, nb::arg("seed"))
        .def_prop_ro(
            "meas_record",
            [](const ucc::SchrodingerState& s) { return std::vector<uint8_t>(s.meas_record); })
        .def_prop_ro(
            "det_record",
            [](const ucc::SchrodingerState& s) { return std::vector<uint8_t>(s.det_record); })
        .def_prop_ro(
            "obs_record",
            [](const ucc::SchrodingerState& s) { return std::vector<uint8_t>(s.obs_record); })
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
        "Execute a compiled program, updating the state in-place.");

    // Get statevector: Program + State -> numpy array
    m.def(
        "get_statevector",
        [](const ucc::CompiledModule& program, const ucc::SchrodingerState& state) {
            auto sv = ucc::get_statevector(program, state);

            size_t n = sv.size();
            auto* data = new std::complex<double>[n];
            std::copy(sv.begin(), sv.end(), data);

            nb::capsule owner(
                data, [](void* p) noexcept { delete[] static_cast<std::complex<double>*>(p); });

            return nb::ndarray<nb::numpy, std::complex<double>, nb::c_contig>(data, {n}, owner);
        },
        nb::arg("program"), nb::arg("state"), "Expand the SVM state into a dense statevector.");
}
