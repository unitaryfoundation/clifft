#include "ucc/backend/backend.h"
#include "ucc/circuit/circuit.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/pass_manager.h"
#include "ucc/optimizer/peephole.h"
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
        // Single-qubit Cliffords
        .value("H", ucc::GateType::H)
        .value("S", ucc::GateType::S)
        .value("S_DAG", ucc::GateType::S_DAG)
        .value("X", ucc::GateType::X)
        .value("Y", ucc::GateType::Y)
        .value("Z", ucc::GateType::Z)
        .value("SQRT_X", ucc::GateType::SQRT_X)
        .value("SQRT_X_DAG", ucc::GateType::SQRT_X_DAG)
        .value("SQRT_Y", ucc::GateType::SQRT_Y)
        .value("SQRT_Y_DAG", ucc::GateType::SQRT_Y_DAG)
        .value("H_XY", ucc::GateType::H_XY)
        .value("H_YZ", ucc::GateType::H_YZ)
        .value("H_NXY", ucc::GateType::H_NXY)
        .value("H_NXZ", ucc::GateType::H_NXZ)
        .value("H_NYZ", ucc::GateType::H_NYZ)
        .value("C_XYZ", ucc::GateType::C_XYZ)
        .value("C_ZYX", ucc::GateType::C_ZYX)
        .value("C_NXYZ", ucc::GateType::C_NXYZ)
        .value("C_NZYX", ucc::GateType::C_NZYX)
        .value("C_XNYZ", ucc::GateType::C_XNYZ)
        .value("C_XYNZ", ucc::GateType::C_XYNZ)
        .value("C_ZNYX", ucc::GateType::C_ZNYX)
        .value("C_ZYNX", ucc::GateType::C_ZYNX)
        // Non-Clifford
        .value("T", ucc::GateType::T)
        .value("T_DAG", ucc::GateType::T_DAG)
        // Two-qubit Cliffords
        .value("CX", ucc::GateType::CX)
        .value("CY", ucc::GateType::CY)
        .value("CZ", ucc::GateType::CZ)
        .value("SWAP", ucc::GateType::SWAP)
        .value("ISWAP", ucc::GateType::ISWAP)
        .value("ISWAP_DAG", ucc::GateType::ISWAP_DAG)
        .value("SQRT_XX", ucc::GateType::SQRT_XX)
        .value("SQRT_XX_DAG", ucc::GateType::SQRT_XX_DAG)
        .value("SQRT_YY", ucc::GateType::SQRT_YY)
        .value("SQRT_YY_DAG", ucc::GateType::SQRT_YY_DAG)
        .value("SQRT_ZZ", ucc::GateType::SQRT_ZZ)
        .value("SQRT_ZZ_DAG", ucc::GateType::SQRT_ZZ_DAG)
        .value("CXSWAP", ucc::GateType::CXSWAP)
        .value("CZSWAP", ucc::GateType::CZSWAP)
        .value("SWAPCX", ucc::GateType::SWAPCX)
        .value("XCX", ucc::GateType::XCX)
        .value("XCY", ucc::GateType::XCY)
        .value("XCZ", ucc::GateType::XCZ)
        .value("YCX", ucc::GateType::YCX)
        .value("YCY", ucc::GateType::YCY)
        .value("YCZ", ucc::GateType::YCZ)
        // Measurements
        .value("M", ucc::GateType::M)
        .value("MX", ucc::GateType::MX)
        .value("MY", ucc::GateType::MY)
        .value("MR", ucc::GateType::MR)
        .value("MRX", ucc::GateType::MRX)
        .value("MRY", ucc::GateType::MRY)
        .value("MPP", ucc::GateType::MPP)
        .value("MXX", ucc::GateType::MXX)
        .value("MYY", ucc::GateType::MYY)
        .value("MZZ", ucc::GateType::MZZ)
        // Resets
        .value("R", ucc::GateType::R)
        .value("RX", ucc::GateType::RX)
        .value("RY", ucc::GateType::RY)
        // Padding
        .value("MPAD", ucc::GateType::MPAD)
        // Identity no-ops
        .value("I", ucc::GateType::I)
        .value("II", ucc::GateType::II)
        .value("I_ERROR", ucc::GateType::I_ERROR)
        .value("II_ERROR", ucc::GateType::II_ERROR)
        // Noise
        .value("X_ERROR", ucc::GateType::X_ERROR)
        .value("Y_ERROR", ucc::GateType::Y_ERROR)
        .value("Z_ERROR", ucc::GateType::Z_ERROR)
        .value("DEPOLARIZE1", ucc::GateType::DEPOLARIZE1)
        .value("DEPOLARIZE2", ucc::GateType::DEPOLARIZE2)
        .value("PAULI_CHANNEL_1", ucc::GateType::PAULI_CHANNEL_1)
        .value("PAULI_CHANNEL_2", ucc::GateType::PAULI_CHANNEL_2)
        .value("READOUT_NOISE", ucc::GateType::READOUT_NOISE)
        // Annotations
        .value("DETECTOR", ucc::GateType::DETECTOR)
        .value("OBSERVABLE_INCLUDE", ucc::GateType::OBSERVABLE_INCLUDE)
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
        .def_prop_ro("arg", [](const ucc::AstNode& n) { return n.args.empty() ? 0.0 : n.args[0]; })
        .def_ro("args", &ucc::AstNode::args)
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
    m.def(
        "parse", [](std::string_view text) { return ucc::parse(text); }, nb::arg("text"),
        "Parse a quantum circuit from a string.");
    m.def(
        "parse", [](std::string_view text, size_t max_ops) { return ucc::parse(text, max_ops); },
        nb::arg("text"), nb::arg("max_ops"),
        "Parse a quantum circuit from a string with an explicit AST node limit.");
    m.def(
        "parse_file", [](const std::string& path) { return ucc::parse_file(path); },
        nb::arg("path"), "Parse a quantum circuit from a file.");
    m.def(
        "parse_file",
        [](const std::string& path, size_t max_ops) { return ucc::parse_file(path, max_ops); },
        nb::arg("path"), nb::arg("max_ops"),
        "Parse a quantum circuit from a file with an explicit AST node limit.");

    // =========================================================================
    // Heisenberg IR
    // =========================================================================

    nb::class_<ucc::HirModule>(m, "HirModule", "Heisenberg Intermediate Representation")
        .def_prop_ro("num_ops", [](const ucc::HirModule& h) { return h.num_ops(); })
        .def_prop_ro("num_t_gates", [](const ucc::HirModule& h) { return h.num_t_gates(); })
        .def_prop_ro("num_qubits", [](const ucc::HirModule& h) { return h.num_qubits; })
        .def_prop_ro("num_measurements", [](const ucc::HirModule& h) { return h.num_measurements; })
        .def_prop_ro("num_detectors", [](const ucc::HirModule& h) { return h.num_detectors; })
        .def_prop_ro("num_observables", [](const ucc::HirModule& h) { return h.num_observables; })
        .def("__repr__", [](const ucc::HirModule& h) {
            return "HirModule(" + std::to_string(h.num_ops()) + " ops, " +
                   std::to_string(h.num_t_gates()) + " T-gates, " + std::to_string(h.num_qubits) +
                   " qubits)";
        });

    // Front-end: Circuit -> HIR
    m.def(
        "trace", [](const ucc::Circuit& circuit) { return ucc::trace(circuit); },
        nb::arg("circuit"),
        "Trace a parsed circuit through the Clifford front-end to produce the "
        "Heisenberg IR.");

    // =========================================================================
    // Optimizer Pass Infrastructure
    // =========================================================================

    nb::class_<ucc::Pass>(m, "Pass", "Abstract base class for HIR optimization passes.");

    nb::class_<ucc::PeepholeFusionPass, ucc::Pass>(
        m, "PeepholeFusionPass",
        "Symplectic peephole fusion: cancels and fuses T/T-dag gates on the "
        "same virtual Pauli axis.")
        .def(nb::init<>())
        .def_prop_ro("cancellations", &ucc::PeepholeFusionPass::cancellations)
        .def_prop_ro("fusions", &ucc::PeepholeFusionPass::fusions)
        .def("__repr__", [](const ucc::PeepholeFusionPass& p) {
            return "PeepholeFusionPass(cancellations=" + std::to_string(p.cancellations()) +
                   ", fusions=" + std::to_string(p.fusions()) + ")";
        });

    nb::class_<ucc::PassManager>(m, "PassManager",
                                 "Runs a sequence of optimization passes over an HirModule.")
        .def(nb::init<>())
        .def(
            "add",
            [](ucc::PassManager& pm, ucc::Pass& pass) {
                // PassManager needs unique_ptr ownership, but Python owns the pass.
                // Use a thin non-owning wrapper that delegates to the Python-owned pass.
                struct BorrowedPass : ucc::Pass {
                    ucc::Pass& ref;
                    explicit BorrowedPass(ucc::Pass& r) : ref(r) {}
                    void run(ucc::HirModule& hir) override { ref.run(hir); }
                };
                pm.add_pass(std::make_unique<BorrowedPass>(pass));
            },
            nb::arg("pass"), nb::keep_alive<1, 2>(),
            "Add an optimization pass. Passes run in the order added.")
        .def(
            "run", [](ucc::PassManager& pm, ucc::HirModule& hir) { pm.run(hir); }, nb::arg("hir"),
            "Run all passes on the HIR module in sequence.");

    m.def(
        "default_pass_manager",
        []() {
            ucc::PassManager pm;
            pm.add_pass(std::make_unique<ucc::PeepholeFusionPass>());
            return pm;
        },
        nb::rv_policy::move,
        "Return a PassManager pre-loaded with the standard optimization passes.");

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

    // Back-end: HIR -> Program
    m.def(
        "lower", [](const ucc::HirModule& hir) { return ucc::lower(hir); }, nb::arg("hir"),
        "Lower a Heisenberg IR module to executable VM bytecode.");

    // Convenience: stim text -> Program (parse + trace + lower, no optimization)
    m.def(
        "compile",
        [](const std::string& stim_text) {
            ucc::Circuit circuit = ucc::parse(stim_text);
            ucc::HirModule hir = ucc::trace(circuit);
            return ucc::lower(hir);
        },
        nb::arg("stim_text"),
        "Compile a quantum circuit string to executable bytecode.\n\n"
        "Convenience function equivalent to lower(trace(parse(text))).\n"
        "For optimization, use the explicit pipeline: parse -> trace -> "
        "PassManager.run -> lower.\n");

    // Zero-copy transfer: move the vector onto the heap and let the capsule own it.
    // Avoids O(N) memcpy for large shot batches.
    auto make_numpy_array = [](std::vector<uint8_t> vec, size_t rows, size_t cols) {
        auto* owner_vec = new std::vector<uint8_t>(std::move(vec));
        nb::capsule owner(owner_vec,
                          [](void* p) noexcept { delete static_cast<std::vector<uint8_t>*>(p); });
        return nb::ndarray<nb::numpy, uint8_t, nb::c_contig>(owner_vec->data(), {rows, cols},
                                                             owner);
    };

    // Sample: Program + shots -> tuple of (measurements, detectors, observables)
    m.def(
        "sample",
        [make_numpy_array](const ucc::CompiledModule& program, uint32_t shots, uint64_t seed) {
            ucc::SampleResult result = ucc::sample(program, shots, seed);

            auto meas_arr =
                make_numpy_array(std::move(result.measurements), shots, program.num_measurements);
            auto det_arr =
                make_numpy_array(std::move(result.detectors), shots, program.num_detectors);
            auto obs_arr =
                make_numpy_array(std::move(result.observables), shots, program.num_observables);

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
