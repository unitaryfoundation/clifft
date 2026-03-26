#include "ucc/backend/backend.h"
#include "ucc/backend/reference_syndrome.h"
#include "ucc/circuit/circuit.h"
#include "ucc/circuit/parser.h"
#include "ucc/frontend/frontend.h"
#include "ucc/optimizer/bytecode_pass.h"
#include "ucc/optimizer/expand_t_pass.h"
#include "ucc/optimizer/hir_pass_manager.h"
#include "ucc/optimizer/multi_gate_pass.h"
#include "ucc/optimizer/noise_block_pass.h"
#include "ucc/optimizer/pass_factory.h"
#include "ucc/optimizer/peephole.h"
#include "ucc/optimizer/remove_noise_pass.h"
#include "ucc/optimizer/single_axis_fusion_pass.h"
#include "ucc/optimizer/statevector_squeeze_pass.h"
#include "ucc/optimizer/swap_meas_pass.h"
#include "ucc/optimizer/tile_axis_fusion_pass.h"
#include "ucc/svm/svm.h"
#include "ucc/util/config.h"
#include "ucc/util/introspection.h"
#include "ucc/util/version.h"

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <sstream>

namespace nb = nanobind;

// Zero-copy transfer: move a std::vector into a numpy array via capsule ownership.
// Uses unique_ptr for exception safety: if capsule construction throws,
// the vector is automatically freed. Ownership transfers to the capsule
// via release() only after the capsule is successfully constructed.
template <typename T>
nb::ndarray<nb::numpy, T, nb::c_contig> vec_to_numpy(std::vector<T> vec,
                                                     std::initializer_list<size_t> shape) {
    auto owner_ptr = std::make_unique<std::vector<T>>(std::move(vec));
    T* data = owner_ptr->data();
    nb::capsule owner(owner_ptr.release(),
                      [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, shape, owner);
}

NB_MODULE(_ucc_core, m) {
    m.doc() = "UCC core C++ extension module";

    // Register ParseError as a Python exception.
    nb::exception<ucc::ParseError>(m, "ParseError");

    // Version info
    m.def("version", []() { return ucc::kVersion; }, "Return the UCC version string");

    m.def(
        "max_sim_qubits", []() { return ucc::kMaxInlineQubits; },
        "Return the maximum number of qubits supported by the simulator");

    m.def(
        "svm_backend", []() { return ucc::svm_backend(); },
        "Return the active SVM dispatch backend: 'avx512', 'avx2', or 'scalar'.\n\n"
        "Reflects the resolved CPUID path or UCC_FORCE_ISA environment override.");

    // Sentinel-based enum counts for defensive binding tests.
    // If a new enum value is added in C++ but not bound in Python,
    // the test_introspection.py tripwire will catch it.
    m.def("_num_optypes", []() { return static_cast<int>(ucc::OpType::NUM_OP_TYPES); });
    m.def("_num_opcodes", []() { return static_cast<int>(ucc::Opcode::NUM_OPCODES); });

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
        // Parameterized rotations
        .value("R_X", ucc::GateType::R_X)
        .value("R_Y", ucc::GateType::R_Y)
        .value("R_Z", ucc::GateType::R_Z)
        .value("U3", ucc::GateType::U3)
        .value("R_XX", ucc::GateType::R_XX)
        .value("R_YY", ucc::GateType::R_YY)
        .value("R_ZZ", ucc::GateType::R_ZZ)
        .value("R_PAULI", ucc::GateType::R_PAULI)
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

    nb::class_<ucc::AstNode>(m, "AstNode", "A single circuit operation")
        .def_ro("gate", &ucc::AstNode::gate)
        .def_ro("targets", &ucc::AstNode::targets)
        .def_prop_ro("arg", [](const ucc::AstNode& n) { return n.args.empty() ? 0.0 : n.args[0]; })
        .def_ro("args", &ucc::AstNode::args)
        .def_ro("source_line", &ucc::AstNode::source_line)
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

    m.def(
        "parse",
        [](std::string_view text) {
            nb::gil_scoped_release release;
            return ucc::parse(text);
        },
        nb::arg("text"), "Parse a quantum circuit from a string.");
    m.def(
        "parse",
        [](std::string_view text, size_t max_ops) {
            nb::gil_scoped_release release;
            return ucc::parse(text, max_ops);
        },
        nb::arg("text"), nb::arg("max_ops"),
        "Parse a quantum circuit from a string with an explicit AST node limit.");
    m.def(
        "parse_file",
        [](const std::string& path) {
            nb::gil_scoped_release release;
            return ucc::parse_file(path);
        },
        nb::arg("path"), "Parse a quantum circuit from a file.");
    m.def(
        "parse_file",
        [](const std::string& path, size_t max_ops) {
            nb::gil_scoped_release release;
            return ucc::parse_file(path, max_ops);
        },
        nb::arg("path"), nb::arg("max_ops"),
        "Parse a quantum circuit from a file with an explicit AST node limit.");

    // =========================================================================
    nb::enum_<ucc::OpType>(m, "OpType", "Heisenberg IR operation types")
        .value("T_GATE", ucc::OpType::T_GATE)
        .value("MEASURE", ucc::OpType::MEASURE)
        .value("CONDITIONAL_PAULI", ucc::OpType::CONDITIONAL_PAULI)
        .value("NOISE", ucc::OpType::NOISE)
        .value("READOUT_NOISE", ucc::OpType::READOUT_NOISE)
        .value("PHASE_ROTATION", ucc::OpType::PHASE_ROTATION)
        .value("DETECTOR", ucc::OpType::DETECTOR)
        .value("OBSERVABLE", ucc::OpType::OBSERVABLE)
        .value("EXP_VAL", ucc::OpType::EXP_VAL);

    nb::class_<ucc::HeisenbergOp>(m, "HeisenbergOp",
                                  "A single abstract operation in the Heisenberg IR")
        .def_prop_ro("op_type", [](const ucc::HeisenbergOp& op) { return op.op_type(); })
        .def_prop_ro("is_dagger", [](const ucc::HeisenbergOp& op) { return op.is_dagger(); })
        .def_prop_ro("is_hidden", [](const ucc::HeisenbergOp& op) { return op.is_hidden(); })
        .def_prop_ro("sign", [](const ucc::HeisenbergOp& op) { return op.sign(); })
        .def_prop_ro("pauli_string",
                     [](const ucc::HeisenbergOp& op) { return ucc::format_pauli_mask(op); })
        .def(
            "as_dict",
            [](const ucc::HeisenbergOp& op) {
                nb::dict d;
                d["op_type"] = ucc::op_type_to_str(op.op_type());
                d["pauli_string"] = ucc::format_pauli_mask(op);
                d["is_dagger"] = op.is_dagger();
                d["is_hidden"] = op.is_hidden();
                d["sign"] = op.sign();

                switch (op.op_type()) {
                    case ucc::OpType::MEASURE:
                        d["meas_record_idx"] = static_cast<uint32_t>(op.meas_record_idx());
                        break;
                    case ucc::OpType::CONDITIONAL_PAULI:
                        d["controlling_meas"] = static_cast<uint32_t>(op.controlling_meas());
                        break;
                    case ucc::OpType::NOISE:
                        d["noise_site_idx"] = static_cast<uint32_t>(op.noise_site_idx());
                        break;
                    case ucc::OpType::READOUT_NOISE:
                        d["readout_noise_idx"] = static_cast<uint32_t>(op.readout_noise_idx());
                        break;
                    case ucc::OpType::DETECTOR:
                        d["detector_idx"] = static_cast<uint32_t>(op.detector_idx());
                        break;
                    case ucc::OpType::OBSERVABLE:
                        d["observable_idx"] = static_cast<uint32_t>(op.observable_idx());
                        d["observable_target_list_idx"] = op.observable_target_list_idx();
                        break;
                    case ucc::OpType::PHASE_ROTATION:
                        d["alpha"] = op.alpha();
                        break;
                    case ucc::OpType::EXP_VAL:
                        d["exp_val_idx"] = static_cast<uint32_t>(op.exp_val_idx());
                        break;
                    default:
                        break;
                }
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__", [](const ucc::HeisenbergOp& op) { return ucc::format_hir_op(op); })
        .def("__repr__", [](const ucc::HeisenbergOp& op) {
            return "<HeisenbergOp: " + ucc::format_hir_op(op) + ">";
        });

    nb::class_<ucc::HirModule>(m, "HirModule", "Heisenberg Intermediate Representation")
        .def_prop_ro("num_ops", [](const ucc::HirModule& h) { return h.num_ops(); })
        .def_prop_ro("num_t_gates", [](const ucc::HirModule& h) { return h.num_t_gates(); })
        .def_prop_ro("num_qubits", [](const ucc::HirModule& h) { return h.num_qubits; })
        .def_prop_ro("num_measurements", [](const ucc::HirModule& h) { return h.num_measurements; })
        .def_prop_ro("num_detectors", [](const ucc::HirModule& h) { return h.num_detectors; })
        .def_prop_ro("num_observables", [](const ucc::HirModule& h) { return h.num_observables; })
        .def_prop_ro(
            "source_map",
            [](const ucc::HirModule& h) {
                nb::list outer;
                for (const auto& lines : h.source_map)
                    outer.append(nb::cast(lines));
                return outer;
            },
            "Source line mapping parallel to ops (list of list of uint32).")
        .def(
            "__len__", [](const ucc::HirModule& h) { return h.ops.size(); },
            "Return the number of HIR operations.")
        .def(
            "__getitem__",
            [](const ucc::HirModule& h, int64_t idx) -> const ucc::HeisenbergOp& {
                int64_t size = static_cast<int64_t>(h.ops.size());
                if (idx < 0)
                    idx += size;
                if (idx < 0 || idx >= size)
                    throw nb::index_error();
                return h.ops[static_cast<size_t>(idx)];
            },
            nb::rv_policy::reference_internal, "Return the HIR operation at the given index.")
        .def(
            "__iter__",
            [](const ucc::HirModule& h) {
                return nb::make_iterator(nb::type<ucc::HirModule>(), "hir_iter", h.ops.begin(),
                                         h.ops.end());
            },
            nb::keep_alive<0, 1>())
        .def(
            "as_dict",
            [](const ucc::HirModule& h) {
                nb::dict d;
                d["num_qubits"] = h.num_qubits;
                d["num_measurements"] = h.num_measurements;
                d["num_detectors"] = h.num_detectors;
                d["num_observables"] = h.num_observables;
                nb::list ops;
                for (const auto& op : h.ops)
                    ops.append(nb::cast(op).attr("as_dict")());
                d["ops"] = ops;
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__",
             [](const ucc::HirModule& h) {
                 std::ostringstream ss;
                 for (size_t i = 0; i < h.ops.size(); ++i)
                     ss << i << ": " << ucc::format_hir_op(h.ops[i]) << "\n";
                 return ss.str();
             })
        .def("__repr__", [](const ucc::HirModule& h) {
            return "HirModule(" + std::to_string(h.num_ops()) + " ops, " +
                   std::to_string(h.num_t_gates()) + " T-gates, " + std::to_string(h.num_qubits) +
                   " qubits)";
        });

    // Front-end: Circuit -> HIR
    m.def(
        "trace",
        [](const ucc::Circuit& circuit) {
            nb::gil_scoped_release release;
            return ucc::trace(circuit);
        },
        nb::arg("circuit"),
        "Trace a parsed circuit through the Clifford front-end to produce the "
        "Heisenberg IR.");

    nb::class_<ucc::HirPass>(m, "HirPass", "Abstract base class for HIR optimization passes.");

    nb::class_<ucc::PeepholeFusionPass, ucc::HirPass>(
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

    nb::class_<ucc::StatevectorSqueezePass, ucc::HirPass>(
        m, "StatevectorSqueezePass",
        "Bidirectional bubble sort: moves measurements leftward and\n"
        "non-Clifford gates rightward to minimize peak active rank.")
        .def(nb::init<>());

    nb::class_<ucc::RemoveNoisePass, ucc::HirPass>(
        m, "RemoveNoisePass",
        "Strips all stochastic noise and readout noise ops from the HIR.\n"
        "Not included in the default pass list. Used internally by\n"
        "compute_reference_syndrome() for noiseless reference shots.")
        .def(nb::init<>());

    m.def(
        "compute_reference_syndrome",
        [](const ucc::HirModule& hir) {
            ucc::ReferenceSyndrome ref;
            {
                nb::gil_scoped_release release;
                ref = ucc::compute_reference_syndrome(hir);
            }
            nb::dict d;
            d["detectors"] = nb::cast(std::move(ref.detectors));
            d["observables"] = nb::cast(std::move(ref.observables));
            return d;
        },
        nb::arg("hir"),
        "Compute noiseless reference syndrome for an HirModule.\n\n"
        "Returns a dict with 'detectors' and 'observables' lists.");

    nb::class_<ucc::HirPassManager>(m, "HirPassManager",
                                    "Runs a sequence of optimization passes over an HirModule.")
        .def(nb::init<>())
        .def(
            "add",
            [](ucc::HirPassManager& pm, ucc::HirPass& pass) {
                // HirPassManager needs unique_ptr ownership, but Python owns the pass.
                // Use a thin non-owning wrapper that delegates to the Python-owned pass.
                struct BorrowedPass : ucc::HirPass {
                    ucc::HirPass& ref;
                    explicit BorrowedPass(ucc::HirPass& r) : ref(r) {}
                    void run(ucc::HirModule& hir) override { ref.run(hir); }
                };
                pm.add_pass(std::make_unique<BorrowedPass>(pass));
            },
            nb::arg("pass"), nb::keep_alive<1, 2>(),
            "Add an optimization pass. Passes run in the order added.")
        .def(
            "run", [](ucc::HirPassManager& pm, ucc::HirModule& hir) { pm.run(hir); },
            nb::arg("hir"), "Run all passes on the HIR module in sequence.");

    m.def(
        "default_hir_pass_manager", []() { return ucc::default_hir_pass_manager(); },
        nb::rv_policy::move,
        "Return an HirPassManager pre-loaded with the standard optimization passes.");

    nb::class_<ucc::BytecodePass>(m, "BytecodePass",
                                  "Abstract base class for bytecode optimization passes.\n\n"
                                  "Each pass receives a mutable Program and may rewrite,\n"
                                  "reorder, or remove instructions.");

    nb::class_<ucc::NoiseBlockPass, ucc::BytecodePass>(
        m, "NoiseBlockPass", "Coalesces contiguous OP_NOISE instructions into OP_NOISE_BLOCK.")
        .def(nb::init<>());

    nb::class_<ucc::ExpandTPass, ucc::BytecodePass>(
        m, "ExpandTPass", "Fuses OP_EXPAND + OP_PHASE_T into single OP_EXPAND_T instructions.")
        .def(nb::init<>());

    nb::class_<ucc::ExpandRotPass, ucc::BytecodePass>(
        m, "ExpandRotPass",
        "Fuses OP_EXPAND + OP_PHASE_ROT into single OP_EXPAND_ROT instructions.")
        .def(nb::init<>());

    nb::class_<ucc::SwapMeasPass, ucc::BytecodePass>(
        m, "SwapMeasPass",
        "Fuses OP_ARRAY_SWAP + OP_MEAS_ACTIVE_INTERFERE into OP_SWAP_MEAS_INTERFERE.")
        .def(nb::init<>());

    nb::class_<ucc::MultiGatePass, ucc::BytecodePass>(
        m, "MultiGatePass", "Fuses sequences of same-type 2-qubit gates into multi-target ops.")
        .def(nb::init<>());

    nb::class_<ucc::TileAxisFusionPass, ucc::BytecodePass>(
        m, "TileAxisFusionPass",
        "Fuses 2-qubit tile sequences into OP_ARRAY_U4 with precomputed 4x4 matrices.")
        .def(nb::init<>());

    nb::class_<ucc::SingleAxisFusionPass, ucc::BytecodePass>(
        m, "SingleAxisFusionPass",
        "Fuses consecutive single-axis ops into OP_ARRAY_U2 with precomputed 2x2 matrices.")
        .def(nb::init<>());

    nb::class_<ucc::BytecodePassManager>(m, "BytecodePassManager",
                                         "Runs a sequence of bytecode optimization passes "
                                         "over a Program.")
        .def(nb::init<>())
        .def(
            "add",
            [](ucc::BytecodePassManager& bpm, ucc::BytecodePass& pass) {
                struct BorrowedBytecodePass : ucc::BytecodePass {
                    ucc::BytecodePass& ref;
                    explicit BorrowedBytecodePass(ucc::BytecodePass& r) : ref(r) {}
                    void run(ucc::CompiledModule& mod) override { ref.run(mod); }
                };
                bpm.add_pass(std::make_unique<BorrowedBytecodePass>(pass));
            },
            nb::arg("pass"), nb::keep_alive<1, 2>(),
            "Add a bytecode optimization pass. Passes run in the order added.")
        .def(
            "run", [](ucc::BytecodePassManager& bpm, ucc::CompiledModule& mod) { bpm.run(mod); },
            nb::arg("program"), "Run all bytecode passes on the program in sequence.");

    m.def(
        "default_bytecode_pass_manager", []() { return ucc::default_bytecode_pass_manager(); },
        nb::rv_policy::move,
        "Return a BytecodePassManager pre-loaded with the default passes:\n"
        "NoiseBlockPass, MultiGatePass, ExpandTPass, ExpandRotPass, SwapMeasPass,\n"
        "SingleAxisFusionPass.");

    nb::enum_<ucc::Opcode>(m, "Opcode", "RISC Virtual Machine opcodes")
        .value("OP_FRAME_CNOT", ucc::Opcode::OP_FRAME_CNOT)
        .value("OP_FRAME_CZ", ucc::Opcode::OP_FRAME_CZ)
        .value("OP_FRAME_H", ucc::Opcode::OP_FRAME_H)
        .value("OP_FRAME_S", ucc::Opcode::OP_FRAME_S)
        .value("OP_FRAME_S_DAG", ucc::Opcode::OP_FRAME_S_DAG)
        .value("OP_FRAME_SWAP", ucc::Opcode::OP_FRAME_SWAP)
        .value("OP_ARRAY_CNOT", ucc::Opcode::OP_ARRAY_CNOT)
        .value("OP_ARRAY_CZ", ucc::Opcode::OP_ARRAY_CZ)
        .value("OP_ARRAY_SWAP", ucc::Opcode::OP_ARRAY_SWAP)
        .value("OP_ARRAY_MULTI_CNOT", ucc::Opcode::OP_ARRAY_MULTI_CNOT)
        .value("OP_ARRAY_MULTI_CZ", ucc::Opcode::OP_ARRAY_MULTI_CZ)
        .value("OP_ARRAY_H", ucc::Opcode::OP_ARRAY_H)
        .value("OP_ARRAY_S", ucc::Opcode::OP_ARRAY_S)
        .value("OP_ARRAY_S_DAG", ucc::Opcode::OP_ARRAY_S_DAG)
        .value("OP_EXPAND", ucc::Opcode::OP_EXPAND)
        .value("OP_PHASE_T", ucc::Opcode::OP_PHASE_T)
        .value("OP_PHASE_T_DAG", ucc::Opcode::OP_PHASE_T_DAG)
        .value("OP_EXPAND_T", ucc::Opcode::OP_EXPAND_T)
        .value("OP_EXPAND_T_DAG", ucc::Opcode::OP_EXPAND_T_DAG)
        .value("OP_PHASE_ROT", ucc::Opcode::OP_PHASE_ROT)
        .value("OP_EXPAND_ROT", ucc::Opcode::OP_EXPAND_ROT)
        .value("OP_ARRAY_U2", ucc::Opcode::OP_ARRAY_U2)
        .value("OP_ARRAY_U4", ucc::Opcode::OP_ARRAY_U4)
        .value("OP_MEAS_DORMANT_STATIC", ucc::Opcode::OP_MEAS_DORMANT_STATIC)
        .value("OP_MEAS_DORMANT_RANDOM", ucc::Opcode::OP_MEAS_DORMANT_RANDOM)
        .value("OP_MEAS_ACTIVE_DIAGONAL", ucc::Opcode::OP_MEAS_ACTIVE_DIAGONAL)
        .value("OP_MEAS_ACTIVE_INTERFERE", ucc::Opcode::OP_MEAS_ACTIVE_INTERFERE)
        .value("OP_SWAP_MEAS_INTERFERE", ucc::Opcode::OP_SWAP_MEAS_INTERFERE)
        .value("OP_APPLY_PAULI", ucc::Opcode::OP_APPLY_PAULI)
        .value("OP_NOISE", ucc::Opcode::OP_NOISE)
        .value("OP_NOISE_BLOCK", ucc::Opcode::OP_NOISE_BLOCK)
        .value("OP_READOUT_NOISE", ucc::Opcode::OP_READOUT_NOISE)
        .value("OP_DETECTOR", ucc::Opcode::OP_DETECTOR)
        .value("OP_POSTSELECT", ucc::Opcode::OP_POSTSELECT)
        .value("OP_OBSERVABLE", ucc::Opcode::OP_OBSERVABLE);

    nb::class_<ucc::Instruction>(m, "Instruction", "A localized RISC operation for the VM")
        .def_prop_ro("opcode", [](const ucc::Instruction& i) { return i.opcode; })
        .def_prop_ro("flags", [](const ucc::Instruction& i) { return i.flags; })
        .def_prop_ro("axis_1", [](const ucc::Instruction& i) { return i.axis_1; })
        .def_prop_ro("axis_2", [](const ucc::Instruction& i) { return i.axis_2; })
        .def(
            "as_dict",
            [](const ucc::Instruction& i) {
                nb::dict d;
                d["opcode"] = ucc::opcode_to_str(i.opcode);
                d["axis_1"] = i.axis_1;
                d["axis_2"] = i.axis_2;
                d["flags"] = i.flags;
                d["description"] = ucc::format_instruction(i);

                if (ucc::is_meas_opcode(i.opcode)) {
                    d["classical_idx"] = i.classical.classical_idx;
                    d["expected_val"] = i.classical.expected_val;
                } else if (i.opcode == ucc::Opcode::OP_APPLY_PAULI) {
                    d["cp_mask_idx"] = i.pauli.cp_mask_idx;
                    d["condition_idx"] = i.pauli.condition_idx;
                } else if (i.opcode == ucc::Opcode::OP_DETECTOR ||
                           i.opcode == ucc::Opcode::OP_POSTSELECT) {
                    d["target_list_index"] = i.pauli.cp_mask_idx;
                    d["detector_index"] = i.pauli.condition_idx;
                } else if (i.opcode == ucc::Opcode::OP_OBSERVABLE) {
                    d["target_list_index"] = i.pauli.cp_mask_idx;
                    d["observable_index"] = i.pauli.condition_idx;
                } else if (i.opcode == ucc::Opcode::OP_NOISE) {
                    d["noise_site_idx"] = i.pauli.cp_mask_idx;
                } else if (i.opcode == ucc::Opcode::OP_NOISE_BLOCK) {
                    d["start_site"] = i.pauli.cp_mask_idx;
                    d["count"] = i.pauli.condition_idx;
                } else if (i.opcode == ucc::Opcode::OP_READOUT_NOISE) {
                    d["readout_noise_idx"] = i.pauli.cp_mask_idx;
                } else if (i.opcode == ucc::Opcode::OP_ARRAY_MULTI_CNOT ||
                           i.opcode == ucc::Opcode::OP_ARRAY_MULTI_CZ) {
                    d["mask"] = i.multi_gate.mask;
                } else if (i.opcode == ucc::Opcode::OP_PHASE_ROT ||
                           i.opcode == ucc::Opcode::OP_EXPAND_ROT) {
                    d["weight_re"] = i.math.weight_re;
                    d["weight_im"] = i.math.weight_im;
                } else if (i.opcode == ucc::Opcode::OP_ARRAY_U2) {
                    d["cp_idx"] = i.u2.cp_idx;
                }
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__", [](const ucc::Instruction& inst) { return ucc::format_instruction(inst); })
        .def("__repr__", [](const ucc::Instruction& inst) {
            return "<Instruction: " + ucc::format_instruction(inst) + ">";
        });

    nb::class_<ucc::CompiledModule>(m, "Program", "A compiled quantum program")
        .def_prop_ro("peak_rank", [](const ucc::CompiledModule& p) { return p.peak_rank; })
        .def_prop_ro("num_measurements",
                     [](const ucc::CompiledModule& p) { return p.num_measurements; })
        .def_prop_ro("num_detectors", [](const ucc::CompiledModule& p) { return p.num_detectors; })
        .def_prop_ro("num_observables",
                     [](const ucc::CompiledModule& p) { return p.num_observables; })
        .def_prop_ro("num_instructions",
                     [](const ucc::CompiledModule& p) { return p.bytecode.size(); })
        .def_prop_ro(
            "source_map",
            [](const ucc::CompiledModule& p) {
                nb::list outer;
                for (size_t i = 0; i < p.source_map.size(); ++i) {
                    auto lines = p.source_map.lines_for(i);
                    outer.append(nb::cast(std::vector<uint32_t>(lines.begin(), lines.end())));
                }
                return outer;
            },
            "Source line mapping parallel to bytecode (list of list of uint32).")
        .def_prop_ro(
            "active_k_history",
            [](const ucc::CompiledModule& p) { return nb::cast(p.source_map.active_k_history()); },
            "Active dimension k after each instruction (list of uint32).")
        .def(
            "__len__", [](const ucc::CompiledModule& p) { return p.bytecode.size(); },
            "Return the number of bytecode instructions.")
        .def(
            "__getitem__",
            [](const ucc::CompiledModule& p, int64_t idx) -> const ucc::Instruction& {
                int64_t size = static_cast<int64_t>(p.bytecode.size());
                if (idx < 0)
                    idx += size;
                if (idx < 0 || idx >= size)
                    throw nb::index_error();
                return p.bytecode[static_cast<size_t>(idx)];
            },
            nb::rv_policy::reference_internal, "Return the instruction at the given index.")
        .def(
            "__iter__",
            [](const ucc::CompiledModule& p) {
                return nb::make_iterator(nb::type<ucc::CompiledModule>(), "program_iter",
                                         p.bytecode.begin(), p.bytecode.end());
            },
            nb::keep_alive<0, 1>())
        .def(
            "as_dict",
            [](const ucc::CompiledModule& p) {
                nb::dict d;
                d["peak_rank"] = p.peak_rank;
                d["num_qubits"] = p.num_qubits;
                d["num_measurements"] = p.num_measurements;
                d["num_detectors"] = p.num_detectors;
                d["num_observables"] = p.num_observables;
                nb::list bytecode;
                for (const auto& i : p.bytecode)
                    bytecode.append(nb::cast(i).attr("as_dict")());
                d["bytecode"] = bytecode;
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__",
             [](const ucc::CompiledModule& p) {
                 std::ostringstream ss;
                 for (size_t i = 0; i < p.bytecode.size(); ++i)
                     ss << i << ": " << ucc::format_instruction(p.bytecode[i]) << "\n";
                 return ss.str();
             })
        .def_prop_ro(
            "noise_site_probabilities",
            [](const ucc::CompiledModule& p) {
                auto probs = ucc::noise_site_probabilities(p);
                size_t n = probs.size();
                return vec_to_numpy(std::move(probs), {n});
            },
            nb::rv_policy::move,
            "Per-site total fault probabilities: quantum noise sites followed by\n"
            "readout noise entries. Use for Poisson-Binomial PMF computation.")
        .def("__repr__", [](const ucc::CompiledModule& p) {
            return "Program(" + std::to_string(p.bytecode.size()) +
                   " instructions, peak_rank=" + std::to_string(p.peak_rank) + ", " +
                   std::to_string(p.num_measurements) + " measurements)";
        });

    // Back-end: HIR -> Program (with optional postselection/normalization masks)
    m.def(
        "lower",
        [](const ucc::HirModule& hir, std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors, std::vector<uint8_t> expected_observables) {
            nb::gil_scoped_release release;
            return ucc::lower(hir, postselection_mask, expected_detectors, expected_observables);
        },
        nb::arg("hir"), nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{},
        "Lower a Heisenberg IR module to executable VM bytecode.\n\n"
        "To optimize the bytecode, use a BytecodePassManager after lowering:\n"
        "    prog = ucc.lower(hir)\n"
        "    bpm = ucc.default_bytecode_pass_manager()\n"
        "    bpm.run(prog)\n\n"
        "Args:\n"
        "    hir: The Heisenberg IR module to lower.\n"
        "    postselection_mask: Optional list of uint8 flags, one per detector.\n"
        "        Detectors where mask[i] != 0 become post-selection checks\n"
        "        that abort the shot early if their parity is non-zero.\n"
        "    expected_detectors: Optional noiseless reference parities for detectors.\n"
        "    expected_observables: Optional noiseless reference parities for observables.\n");

    // Convenience: stim text -> Program (parse + trace + lower)
    m.def(
        "compile",
        [](const std::string& stim_text, std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors, std::vector<uint8_t> expected_observables,
           bool normalize_syndromes, ucc::HirPassManager* hir_passes,
           ucc::BytecodePassManager* bytecode_passes) {
            nb::gil_scoped_release release;
            ucc::Circuit circuit = ucc::parse(stim_text);
            ucc::HirModule hir = ucc::trace(circuit);
            if (hir_passes)
                hir_passes->run(hir);

            if (normalize_syndromes) {
                if (!expected_detectors.empty() || !expected_observables.empty()) {
                    throw std::invalid_argument(
                        "Cannot provide expected parities when normalize_syndromes=True");
                }
                auto ref = ucc::compute_reference_syndrome(hir);
                expected_detectors = std::move(ref.detectors);
                expected_observables = std::move(ref.observables);
            }

            auto program =
                ucc::lower(hir, postselection_mask, expected_detectors, expected_observables);
            if (bytecode_passes)
                bytecode_passes->run(program);
            return program;
        },
        nb::arg("stim_text"), nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{},
        nb::arg("normalize_syndromes") = false, nb::arg("hir_passes") = nb::none(),
        nb::arg("bytecode_passes") = nb::none(),
        "Compile a quantum circuit string to executable bytecode.\n\n"
        "Runs the full pipeline: parse -> trace -> [HIR optimize] ->\n"
        "lower -> [bytecode optimize].  Pass manager arguments are\n"
        "optional; when None the corresponding optimization stage is\n"
        "skipped (matching the previous default behavior).\n"
        "\n"
        "When normalize_syndromes=True, a noiseless reference shot is\n"
        "executed internally to extract expected detector and observable\n"
        "parities. Detectors and observables are then XOR-normalized so\n"
        "that 0 means 'matches noiseless reference' and 1 means 'error'.\n"
        "\n"
        "Args:\n"
        "    stim_text: Circuit in .stim text format.\n"
        "    postselection_mask: Optional list of uint8 flags, one per detector.\n"
        "        Detectors where mask[i] != 0 become post-selection checks\n"
        "        that abort the shot early if their parity is non-zero.\n"
        "    expected_detectors: Optional noiseless reference parities for detectors.\n"
        "    expected_observables: Optional noiseless reference parities for observables.\n"
        "    normalize_syndromes: If True, auto-compute reference parities from a\n"
        "        noiseless reference shot (mutually exclusive with explicit parities).\n"
        "    hir_passes: Optional HirPassManager to run on the HIR before lowering.\n"
        "    bytecode_passes: Optional BytecodePassManager to run after lowering.\n");

    // Sample: Program + shots -> SampleResult
    m.def(
        "sample",
        [](const ucc::CompiledModule& program, uint32_t shots, std::optional<uint64_t> seed) {
            ucc::SampleResult result;
            {
                nb::gil_scoped_release release;
                result = ucc::sample(program, shots, seed);
            }

            auto meas_arr =
                vec_to_numpy(std::move(result.measurements), {shots, program.num_measurements});
            auto det_arr =
                vec_to_numpy(std::move(result.detectors), {shots, program.num_detectors});
            auto obs_arr =
                vec_to_numpy(std::move(result.observables), {shots, program.num_observables});

            // Import the Python SampleResult class and construct it.
            nb::object mod = nb::module_::import_("ucc._sample_result");
            return mod.attr("SampleResult")(meas_arr, det_arr, obs_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        "Run a compiled program and return a SampleResult.\n\n"
        "If seed is None (default), uses 256-bit OS hardware entropy.\n\n"
        "Returns a SampleResult with .measurements, .detectors, .observables attributes.\n"
        "Supports tuple unpacking: m, d, o = ucc.sample(prog, shots)");

    // Sample with forced k-faults (importance sampling).
    m.def(
        "sample_k",
        [](const ucc::CompiledModule& program, uint32_t shots, uint32_t k,
           std::optional<uint64_t> seed) {
            ucc::SampleResult result;
            {
                nb::gil_scoped_release release;
                result = ucc::sample_k(program, shots, k, seed);
            }

            auto meas_arr =
                vec_to_numpy(std::move(result.measurements), {shots, program.num_measurements});
            auto det_arr =
                vec_to_numpy(std::move(result.detectors), {shots, program.num_detectors});
            auto obs_arr =
                vec_to_numpy(std::move(result.observables), {shots, program.num_observables});

            nb::object mod = nb::module_::import_("ucc._sample_result");
            return mod.attr("SampleResult")(meas_arr, det_arr, obs_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("k"), nb::arg("seed") = nb::none(),
        "Sample with exactly k forced faults per shot (importance sampling).\n\n"
        "Sites are drawn from the exact conditional Poisson-Binomial\n"
        "distribution. Results are conditioned on K=k and must be combined\n"
        "across strata with P(K=k) weights for correct error rate estimation.\n"
        "For post-selected circuits, weight numerator and denominator\n"
        "separately: p_fail = sum(P(K=k)*errors_k/shots_k) / sum(P(K=k)*passed_k/shots_k).\n\n"
        "Raises ValueError if the k-fault stratum has zero probability mass\n"
        "(e.g. k exceeds the number of non-zero-probability sites).\n\n"
        "When all site probabilities are equal, an O(k) Fisher-Yates\n"
        "sampler is used automatically.\n\n"
        "Returns a SampleResult with .measurements, .detectors, .observables attributes.\n"
        "Supports tuple unpacking: m, d, o = ucc.sample_k(prog, shots, k)");

    // Helper lambda to populate a SampleResult Python object from C++ survivor data.
    auto make_survivor_result = [](ucc::SurvivorResult result, const ucc::CompiledModule& program,
                                   bool keep_records) -> nb::object {
        size_t num_obs = result.observable_ones.size();
        auto obs_ones_arr = vec_to_numpy(std::move(result.observable_ones), {num_obs});

        nb::object mod = nb::module_::import_("ucc._sample_result");
        nb::object cls = mod.attr("SampleResult");

        std::vector<uint8_t> meas_storage =
            keep_records ? std::move(result.measurements) : std::vector<uint8_t>{};
        std::vector<uint8_t> det_storage =
            keep_records ? std::move(result.detectors) : std::vector<uint8_t>{};
        std::vector<uint8_t> obs_storage =
            keep_records ? std::move(result.observables) : std::vector<uint8_t>{};

        size_t rows = keep_records ? result.passed_shots : 0;
        auto meas_arr = vec_to_numpy(std::move(meas_storage), {rows, program.num_measurements});
        auto det_arr = vec_to_numpy(std::move(det_storage), {rows, program.num_detectors});
        auto obs_arr = vec_to_numpy(std::move(obs_storage), {rows, program.num_observables});
        return cls(meas_arr, det_arr, obs_arr, result.total_shots, result.passed_shots,
                   result.logical_errors, obs_ones_arr);
    };

    // Sample survivors with forced k-faults.
    m.def(
        "sample_k_survivors",
        [make_survivor_result](const ucc::CompiledModule& program, uint32_t shots, uint32_t k,
                               std::optional<uint64_t> seed, bool keep_records) {
            ucc::SurvivorResult result;
            {
                nb::gil_scoped_release release;
                result = ucc::sample_k_survivors(program, shots, k, seed, keep_records);
            }
            return make_survivor_result(std::move(result), program, keep_records);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("k"), nb::arg("seed") = nb::none(),
        nb::arg("keep_records") = false,
        "Sample survivors with exactly k forced faults per shot.\n\n"
        "Results are conditioned on K=k. To estimate the overall logical\n"
        "error rate across strata, weight numerator and denominator\n"
        "separately to account for k-dependent survival probability:\n"
        "  p_fail = sum(P(K=k)*logical_errors_k/shots_k)\n"
        "         / sum(P(K=k)*passed_k/shots_k)\n\n"
        "Raises ValueError if the k-fault stratum has zero probability mass.\n\n"
        "Returns a SampleResult whose measurements/detectors/observables arrays\n"
        "contain only surviving shots. Survivor metadata is available via\n"
        ".total_shots, .passed_shots, .discards, .logical_errors, and\n"
        ".observable_ones.");

    // Sample survivors: only non-discarded shots contribute to output.
    m.def(
        "sample_survivors",
        [make_survivor_result](const ucc::CompiledModule& program, uint32_t shots,
                               std::optional<uint64_t> seed, bool keep_records) {
            ucc::SurvivorResult result;
            {
                nb::gil_scoped_release release;
                result = ucc::sample_survivors(program, shots, seed, keep_records);
            }
            return make_survivor_result(std::move(result), program, keep_records);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        nb::arg("keep_records") = false,
        "Sample shots and return results only for surviving (non-discarded) shots.\n\n"
        "If seed is None (default), uses 256-bit OS hardware entropy.\n\n"
        "Returns a SampleResult whose measurements/detectors/observables arrays\n"
        "contain only surviving shots. Survivor metadata is available via\n"
        ".total_shots, .passed_shots, .discards, .logical_errors, and\n"
        ".observable_ones.");

    nb::class_<ucc::SchrodingerState>(m, "State", "Schrodinger VM execution state")
        .def(nb::init<uint32_t, uint32_t, uint32_t, uint32_t, std::optional<uint64_t>>(),
             nb::arg("peak_rank"), nb::arg("num_measurements"), nb::arg("num_detectors") = 0,
             nb::arg("num_observables") = 0, nb::arg("seed") = nb::none())
        .def("reset", &ucc::SchrodingerState::reset)
        .def("reseed", &ucc::SchrodingerState::reseed, nb::arg("seed"))
        .def_prop_ro("dust_clamps", [](const ucc::SchrodingerState& s) { return s.dust_clamps; })
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
            nb::gil_scoped_release release;
            ucc::execute(program, state);
        },
        nb::arg("program"), nb::arg("state"),
        "Execute a compiled program, updating the state in-place.");

    // Get statevector: Program + State -> numpy array
    m.def(
        "get_statevector",
        [](const ucc::CompiledModule& program, const ucc::SchrodingerState& state) {
            std::vector<std::complex<double>> sv;
            {
                nb::gil_scoped_release release;
                sv = ucc::get_statevector(program, state);
            }
            size_t n = sv.size();
            return vec_to_numpy(std::move(sv), {n});
        },
        nb::arg("program"), nb::arg("state"), "Expand the SVM state into a dense statevector.");
}
