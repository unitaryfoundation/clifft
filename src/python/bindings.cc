#include "clifft/api/reference_syndrome.h"
#include "clifft/backend/backend.h"
#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/bytecode_pass.h"
#include "clifft/optimizer/expand_t_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/multi_gate_pass.h"
#include "clifft/optimizer/noise_block_pass.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/optimizer/single_axis_fusion_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/optimizer/swap_meas_pass.h"
#include "clifft/optimizer/tile_axis_fusion_pass.h"
#include "clifft/svm/svm.h"
#include "clifft/util/config.h"
#include "clifft/util/introspection.h"
#include "clifft/util/version.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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

NB_MODULE(_clifft_core, m) {
    m.doc() = "Clifft core C++ extension module";

    nb::exception<clifft::ParseError>(m, "ParseError");

    m.def("version", []() { return clifft::kVersion; }, "Return the Clifft version string");

    m.def(
        "max_sim_qubits", []() { return clifft::kMaxInlineQubits; },
        "Return the maximum number of qubits supported by the simulator");

    m.def(
        "svm_backend", []() { return clifft::svm_backend(); },
        "Return the active SVM dispatch backend: 'avx512', 'avx2', or 'scalar'.\n\n"
        "Reflects the resolved runtime kernel path or CLIFFT_FORCE_ISA environment override. "
        "'scalar' names the generic/base SVM path.");

    m.def(
        "set_num_threads",
        [](int n) {
#ifdef _OPENMP
            omp_set_num_threads(n);
#else
            (void)n;
#endif
        },
        nb::arg("num_threads"),
        "Set the number of OpenMP threads for multi-core statevector operations.\n\n"
        "Threading only activates for circuits with peak rank >= 18\n"
        "(statevector >= 4 MB). Pure Clifford and low-T-count circuits\n"
        "run single-threaded regardless of this setting.\n\n"
        "When using multiprocessing (e.g. sinter), set num_threads=1 in\n"
        "each worker to avoid oversubscription.\n\n"
        "Has no effect if Clifft was built without OpenMP support.");

    m.def(
        "get_num_threads",
        []() -> int {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        },
        "Return the current maximum number of OpenMP threads.\n\n"
        "Threading only activates for circuits with peak rank >= 18\n"
        "(statevector >= 4 MB).\n\n"
        "Returns 1 if Clifft was built without OpenMP support.");

    // Sentinel-based enum counts for defensive binding tests.
    // If a new enum value is added in C++ but not bound in Python,
    // the test_introspection.py tripwire will catch it.
    m.def("_num_optypes", []() { return static_cast<int>(clifft::OpType::NUM_OP_TYPES); });
    m.def("_num_opcodes", []() { return static_cast<int>(clifft::Opcode::NUM_OPCODES); });

    nb::enum_<clifft::GateType>(m, "GateType", "Quantum gate types")
        // Single-qubit Cliffords
        .value("H", clifft::GateType::H)
        .value("S", clifft::GateType::S)
        .value("S_DAG", clifft::GateType::S_DAG)
        .value("X", clifft::GateType::X)
        .value("Y", clifft::GateType::Y)
        .value("Z", clifft::GateType::Z)
        .value("SQRT_X", clifft::GateType::SQRT_X)
        .value("SQRT_X_DAG", clifft::GateType::SQRT_X_DAG)
        .value("SQRT_Y", clifft::GateType::SQRT_Y)
        .value("SQRT_Y_DAG", clifft::GateType::SQRT_Y_DAG)
        .value("H_XY", clifft::GateType::H_XY)
        .value("H_YZ", clifft::GateType::H_YZ)
        .value("H_NXY", clifft::GateType::H_NXY)
        .value("H_NXZ", clifft::GateType::H_NXZ)
        .value("H_NYZ", clifft::GateType::H_NYZ)
        .value("C_XYZ", clifft::GateType::C_XYZ)
        .value("C_ZYX", clifft::GateType::C_ZYX)
        .value("C_NXYZ", clifft::GateType::C_NXYZ)
        .value("C_NZYX", clifft::GateType::C_NZYX)
        .value("C_XNYZ", clifft::GateType::C_XNYZ)
        .value("C_XYNZ", clifft::GateType::C_XYNZ)
        .value("C_ZNYX", clifft::GateType::C_ZNYX)
        .value("C_ZYNX", clifft::GateType::C_ZYNX)
        // Non-Clifford
        .value("T", clifft::GateType::T)
        .value("T_DAG", clifft::GateType::T_DAG)
        // Parameterized rotations
        .value("R_X", clifft::GateType::R_X)
        .value("R_Y", clifft::GateType::R_Y)
        .value("R_Z", clifft::GateType::R_Z)
        .value("U3", clifft::GateType::U3)
        .value("R_XX", clifft::GateType::R_XX)
        .value("R_YY", clifft::GateType::R_YY)
        .value("R_ZZ", clifft::GateType::R_ZZ)
        .value("R_PAULI", clifft::GateType::R_PAULI)
        // Two-qubit Cliffords
        .value("CX", clifft::GateType::CX)
        .value("CY", clifft::GateType::CY)
        .value("CZ", clifft::GateType::CZ)
        .value("SWAP", clifft::GateType::SWAP)
        .value("ISWAP", clifft::GateType::ISWAP)
        .value("ISWAP_DAG", clifft::GateType::ISWAP_DAG)
        .value("SQRT_XX", clifft::GateType::SQRT_XX)
        .value("SQRT_XX_DAG", clifft::GateType::SQRT_XX_DAG)
        .value("SQRT_YY", clifft::GateType::SQRT_YY)
        .value("SQRT_YY_DAG", clifft::GateType::SQRT_YY_DAG)
        .value("SQRT_ZZ", clifft::GateType::SQRT_ZZ)
        .value("SQRT_ZZ_DAG", clifft::GateType::SQRT_ZZ_DAG)
        .value("CXSWAP", clifft::GateType::CXSWAP)
        .value("CZSWAP", clifft::GateType::CZSWAP)
        .value("SWAPCX", clifft::GateType::SWAPCX)
        .value("XCX", clifft::GateType::XCX)
        .value("XCY", clifft::GateType::XCY)
        .value("XCZ", clifft::GateType::XCZ)
        .value("YCX", clifft::GateType::YCX)
        .value("YCY", clifft::GateType::YCY)
        .value("YCZ", clifft::GateType::YCZ)
        // Measurements
        .value("M", clifft::GateType::M)
        .value("MX", clifft::GateType::MX)
        .value("MY", clifft::GateType::MY)
        .value("MR", clifft::GateType::MR)
        .value("MRX", clifft::GateType::MRX)
        .value("MRY", clifft::GateType::MRY)
        .value("MPP", clifft::GateType::MPP)
        .value("MXX", clifft::GateType::MXX)
        .value("MYY", clifft::GateType::MYY)
        .value("MZZ", clifft::GateType::MZZ)
        // Resets
        .value("R", clifft::GateType::R)
        .value("RX", clifft::GateType::RX)
        .value("RY", clifft::GateType::RY)
        // Padding
        .value("MPAD", clifft::GateType::MPAD)
        // Identity no-ops
        .value("I", clifft::GateType::I)
        .value("II", clifft::GateType::II)
        .value("I_ERROR", clifft::GateType::I_ERROR)
        .value("II_ERROR", clifft::GateType::II_ERROR)
        // Noise
        .value("X_ERROR", clifft::GateType::X_ERROR)
        .value("Y_ERROR", clifft::GateType::Y_ERROR)
        .value("Z_ERROR", clifft::GateType::Z_ERROR)
        .value("DEPOLARIZE1", clifft::GateType::DEPOLARIZE1)
        .value("DEPOLARIZE2", clifft::GateType::DEPOLARIZE2)
        .value("PAULI_CHANNEL_1", clifft::GateType::PAULI_CHANNEL_1)
        .value("PAULI_CHANNEL_2", clifft::GateType::PAULI_CHANNEL_2)
        .value("READOUT_NOISE", clifft::GateType::READOUT_NOISE)
        // Annotations
        .value("DETECTOR", clifft::GateType::DETECTOR)
        .value("OBSERVABLE_INCLUDE", clifft::GateType::OBSERVABLE_INCLUDE)
        .value("TICK", clifft::GateType::TICK)
        .value("UNKNOWN", clifft::GateType::UNKNOWN);

    nb::class_<clifft::Target>(m, "Target", "Encoded quantum target")
        .def_prop_ro("value", [](const clifft::Target& t) { return t.value(); })
        .def_prop_ro("is_rec", [](const clifft::Target& t) { return t.is_rec(); })
        .def_prop_ro("is_inverted", [](const clifft::Target& t) { return t.is_inverted(); })
        .def_prop_ro("has_pauli", [](const clifft::Target& t) { return t.has_pauli(); })
        .def_prop_ro("pauli", [](const clifft::Target& t) { return t.pauli(); })
        .def_prop_ro("pauli_char",
                     [](const clifft::Target& t) { return std::string(1, t.pauli_char()); })
        .def("__repr__", [](const clifft::Target& t) {
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

    nb::class_<clifft::AstNode>(m, "AstNode", "A single circuit operation")
        .def_ro("gate", &clifft::AstNode::gate)
        .def_ro("targets", &clifft::AstNode::targets)
        .def_prop_ro("arg",
                     [](const clifft::AstNode& n) { return n.args.empty() ? 0.0 : n.args[0]; })
        .def_ro("args", &clifft::AstNode::args)
        .def_ro("source_line", &clifft::AstNode::source_line)
        .def("__repr__", [](const clifft::AstNode& n) {
            std::string result = std::string(clifft::gate_name(n.gate));
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

    nb::class_<clifft::Circuit>(m, "Circuit", "A parsed quantum circuit")
        .def_ro("nodes", &clifft::Circuit::nodes)
        .def_ro("num_qubits", &clifft::Circuit::num_qubits)
        .def_ro("num_measurements", &clifft::Circuit::num_measurements)
        .def("__len__", [](const clifft::Circuit& c) { return c.nodes.size(); })
        .def("__repr__", [](const clifft::Circuit& c) {
            return "Circuit(" + std::to_string(c.nodes.size()) + " ops, " +
                   std::to_string(c.num_qubits) + " qubits, " + std::to_string(c.num_measurements) +
                   " measurements)";
        });

    m.def(
        "parse",
        [](std::string_view text) {
            nb::gil_scoped_release release;
            return clifft::parse(text);
        },
        nb::arg("text"), "Parse a quantum circuit from a string.");
    m.def(
        "parse",
        [](std::string_view text, size_t max_ops) {
            nb::gil_scoped_release release;
            return clifft::parse(text, max_ops);
        },
        nb::arg("text"), nb::arg("max_ops"),
        "Parse a quantum circuit from a string with an explicit AST node limit.");
    m.def(
        "parse_file",
        [](const std::string& path) {
            nb::gil_scoped_release release;
            return clifft::parse_file(path);
        },
        nb::arg("path"), "Parse a quantum circuit from a file.");
    m.def(
        "parse_file",
        [](const std::string& path, size_t max_ops) {
            nb::gil_scoped_release release;
            return clifft::parse_file(path, max_ops);
        },
        nb::arg("path"), nb::arg("max_ops"),
        "Parse a quantum circuit from a file with an explicit AST node limit.");

    nb::enum_<clifft::OpType>(m, "OpType", "Heisenberg IR operation types")
        .value("T_GATE", clifft::OpType::T_GATE)
        .value("MEASURE", clifft::OpType::MEASURE)
        .value("CONDITIONAL_PAULI", clifft::OpType::CONDITIONAL_PAULI)
        .value("NOISE", clifft::OpType::NOISE)
        .value("READOUT_NOISE", clifft::OpType::READOUT_NOISE)
        .value("PHASE_ROTATION", clifft::OpType::PHASE_ROTATION)
        .value("DETECTOR", clifft::OpType::DETECTOR)
        .value("OBSERVABLE", clifft::OpType::OBSERVABLE)
        .value("EXP_VAL", clifft::OpType::EXP_VAL);

    // Python view of a HeisenbergOp paired with the HirModule that owns its
    // mask data. Holding the module reference lets the Python API expose
    // sign / pauli_string without depending on (now-removed) per-op storage.
    struct PyHeisenbergOp {
        const clifft::HeisenbergOp* op;
        const clifft::HirModule* hir;
    };

    nb::class_<PyHeisenbergOp>(m, "HeisenbergOp",
                               "A single abstract operation in the Heisenberg IR")
        .def_prop_ro("op_type", [](const PyHeisenbergOp& w) { return w.op->op_type(); })
        .def_prop_ro("is_dagger", [](const PyHeisenbergOp& w) { return w.op->is_dagger(); })
        .def_prop_ro("is_hidden", [](const PyHeisenbergOp& w) { return w.op->is_hidden(); })
        .def_prop_ro(
            "sign",
            [](const PyHeisenbergOp& w) { return w.op->has_mask() ? w.hir->sign(*w.op) : false; })
        .def_prop_ro("pauli_string",
                     [](const PyHeisenbergOp& w) {
                         return w.op->has_mask() ? clifft::format_pauli_mask(*w.op, *w.hir)
                                                 : std::string("+I");
                     })
        .def(
            "as_dict",
            [](const PyHeisenbergOp& w) {
                const auto& op = *w.op;
                nb::dict d;
                d["op_type"] = clifft::op_type_to_str(op.op_type());
                d["pauli_string"] =
                    op.has_mask() ? clifft::format_pauli_mask(op, *w.hir) : std::string("+I");
                d["is_dagger"] = op.is_dagger();
                d["is_hidden"] = op.is_hidden();
                d["sign"] = op.has_mask() ? w.hir->sign(op) : false;

                switch (op.op_type()) {
                    case clifft::OpType::MEASURE:
                        d["meas_record_idx"] = static_cast<uint32_t>(op.meas_record_idx());
                        break;
                    case clifft::OpType::CONDITIONAL_PAULI:
                        d["controlling_meas"] = static_cast<uint32_t>(op.controlling_meas());
                        break;
                    case clifft::OpType::NOISE:
                        d["noise_site_idx"] = static_cast<uint32_t>(op.noise_site_idx());
                        break;
                    case clifft::OpType::READOUT_NOISE:
                        d["readout_noise_idx"] = static_cast<uint32_t>(op.readout_noise_idx());
                        break;
                    case clifft::OpType::DETECTOR:
                        d["detector_idx"] = static_cast<uint32_t>(op.detector_idx());
                        break;
                    case clifft::OpType::OBSERVABLE:
                        d["observable_idx"] = static_cast<uint32_t>(op.observable_idx());
                        d["observable_target_list_idx"] = op.observable_target_list_idx();
                        break;
                    case clifft::OpType::PHASE_ROTATION:
                        d["alpha"] = op.alpha();
                        break;
                    case clifft::OpType::EXP_VAL:
                        d["exp_val_idx"] = static_cast<uint32_t>(op.exp_val_idx());
                        break;
                    default:
                        break;
                }
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__",
             [](const PyHeisenbergOp& w) { return clifft::format_hir_op(*w.op, *w.hir); })
        .def("__repr__", [](const PyHeisenbergOp& w) {
            return "<HeisenbergOp: " + clifft::format_hir_op(*w.op, *w.hir) + ">";
        });

    nb::class_<clifft::HirModule>(m, "HirModule", "Heisenberg Intermediate Representation")
        .def_prop_ro("num_ops", [](const clifft::HirModule& h) { return h.num_ops(); })
        .def_prop_ro("num_t_gates", [](const clifft::HirModule& h) { return h.num_t_gates(); })
        .def_prop_ro("num_qubits", [](const clifft::HirModule& h) { return h.num_qubits; })
        .def_prop_ro("num_measurements",
                     [](const clifft::HirModule& h) { return h.num_measurements; })
        .def_prop_ro("num_detectors", [](const clifft::HirModule& h) { return h.num_detectors; })
        .def_prop_ro("num_observables",
                     [](const clifft::HirModule& h) { return h.num_observables; })
        .def_prop_ro("num_exp_vals", [](const clifft::HirModule& h) { return h.num_exp_vals; })
        .def_prop_ro(
            "source_map",
            [](const clifft::HirModule& h) {
                nb::list outer;
                for (const auto& lines : h.source_map)
                    outer.append(nb::cast(lines));
                return outer;
            },
            "Source line mapping parallel to ops (list of list of uint32).")
        .def(
            "__len__", [](const clifft::HirModule& h) { return h.ops.size(); },
            "Return the number of HIR operations.")
        .def(
            "__getitem__",
            [](const clifft::HirModule& h, int64_t idx) {
                int64_t size = static_cast<int64_t>(h.ops.size());
                if (idx < 0)
                    idx += size;
                if (idx < 0 || idx >= size)
                    throw nb::index_error();
                return PyHeisenbergOp{&h.ops[static_cast<size_t>(idx)], &h};
            },
            nb::keep_alive<0, 1>(), "Return the HIR operation at the given index.")
        .def("__iter__",
             [](const clifft::HirModule& h) {
                 nb::list items;
                 for (const auto& op : h.ops)
                     items.append(nb::cast(PyHeisenbergOp{&op, &h}));
                 return items.attr("__iter__")();
             })
        .def(
            "as_dict",
            [](const clifft::HirModule& h) {
                nb::dict d;
                d["num_qubits"] = h.num_qubits;
                d["num_measurements"] = h.num_measurements;
                d["num_detectors"] = h.num_detectors;
                d["num_observables"] = h.num_observables;
                nb::list ops;
                for (const auto& op : h.ops) {
                    PyHeisenbergOp w{&op, &h};
                    ops.append(nb::cast(w).attr("as_dict")());
                }
                d["ops"] = ops;
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__",
             [](const clifft::HirModule& h) {
                 std::ostringstream ss;
                 for (size_t i = 0; i < h.ops.size(); ++i)
                     ss << i << ": " << clifft::format_hir_op(h.ops[i], h) << "\n";
                 return ss.str();
             })
        .def("__repr__", [](const clifft::HirModule& h) {
            return "HirModule(" + std::to_string(h.num_ops()) + " ops, " +
                   std::to_string(h.num_t_gates()) + " T-gates, " + std::to_string(h.num_qubits) +
                   " qubits)";
        });

    m.def(
        "trace",
        [](const clifft::Circuit& circuit) {
            nb::gil_scoped_release release;
            return clifft::trace(circuit);
        },
        nb::arg("circuit"),
        "Trace a parsed circuit through the Clifford front-end to produce the "
        "Heisenberg IR.");

    nb::class_<clifft::HirPass>(m, "HirPass", "Abstract base class for HIR optimization passes.");

    nb::class_<clifft::PeepholeFusionPass, clifft::HirPass>(
        m, "PeepholeFusionPass",
        "Symplectic peephole fusion: cancels and fuses T/T-dag gates on the "
        "same virtual Pauli axis.")
        .def(nb::init<>())
        .def_prop_ro("cancellations", &clifft::PeepholeFusionPass::cancellations)
        .def_prop_ro("fusions", &clifft::PeepholeFusionPass::fusions)
        .def("__repr__", [](const clifft::PeepholeFusionPass& p) {
            return "PeepholeFusionPass(cancellations=" + std::to_string(p.cancellations()) +
                   ", fusions=" + std::to_string(p.fusions()) + ")";
        });

    nb::class_<clifft::StatevectorSqueezePass, clifft::HirPass>(
        m, "StatevectorSqueezePass",
        "Bidirectional bubble sort: moves measurements leftward and\n"
        "non-Clifford gates rightward to minimize peak active rank.")
        .def(nb::init<>());

    nb::class_<clifft::RemoveNoisePass, clifft::HirPass>(
        m, "RemoveNoisePass",
        "Strips all stochastic noise and readout noise ops from the HIR.\n"
        "Not included in the default pass list. Used internally by\n"
        "compute_reference_syndrome() for noiseless reference shots.")
        .def(nb::init<>());

    m.def(
        "compute_reference_syndrome",
        [](const clifft::HirModule& hir) {
            clifft::ReferenceSyndrome ref;
            {
                nb::gil_scoped_release release;
                ref = clifft::compute_reference_syndrome(hir);
            }
            nb::dict d;
            d["detectors"] = nb::cast(std::move(ref.detectors));
            d["observables"] = nb::cast(std::move(ref.observables));
            return d;
        },
        nb::arg("hir"),
        "Compute noiseless reference syndrome for an HirModule.\n\n"
        "Returns a dict with 'detectors' and 'observables' lists.");

    nb::class_<clifft::HirPassManager>(m, "HirPassManager",
                                       "Runs a sequence of optimization passes over an HirModule.")
        .def(nb::init<>())
        .def(
            "add",
            [](clifft::HirPassManager& pm, clifft::HirPass& pass) {
                // HirPassManager needs unique_ptr ownership, but Python owns the pass.
                // Use a thin non-owning wrapper that delegates to the Python-owned pass.
                struct BorrowedPass : clifft::HirPass {
                    clifft::HirPass& ref;
                    explicit BorrowedPass(clifft::HirPass& r) : ref(r) {}
                    void run(clifft::HirModule& hir) override { ref.run(hir); }
                };
                pm.add_pass(std::make_unique<BorrowedPass>(pass));
            },
            nb::arg("pass"), nb::keep_alive<1, 2>(),
            "Add an optimization pass. Passes run in the order added.")
        .def(
            "run", [](clifft::HirPassManager& pm, clifft::HirModule& hir) { pm.run(hir); },
            nb::arg("hir"), "Run all passes on the HIR module in sequence.");

    m.def(
        "default_hir_pass_manager", []() { return clifft::default_hir_pass_manager(); },
        nb::rv_policy::move, "Return an HirPassManager pre-loaded with the default passes.");

    nb::class_<clifft::BytecodePass>(m, "BytecodePass",
                                     "Abstract base class for bytecode optimization passes.\n\n"
                                     "Each pass receives a mutable Program and may rewrite,\n"
                                     "reorder, or remove instructions.");

    nb::class_<clifft::NoiseBlockPass, clifft::BytecodePass>(
        m, "NoiseBlockPass", "Coalesces contiguous OP_NOISE instructions into OP_NOISE_BLOCK.")
        .def(nb::init<>());

    nb::class_<clifft::ExpandTPass, clifft::BytecodePass>(
        m, "ExpandTPass", "Fuses OP_EXPAND + OP_ARRAY_T into single OP_EXPAND_T instructions.")
        .def(nb::init<>());

    nb::class_<clifft::ExpandRotPass, clifft::BytecodePass>(
        m, "ExpandRotPass",
        "Fuses OP_EXPAND + OP_ARRAY_ROT into single OP_EXPAND_ROT instructions.")
        .def(nb::init<>());

    nb::class_<clifft::SwapMeasPass, clifft::BytecodePass>(
        m, "SwapMeasPass",
        "Fuses OP_ARRAY_SWAP + OP_MEAS_ACTIVE_INTERFERE into OP_SWAP_MEAS_INTERFERE.")
        .def(nb::init<>());

    nb::class_<clifft::MultiGatePass, clifft::BytecodePass>(
        m, "MultiGatePass", "Fuses sequences of same-type 2-qubit gates into multi-target ops.")
        .def(nb::init<>());

    nb::class_<clifft::TileAxisFusionPass, clifft::BytecodePass>(
        m, "TileAxisFusionPass",
        "Fuses 2-qubit tile sequences into OP_ARRAY_U4 with precomputed 4x4 matrices.")
        .def(nb::init<>());

    nb::class_<clifft::SingleAxisFusionPass, clifft::BytecodePass>(
        m, "SingleAxisFusionPass",
        "Fuses consecutive single-axis ops into OP_ARRAY_U2 with precomputed 2x2 matrices.")
        .def(nb::init<>());

    nb::class_<clifft::BytecodePassManager>(m, "BytecodePassManager",
                                            "Runs a sequence of bytecode optimization passes "
                                            "over a Program.")
        .def(nb::init<>())
        .def(
            "add",
            [](clifft::BytecodePassManager& bpm, clifft::BytecodePass& pass) {
                struct BorrowedBytecodePass : clifft::BytecodePass {
                    clifft::BytecodePass& ref;
                    explicit BorrowedBytecodePass(clifft::BytecodePass& r) : ref(r) {}
                    void run(clifft::CompiledModule& mod) override { ref.run(mod); }
                };
                bpm.add_pass(std::make_unique<BorrowedBytecodePass>(pass));
            },
            nb::arg("pass"), nb::keep_alive<1, 2>(),
            "Add a bytecode optimization pass. Passes run in the order added.")
        .def(
            "run",
            [](clifft::BytecodePassManager& bpm, clifft::CompiledModule& mod) { bpm.run(mod); },
            nb::arg("program"), "Run all bytecode passes on the program in sequence.");

    m.def(
        "default_bytecode_pass_manager", []() { return clifft::default_bytecode_pass_manager(); },
        nb::rv_policy::move, "Return a BytecodePassManager pre-loaded with the default passes.");

    nb::enum_<clifft::Opcode>(m, "Opcode", "Virtual Machine opcodes")
        .value("OP_FRAME_CNOT", clifft::Opcode::OP_FRAME_CNOT)
        .value("OP_FRAME_CZ", clifft::Opcode::OP_FRAME_CZ)
        .value("OP_FRAME_H", clifft::Opcode::OP_FRAME_H)
        .value("OP_FRAME_S", clifft::Opcode::OP_FRAME_S)
        .value("OP_FRAME_S_DAG", clifft::Opcode::OP_FRAME_S_DAG)
        .value("OP_FRAME_SWAP", clifft::Opcode::OP_FRAME_SWAP)
        .value("OP_ARRAY_CNOT", clifft::Opcode::OP_ARRAY_CNOT)
        .value("OP_ARRAY_CZ", clifft::Opcode::OP_ARRAY_CZ)
        .value("OP_ARRAY_SWAP", clifft::Opcode::OP_ARRAY_SWAP)
        .value("OP_ARRAY_MULTI_CNOT", clifft::Opcode::OP_ARRAY_MULTI_CNOT)
        .value("OP_ARRAY_MULTI_CZ", clifft::Opcode::OP_ARRAY_MULTI_CZ)
        .value("OP_ARRAY_H", clifft::Opcode::OP_ARRAY_H)
        .value("OP_ARRAY_S", clifft::Opcode::OP_ARRAY_S)
        .value("OP_ARRAY_S_DAG", clifft::Opcode::OP_ARRAY_S_DAG)
        .value("OP_EXPAND", clifft::Opcode::OP_EXPAND)
        .value("OP_ARRAY_T", clifft::Opcode::OP_ARRAY_T)
        .value("OP_ARRAY_T_DAG", clifft::Opcode::OP_ARRAY_T_DAG)
        .value("OP_EXPAND_T", clifft::Opcode::OP_EXPAND_T)
        .value("OP_EXPAND_T_DAG", clifft::Opcode::OP_EXPAND_T_DAG)
        .value("OP_ARRAY_ROT", clifft::Opcode::OP_ARRAY_ROT)
        .value("OP_EXPAND_ROT", clifft::Opcode::OP_EXPAND_ROT)
        .value("OP_ARRAY_U2", clifft::Opcode::OP_ARRAY_U2)
        .value("OP_ARRAY_U4", clifft::Opcode::OP_ARRAY_U4)
        .value("OP_MEAS_DORMANT_STATIC", clifft::Opcode::OP_MEAS_DORMANT_STATIC)
        .value("OP_MEAS_DORMANT_RANDOM", clifft::Opcode::OP_MEAS_DORMANT_RANDOM)
        .value("OP_MEAS_ACTIVE_DIAGONAL", clifft::Opcode::OP_MEAS_ACTIVE_DIAGONAL)
        .value("OP_MEAS_ACTIVE_INTERFERE", clifft::Opcode::OP_MEAS_ACTIVE_INTERFERE)
        .value("OP_SWAP_MEAS_INTERFERE", clifft::Opcode::OP_SWAP_MEAS_INTERFERE)
        .value("OP_APPLY_PAULI", clifft::Opcode::OP_APPLY_PAULI)
        .value("OP_NOISE", clifft::Opcode::OP_NOISE)
        .value("OP_NOISE_BLOCK", clifft::Opcode::OP_NOISE_BLOCK)
        .value("OP_READOUT_NOISE", clifft::Opcode::OP_READOUT_NOISE)
        .value("OP_DETECTOR", clifft::Opcode::OP_DETECTOR)
        .value("OP_POSTSELECT", clifft::Opcode::OP_POSTSELECT)
        .value("OP_OBSERVABLE", clifft::Opcode::OP_OBSERVABLE)
        .value("OP_EXP_VAL", clifft::Opcode::OP_EXP_VAL);

    nb::class_<clifft::Instruction>(m, "Instruction", "A localized VM operation")
        .def_prop_ro("opcode", [](const clifft::Instruction& i) { return i.opcode; })
        .def_prop_ro("flags", [](const clifft::Instruction& i) { return i.flags; })
        .def_prop_ro("axis_1", [](const clifft::Instruction& i) { return i.axis_1; })
        .def_prop_ro("axis_2", [](const clifft::Instruction& i) { return i.axis_2; })
        .def(
            "as_dict",
            [](const clifft::Instruction& i) {
                nb::dict d;
                d["opcode"] = clifft::opcode_to_str(i.opcode);
                d["axis_1"] = i.axis_1;
                d["axis_2"] = i.axis_2;
                d["flags"] = i.flags;
                d["description"] = clifft::format_instruction(i);

                if (clifft::is_meas_opcode(i.opcode)) {
                    d["classical_idx"] = i.classical.classical_idx;
                    d["expected_val"] = i.classical.expected_val;
                } else if (i.opcode == clifft::Opcode::OP_APPLY_PAULI) {
                    d["cp_mask_idx"] = i.pauli.cp_mask_idx;
                    d["condition_idx"] = i.pauli.condition_idx;
                } else if (i.opcode == clifft::Opcode::OP_DETECTOR ||
                           i.opcode == clifft::Opcode::OP_POSTSELECT) {
                    d["target_list_index"] = i.pauli.cp_mask_idx;
                    d["detector_index"] = i.pauli.condition_idx;
                } else if (i.opcode == clifft::Opcode::OP_OBSERVABLE) {
                    d["target_list_index"] = i.pauli.cp_mask_idx;
                    d["observable_index"] = i.pauli.condition_idx;
                } else if (i.opcode == clifft::Opcode::OP_NOISE) {
                    d["noise_site_idx"] = i.pauli.cp_mask_idx;
                } else if (i.opcode == clifft::Opcode::OP_NOISE_BLOCK) {
                    d["start_site"] = i.pauli.cp_mask_idx;
                    d["count"] = i.pauli.condition_idx;
                } else if (i.opcode == clifft::Opcode::OP_READOUT_NOISE) {
                    d["readout_noise_idx"] = i.pauli.cp_mask_idx;
                } else if (i.opcode == clifft::Opcode::OP_ARRAY_MULTI_CNOT ||
                           i.opcode == clifft::Opcode::OP_ARRAY_MULTI_CZ) {
                    d["mask"] = i.multi_gate.mask;
                } else if (i.opcode == clifft::Opcode::OP_ARRAY_ROT ||
                           i.opcode == clifft::Opcode::OP_EXPAND_ROT) {
                    d["weight_re"] = i.math.weight_re;
                    d["weight_im"] = i.math.weight_im;
                } else if (i.opcode == clifft::Opcode::OP_ARRAY_U2) {
                    d["cp_idx"] = i.u2.cp_idx;
                } else if (i.opcode == clifft::Opcode::OP_EXP_VAL) {
                    d["cp_exp_val_idx"] = i.exp_val.cp_exp_val_idx;
                    d["exp_val_idx"] = i.exp_val.exp_val_idx;
                }
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__",
             [](const clifft::Instruction& inst) { return clifft::format_instruction(inst); })
        .def("__repr__", [](const clifft::Instruction& inst) {
            return "<Instruction: " + clifft::format_instruction(inst) + ">";
        });

    nb::class_<clifft::CompiledModule>(m, "Program", "A compiled quantum program")
        .def_prop_ro("peak_rank", [](const clifft::CompiledModule& p) { return p.peak_rank; })
        .def_prop_ro("num_measurements",
                     [](const clifft::CompiledModule& p) { return p.num_measurements; })
        .def_prop_ro("num_detectors",
                     [](const clifft::CompiledModule& p) { return p.num_detectors; })
        .def_prop_ro("num_observables",
                     [](const clifft::CompiledModule& p) { return p.num_observables; })
        .def_prop_ro("num_exp_vals", [](const clifft::CompiledModule& p) { return p.num_exp_vals; })
        .def_prop_ro(
            "has_postselection",
            [](const clifft::CompiledModule& p) { return p.has_postselection; },
            "True if the program contains OP_POSTSELECT instructions.")
        .def_prop_ro("num_instructions",
                     [](const clifft::CompiledModule& p) { return p.bytecode.size(); })
        .def_prop_ro(
            "source_map",
            [](const clifft::CompiledModule& p) {
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
            [](const clifft::CompiledModule& p) {
                return nb::cast(p.source_map.active_k_history());
            },
            "Active dimension k after each instruction (list of uint32).")
        .def(
            "__len__", [](const clifft::CompiledModule& p) { return p.bytecode.size(); },
            "Return the number of bytecode instructions.")
        .def(
            "__getitem__",
            [](const clifft::CompiledModule& p, int64_t idx) -> const clifft::Instruction& {
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
            [](const clifft::CompiledModule& p) {
                return nb::make_iterator(nb::type<clifft::CompiledModule>(), "program_iter",
                                         p.bytecode.begin(), p.bytecode.end());
            },
            nb::keep_alive<0, 1>())
        .def(
            "as_dict",
            [](const clifft::CompiledModule& p) {
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
             [](const clifft::CompiledModule& p) {
                 std::ostringstream ss;
                 for (size_t i = 0; i < p.bytecode.size(); ++i)
                     ss << i << ": " << clifft::format_instruction(p.bytecode[i]) << "\n";
                 return ss.str();
             })
        .def_prop_ro(
            "noise_site_probabilities",
            [](const clifft::CompiledModule& p) {
                auto probs = clifft::noise_site_probabilities(p);
                size_t n = probs.size();
                return vec_to_numpy(std::move(probs), {n});
            },
            nb::rv_policy::move,
            "Per-site total fault probabilities: quantum noise sites followed by\n"
            "readout noise entries. Use for Poisson-Binomial PMF computation.")
        .def("__repr__", [](const clifft::CompiledModule& p) {
            return "Program(" + std::to_string(p.bytecode.size()) +
                   " instructions, peak_rank=" + std::to_string(p.peak_rank) + ", " +
                   std::to_string(p.num_measurements) + " measurements)";
        });

    m.def(
        "lower",
        [](const clifft::HirModule& hir, std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors, std::vector<uint8_t> expected_observables) {
            nb::gil_scoped_release release;
            return clifft::lower(hir, postselection_mask, expected_detectors, expected_observables);
        },
        nb::arg("hir"), nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{},
        "Lower a Heisenberg IR module to executable VM bytecode.\n\n"
        "To optimize the bytecode, use a BytecodePassManager after lowering:\n"
        "    prog = clifft.lower(hir)\n"
        "    bpm = clifft.default_bytecode_pass_manager()\n"
        "    bpm.run(prog)\n\n"
        "Args:\n"
        "    hir: The Heisenberg IR module to lower.\n"
        "    postselection_mask: Optional list of uint8 flags, one per detector.\n"
        "        Detectors where mask[i] != 0 become post-selection checks\n"
        "        that abort the shot early if their parity is non-zero.\n"
        "    expected_detectors: Optional noiseless reference parities for detectors.\n"
        "    expected_observables: Optional noiseless reference parities for observables.\n");

    m.def(
        "compile",
        [](const std::string& stim_text, std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors, std::vector<uint8_t> expected_observables,
           bool normalize_syndromes, clifft::HirPassManager* hir_passes,
           clifft::BytecodePassManager* bytecode_passes) {
            nb::gil_scoped_release release;
            clifft::Circuit circuit = clifft::parse(stim_text);
            clifft::HirModule hir = clifft::trace(circuit);
            if (hir_passes)
                hir_passes->run(hir);

            if (normalize_syndromes) {
                if (!expected_detectors.empty() || !expected_observables.empty()) {
                    throw std::invalid_argument(
                        "Cannot provide expected parities when normalize_syndromes=True");
                }
                auto ref = clifft::compute_reference_syndrome(hir);
                expected_detectors = std::move(ref.detectors);
                expected_observables = std::move(ref.observables);
            }

            auto program =
                clifft::lower(hir, postselection_mask, expected_detectors, expected_observables);
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
        "lower -> [bytecode optimize]. Pass manager arguments are\n"
        "optional; when None the corresponding optimization stage is\n"
        "skipped.\n"
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

    m.def(
        "sample",
        [](const clifft::CompiledModule& program, uint32_t shots, std::optional<uint64_t> seed) {
            if (program.has_postselection) {
                throw nb::value_error(
                    "sample() cannot be used with post-selected programs because it "
                    "returns a fixed number of rows and cannot discard shots. "
                    "Use sample_survivors(program, shots, keep_records=True) instead.");
            }
            clifft::SampleResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sample(program, shots, seed);
            }

            auto meas_arr =
                vec_to_numpy(std::move(result.measurements), {shots, program.num_measurements});
            auto det_arr =
                vec_to_numpy(std::move(result.detectors), {shots, program.num_detectors});
            auto obs_arr =
                vec_to_numpy(std::move(result.observables), {shots, program.num_observables});
            auto ev_arr = vec_to_numpy(std::move(result.exp_vals), {shots, program.num_exp_vals});

            nb::object mod = nb::module_::import_("clifft._sample_result");
            return mod.attr("SampleResult")(meas_arr, det_arr, obs_arr, nb::none(), nb::none(),
                                            nb::none(), nb::none(), ev_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        "Run a compiled program and return a SampleResult.\n\n"
        "Raises ValueError for post-selected programs because fixed-row output\n"
        "cannot represent discarded shots. Use sample_survivors() instead.\n\n"
        "If seed is None (default), uses hardware entropy.\n\n"
        "Returns a SampleResult with .measurements, .detectors, .observables attributes.\n"
        "Supports tuple unpacking: m, d, o = clifft.sample(prog, shots)");

    m.def(
        "sample_k",
        [](const clifft::CompiledModule& program, uint32_t shots, uint32_t k,
           std::optional<uint64_t> seed) {
            if (program.has_postselection) {
                throw nb::value_error(
                    "sample_k() cannot be used with post-selected programs because it "
                    "returns a fixed number of rows and cannot discard shots. "
                    "Use sample_k_survivors(program, shots, k, keep_records=True) instead.");
            }
            clifft::SampleResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sample_k(program, shots, k, seed);
            }

            auto meas_arr =
                vec_to_numpy(std::move(result.measurements), {shots, program.num_measurements});
            auto det_arr =
                vec_to_numpy(std::move(result.detectors), {shots, program.num_detectors});
            auto obs_arr =
                vec_to_numpy(std::move(result.observables), {shots, program.num_observables});
            auto ev_arr = vec_to_numpy(std::move(result.exp_vals), {shots, program.num_exp_vals});

            nb::object mod = nb::module_::import_("clifft._sample_result");
            return mod.attr("SampleResult")(meas_arr, det_arr, obs_arr, nb::none(), nb::none(),
                                            nb::none(), nb::none(), ev_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("k"), nb::arg("seed") = nb::none(),
        "Sample with exactly k forced faults per shot (importance sampling).\n\n"
        "Sites are drawn from the exact conditional Poisson-Binomial\n"
        "distribution. Results are conditioned on K=k and must be combined\n"
        "across strata with P(K=k) weights for correct error rate estimation.\n"
        "Raises ValueError for post-selected programs because fixed-row output\n"
        "cannot represent discarded shots. Use sample_k_survivors() instead.\n\n"
        "For post-selected circuits, weight numerator and denominator\n"
        "separately via sample_k_survivors(): p_fail =\n"
        "sum(P(K=k)*errors_k/shots_k) / sum(P(K=k)*passed_k/shots_k).\n\n"
        "Raises ValueError if the k-fault stratum has zero probability mass\n"
        "(e.g. k exceeds the number of non-zero-probability sites).\n\n"
        "When all site probabilities are equal, an O(k) Fisher-Yates\n"
        "sampler is used automatically.\n\n"
        "Returns a SampleResult with .measurements, .detectors, .observables attributes.\n"
        "Supports tuple unpacking: m, d, o = clifft.sample_k(prog, shots, k)");

    auto make_survivor_result = [](clifft::SurvivorResult result,
                                   const clifft::CompiledModule& program,
                                   bool keep_records) -> nb::object {
        size_t num_obs = result.observable_ones.size();
        auto obs_ones_arr = vec_to_numpy(std::move(result.observable_ones), {num_obs});

        nb::object mod = nb::module_::import_("clifft._sample_result");
        nb::object cls = mod.attr("SampleResult");

        std::vector<uint8_t> meas_storage =
            keep_records ? std::move(result.measurements) : std::vector<uint8_t>{};
        std::vector<uint8_t> det_storage =
            keep_records ? std::move(result.detectors) : std::vector<uint8_t>{};
        std::vector<uint8_t> obs_storage =
            keep_records ? std::move(result.observables) : std::vector<uint8_t>{};
        std::vector<double> ev_storage =
            keep_records ? std::move(result.exp_vals) : std::vector<double>{};

        size_t rows = keep_records ? result.passed_shots : 0;
        auto meas_arr = vec_to_numpy(std::move(meas_storage), {rows, program.num_measurements});
        auto det_arr = vec_to_numpy(std::move(det_storage), {rows, program.num_detectors});
        auto obs_arr = vec_to_numpy(std::move(obs_storage), {rows, program.num_observables});
        auto ev_arr =
            vec_to_numpy(std::move(ev_storage), {rows, static_cast<size_t>(program.num_exp_vals)});
        return cls(meas_arr, det_arr, obs_arr, result.total_shots, result.passed_shots,
                   result.logical_errors, obs_ones_arr, ev_arr);
    };

    m.def(
        "sample_k_survivors",
        [make_survivor_result](const clifft::CompiledModule& program, uint32_t shots, uint32_t k,
                               std::optional<uint64_t> seed, bool keep_records) {
            clifft::SurvivorResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sample_k_survivors(program, shots, k, seed, keep_records);
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
        "Returns a SampleResult. Survivor metadata is always populated via\n"
        ".total_shots, .passed_shots, .discards, .logical_errors, and\n"
        ".observable_ones. Per-shot record arrays\n"
        "(.measurements, .detectors, .observables, .exp_vals) are only\n"
        "filled when keep_records=True; otherwise they are empty (rows=0).");

    m.def(
        "sample_survivors",
        [make_survivor_result](const clifft::CompiledModule& program, uint32_t shots,
                               std::optional<uint64_t> seed, bool keep_records) {
            clifft::SurvivorResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sample_survivors(program, shots, seed, keep_records);
            }
            return make_survivor_result(std::move(result), program, keep_records);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        nb::arg("keep_records") = false,
        "Sample shots and return results only for surviving (non-discarded) shots.\n\n"
        "If seed is None (default), uses hardware entropy.\n\n"
        "Returns a SampleResult. Survivor metadata is always populated via\n"
        ".total_shots, .passed_shots, .discards, .logical_errors, and\n"
        ".observable_ones. Per-shot record arrays\n"
        "(.measurements, .detectors, .observables, .exp_vals) are only\n"
        "filled when keep_records=True; otherwise they are empty (rows=0).");

    nb::class_<clifft::SchrodingerState>(m, "State", "Schrodinger VM execution state")
        .def(nb::new_([](uint32_t peak_rank, uint32_t num_measurements, uint32_t num_detectors,
                         uint32_t num_observables, uint32_t num_exp_vals,
                         std::optional<uint64_t> seed) {
                 return clifft::SchrodingerState({.peak_rank = peak_rank,
                                                  .num_measurements = num_measurements,
                                                  .num_detectors = num_detectors,
                                                  .num_observables = num_observables,
                                                  .num_exp_vals = num_exp_vals,
                                                  .seed = seed});
             }),
             nb::kw_only(), nb::arg("peak_rank"), nb::arg("num_measurements"),
             nb::arg("num_detectors") = 0, nb::arg("num_observables") = 0,
             nb::arg("num_exp_vals") = 0, nb::arg("seed") = nb::none())
        .def("reset", &clifft::SchrodingerState::reset)
        .def("reseed", &clifft::SchrodingerState::reseed, nb::arg("seed"))
        .def_prop_ro("dust_clamps", [](const clifft::SchrodingerState& s) { return s.dust_clamps; })
        .def_prop_ro(
            "meas_record",
            [](const clifft::SchrodingerState& s) { return std::vector<uint8_t>(s.meas_record); })
        .def_prop_ro(
            "det_record",
            [](const clifft::SchrodingerState& s) { return std::vector<uint8_t>(s.det_record); })
        .def_prop_ro(
            "obs_record",
            [](const clifft::SchrodingerState& s) { return std::vector<uint8_t>(s.obs_record); })
        .def_prop_ro(
            "exp_vals",
            [](const clifft::SchrodingerState& s) { return std::vector<double>(s.exp_vals); })
        .def("__repr__", [](const clifft::SchrodingerState& s) {
            return "State(array_size=" + std::to_string(s.array_size()) + ")";
        });

    m.def(
        "execute",
        [](const clifft::CompiledModule& program, clifft::SchrodingerState& state) {
            nb::gil_scoped_release release;
            clifft::execute(program, state);
        },
        nb::arg("program"), nb::arg("state"),
        "Execute a compiled program, updating the state in-place.");

    m.def(
        "get_statevector",
        [](const clifft::CompiledModule& program, const clifft::SchrodingerState& state) {
            std::vector<std::complex<double>> sv;
            {
                nb::gil_scoped_release release;
                sv = clifft::get_statevector(program, state);
            }
            size_t n = sv.size();
            return vec_to_numpy(std::move(sv), {n});
        },
        nb::arg("program"), nb::arg("state"), "Expand the SVM state into a dense statevector.");
}
