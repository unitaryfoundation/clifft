#include "clifft/api/reference_syndrome.h"
#include "clifft/circuit/circuit.h"
#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/noncomp/level.h"
#include "clifft/noncomp/model.h"
#include "clifft/noncomp/policy.h"
#include "clifft/noncomp/sample.h"
#include "clifft/optimizer/drop_non_unitary_pass.h"
#include "clifft/optimizer/hir_pass_manager.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/optimizer/peephole.h"
#include "clifft/optimizer/remove_noise_pass.h"
#include "clifft/optimizer/statevector_squeeze_pass.h"
#include "clifft/sampling/planner.h"
#include "clifft/sampling/sampler.h"
#include "clifft/sampling/state_queries.h"
#include "clifft/util/config.h"
#include "clifft/util/hir_introspection.h"
#include "clifft/util/runtime_isa.h"
#include "clifft/util/version.h"

#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <stdexcept>
#include <variant>

namespace nb = nanobind;

namespace {

using ThreadOption = std::variant<int64_t, std::string>;

uint32_t parse_thread_option(const ThreadOption& option) {
    if (const auto* name = std::get_if<std::string>(&option)) {
        if (*name == "auto") {
            return 0;
        }
        throw std::invalid_argument("threads must be a positive integer or 'auto'");
    }
    const int64_t count = std::get<int64_t>(option);
    if (count <= 0 || static_cast<uint64_t>(count) > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("threads must be a positive integer or 'auto'");
    }
    return static_cast<uint32_t>(count);
}

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

clifft::HirModule prepare_hir_for_lowering(const std::string& stim_text, bool normalize_syndromes,
                                           clifft::HirPassManager* hir_passes,
                                           std::vector<uint8_t>& expected_detectors,
                                           std::vector<uint8_t>& expected_observables) {
    clifft::HirModule hir = clifft::trace(clifft::parse(stim_text));
    if (hir_passes != nullptr) {
        hir_passes->run(hir);
    }
    if (!normalize_syndromes) {
        return hir;
    }
    if (!expected_detectors.empty() || !expected_observables.empty()) {
        throw std::invalid_argument(
            "Cannot provide expected parities when normalize_syndromes=True");
    }
    auto reference = clifft::compute_reference_syndrome(hir);
    expected_detectors = std::move(reference.detectors);
    expected_observables = std::move(reference.observables);
    return hir;
}

}  // namespace

nb::tuple noncomp_sample_to_python(clifft::NonComputationalSample r, uint32_t shots) {
    // These values match Python's QubitStatus enum, including the distinct
    // leaked substates.
    std::vector<uint8_t> status(r.final_status.size());
    for (size_t i = 0; i < r.final_status.size(); ++i) {
        status[i] = static_cast<uint8_t>(r.final_status[i]);
    }
    auto meas = vec_to_numpy(std::move(r.measurements), {shots, r.num_measurements});
    auto det = vec_to_numpy(std::move(r.detectors), {shots, r.num_detectors});
    auto obs = vec_to_numpy(std::move(r.observables), {shots, r.num_observables});
    auto fs = vec_to_numpy(std::move(status), {shots, r.num_qubits});
    auto her = vec_to_numpy(std::move(r.heralds), {shots, r.num_measurements});
    return nb::make_tuple(meas, det, obs, fs, her, r.num_qubits, r.num_measurements,
                          r.num_detectors, r.num_observables);
}

// Noncomputational (leakage/loss) bindings. Raw spec-builder + sampler entry
// points; the ergonomic surface (Model, Classifier, sample) lives in the
// clifft.noncomp Python wrapper. The model is an opaque handle.
void register_noncomp(nb::module_& m) {
    nb::class_<clifft::NonComputationalModel>(m, "_NonComputationalModel");

    m.def(
        "_build_noncomp_model",
        [](std::vector<double> initial_state,
           std::map<std::string, std::vector<std::vector<double>>> transitions,
           std::optional<std::vector<std::vector<double>>> classifier_matrix,
           bool reset_restores_lost, const std::string& damping) {
            clifft::NonComputationalPolicy policy;
            policy.reset_restores_lost = reset_restores_lost;
            if (damping == "exact") {
                policy.damping = clifft::DampingPolicy::Exact;
            } else if (damping == "neglect") {
                policy.damping = clifft::DampingPolicy::Neglect;
            } else {
                throw std::invalid_argument(
                    "noncomp model: damping must be 'exact' or 'neglect', got '" + damping + "'");
            }

            return clifft::NonComputationalModel::from_spec(std::move(initial_state), transitions,
                                                            std::move(classifier_matrix), policy);
        },
        nb::arg("initial_state"), nb::arg("transitions"), nb::arg("classifier_matrix") = nb::none(),
        nb::arg("reset_restores_lost") = false, nb::arg("damping") = "exact",
        "Build the built-in five-level NonComputationalModel from raw matrices. See "
        "clifft.noncomp.Model.");

    m.def(
        "_sample_noncomputational",
        [](const clifft::Circuit& circuit, const clifft::NonComputationalModel& model,
           uint32_t shots, std::optional<uint64_t> seed, std::optional<uint32_t> max_active_width) {
            clifft::NonComputationalSample r;
            {
                nb::gil_scoped_release release;
                r = clifft::sample_noncomputational(circuit, model, shots, seed, max_active_width);
            }
            return noncomp_sample_to_python(std::move(r), shots);
        },
        nb::arg("circuit"), nb::arg("model"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        nb::arg("max_active_width") = nb::none(),
        "Sample a noncomputational model with Clifft's sampler.");
}

NB_MODULE(_clifft_core, m) {
    m.doc() = "Clifft core C++ extension module";

    nb::exception<clifft::ParseError>(m, "ParseError");

    m.def("version", []() { return clifft::kVersion; }, "Return the Clifft version string");
    m.def(
        "runtime_isa",
        []() { return clifft::internal::runtime_isa_name(clifft::internal::runtime_isa()); },
        "Return the resolved kernel ISA: 'scalar', 'avx2', 'avx512', or a 'trap:...' value when "
        "CLIFFT_FORCE_ISA requests an unavailable backend");

    register_noncomp(m);

    // Sentinel-based enum counts for defensive binding tests.
    // If a new enum value is added in C++ but not bound in Python,
    // the test_introspection.py tripwire will catch it.
    m.def("_num_optypes", []() { return static_cast<int>(clifft::OpType::NUM_OP_TYPES); });
    m.def("_num_gate_types", []() {
        return static_cast<int>(sizeof(clifft::detail::kGateTraitsData) /
                                sizeof(clifft::detail::kGateTraitsData[0]));
    });

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
        .value("SPP", clifft::GateType::SPP)
        .value("SPP_DAG", clifft::GateType::SPP_DAG)
        // Non-Clifford
        .value("T", clifft::GateType::T)
        .value("T_DAG", clifft::GateType::T_DAG)
        .value("TPP", clifft::GateType::TPP)
        .value("TPP_DAG", clifft::GateType::TPP_DAG)
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
        .value("DEPOLARIZE3", clifft::GateType::DEPOLARIZE3)
        .value("PAULI_CHANNEL_1", clifft::GateType::PAULI_CHANNEL_1)
        .value("PAULI_CHANNEL_2", clifft::GateType::PAULI_CHANNEL_2)
        .value("PAULI_CHANNEL_3", clifft::GateType::PAULI_CHANNEL_3)
        .value("CORRELATED_ERROR", clifft::GateType::CORRELATED_ERROR)
        .value("ELSE_CORRELATED_ERROR", clifft::GateType::ELSE_CORRELATED_ERROR)
        .value("READOUT_NOISE", clifft::GateType::READOUT_NOISE)
        // Annotations
        .value("DETECTOR", clifft::GateType::DETECTOR)
        .value("OBSERVABLE_INCLUDE", clifft::GateType::OBSERVABLE_INCLUDE)
        .value("TICK", clifft::GateType::TICK)
        // Noncomputational trajectory annotations
        .value("LEVEL_TRANSITION", clifft::GateType::LEVEL_TRANSITION)
        .value("LEAKAGE", clifft::GateType::LEAKAGE)
        .value("LOSS", clifft::GateType::LOSS)
        // Simulation-only probes
        .value("EXP_VAL", clifft::GateType::EXP_VAL)
        // Parse-time rewrites: no AST nodes carry these types
        .value("CH", clifft::GateType::CH)
        .value("CCX", clifft::GateType::CCX)
        .value("CCZ", clifft::GateType::CCZ)
        // Sentinel for unknown/unsupported gates
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
        .def_ro("tag", &clifft::AstNode::tag)
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
        .value("EXP_VAL", clifft::OpType::EXP_VAL)
        .value("INSTRUMENT", clifft::OpType::INSTRUMENT);

    // Python view of a HeisenbergOp paired with the HirModule that owns its
    // mask data. Holds an nb::object ref to the module so the Python
    // wrapper keeps it alive: any wrapper handed out by __getitem__ or
    // __iter__ stays valid until the wrapper itself is collected.
    struct PyHeisenbergOp {
        const clifft::HeisenbergOp* op;
        const clifft::HirModule* hir;
        nb::object module_owner;
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
                         return w.op->has_mask()
                                    ? clifft::format_pauli_mask(w.hir->mask_view(*w.op))
                                    : std::string("+I");
                     })
        .def(
            "as_dict",
            [](const PyHeisenbergOp& w) {
                const auto& op = *w.op;
                nb::dict d;
                d["op_type"] = clifft::op_type_to_str(op.op_type());
                d["pauli_string"] = op.has_mask() ? clifft::format_pauli_mask(w.hir->mask_view(op))
                                                  : std::string("+I");
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
             [](const PyHeisenbergOp& w) {
                 auto mask =
                     w.op->has_mask() ? std::optional{w.hir->mask_view(*w.op)} : std::nullopt;
                 return clifft::format_hir_op(*w.op, mask);
             })
        .def("__repr__", [](const PyHeisenbergOp& w) {
            auto mask = w.op->has_mask() ? std::optional{w.hir->mask_view(*w.op)} : std::nullopt;
            return "<HeisenbergOp: " + clifft::format_hir_op(*w.op, mask) + ">";
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
            [](nb::handle self, int64_t idx) {
                const auto& h = nb::cast<const clifft::HirModule&>(self);
                int64_t size = static_cast<int64_t>(h.ops.size());
                if (idx < 0)
                    idx += size;
                if (idx < 0 || idx >= size)
                    throw nb::index_error();
                return PyHeisenbergOp{&h.ops[static_cast<size_t>(idx)], &h, nb::borrow(self)};
            },
            "Return the HIR operation at the given index.")
        .def("__iter__",
             [](nb::handle self) {
                 const auto& h = nb::cast<const clifft::HirModule&>(self);
                 nb::list items;
                 for (const auto& op : h.ops)
                     items.append(nb::cast(PyHeisenbergOp{&op, &h, nb::borrow(self)}));
                 return items.attr("__iter__")();
             })
        .def(
            "as_dict",
            [](nb::handle self) {
                const auto& h = nb::cast<const clifft::HirModule&>(self);
                nb::dict d;
                d["num_qubits"] = h.num_qubits;
                d["num_measurements"] = h.num_measurements;
                d["num_detectors"] = h.num_detectors;
                d["num_observables"] = h.num_observables;
                nb::list ops;
                for (const auto& op : h.ops) {
                    PyHeisenbergOp w{&op, &h, nb::borrow(self)};
                    ops.append(nb::cast(w).attr("as_dict")());
                }
                d["ops"] = ops;
                return d;
            },
            "Return a JSON-friendly dictionary representation.")
        .def("__str__",
             [](const clifft::HirModule& h) {
                 std::ostringstream ss;
                 for (size_t i = 0; i < h.ops.size(); ++i) {
                     const auto& op = h.ops[i];
                     auto mask = op.has_mask() ? std::optional{h.mask_view(op)} : std::nullopt;
                     ss << i << ": " << clifft::format_hir_op(op, mask) << "\n";
                 }
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
        "Symplectic peephole optimization: cancels and fuses T/T-dag gates, "
        "and removes terminal phases consumed by same-axis measurements.")
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
        "non-Clifford gates rightward to minimize peak active width.")
        .def(nb::init<>());

    nb::class_<clifft::RemoveNoisePass, clifft::HirPass>(
        m, "RemoveNoisePass",
        "Strips all stochastic noise and readout noise ops from the HIR.\n"
        "Not included in the default pass list. Used internally by\n"
        "compute_reference_syndrome() for noiseless reference shots.")
        .def(nb::init<>());

    nb::class_<clifft::DropNonUnitaryPass, clifft::HirPass>(
        m, "DropNonUnitaryPass",
        "Drops non-evolution HIR ops so the remaining program is a unitary skeleton.\n"
        "Not included in the default pass list and not semantics-preserving.")
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

    nb::class_<clifft::sampling::ExecutablePlan>(m, "Program",
                                                 "A reusable compiled sampling program")
        .def_prop_ro("peak_active_width", &clifft::sampling::ExecutablePlan::peak_active_width,
                     "Largest active width reached by the compiled program.")
        .def_prop_ro(
            "peak_rank",
            [](const clifft::sampling::ExecutablePlan& p) {
                if (PyErr_WarnEx(PyExc_DeprecationWarning,
                                 "Program.peak_rank is deprecated; use peak_active_width", 1) < 0) {
                    throw nb::python_error();
                }
                return p.peak_active_width();
            },
            "Deprecated alias for peak_active_width.")
        .def_prop_ro("num_qubits", &clifft::sampling::ExecutablePlan::num_qubits)
        .def_prop_ro("num_measurements", &clifft::sampling::ExecutablePlan::num_visible_records)
        .def_prop_ro("num_hidden_measurements",
                     &clifft::sampling::ExecutablePlan::num_hidden_records)
        .def_prop_ro("num_detectors", &clifft::sampling::ExecutablePlan::num_detectors)
        .def_prop_ro("num_observables", &clifft::sampling::ExecutablePlan::num_observables)
        .def_prop_ro("num_exp_vals", &clifft::sampling::ExecutablePlan::num_exp_vals)
        .def_prop_ro("has_postselection", &clifft::sampling::ExecutablePlan::has_postselection)
        .def_prop_ro("num_actions", &clifft::sampling::ExecutablePlan::num_actions)
        .def_prop_ro(
            "noise_site_probabilities",
            [](const clifft::sampling::ExecutablePlan& p) {
                auto probabilities = p.noise_site_probabilities();
                const size_t size = probabilities.size();
                return vec_to_numpy(std::move(probabilities), {size});
            },
            nb::rv_policy::move,
            "Per-site total fault probabilities: quantum noise sites followed by readout noise.")
        .def("inspect", &clifft::sampling::ExecutablePlan::inspect,
             "Deterministic human-readable diagnostic text for the whole lowered "
             "CPU program.\n\n"
             "The format is diagnostic output for debugging and tooling, not a "
             "stable machine-readable interface.")
        .def("inspect_action", &clifft::sampling::ExecutablePlan::inspect_action, nb::arg("action"),
             "Deterministic human-readable diagnostic text for a single action "
             "in the lowered CPU program.\n\n"
             "The format is diagnostic output for debugging and tooling, not a "
             "stable machine-readable interface.")
        .def("__repr__", [](const clifft::sampling::ExecutablePlan& p) {
            return "Program(" + std::to_string(p.num_actions()) +
                   " actions, peak_active_width=" + std::to_string(p.peak_active_width()) + ", " +
                   std::to_string(p.num_visible_records()) + " measurements)";
        });

    m.def(
        "lower",
        [](const clifft::HirModule& hir, std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors, std::vector<uint8_t> expected_observables) {
            nb::gil_scoped_release release;
            return clifft::sampling::ExecutablePlan(clifft::sampling::plan_sampling(
                hir, {postselection_mask, expected_detectors, expected_observables}));
        },
        nb::arg("hir"), nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{},
        "Lower a Heisenberg IR module to an executable sampling program.\n\n"
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
           bool normalize_syndromes, clifft::HirPassManager* hir_passes) {
            nb::gil_scoped_release release;
            clifft::HirModule hir =
                prepare_hir_for_lowering(stim_text, normalize_syndromes, hir_passes,
                                         expected_detectors, expected_observables);

            return clifft::sampling::ExecutablePlan(clifft::sampling::plan_sampling(
                hir, {postselection_mask, expected_detectors, expected_observables}));
        },
        nb::arg("stim_text"), nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{},
        nb::arg("normalize_syndromes") = false, nb::arg("hir_passes") = nb::none(),
        "Compile a quantum circuit string to an executable program.\n\n"
        "Compilation plans optimized HIR and prepares it for Clifft's sampler.\n"
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
        "    hir_passes: Optional HirPassManager to run on the HIR before lowering.\n");

    m.def(
        "sample",
        [](const clifft::sampling::ExecutablePlan& program, uint32_t shots,
           std::optional<uint64_t> seed, const ThreadOption& thread_option) {
            if (program.has_postselection()) {
                throw nb::value_error(
                    "sample() cannot be used with post-selected programs because it "
                    "returns a fixed number of rows and cannot discard shots. "
                    "Use sample_survivors(program, shots, keep_records=True) instead.");
            }
            const uint32_t threads = parse_thread_option(thread_option);
            clifft::sampling::SamplingResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sampling::sample(program, shots, seed, threads);
            }

            auto meas_arr = vec_to_numpy(std::move(result.measurements),
                                         {shots, program.num_visible_records()});
            auto det_arr =
                vec_to_numpy(std::move(result.detectors), {shots, program.num_detectors()});
            auto obs_arr =
                vec_to_numpy(std::move(result.observables), {shots, program.num_observables()});
            auto ev_arr = vec_to_numpy(std::move(result.exp_vals), {shots, program.num_exp_vals()});

            nb::object mod = nb::module_::import_("clifft._sample_result");
            return mod.attr("SampleResult")(meas_arr, det_arr, obs_arr, nb::none(), nb::none(),
                                            nb::none(), nb::none(), ev_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        nb::arg("threads") = int64_t{1},
        "Run a compiled program and return a SampleResult.\n\n"
        "Raises ValueError for post-selected programs because fixed-row output\n"
        "cannot represent discarded shots. Use sample_survivors() instead.\n\n"
        "If seed is None (default), uses hardware entropy. threads is a positive\n"
        "worker count or 'auto' to use the implementation-reported hardware\n"
        "concurrency; it defaults to 1.\n\n"
        "Returns a SampleResult with .measurements, .detectors, .observables attributes.\n"
        "Supports tuple unpacking: m, d, o = clifft.sample(prog, shots)");

    m.def(
        "sample_k",
        [](const clifft::sampling::ExecutablePlan& program, uint32_t shots, uint32_t k,
           std::optional<uint64_t> seed, const ThreadOption& thread_option) {
            if (program.has_postselection()) {
                throw nb::value_error(
                    "sample_k() cannot be used with post-selected programs because it "
                    "returns a fixed number of rows and cannot discard shots. "
                    "Use sample_k_survivors(program, shots, k, keep_records=True) instead.");
            }
            const uint32_t threads = parse_thread_option(thread_option);
            clifft::sampling::SamplingResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sampling::sample_k(program, shots, k, seed, threads);
            }

            auto meas_arr = vec_to_numpy(std::move(result.measurements),
                                         {shots, program.num_visible_records()});
            auto det_arr =
                vec_to_numpy(std::move(result.detectors), {shots, program.num_detectors()});
            auto obs_arr =
                vec_to_numpy(std::move(result.observables), {shots, program.num_observables()});
            auto ev_arr = vec_to_numpy(std::move(result.exp_vals), {shots, program.num_exp_vals()});

            nb::object mod = nb::module_::import_("clifft._sample_result");
            return mod.attr("SampleResult")(meas_arr, det_arr, obs_arr, nb::none(), nb::none(),
                                            nb::none(), nb::none(), ev_arr);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("k"), nb::arg("seed") = nb::none(),
        nb::arg("threads") = int64_t{1},
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
        "sampler is used automatically. threads is a positive worker count\n"
        "or 'auto' to use the implementation-reported hardware concurrency;\n"
        "it defaults to 1.\n\n"
        "Returns a SampleResult with .measurements, .detectors, .observables attributes.\n"
        "Supports tuple unpacking: m, d, o = clifft.sample_k(prog, shots, k)");

    auto make_survivor_result = [](clifft::sampling::SamplingSurvivorResult result,
                                   const clifft::sampling::ExecutablePlan& program,
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
        auto meas_arr =
            vec_to_numpy(std::move(meas_storage), {rows, program.num_visible_records()});
        auto det_arr = vec_to_numpy(std::move(det_storage), {rows, program.num_detectors()});
        auto obs_arr = vec_to_numpy(std::move(obs_storage), {rows, program.num_observables()});
        auto ev_arr = vec_to_numpy(std::move(ev_storage), {rows, program.num_exp_vals()});
        return cls(meas_arr, det_arr, obs_arr, result.total_shots, result.passed_shots,
                   result.logical_errors, obs_ones_arr, ev_arr);
    };

    m.def(
        "sample_k_survivors",
        [make_survivor_result](const clifft::sampling::ExecutablePlan& program, uint32_t shots,
                               uint32_t k, std::optional<uint64_t> seed, bool keep_records,
                               const ThreadOption& thread_option) {
            const uint32_t threads = parse_thread_option(thread_option);
            clifft::sampling::SamplingSurvivorResult result;
            {
                nb::gil_scoped_release release;
                result = clifft::sampling::sample_k_survivors(program, shots, k, seed, keep_records,
                                                              threads);
            }
            return make_survivor_result(std::move(result), program, keep_records);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("k"), nb::arg("seed") = nb::none(),
        nb::arg("keep_records") = false, nb::arg("threads") = int64_t{1},
        "Sample survivors with exactly k forced faults per shot.\n\n"
        "Results are conditioned on K=k. To estimate the overall logical\n"
        "error rate across strata, weight numerator and denominator\n"
        "separately to account for k-dependent survival probability:\n"
        "  p_fail = sum(P(K=k)*logical_errors_k/shots_k)\n"
        "         / sum(P(K=k)*passed_k/shots_k)\n\n"
        "Raises ValueError if the k-fault stratum has zero probability mass.\n"
        "threads is a positive worker count or 'auto' to use the\n"
        "implementation-reported hardware concurrency; it defaults to 1.\n\n"
        "Returns a SampleResult. Survivor metadata is always populated via\n"
        ".total_shots, .passed_shots, .discards, .logical_errors, and\n"
        ".observable_ones. Per-shot record arrays\n"
        "(.measurements, .detectors, .observables, .exp_vals) are only\n"
        "filled when keep_records=True; otherwise they are empty (rows=0).");

    m.def(
        "sample_survivors",
        [make_survivor_result](const clifft::sampling::ExecutablePlan& program, uint32_t shots,
                               std::optional<uint64_t> seed, bool keep_records,
                               const ThreadOption& thread_option) {
            const uint32_t threads = parse_thread_option(thread_option);
            clifft::sampling::SamplingSurvivorResult result;
            {
                nb::gil_scoped_release release;
                result =
                    clifft::sampling::sample_survivors(program, shots, seed, keep_records, threads);
            }
            return make_survivor_result(std::move(result), program, keep_records);
        },
        nb::arg("program"), nb::arg("shots"), nb::arg("seed") = nb::none(),
        nb::arg("keep_records") = false, nb::arg("threads") = int64_t{1},
        "Sample shots and return results only for surviving (non-discarded) shots.\n\n"
        "If seed is None (default), uses hardware entropy. threads is a positive\n"
        "worker count or 'auto' to use the implementation-reported hardware\n"
        "concurrency; it defaults to 1.\n\n"
        "Returns a SampleResult. Survivor metadata is always populated via\n"
        ".total_shots, .passed_shots, .discards, .logical_errors, and\n"
        ".observable_ones. Per-shot record arrays\n"
        "(.measurements, .detectors, .observables, .exp_vals) are only\n"
        "filled when keep_records=True; otherwise they are empty (rows=0).");

    m.def(
        "get_statevector",
        [](const clifft::sampling::ExecutablePlan& program) {
            std::vector<std::complex<double>> statevector;
            {
                nb::gil_scoped_release release;
                statevector = clifft::sampling::get_statevector(program);
            }
            const size_t size = statevector.size();
            return vec_to_numpy(std::move(statevector), {size});
        },
        nb::arg("program"),
        "Return the dense final statevector of a compiled pure-unitary program.");

    m.def(
        "_basis_probabilities_from_bitmasks",
        [](const clifft::sampling::ExecutablePlan& program,
           nb::ndarray<nb::numpy, const uint64_t, nb::shape<-1, -1>, nb::c_contig> basis_masks) {
            std::vector<double> probs;
            {
                nb::gil_scoped_release release;
                probs = clifft::sampling::basis_probabilities(
                    program, std::span<const uint64_t>(basis_masks.data(), basis_masks.size()),
                    basis_masks.shape(0), basis_masks.shape(1));
            }
            size_t n = probs.size();
            return vec_to_numpy(std::move(probs), {n});
        },
        nb::arg("program"), nb::arg("basis_masks"),
        "Internal helper for clifft.basis_probabilities().");

    m.def(
        "_record_probabilities_from_records",
        [](const clifft::sampling::ExecutablePlan& program,
           nb::ndarray<nb::numpy, const uint8_t, nb::shape<-1, -1>, nb::c_contig> records) {
            std::vector<double> log_probs;
            {
                nb::gil_scoped_release release;
                log_probs = clifft::sampling::record_log_probabilities(
                    program, std::span<const uint8_t>(records.data(), records.size()),
                    records.shape(0));
            }
            size_t n = log_probs.size();
            return vec_to_numpy(std::move(log_probs), {n});
        },
        nb::arg("program"), nb::arg("records"),
        "Internal helper for clifft.record_probabilities(). Returns log-probabilities; "
        "the Python wrapper exponentiates to linear unless return_log=True.");
}
