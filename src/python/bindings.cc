#include "ucc/util/config.h"
#include "ucc/util/version.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(_ucc_core, m) {
    m.doc() = "UCC core C++ extension module";

    // Version info (from generated header, source of truth is pyproject.toml)
    m.def("version", []() { return ucc::kVersion; }, "Return the UCC version string");

    // Expose compile-time constants
    m.def(
        "max_sim_qubits", []() { return ucc::kMaxInlineQubits; },
        "Return the maximum number of qubits supported by the simulator");
}
