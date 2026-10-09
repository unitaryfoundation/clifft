// Export an optimized, unmeasured zero-input preparation for research reuse.

#include "clifft/circuit/parser.h"
#include "clifft/frontend/frontend.h"
#include "clifft/optimizer/pass_factory.h"
#include "clifft/util/hir_introspection.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string dense(clifft::PauliStringView p) {
    std::string result = p.sign() ? "-" : "+";
    for (uint32_t q = 0; q < p.num_qubits(); ++q)
        result += "IXZY"[p.x().bit_get(q) + 2 * p.z().bit_get(q)];
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Usage: export_optimized_prefix CIRCUIT");
        std::ifstream input(argv[1]);
        if (!input)
            throw std::invalid_argument("Cannot read input circuit");
        std::ostringstream source;
        source << input.rdbuf();
        auto hir = clifft::trace(clifft::parse(source.str()));
        const auto require_unitary = [&] {
            if (hir.num_measurements || hir.num_hidden_measurements || hir.num_detectors ||
                hir.num_observables || hir.num_exp_vals || !hir.noise_sites.empty() ||
                !hir.readout_noise.empty() || !hir.instrument_sites.empty() || !hir.final_tableau)
                throw std::invalid_argument("Expected an unmeasured deterministic preparation");
            for (const auto& op : hir.ops)
                if (op.op_type() != clifft::OpType::T_GATE)
                    throw std::invalid_argument("Preparation contains an unsupported operation");
        };
        require_unitary();
        const auto input_t = hir.num_t_gates();
        for (const auto* name : {"PeepholeFusionPass", "PhasePolynomialPass",
                                 "RotationSimplificationPass", "StatevectorSqueezePass"})
            clifft::make_hir_pass(name)->run(hir);
        require_unitary();
        std::cout << "{\"qubits\":" << hir.num_qubits << ",\"input_t\":" << input_t
                  << ",\"output_t\":" << hir.num_t_gates() << ",\"rotations\":[";
        bool first = true;
        for (const auto& op : hir.ops) {
            auto axis = clifft::format_pauli_mask(hir.mask_view(op));
            if (axis == "+I" || axis == "-I")
                continue;
            if (!first)
                std::cout << ',';
            first = false;
            std::cout << '"' << ((op.is_dagger() ^ hir.sign(op)) ? "TPP_DAG " : "TPP ")
                      << axis.substr(1) << '"';
        }
        for (const auto* label : {"xs", "zs"}) {
            std::cout << (std::string(label) == "xs" ? "],\"xs\":[" : "],\"zs\":[");
            for (uint32_t q = 0; q < hir.num_qubits; ++q) {
                if (q)
                    std::cout << ',';
                const auto p = std::string(label) == "xs" ? hir.final_tableau->x_output(q)
                                                          : hir.final_tableau->z_output(q);
                std::cout << '"' << dense(p) << '"';
            }
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
