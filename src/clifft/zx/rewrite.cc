#include "clifft/zx/rewrite.h"

#include "clifft/zx/emit.h"
#include "clifft/zx/from_hir.h"
#include "clifft/zx/pbc.h"
#include "clifft/zx/schedule.h"
#include "clifft/zx/simplify.h"

#include <utility>
#include <vector>

namespace clifft::zx {

std::optional<HirModule> rewrite_hir(const HirModule& hir, const RewriteOptions& options,
                                     std::string* reason) {
    auto reject = [&](std::string why) -> std::optional<HirModule> {
        if (reason) {
            *reason = std::move(why);
        }
        return std::nullopt;
    };
    if (hir.forced_traceout_slot) {
        return reject("a forced trace-out names a hidden measurement");
    }
    if (hir.num_measurements == 0) {
        return reject("the circuit has no measurement records");
    }
    const auto diagram = build_single_copy_diagram(hir, reason, options.max_qubits);
    if (!diagram) {
        return std::nullopt;
    }
    Graph reduced = diagram->graph;
    if (!full_reduce(reduced, {.max_work = options.max_reduce_work})) {
        return reject("the diagram is too dense to reduce within the work limit");
    }
    const auto program = read_pbc(*diagram, reduced, reason);
    if (!program) {
        return std::nullopt;
    }
    std::vector<std::vector<uint32_t>> orders;
    orders.reserve(program->groups.size());
    for (const PbcGroup& group : program->groups) {
        orders.push_back(greedy_schedule(group).order);
    }
    auto rewritten = emit_hir(hir, *diagram, *program, orders, reason);
    if (rewritten && rewritten->num_qubits > options.max_qubits) {
        return reject("the rewrite would exceed the qubit limit");
    }
    return rewritten;
}

}  // namespace clifft::zx
