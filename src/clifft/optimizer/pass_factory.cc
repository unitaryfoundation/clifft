#include "clifft/optimizer/pass_factory.h"

#include "clifft/optimizer/pass_registry.h"

#include <stdexcept>
#include <string>

namespace clifft {

std::unique_ptr<HirPass> make_hir_pass(std::string_view name) {
    for (const auto& info : kRegisteredPasses) {
        if (info.name == name) {
            return info.make();
        }
    }
    throw std::invalid_argument("Unknown HIR pass: " + std::string(name));
}

HirPassManager default_hir_pass_manager() {
    HirPassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.default_enabled) {
            pm.add_pass(info.make());
        }
    }
    return pm;
}

std::string pass_registry_json() {
    std::string out = "[";
    for (size_t i = 0; i < kNumRegisteredPasses; ++i) {
        const auto& p = kRegisteredPasses[i];
        if (i > 0)
            out += ',';
        out += "{\"name\":\"";
        out += p.name;
        out += "\",\"kind\":\"";
        out += "hir";
        out += "\",\"default\":";
        out += p.default_enabled ? "true" : "false";
        out += ",\"preserves_record_order\":";
        out += p.record_order.preserved ? "true" : "false";
        out += ",\"preserves_instrument_prefix\":";
        out += p.instrument_prefix.preserved ? "true" : "false";
        out += "}";
    }
    out += ']';
    return out;
}

}  // namespace clifft
