#include "clifft/optimizer/pass_factory.h"

#include "clifft/optimizer/pass_registry.h"

#include <stdexcept>
#include <string>

namespace clifft {

std::unique_ptr<HirPass> make_hir_pass(std::string_view name) {
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::HIR && info.name == name) {
            return info.make_hir();
        }
    }
    throw std::invalid_argument("Unknown HIR pass: " + std::string(name));
}

std::unique_ptr<BytecodePass> make_bytecode_pass(std::string_view name) {
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::Bytecode && info.name == name) {
            return info.make_bc();
        }
    }
    throw std::invalid_argument("Unknown bytecode pass: " + std::string(name));
}

HirPassManager default_hir_pass_manager() {
    HirPassManager pm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::HIR && info.default_enabled) {
            pm.add_pass(info.make_hir());
        }
    }
    return pm;
}

BytecodePassManager default_bytecode_pass_manager() {
    BytecodePassManager bpm;
    for (const auto& info : kRegisteredPasses) {
        if (info.kind == PassKind::Bytecode && info.default_enabled) {
            bpm.add_pass(info.make_bc());
        }
    }
    return bpm;
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
        out += (p.kind == PassKind::HIR) ? "hir" : "bytecode";
        out += "\",\"default\":";
        out += p.default_enabled ? "true" : "false";
        out += "}";
    }
    out += ']';
    return out;
}

}  // namespace clifft
