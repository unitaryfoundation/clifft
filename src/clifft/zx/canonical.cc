#include "clifft/zx/canonical.h"

#include <cmath>
#include <cstdio>

namespace clifft::zx {
namespace {

void append_phase(std::string& out, const Phase& phase) {
    const double value = phase.value;
    char buffer[32];
    if (phase.is_dyadic()) {
        std::snprintf(buffer, sizeof(buffer), "g%d", static_cast<int>(std::lround(value * 4.0)));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.9f", value);
    }
    out += buffer;
}

}  // namespace

std::string canonical_text(const Graph& g) {
    std::string out;
    for (Vertex v : g.vertices()) {
        out += std::to_string(v);
        out += g.type(v) == VertexType::Z ? " Z " : " X ";
        append_phase(out, g.phase(v));
        out += ' ';
        bool first = true;
        for (uint32_t p : g.phase(v).params.ids()) {
            if (!first) {
                out += ',';
            }
            out += std::to_string(p);
            first = false;
        }
        out += " |";
        for (const Neighbor& n : g.neighbors(v)) {
            out += ' ';
            out += std::to_string(n.vertex);
            out += n.type == EdgeType::Simple ? 's' : 'h';
        }
        out += '\n';
    }
    return out;
}

uint64_t fnv1a64(const std::string& text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

}  // namespace clifft::zx
