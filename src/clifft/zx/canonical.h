#pragma once

// An implementation-independent text form of a graph and its FNV-1a hash.
//
// Fixtures generated from zxparam pin every rule pass by this hash, so the
// format is part of the fixture contract (tools/zx_fixtures/generate.py writes
// the same text). One line per live vertex in ID order:
//
//   <id> <Z|X> <phase> <param,...> | <neighbor><s|h> ...
//
// with neighbors in neighbor order. A phase on the pi/4 grid prints as g<k>
// for k quarter turns of pi; any other phase prints with nine decimals.

#include "clifft/zx/graph.h"

#include <cstdint>
#include <string>

namespace clifft::zx {

[[nodiscard]] std::string canonical_text(const Graph& g);
[[nodiscard]] uint64_t fnv1a64(const std::string& text);

}  // namespace clifft::zx
