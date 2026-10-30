#pragma once

// Graph-like reduction of single-copy parametric ZX diagrams.
//
// A port of zxparam's full_reduce strategy and its rules, restricted to
// classical-domain (single-copy) vertices. The rule set and its order are
// zxparam's: to_gh, spider fusion, copy, identity removal, pivoting, local
// complementation, gadgetization, gadget normalization and gadget fusion, and
// folding isolated pieces into the scalar. Rules that only act across quantum
// and classical domains (decohere, parity_tie, ground_pauli, ground_gadget) and
// boundary pivots never match on closed single-copy diagrams and are omitted.
//
// Each rule pass collects every match first, then applies each one that still
// checks, as zxparam does; the order of matches is vertex then neighbor order.

#include "clifft/zx/graph.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>

namespace clifft::zx {

struct ReduceOptions {
    // Keep isolated pieces whose scalar factor has a parameter-dependent modulus
    // (a non-Pauli phase with parameters). Such a factor weights trajectories,
    // so it must stay a quantum operation rather than become a dropped scalar.
    // Off reproduces zxparam exactly, which folds every isolated piece.
    bool keep_weighted_isolated = true;
    // Hadamard-edge toggles allowed (see Graph::work). Once spent, reduction
    // stops between rewrites, leaving a sound but partly reduced graph.
    uint64_t max_work = std::numeric_limits<uint64_t>::max();
};

// Called after every leaf step of the strategy with the step's rule name and
// whether it changed the graph.
using ReduceTrace = std::function<void(std::string_view rule, bool changed, const Graph& g)>;

// Returns false if the work budget ran out before the graph was reduced.
bool full_reduce(Graph& g, const ReduceOptions& options = {}, const ReduceTrace& trace = {});

// Individual rule passes, each returning whether it changed the graph.
namespace rules {
bool to_gh(Graph& g);
bool spider_fusion(Graph& g);
bool copy(Graph& g);
bool id_remove(Graph& g);
bool pivot(Graph& g);
bool lcomp(Graph& g);
bool pivot_gadget(Graph& g);
bool gadget_normalize(Graph& g);
bool gadget_fusion(Graph& g);
bool remove_isolated(Graph& g, const ReduceOptions& options);
}  // namespace rules

}  // namespace clifft::zx
