<!--pytest-codeblocks:skipfile-->
# Exact State-Dependent Jumps — Design Note

Design for adding exact, state-dependent loss/leakage transitions to the
noncomputational model. Builds on the semantics in
`design/noncomputational-mvp.md` (status kinds, policy tables, rewrite rules);
prior discussion in #62 (structural loss) and #103 (state-dependent
transitions and the compile-time frame constraint). The API surface described
here is experimental until stabilized.

## 1. Goal

A *state-dependent jump* is a loss/leakage transition whose firing probability
depends on the quantum state of the target qubit — e.g. a gate that leaks
population out of `e` at rate `p` fires with probability `p · ⟨P_e⟩`, where
`⟨P_e⟩` is the population of `e` at that point in the circuit, including
through superposition and entanglement.

Today `sample_noncomputational` resolves all transitions ahead of time (AOT):
the history sampler draws jump outcomes per shot before compilation, and the
rewriter edits the circuit accordingly. This is *exact* when the source is
classical — a known level, or a transition whose rate is the same from every
computational level — and otherwise either rejects (default) or approximates
(`unknown_source_policy="equalize_rates"`). The equalized mode has three
documented residual approximations:

1. **Joint correlations** — the AOT source draw is independent of entangled
   partners and measurement outcomes.
2. **Gate-determined-but-unknown** — a qubit whose level is deterministic by
   circuit algebra (e.g. `H·H`) but not instruction-known is treated as
   unknown, so a jump can fire from a level the qubit provably does not occupy.
3. **Equalization dephasing** — the diagonal padding that equalizes per-source
   rates adds dephasing the physical channel does not have.

This design adds a third policy value, `unknown_source_policy="exact"`, that
removes all three at once by evaluating jumps against the live simulator
state. Once validated, `exact` becomes the default. `reject` and
`equalize_rates` remain available (strictness and compatibility/performance
modes respectively).

A large side benefit falls out of the architecture: in `exact` mode the
common no-jump execution path is compiled **once per model** instead of once
per shot, so typical workloads get faster, not slower (§4.5).

## 2. The physical model, and what it costs

### 2.1 The per-site instrument

Attach a transition with per-source jump probabilities `p_g`, `p_e` (to some
destination level per source) to an operation on qubit `q`. The exact channel
applied immediately after the ideal operation is a quantum instrument with
Kraus operators

```
K_jump,s = sqrt(p_s) |dest_s><s|          (one per source s in {g, e})
K_stay   = sqrt(1-p_g) P_g + sqrt(1-p_e) P_e
```

Two consequences drive everything else:

- **Fire probability is runtime state.** The jump branch fires with
  probability `p_g·⟨P_g⟩ + p_e·⟨P_e⟩` evaluated on the live state. It cannot
  be drawn before execution.
- **The no-jump branch is not the identity.** `K_stay` is a *weak damping
  filter* (`a·I + b·Z_q` in spectral form). Physically: surviving a lossy
  channel is itself evidence about the state, and it damps `g`/`e` coherence
  by `sqrt((1-p_g)(1-p_e))`. This back-action must be applied at **every**
  annotated site on **every** shot — it cannot be confined to sparsely
  sampled candidate sites (see FAQ for the proof sketch). This single fact
  rules out "pre-sample where jumps fire, compile a small branch tree per
  shot" as an exact architecture.

The special case that stays cheap: when `p_g = p_e = p` (source-independent
total rate, e.g. uniform atom loss), `K_stay = sqrt(1-p)·I` is a scalar — no
back-action, and the fire probability `p` is state-independent. This is
exactly why the existing AOT path is already exact for source-independent
transitions.

### 2.2 Cost taxonomy

The damping filter is non-Clifford in general, but its cost at a given site is
determined by how the compiler represents the qubit at that point — which is
known statically (the same active/dormant + localized-basis classification
that selects measurement opcodes today):

| Qubit at the site               | Damp filter                          | Fire evaluation                 | Cost |
| ------------------------------- | ------------------------------------ | ------------------------------- | ---- |
| dormant, outcome deterministic  | scalar — normalizes away             | frame-bit-conditioned Bernoulli | ~free; poolable in the noise hazard table |
| active axis                     | `diag(1, r)` in place + renormalize  | fused into the same `O(2^k)` pass | one array pass, like a T-gate site |
| dormant, outcome random         | genuinely non-Clifford               | requires expansion first        | **+1 to k** (see §4.6 policy) |

This generalizes clifft's core contract honestly: runtime cost is exponential
in *(T-count + number of damped coherent sites)*. Small circuits (Bell pairs,
analytic micro-cases, few-qubit codes) are exact and cheap. Large circuits
where every data qubit is coherent at every annotated site are exponentially
expensive *because the exact physics is*; §4.6 gives those users an explicit,
machine-visible, per-site approximation knob rather than a silent one.

## 3. Strategy (high level)

Three moves, in one sentence each:

1. **Every source-dependent site compiles to a straight-line `INSTRUMENT`
   instruction** that evaluates the fire probability on the live state,
   applies the damping filter on the (overwhelmingly common) no-fire path, and
   *falls through* — no control transfer.
2. **A fire that changes the qubit's status to Leaked/Lost is a resumable
   trap**: execution halts, the host driver rewrites the remaining circuit at
   source level under the now-known statuses (the existing rewriter, in its
   existing "statuses known at compile time" regime), recompiles, and resumes
   execution at the trap site's bytecode offset.
3. **Continuations are cached across shots**, keyed by (site, status
   outcome), so rare fires amortize; the no-fire main line is a single module
   shared by all shots.

Equivalently: the MVP invariant "statuses are known when the compiler runs"
is preserved — exact mode just makes the boundary where a compilation
*starts* dynamic, at fire events.

### 3.1 Why this shape and not the alternatives

- **Not precompiled branch arms.** Fire decisions depend on runtime state at
  every annotated site (§2.1), so enumerating arms ahead of execution is
  `2^(sites)`, not `2^(fires)`. Fires are rare at physical rates, so lazy
  compilation with caching is both necessary and sufficient.
- **No jump instructions in the bytecode, no blocks/regions in the HIR.**
  Branch arms here never rejoin: after a fork, the two paths' virtual frames
  (`V_cum`) diverge permanently, so a merge point would need frame
  re-synchronization gates for no benefit. A control-flow graph over
  non-rejoining rare branches degenerates to exactly what "flat programs +
  lazily compiled suffixes" already is. What the IR actually needs is the
  small useful fragment of basic blocks — **fence points** (§4.2) — and what
  the runtime needs is **trap/resume**. Control flow itself lives host-side
  in the orchestrator, keeping the architecture's isolation invariant: the VM
  remains a dumb flat loop over 32-byte instructions.
- **Frame safety by reconstruction, not patching.** The compile-time Clifford
  frame means runtime code cannot be skipped or edited in place (#103's
  constraint). Here, no compiled artifact is ever executed under a frame it
  was not compiled for: the continuation is a fresh deterministic compile,
  and re-entry is valid because its prefix is bit-identical to what already
  ran (§4.3).

## 4. Detailed design

### 4.1 HIR: `OpType::INSTRUMENT` + side-table

One new mask-carrying op type (like `MEASURE` — the source projector `Z_q`
maps through the virtual frame exactly like a measurement mask), whose
payload indexes an `InstrumentSite` side-table on `HirModule` (the
`NoiseSite` pattern):

```
struct InstrumentSite {
    uint32_t site_id;        // stable id, keys the trap protocol + offset table
    uint32_t qubit;
    // Per computational source s: total fire probability p_s, split into
    // computational-destination sub-probabilities (with the destination bit)
    // and the trap (Leaked/Lost destination) remainder.
    ...
    // Damp coefficients sqrt(1-p_g), sqrt(1-p_e).
};
```

Sites are materialized at trace time from the model's transition hooks, one
per (annotated operation, qubit operand), placed **immediately after** the
ideal operation (the standard op-then-noise circuit convention; source
populations are post-op — see FAQ). Only computational-source content lowers
to instruments; transitions whose source is a classical status (Leaked/Lost,
e.g. seepage) remain host-side (§4.5).

Compile-time specialization and elision:

- Qubit instruction-known at the site with `p_level = 0` → **elided**.
- Source-independent (`p_g = p_e`) → damp elided (scalar), Bernoulli fire
  only, pooled into the existing noise hazard table for `O(1)` skip.
- Otherwise → one of the three §2.2 forms, selected by the localization
  result.

### 4.2 Instrument sites are optimization fences

Instrument sites are **optimization barriers in the original program**: no
HIR or bytecode pass may move any operation across one. This is enforced
structurally, not by per-pass discipline — the pass managers split the op
stream into instrument-delimited segments and run passes per segment.

Fences are required for correctness independent of performance
considerations:

1. **Sound source-level cuts.** On a trap at site `s`, the remaining work is
   defined as "the circuit's operations after `s`", and the rewriter operates
   at source level. If a pass had moved an op across `s`, that op would be
   either double-applied or dropped by the suffix rewrite.
2. **Prefix identity for re-entry.** Making prefix compilation a function of
   the prefix alone (no cross-fence lookahead) is what guarantees the
   recompiled continuation's prefix is bit-identical to the code that already
   executed (§4.3).

The performance cost of fencing is measured before the rest is built (§6).

### 4.3 Backend: lowering, and frame composition across the trap boundary

Lowering threads the virtual frame `V_cum` through instruments exactly as it
does through measurements — localize the source Pauli, emit the specialized
opcode against the pivot axis. **No changes to the frame machinery.** The
backend additionally emits a `site_id → bytecode offset` table into the
compiled module.

The trap boundary is the crux, and the rule is: **never patch — recompile and
re-enter.**

- The continuation is the *full* circuit — unchanged prefix + rewritten
  suffix — compiled from scratch. Execution re-enters at the trap site's
  offset in the new module.
- Re-entry is valid iff the new module's prefix (everything before the
  offset) is identical to what already executed, so that the live runtime
  state (dense array, Pauli frame bits, `active_k`, record slots, gamma) is
  exactly the state the continuation's code expects at that offset. Two
  properties deliver this:
  1. the compiler is deterministic, and
  2. fences (§4.2) make prefix compilation independent of anything after the
     fence — including the suffix edits.
- Consequently the leaked/lost qubit's downstream operations are removed at
  source level before compilation; the no-fire path's baked-in Clifford
  propagation for those operations simply never executes on the fire path.
  Runtime code is never skipped.
- A debug-mode assertion recompiles the prefix and byte-compares it against
  the executed module on every cache insertion, so a determinism regression
  is caught loudly rather than as state corruption.

Compiling only the suffix from a saved frame snapshot (rather than the full
circuit) is a straightforward later optimization — the `VirtualFrame` gate
log can replay `V_cum` up to any site — but v1 uses full recompilation:
fewer new pipeline entry points, and the cost is per *unique* trap, not per
shot, thanks to caching.

Branch paths never rejoin (§3.1), so per shot an execution is a chain:
main line, then one segment per fired trap. Phase bookkeeping is a non-issue
by existing contract: instrumented programs contain measurements, so results
are already defined only up to global phase, and per-continuation
`global_weight` differences are unobservable.

### 4.4 SVM runtime

Per instrument instruction:

- **Evaluate** the source populations. Active axis: one pass over the array
  (fused with the damp below). Dormant-deterministic: read the frame bit.
- **Draw** fire / no-fire (and on fire, the destination category) from the
  shot's RNG stream.
- **No-fire** (common case): apply the damp — `diag(r_g, r_e)` on the active
  axis with renormalization via the existing `scale_magnitude` machinery, or
  a no-op scalar on deterministic qubits — and fall through.
- **Fire, computational destination** (e.g. relaxation `e→g`): resolved
  entirely in-line, **no trap**: collapse onto the source (the forced-outcome
  measurement kernels already implement weighted forced collapse), apply the
  Pauli fixup if `dest ≠ source`, and continue. The collapse projects in
  place without deactivating the axis, preserving the compiled layout.
  Downstream code is unaffected because computational statuses never change
  the rewrite.
- **Fire, noncomputational destination**: collapse onto the source, then
  **trap**: `execute()` returns a resumable `TrapResult{site_id, source}`
  with the `SchrodingerState` intact. A new `resume(module, state, offset)`
  entry point continues execution; measurement records, detector/observable
  records, frame bits, gamma, and the RNG all already live in the state
  object and survive the module switch.

Dormant-random sites are handled per the damping policy (§4.6): under
`exact`, the compiler emits an expansion before the instrument (the fused
expand-plus-diagonal pattern already exists for rotations), raising `k` by
one at compile time; under `collapse`, it emits a frame-based collapse (k
stable) followed by the deterministic-qubit form.

New opcodes land in the shared kernel include and are compiled per-ISA like
every other opcode; the instruction's 24-byte payload carries the axis, the
constant-pool site index, and the damp coefficients.

### 4.5 Orchestrator: the driver loop and the continuation cache

```
main = cache.main_line(model)                 # compiled once per model
for shot:
    state = init(shot_seed)
    preload initial computational levels       # Pauli-frame preload, no recompile
    result = execute(main, state)
    while result is Trap(site, source):
        dest    = draw destination from column(source)          # host RNG
        events  = sample classical-source follow-ons (seepage,  # host RNG,
                  restore) over the remaining ops               # existing sampler
        key     = (site, dest, events)
        module, offset = cache.get_or_compile(key)
            # rewrite(circuit, statuses) on ops after site  — existing rewriter:
            # drops, trace-out R, carrier materialization, classifier handling
        result = resume(module, state, offset)
    emit records + status sidecar (from the trap chain)
```

Notes:

- **In exact mode, all transition firing moves to runtime** — including
  source-independent sites (as hazard-pooled Bernoulli instruments). This is
  what makes the main line shot-invariant; statistically it is identical to
  the AOT draw (memoryless Bernoulli either way). The AOT history sampler's
  remaining jobs are initial statuses and, at trap time, transitions whose
  source is a classical status.
- **Classifier injection becomes runtime-stochastic.** Today's injector
  pre-draws a per-shot bit and bakes it into `MPAD`; that is incompatible
  with sharing modules across shots. Instead the rewrite emits `MPAD(0)`
  followed by `READOUT_NOISE(P(symbol=1 | level))`, moving the draw into the
  VM. (This applies to the AOT modes too, and retires the per-shot replay in
  the injector.)
- **Initial computational population errors do not fragment the cache.**
  A `|1⟩` initial level is an X at time zero, i.e. a Pauli — and compiled
  programs are Pauli-frame covariant, so it is applied as a per-shot frame
  preload on the initial state, not a rewrite. Only initial Leaked/Lost draws
  (rare) require their own cached modules.
- **Cache** is keyed by the status-outcome delta as above, LRU-capped, and
  in-memory per sampler call. At physical rates the hit rate is high: most
  shots take zero traps; single-trap shots cluster on few (site, dest) pairs.
- **Seeding** follows the existing domain-separated scheme: per-shot
  sub-seeds for the SVM stream and the host-side draws. Same-version, same
  seed runs reproduce exactly; the draw *order* differs from the AOT modes,
  so seeds are not stable across modes or feature versions.
- v1 handles trapped shots serially after the (parallelizable) main-line
  pass; batching trapped shots by cache key is a later optimization.

### 4.6 Policy and API surface

```python
model = noncomp.Model(
    ...,
    unknown_source_policy="exact",   # new value; joins "reject" (default today,
                                     # "exact" becomes default after validation)
                                     # and "equalize_rates"
    damping="exact",                 # exact-mode only: "exact" (default) |
                                     # "collapse"
)
```

- `unknown_source_policy="exact"` — the mode this note adds.
- `damping` controls the no-jump filter at sites where the qubit is coherent
  but dormant (§2.2 row 3):
  - `"exact"` (default): expand and apply the filter exactly. `k` grows by
    one per such site; if the compile exceeds the rank cap, it fails with an
    error **naming the site and qubit**, so the cost is visible and
    attributable rather than silently approximated.
  - `"collapse"`: collapse the qubit (frame-based measurement, `k` stable)
    and fire on the outcome. Populations and fire statistics stay exact;
    `g`/`e` coherence at that site is destroyed. This is the approximation
    fast-path stabilizer simulators commonly make at *every* jump site;
    here it is opt-in, per-model, and machine-visible.
- Restrictions in v1 (explicit errors, all lift-able later): instrumented
  programs do not support `get_statevector` (the final tableau is
  per-continuation), `EXP_VAL` (already rejected by the noncomputational
  path), or `record_probabilities`/forced-fault execution.

## 5. What exact mode removes (validation targets)

| Residual approximation | Why it disappears |
| --- | --- |
| Joint correlations | The fire draw conditions on `⟨P_s⟩` of the live, entangled state; the collapse is a genuine Born projection correlated with partners and subsequent measurements. |
| Gate-determined-but-unknown | Sites where the tableau makes the level deterministic compile to the frame-conditioned form; a provably-`g` qubit has fire probability exactly 0 (and the instrument is often elided outright). |
| Equalization dephasing | There is no rate padding; the only non-unitary no-fire effect is the physical damping filter itself. |

Validation plan (extends the existing oracle/probe suite):

1. Add the exact per-site channel to the density-matrix test oracle and
   cross-check every supported scenario end-to-end.
2. Micro-probes, closed form: the gate-determined probe flips from
   "equalized fires ~p/2" to "fires exactly 0" — the existing pinned
   divergence test is *expected* to change, consciously. The Bell-pair joint
   probe flips from TVD 0.5 to 0.
3. New discriminating probe: prepare `|+⟩`, pass one no-fire
   leak-from-`e` site, measure X. Exact coherence is `sqrt(1-p)`; the
   current AOT mode gives 1; per-site collapse gives 0. Only the exact mode
   lands on the dense reference.
4. Repetition-code circuits at published parameter magnitudes: total
   variation distance to the dense reference should reach shot noise.
5. Performance gates: fence overhead on the standard compile benchmarks
   (§6 step 2), main-line runtime overhead per instrument form, and the
   expected end-to-end speedup from the shared main line vs. today's
   per-shot compile.

## 6. Implementation plan

Ordered, each step a reviewable PR. Two steps are deliberately pulled to the
front: the classifier-injection conversion, because it is a standalone
improvement the existing AOT modes benefit from immediately, and the
fence-overhead measurement, because it is the one result that could force a
redesign. Steps 1–3 are mutually independent and can proceed in parallel
with unrelated roadmap work.

1. **Runtime-stochastic classifier injection.** Replace the injector's
   per-shot pre-drawn `MPAD` bit with `MPAD(0)` + `READOUT_NOISE` in all
   modes (§4.5). Statistically identical, retires the injector's replay of
   the transition record wholesale (and with it that replay's validation
   gap), and is a prerequisite for sharing compiled modules across shots.
   Independently landable and testable.
2. **Fence de-risk spike.** Pass-manager segmentation plus inert fence ops
   inserted at realistic site densities; measure compile-time and runtime
   deltas on the standard benchmarks. **Go/no-go for the fencing approach**,
   before any of the machinery below is built.
3. **SVM kernels.** Damp/evaluate (fused active-axis pass), expand+damp,
   forced-source collapse variants, renormalization. Standalone,
   oracle-tested against dense results. No user surface.
4. **HIR `INSTRUMENT` + fences + lowering + offset table.** Inert until the
   orchestrator uses it; instrument-form emission validated against the
   spike's segmentation.
5. **Trap/resume.** Trap return path from `execute()`, `resume(offset)`,
   state persistence across module switches (records buffer sized to the
   max slot count across a chain; noise-gap cursor re-anchored at the entry
   offset — exact by memorylessness).
6. **Orchestrator driver + continuation cache + frame-preload initial
   states + seeding.** Exact mode becomes usable here; includes the debug
   prefix-identity assertion.
7. **Validation campaign, docs, default flip.** Oracle extension, probe
   flips, rep-code TVD runs, performance table; then flip the default
   `unknown_source_policy` to `"exact"`.

Mechanics: new core `.cc` files must be added to both `src/clifft/` and
`src/wasm/` source lists; new opcodes/op types must be reflected in the
introspection tables and binding completeness checks.

## 7. FAQ

**Why can't the damping be applied only at pre-sampled candidate sites?**
Suppose sites are AOT-tagged as candidates with probability `q` and
non-candidates execute the identity. Trace preservation forces each
classical group's Kraus sum to be proportional to the identity, and the
channel's `g`/`e` coherence coefficient `sqrt(1-p)` then bounds as
`(1-q)·1 + sqrt(q(q-p))`, which is strictly below `sqrt(1-p)` for every
`q < 1` (Cauchy–Schwarz within each group). The best such scheme is
therefore the exact channel *plus* parasitic dephasing of order `p²/q` per
site — order `p` when `q ~ p`, i.e. as large as the physics being modeled.
Saturating Cauchy–Schwarz also shows every no-jump Kraus operator in *any*
decomposition of this instrument is proportional to the damping filter: there
is no Clifford-friendly unraveling to switch to. Hence: damp everywhere, and
in return the fire test may run everywhere at the full physical rate with no
AOT thinning at all.

**Why not importance weights instead of damping?** Weighted trajectories can
correct branch *probabilities*, not a wrong post-branch *state*; skipping the
damp leaves the state wrong at `O(p)` per site. It would also change the
product semantics — `sample()` returns unweighted records.

**Why do source-independent sites move to runtime too? The AOT path was
already exact for them.** Statistically nothing changes (a memoryless
Bernoulli draw is the same before or during execution). Architecturally it is
what makes the no-fire main line identical across shots, which is what makes
it cacheable. The runtime cost is absorbed by the existing noise hazard
table (`O(1)` skip past silent sites).

**When is a fire resolved without a trap?** When the destination is
computational (e.g. relaxation `e→g`): collapse + Pauli fixup in-line, no
recompile, because computational statuses never change the downstream
rewrite. Only Leaked/Lost destinations trap. Relaxation-heavy models
therefore rarely trap at all.

**Where exactly does the instrument sit, and what state does it probe?** The
transition channel acts immediately after the ideal operation, so source
populations are post-op. This matches the standard circuit-noise convention
(noise follows the ideal gate) and is what makes gate-determined cases exact.
Note the AOT modes select the source column by *entry* status; the two
conventions agree except when the annotated operation itself maps between
computational levels (e.g. `X`). Aligning or explicitly documenting the AOT
corner is called out as a conscious semantic pin during implementation.

**What about detectors whose targets span a trap boundary?** Record slots
are absolute indices into a record buffer owned by the runtime state, which
persists across `resume`. A detector in a continuation reads prefix-era slots
directly. The visible record layout is invariant across all continuations
(measurements are never dropped by the rewriter — the existing MVP
invariant); hidden slot counts may differ per continuation, so the buffer is
sized to the max across the chain.

**Do branch paths ever rejoin, and could they?** No, and by design: after a
fork the virtual frames diverge permanently, so a join would require emitting
frame re-synchronization Cliffords for no benefit — every path runs to shot
end. This is also the core reason blocks/regions and bytecode jump
instructions buy nothing here (§3.1).

**What if `k` hits the rank cap under `damping="exact"`?** Compilation fails
with an error naming the site and qubit. That is the honest answer: the
requested physics is exponentially expensive at that density of coherent
lossy sites. Options, in order: annotate fewer sites, set
`damping="collapse"`, or use `equalize_rates` for large-scale runs and
reserve `exact` for the smaller circuits that validate them.

**Two-qubit gates with transitions on both operands?** Two instruments, one
per operand, in operand order — matching the existing per-operand AOT
semantics. Correlated multi-qubit transition models are out of scope; the
trap protocol extends naturally (a trap may carry a multi-qubit status
delta) if a model ever needs it.

**Measurements and resets with transitions?** Unchanged op-relative
semantics from the base design note (§5.2): the instrument follows the
operation; Z-basis measurement keeps its pre-measurement instruction-known
status for classical bookkeeping, while the instrument's quantum probe acts
on the post-op state. X/Y/MPP measurements on noncomputational operands
continue to reject.

**Does the RNG reproduce across modes/versions?** Within one mode and
version, same seed → same results, as today (domain-separated per-shot
streams). Exact mode consumes randomness in a different order than the AOT
modes by construction, so cross-mode seed stability is explicitly not a
contract.

**How does this interact with per-shot compile costs today?** It removes
them for typical shots: today `sample_noncomputational` compiles inside the
shot loop; exact mode compiles the main line once per model and per-trap
continuations on demand (cached). A trap costs one full compile on cache
miss — a per-unique-outcome cost, not per shot — and the full-recompile
strategy can later be swapped for suffix-only compilation from a frame
snapshot (the gate log already supports replaying `V_cum` to any site)
without changing any semantics.

**Why is `equalize_rates` kept at all?** As the explicit, documented
approximation for workloads that cannot afford exactness — and as the
compatibility reference for fast-path comparisons. The refuse-by-default
principle stands: nothing silently degrades; every approximation is a named
knob.
