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

Before this design, `sample_noncomputational` resolved all transitions ahead
of time (AOT): a history sampler drew jump outcomes per shot before
compilation, and the rewriter edited the circuit accordingly. That was
*exact* when the source was classical — a known level, or a transition whose
rate is the same from every computational level — and otherwise rejected. An
`unknown_source_policy="equalize_rates"` mode approximated the case instead
(retired — §6 step 8, executed early); it had three documented residual
approximations:

1. **Joint correlations** — the AOT source draw is independent of entangled
   partners and measurement outcomes.
2. **Gate-determined-but-unknown** — a qubit whose level is deterministic by
   circuit algebra (e.g. `H·H`) but not instruction-known is treated as
   unknown, so a jump can fire from a level the qubit provably does not occupy.
3. **Equalization dephasing** — the diagonal padding that equalizes per-source
   rates adds dephasing the physical channel does not have.

This design adds exact runtime resolution that removes all three at once by
evaluating jumps against the live simulator state. Validated by the step-7
campaign, it is now the **only** sampling path: pre-release, both
alternatives were removed rather than carried — `equalize_rates` for its
approximations (strictly dominated by `damping="neglect"`, §4.6, FAQ), and
the `reject` strict guard because a contract check does not need to be a
sampling policy (a future `noncomp.validate_static(circuit, model)` is the
named home for "this model should be statically resolvable").

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
| dormant, outcome random         | genuinely non-Clifford — needs expansion, or the `neglect` fallback (§4.6) | exactly `(p_g+p_e)/2` — state-independent, poolable | **+1 to k** exact, ~free under `neglect` |

(The dormant-random fire probability is exactly `(p_g+p_e)/2` because such a
qubit's source populations are exactly half–half — the same stabilizer
property that makes its measurement a fair coin. Only the damp filter is
ever non-Clifford there.)

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
HIR or bytecode pass may move any operation across one, and no pass
decision about operations before an instrument may depend on anything
after it. The enforcement mechanism is the positional-barrier treatment
the pipeline already gives `EXP_VAL` ("a positional probe: never reorder
anything across it"): `INSTRUMENT` joins that clause in the two places
motion happens — the reordering pass's commutation gate and the
peephole's commute-past check. The addition must be explicit, because
both otherwise decide by Pauli-mask commutation and an instrument
carries a mask; the adjacency-driven bytecode passes need nothing, since
an unrecognized opcode already ends their fusion runs. Two pinned tests
guard the contract: a barrier test (no fusion or motion across an
instrument) and a prefix-identity test — compile the same prefix against
two different suffixes and assert the bytecode is bit-identical up to
the instrument's offset — which also guards passes written later.

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

The fencing cost was measured before building the machinery (§6 step 2)
by running the default pass pipelines per fence-delimited segment on
representative workloads (surface d7r7 at p=1e-3, d5r5 at p=0.05,
cultivation d5) — segmentation being a mechanism-independent way to
produce the barrier effects before the op type existed; the lost-fusion
costs are identical under the positional-barrier treatment. At the
realistic density (fences clustered at gate positions, modeled as
noise-run starts) compile time, sampling throughput, and peak rank were
unchanged within measurement noise. At the atomized upper bound (a fence
at every noise site) sampling slowed 7-25%, dominated by lost
noise-block coalescing — the regime instrument hazard-pooling (§4.4) is
designed to recover. Verdict: go.

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
  object and survive the module switch. A continuation compiled with a
  larger peak rank grows the state's amplitude array at that boundary —
  the single sanctioned exception to the VM's allocate-once invariant,
  host-side between dispatch entries, never inside a kernel; a driver
  reusing one state across shots amortizes growth to the chain maximum,
  and its between-shot rebuilds size to the running maxima, never down
  to the triggering module. Preallocating instead of growing would mean
  bounding the continuation family ahead of time — the §3.1 dead end.
  When memory must be capped, `max_rank` is the hard bound: any module
  beyond it fails compilation loudly, so growth never exceeds
  `2^max_rank` amplitudes.

Dormant-random sites are handled per the damping policy (§4.6): under
`exact`, the compiler emits an expansion before the instrument (the fused
expand-plus-diagonal pattern already exists for rotations), raising `k` by
one at compile time; under `neglect`, the site keeps only its exact
ingredients — the state-independent Bernoulli fire draw at `(p_g+p_e)/2`
(pooled into the noise hazard table) — and applies no no-fire back-action
(`k` stable). One implementation refinement over the sketch above: at a
neglect-mode dormant-random site *every* fire traps, including
computational destinations. The in-line collapse of a dormant-random
qubit re-anchors the Pauli frame, and the compiler cannot append the
coordinate-aligning virtual Hadamard for an anchor that only happens on a
runtime-conditional branch (a measurement's anchor is unconditional, so
its compile-time alignment is sound). The trap's suffix rewrite handles
the collapse at source level instead; fires are rare, so the cost is a
cached continuation, not a hot path.

The uncollapsed handover costs no correlation, because past the trap
the continuation is fire-branch-only and the conditional convention
shift becomes unconditional there: the driver's suffix rewrite emits
the leaked qubit's trace-out as a hidden measurement **forced to the
reported source**, so entangled partners collapse consistently with the
destination-side effects. (A bare frame anchor at trap time would be
wrong — after `H 0`, anchoring the dormant representation to `|s⟩`
yields the physical state `H|s⟩`, not `|s⟩`; the collapse must execute
inside compiled code that carries the basis alignment.) With that,
`neglect`'s only approximation is the omitted no-fire back-action.
Everywhere the collapse *can* happen in-line (active, expand, and
dormant-deterministic forms), the runtime collapses onto the drawn
source *before* trapping, which makes the continuation's trace-out
deterministic and keeps the correlation exact with no forcing needed.

New opcodes land in the shared kernel include and are compiled per-ISA like
every other opcode; the instruction's 24-byte payload carries the axis, the
constant-pool site index, and the damp coefficients.

### 4.5 Orchestrator: the driver loop and the continuation cache

```
main = cache.main_line(model)                 # compiled once per model
for shot:
    state = init(shot_seed)
    preload initial computational levels       # Pauli-frame preload, no recompile
    execute(main, state)
    while state.pending_trap is Trap(site, source, destination_pending):
        if destination_pending:                # neglect-form site: the VM drew
            dest = draw from column(source)    #   nothing; full column, computational
        else:                                  #   destinations included (host RNG)
            dest = draw from trap remainder    # class already drawn leaked/lost; only
                   of column(source)           #   the level remains (host RNG)
        events  = sample classical-source follow-ons (seepage,  # host RNG,
                  restore) over the remaining ops               # existing sampler
        key     = (site, dest, events)
        module, offset = cache.get_or_compile(key)
            # rewrite(circuit, statuses) on ops after site  — existing rewriter:
            # drops, trace-out R (forced to `source` at a neglect-form site),
            # carrier materialization, classifier handling
        resume(module, state, offset)          # offset follows the trapped site
    emit records + status sidecar (from the trap chain)
```

Notes:

- **In exact mode, all transition firing moves to runtime** — including
  source-independent sites (as hazard-pooled Bernoulli instruments). This is
  what makes the main line shot-invariant; statistically it is identical to
  an ahead-of-time draw (memoryless Bernoulli either way). The driver's own
  draws are initial statuses and, at trap time, transitions whose source is
  a classical status.
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
    damping="exact",                 # "exact" (default) | "neglect"
)
```

- Exact runtime resolution is the only sampling path; there is no
  unknown-source policy knob. The `"equalize_rates"` approximation and the
  `"reject"` strict guard were both removed pre-release (§6 step 8 and the
  post-plan note): the former is strictly dominated by `damping="neglect"`
  below (see FAQ), and the latter's contract check — "this model should
  never need runtime resolution" — is a validator's job, not a sampling
  mode's, with `noncomp.validate_static(circuit, model)` as its named
  future home.
- `damping` controls the no-jump filter at sites where the qubit is coherent
  but dormant (§2.2 row 3) — the only place exactness ever costs `k` growth:
  - `"exact"` (default): expand and apply the filter exactly. `k` grows by
    one per such site; if the compile exceeds the rank cap, it fails with an
    error **naming the site and qubit**, so the cost is visible and
    attributable rather than silently approximated.
  - `"neglect"`: omit the no-fire back-action; everything else stays exact —
    the fire probability (exactly `(p_g+p_e)/2` at such sites,
    state-independent, so it pools into the hazard table at no marginal
    cost) and the on-fire Born collapse onto the source. The omission is a
    pure amplitude tilt of the surviving state toward the lower-rate level:
    an `O(|p_g−p_e|)` effect per no-fire site, exactly zero for
    source-independent rates, with no dephasing added. Uniform per model,
    machine-visible. The FAQ explains why this fallback and not others.
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
2. Micro-probes, closed form: the gate-determined probe pins "fires
   exactly 0" and the Bell-pair joint probe pins TVD 0. (The equalized
   mode's counterpart pins — fires ~p/2, TVD 0.5 — were removed with the
   mode itself in step 8, executed early.)
3. New discriminating probe: prepare `|+⟩`, pass one no-fire
   leak-from-`e` site, measure X. Exact coherence is `sqrt(1-p)`;
   `damping="neglect"` gives 1 — this probe is the direct measurement of
   the fallback's omission and pins the boundary between the two damping
   modes. Only the exact mode lands on the dense reference.
4. Repetition-code circuits at published parameter magnitudes: total
   variation distance to the dense reference should reach shot noise.
5. Performance gates: fence overhead on the standard compile benchmarks
   (§6 step 2), main-line runtime overhead per instrument form, and the
   expected end-to-end speedup from the shared main line vs. today's
   per-shot compile.

### Step-7 results

Items 1-4 above are implemented in `tests/python` (the oracle's per-site
Kraus channel and its self-checks, the `utils_noncomp_enumerator`
reference, the closed-form probe suite, and the repetition-code TVD runs
under both damping modes). The campaign also caught a real driver bug --
cross-shot state reuse sized rebuilds to the triggering module instead of
the running maxima, an out-of-bounds write in Release -- which is its own
argument for keeping it.

Performance (item 5), measured on a Linux VM with a GCC 13 Release build
via `tools/bench/test_bench_noncomp.py` and small ad-hoc sweeps; informal
best-of-run magnitudes, not microbenchmark-grade:

| Measurement | Result |
| --- | --- |
| Lossless noncomp end-to-end, per-shot AOT (`reject`) vs shared main line (`exact`) | d3-r3: 5.09 ms vs 0.39 ms per 200 shots (**13x**); d17-r5: 32.0 ms vs 2.24 ms per 100 shots (**14x**) |
| Exact-mode overhead over plain sampling (lossless model) | ~22 us/shot at d17-r5 (vs ~1 us plain), dominated by the per-shot classical status walk -- optimization headroom, not a blocker |
| Main-line cost per instrument site per shot (8-site micro circuits, non-firing rates) | dormant-static 23 ns; dormant-random `damping="exact"` 23 ns; `"neglect"` 7 ns; active 22 ns; known source 8 ns |
| Realistic leak+loss (p = 0.01 hooked S layer, traps live) | d3-r3: 0.68 ms per 200 shots; d17-r5: 40.7 ms per 100 shots -- trap-chain compiles dominate at low shot counts and amortize per unique outcome, as designed |
| Main-line peak rank, 11-site d3 repetition round | `damping="exact"`: k = 3 (one expansion per data qubit; MR compaction bounds the growth); `"neglect"`: k = 0 |

The statevector-squeeze exclusion (§6 step 6) is free for the
Clifford-plus-instrument workloads that exist today: instrument fences
already forbid motion across sites, and rank growth happens at the
fences themselves, so there is nothing squeeze could compact that the
exclusion loses (the peak-rank probe above confirms rank equals the
per-qubit expansion count). A measurable cost requires non-Clifford
segments between sites; restoring squeeze behind a pinned-measurement
barrier stays a parked follow-up until such workloads exist.

Deferred-optimization arbitration on these numbers: hazard pooling,
suffix-only compilation, LRU capping of the continuation cache, and
trapped-shot batching all stay deferred -- nothing is dramatically
slower, the shared main line already beats the old per-shot pipeline by
an order of magnitude, and trap costs amortize with shots.

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
2. **Fence de-risk spike (done — go).** Measured by running the default
   pipelines per fence-delimited segment at two densities; numbers and
   verdict in §4.2. The segmented-run mechanism was measurement
   scaffolding, not part of the shipped design: fences are the §4.2
   positional-barrier clauses. The spike's barrier tests, exact
   record-probability equivalence test, and benchmark harness are
   rebuilt against the real op type in step 4.
3. **SVM kernels.** Damp/evaluate (fused active-axis pass), expand+damp,
   forced-source collapse variants, renormalization. Standalone,
   oracle-tested against dense results. No user surface.
4. **HIR `INSTRUMENT` + fences + lowering + offset table.** Inert until the
   orchestrator uses it. Fences land as the §4.2 positional-barrier
   clauses, with the pinned barrier and prefix-identity tests.
5. **Trap/resume.** Trap return path from `execute()`, `resume(offset)`,
   state persistence across module switches (records buffer sized to the
   max slot count across a chain; noise-gap cursor re-anchored at the entry
   offset — exact by memorylessness).
6. **Orchestrator driver + continuation cache + frame-preload initial
   states + seeding.** Exact mode becomes usable here; includes the debug
   prefix-identity assertion.
7. **Validation campaign, docs, default flip (done).** The campaign landed
   as the dense oracle's per-site Kraus channel, the first-principles
   enumerator, and the closed-form probes, with rep-code TVD runs under
   both damping modes; the performance results live at the end of §5;
   exact runtime resolution became the default (and, per the post-plan
   note below, then the only path).
8. **Retire `equalize_rates` (done — executed before step 7).** Removed
   the equalized mode: its sampler code path, policy value, tests, notebook
   coverage, and the base design note's policy text for it. Pre-release,
   nothing depended on the mode and `damping="neglect"` is the designated
   fallback, so retiring first let the validation campaign judge the final
   surface in one pass instead of validating a doomed mode's pins alongside
   new ones.

Post-plan (pre-release): with the campaign green and the default flipped,
the `reject` strict guard and the ahead-of-time trajectory pipeline behind
it were removed as well — exact runtime resolution is the only sampling
path, and the final policy surface is `damping` plus `reset_restores_lost`.
(The `lost_leaked_ops` reject/drop knob was later removed too: dropping an
op with no representable effect on a vacated site is the only behavior, so
it needs no policy — see noncomputational-mvp.md §5.2.3.) The AOT
pipeline's differential-oracle role passed
to the first-principles enumerator (`tests/python`), its per-shot-compile
baseline is recorded in the step-7 results above, and the strict guard's
contract check is a future validator's job. Every future feature targets
one pipeline.

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

**Why not importance weights to get exactness without the damp?** Weighted
trajectories can correct branch *probabilities*, not a wrong post-branch
*state*: with the damp omitted, the surviving state itself is wrong (that
omission, made explicit, is the `neglect` fallback), and no reweighting fixes
a state. Weights would also change the product semantics — `sample()`
returns unweighted records.

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
lossy sites. Options, in order: annotate fewer sites, or set
`damping="neglect"` for the large-scale runs and reserve full exactness for
the smaller circuits that validate them.

**Which fallbacks were considered for the no-fire back-action, and why
`neglect`?** First, the scope of the question: the only thing that ever
needs approximating is the no-fire damp at coherent-but-dormant sites. The
fire probability there is exactly `(p_g+p_e)/2` (§2.2), and the on-fire
collapse is a real physical projection — both stay exact in every mode. And
the exact no-fire back-action, *conditioned on surviving*, is a pure filter:
it tilts the surviving amplitudes toward the lower-rate level by
`O(|p_g−p_e|)` and adds no dephasing. The candidates for replacing it:

- **Neglect it** (chosen). The error is exactly and only the missed
  survivorship tilt: `O(|p_g−p_e|)` per no-fire site, zero for
  source-independent rates, no dephasing introduced. The tilt is non-unital,
  so no Pauli/dephasing channel can reproduce it — the only exact
  representation is the filter itself (expansion), which is what
  `damping="exact"` does.
- **Collapse at the site** (measure `Z_q`, then fire on the outcome).
  Locally exact populations, but it fully dephases the qubit at *every*
  site, fired or not — an `O(1)` error wherever downstream interference
  matters. Concretely, in `H; site; H; M(Z)` with leak-from-`e` at rate `p`:
  exact gives `P(M=1, no leak) ≈ p²/16`; `neglect` gives 0 (error `O(p²)`);
  collapse gives **1/2** (error `O(1)`). In a stabilizer-code circuit it
  Z-projects data qubits mid-round, randomizing the opposite-basis checks.
  Note the superficially similar prior art in fast-path simulators collapses
  *on fire* — which is exact, and which this design already does — not on
  the no-fire path. Rejected.
- **Rate equalization** (the retired AOT approximation). Its no-fire branch
  is also the identity, so it misses the same tilt — and then adds source
  misattribution, padding dephasing, and gate-determined misfires on top,
  with no cost advantage over `neglect`. Strictly dominated; hence §6
  step 8.
- **Importance weights** — see the previous answer.
- **A hybrid budget policy** (exact until a `k` budget, then neglect).
  Declined for now: it makes the approximation depend on where in the shot
  the budget ran out, which is miserable to reason about. Revisit only if
  real profiles demand it.

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

**Why retire `equalize_rates` rather than keep it as the cheap mode?**
Because `damping="neglect"` is the better cheap mode on every axis (previous
answer) at the same or lower cost, and there is no released user base
creating a compatibility obligation. Keeping two approximations that differ
only in which extra errors they add works against the model's legibility.
The nothing-inexact-by-default principle is unchanged: the pipeline is
exact end to end, and the one approximation that remains is a named,
machine-visible knob (`damping="neglect"`).
