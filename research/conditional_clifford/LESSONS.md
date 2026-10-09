# Lessons from approaches set aside

These are the decisions worth retaining from the incremental work. They rule
out particular shortcuts or priorities, not all possible implementations of
the same broad idea. The complete historical records remain at
[`d126c29e`](https://github.com/unitaryfoundation/clifft/tree/d126c29ec60d9dfcecfcbe0171632fd91f4809dc/research/conditional_clifford).

## Complete fault-history caching does not scale with circuit-wide noise

At p=0.001, the original 81-site BT27 layer repeated histories on 97.34% of
10,000 draws. Adding noise throughout BT27 produced 4,170 sites and only 1.74%
repeats; BT81's 24,492 sites produced none. These count ideal unlimited-cache
opportunities in that stream, not measured cache performance or universal hit
rates. Different histories can still have equivalent reduced effects.

Decision: reuse algebra, fault responses and compiled fragments even for fresh
histories. Small per-site probability is not a reason to assume few faults or
frequent complete-history repetition. The current method samples every history
without low-weight truncation.

## A fixed noiseless plan plus output flips is insufficient

A scored BT27 decoder X fault changes the exact visible-law dimension from
30 to 34 random bits. Changing only the offset of `A*r XOR b` cannot change
rank(A). Even equal logical scoring shifts can correspond to different full
record laws. Later audits also found distinct plans at equal active width,
after clearing constant signs and even with a stable squeeze ordering.

Decision: keep fresh planning. A future richer conditional executor might
represent varying supports, but a sign-only patch to one existing plan does
not solve this problem. An identical width, action count or diagnostic hash
is not a plan-equivalence certificate.

## Enumerating conditional sampling maps is not the full-noise solution

Exact Boolean maps were feasible for slices of at most eight fault switches.
Their construction enumerated every setting. The full coefficient-response
rank was 219 for BT27 and 651 for BT81, making enumeration of every distinct
coefficient setting unsuitable. Those ranks are not lower bounds on the number
of measured laws or on the size of a better factored representation.

Decision: retain polynomial-size algebraic responses and sample the varying
Clifford computation directly instead of tabulating all control settings.

## Output-law reduction alone cannot support a quantum continuation

Early terminal samplers could absorb basis changes into measurements or turn
fixed quantum coordinates into classical outputs. Those operations need not
preserve the post-measurement physical state required by later gates. Similarly,
optimized rotations without their final physical frame are an incomplete exit.

Decision: use the state-preserving renderer and validate complete conditional
quantum instruments, including reset mixtures, records and feedback. This is
what permits safe composition, not simply matching a final histogram.

## Transporting one rotation does not make another phase synthesis useful

Cultivation's surviving rotation can cross a supported measured Clifford
bridge. A second generic phase synthesis then produced 39 T gates without
improving complete d3/d5 widths over transport followed by ordinary Clifft.
A small control expanded two input T gates to thirteen, increasing raw width
from two to four. These are representation/synthesis costs, not intrinsic
increases in the state's quantum resource.

Decision: retain the one-rotation carrier, hand the remaining circuit to the
ordinary optimizer, and defer a larger carrier until a useful hard circuit
identifies the actual obstruction. In the latest survey d3 improves from
width four to one; d5 remains at ten.

## More planner caching is not uniformly faster

The recent coordinate audit found that many frames survive only one query;
repeated full Pauli queries were predominantly identity queries. A general
identity shortcut helped the factory and BT. Lazy inverse columns helped BT
further but had mixed results on other inputs; early full inversion was not
a useful uniform policy. Trace reuse alone also had mixed end-to-end results
before boundary composition was reduced.

Decision: the survey uniformly selects the identity shortcut, with guarded
squeeze reuse and fresh planning. It has no family-specific policy lookup.
[The retained paired coordinate study](coordinate-reuse-study.json) supports
this decision; its timings should not be compared directly with a later
unpaired survey to infer a regression.

## A smaller first region is not the end-to-end success metric

Cultivation d5's first region has one rotation, yet the complete circuit
retains width ten. BH distillation shrinks from width five to one but ordinary
Clifft is already much faster. All 24 survey cases within the ordinary width
budget favor ordinary Clifft's per-shot cost.

Decision: measure complete circuits, setup, all visible records, final active
width and sampling cost. The value proposition is extending practical coverage.
Large realistic gains from chaining multiple interacting non-Clifford regions
remain the next structural question. Noise-free BT81 is a separate positive
result, so useful noise-free optimization remains in scope.
