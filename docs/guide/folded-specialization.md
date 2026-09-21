# Experimental folded MSC specialization

`clifft.compile(text, specialize_folded=True)` optionally recognizes a narrow
family of terminal fold-transversal magic-state cultivation regions and uses
a specialized kernel. It returns the usual `Program`, for the usual `sample`
and `sample_survivors` calls. The option defaults to `False`.

```python
program = clifft.compile(circuit_text, specialize_folded=True)
print(program.has_folded_regions)
print(program.inspect())
result = clifft.sample(program, shots=100, seed=123, threads=2)
```

For postselection, supply the same detector mask and expected parities as in
ordinary compilation:

```python
program = clifft.compile(
    circuit_text,
    specialize_folded=True,
    postselection_mask=detector_mask,
    normalize_syndromes=True,
)
survivors = clifft.sample_survivors(program, shots=1000, seed=123, keep_records=True)
```

Measurements, detectors, observables, noise-site probabilities, and survivor
accounting retain their normal meanings. Seeds reproduce results across shot
worker counts. Specialized and ordinary execution need not produce identical
shots for the same seed, since they sample quantum outcomes differently.

## Recognition and supported execution

The recognizer matches the reconstructed distance-3, distance-5, and distance-7
terminal regions: two folded logical checks, the following CSS extraction,
final noiseless syndrome extraction, and the final logical check. It accepts
qubit renumbering, arbitrary probabilities at the supported physical noise
locations, and omission of those noise locations. Comments, coordinates,
detector definitions, and observable definitions do not identify the block.
The recognized gate schedule is intentionally strict; equivalent reordered
circuits are not necessarily recognized.

The ordinary compiler processes the prefix and certifies that its output is
one logical qubit in the required signed CSS code, with classical ancillas.
If gate matching or boundary certification fails, the whole circuit follows
ordinary compilation. `program.inspect()` explains the outcome. No fixtures,
research scripts, or external simulator packages are loaded by compilation.

The initial implementation supports CPU scalar shots and parallel shot
workers. Automatic batching chooses scalar execution. Explicit packed batches,
HIP execution, fixed-fault sampling (`sample_k`, `sample_k_survivors`), and
record-probability replay are unsupported for specialized programs and report
an error. Use ordinary compilation for these workflows. Final-state queries
are unavailable because the kernel represents a terminal measurement region.

Default HIR optimization passes and `hir_passes=None` are supported. An
explicit custom HIR pass manager causes ordinary compilation: its passes must
apply to the entire circuit, including any region they might transform.
OpenQASM input also follows ordinary compilation.

## Implementation and current limits

The Python compiler precomputes code translations, fault effects, record
slots, and contraction gathers. It passes an in-memory plan to C++, which
certifies the prefix boundary and creates a `SampleFoldedRegion` sampling-plan
action. Lowering prepares boundary expectation probes and maps its physical
fault channels to the normal presampled noise symbols.

Each worker owns preallocated mutable contraction scratch and shares the
immutable contraction plan. During dispatch, the kernel obtains the logical
state and code signs from the probes, binds the sampled faults, contracts the
folded checks, and writes the original record slots. Detector and observable
actions remain ordinary Clifft actions. The hot path performs no topology
planning or heap allocation.

This integration currently keeps injection and growth in the ordinary prefix.
It does **not** yet include the research prototype's specialized f5-to-f7 growth
handoff. Prefix detector rejection happens before the region; rejection based
on its records happens after the whole region. There is no early rejection
inside the kernel yet. `peak_active_width` describes the ordinary coefficient
storage and does not include contraction-plan or workspace memory.

This is an opt-in integration experiment, without a performance cost model.
Small distance-3 circuits can be faster with ordinary Clifft. It should not be
read as a new speedup claim against the PR 471/472 optimized baseline.

The fixtures are reconstructed cultivation circuits through fictitious error
detection and a final noiseless logical measurement, not the complete protocol
with escape and decoding. Their provenance and schedule assumptions are recorded in
`tests/fixtures/folded/README.md` in the source checkout.
