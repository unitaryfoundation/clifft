# Logical-block simulation with an alternative growth implementation

The full f7 prototype accepts a local alternative to its d5-to-d7 growth
encoder, both alone and combined with the changed syndrome schedule from the
[previous study](fold_schedule.md). Both circuits match the independent
coherent reference on 199 diagnostic histories each. The existing native
executor and contraction kernels are unchanged.

This is a bounded positive result for implementation flexibility. The new
encoder changes preparations, CNOT directions, and physical fault maps, but
its compiled syndrome-transfer map is the same as the original. It does not
establish coverage of arbitrary code growth or different fold graphs.

## The alternative encoder

`fold_growth.reconstruction_with_growth` constructs the CSS-dual, quarter-turned
version of the existing regular growth stage. It rotates coordinates by
90 degrees, reverses each CNOT, and exchanges the fresh zero/plus preparation
roles under that rotation. Algebraically this comes from applying code duality
on both sides of the growth isometry; the Hadamards on the old data cancel
when the dual circuit is expressed using reversed CNOTs and fresh preparations.
Signed logical transport is verified from the resulting gates rather than
assumed from this symmetry argument.

For the specific square layout, the transformed circuit has:

- The same 44 fresh resets, 22 Hadamards, and 80 CNOTs.
- Four disjoint local CNOT layers of sizes 22, 22, 18, and 18.
- The same 80 undirected couplers, with four CNOT directions changed.
- Eight fresh qubits whose zero/plus preparation assignment changes.
- A different full Clifford tableau when the reset initializations are removed.

The full circuit still has 99 physical qubits, 221 non-Clifford gates, and
2,205 physical noise locations. Injection, the earlier d3-to-d5 growth, cats,
and all fold checks remain fixed. `dual_rotate` changes only the final growth
stage. The combined variant additionally uses `reverse_checks`, which reverses
each syndrome gadget's CNOT order and the sequence of complete checks.

The noisy circuits are validated against their own fault histories. A changed
gate realization cannot simply reuse the original physical fault maps, even
when its ideal encoded action agrees. Neither circuit is claimed to be an
authors' artifact or a verified fault-distance-seven implementation.

## What the compiler did and did not need

The existing growth certificate already verifies that output stabilizers pull
back to signed products of input stabilizers and fresh initialization
stabilizers, and that both logical axes are transported correctly. The supplied
gate sequence determines the physical Pauli fault maps, selected syndrome
sectors, and transported Pauli representatives. The new encoder required no
special-case contraction, executor action, or representation change.

The study exposed one validation omission in the structured-input planner:
removing resets to form the ideal growth unitary is sound only if every fresh
wire is reset before it participates in a unitary. Construction now checks
that ordering and rejects unrelated wires as well. Existing checks still
reject live-data resets, repeated resets, missing fresh initializations, and
growth that fails signed code/logical transport. These checks occur offline.

For this alternative, all 40 transported input-syndrome representatives,
inverse syndrome-map rows, and output constraints are unchanged. Physical
fault maps do change. This limits the generalization result: it exercises a
different noisy encoder implementation and its composition with changed
measurement order, but not a different ideal syndrome-transfer relation.

The mathematical isometry certificate is broader than this example, but its
broader applicability remains an untested capability, not a measured result.
The fixed-capacity native representation remains at four coherent terms,
13 contractions on a full traversal, 6,199 encoded fault bits, and 2,851 complex
contraction scratch slots. No runtime planning or allocation was introduced.

## Validation

The 199 fixed histories for each variant contain the previous 115-case suite
plus 84 growth-specific cases. Those additional cases insert each supported
single-qubit Pauli fault on both outputs of selected CNOTs near the beginning,
middle, and end of every growth layer, and sample reset/Hadamard fault sites
across the fresh preparations. They supplement natural and stress histories,
paired fold hooks, logical-error tails, flag diagnostics, and nonzero syndrome
sectors; they are not a weighted logical-error-rate estimate.

For both variants, maximum discrepancy against the independent phase-preserving
CH/tableau reference is 1.12e-16 in acceptance and 2.23e-16 in conditional
logical XYZ. Both nonzero growth-sector diagnostics retain acceptance 1/4.
Native execution matches all 199 fixtures per variant to 2.23e-16. The new
growth-only native fixture set and 2,500 sampled attempts pass ASAN/UBSAN with
leak detection disabled.

Independent forward Stim checks prepare all six logical X/Y/Z eigenstates and
verify the output code and logical axes. Another test excites every one of the
40 input syndrome generators, checks the predicted output syndrome, and applies
the compiled transported frame to recover the code and logical Y state. These
checks complement the compiler's backward conjugation certificate.

The full research suite passes 83 tests, including seven new growth tests.
Negative controls cover a removed CNOT, a live reset, use of a fresh wire before
reset, and unrelated ancilla operations that would otherwise cancel. Native
catalog checks verify that the growth catalog accepts its own full circuit and
declines the original and the combined-schedule circuit.

## Sustained sampling

| Circuit | Median us/attempt | Range us/attempt | Survivors per 100,000 |
| --- | ---: | ---: | ---: |
| Original | 45.27 | 45.24-48.25 | 12,667 |
| Alternative growth | 45.56 | 45.26-45.93 | 12,654 |
| Alternative growth and syndrome ordering | 47.12 | 46.98-48.07 | 12,612 |

Each row is three sequential process runs of 100,000 attempted shots, with
p=0.001, seed 19331, full survivor records, and logical XYZ probes. Repetitions
reuse a seed for timing stability and are not independent statistical datasets.
The sampling timer includes setup and output allocation, all physical noise
draws, acceptance, conditional flag records, and probe output. Parsing,
recognition, offline plan generation, C++ compilation, and JSON output are
excluded. Tests and builds finished before timing began on the unpinned VM.

Ordinary Clifft plans peak active width 44 on both changed circuits; dense
sampling was skipped. No f7 speedup ratio or best-in-class result is claimed.
Raw comparisons, certificates, native checks, hashes, and timings are stored in
[fold_growth_data.json](fold_growth_data.json).

## Remaining gap and next step

The study now covers changed syndrome extraction and a changed physical growth
implementation without changing the hot representation. Whole-circuit
recognition still uses a generated schedule catalog. The next useful bounded
step is an offline adapter that locates the supported syndrome/growth regions
in a parsed circuit, certifies the supplied gates, and builds their maps without
requiring a new catalog executable for each schedule.

Keep that adapter outside production initially. It must reject unsupported
structures before execution and preserve the existing survivor output contract.
Its success should be tested on a held-out combination of supported changes,
not just the catalogs used to build it. This is distinct from live-state handoff,
general decoding, a new fold geometry, or a growth map with different syndrome
transport; those remain open. The existing MSC corpus and unavailable authors'
artifacts remain broader applicability targets.

## Reproduction

Use the existing research environment with Stim, NumPy, Cirq Core, and Aer:

```sh
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_schedule.py --schedule original --growth dual_rotate --output /tmp/f7-growth
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_schedule.py --schedule reverse_checks --growth dual_rotate --output /tmp/f7-growth-combined
c++ -std=c++20 -O3 -march=native /tmp/f7-growth/native.cpp -o /tmp/f7-growth/native
/tmp/f7-growth/native 1 10
OPENBLAS_NUM_THREADS=1 python tools/profile/export_fold_recognizer.py --growth dual_rotate --output /tmp/f7-growth/recognizer.cpp
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -I src -I build-research/generated /tmp/f7-growth/recognizer.cpp build-research/src/clifft/libclifft_core.a -o /tmp/f7-growth/recognizer
/tmp/f7-growth/recognizer /tmp/f7-growth/circuit.stim 100000 19331 1 100 ordinary
CLIFFT_RECOGNIZER_NATIVE=/tmp/recognizer CLIFFT_SCHEDULE_RECOGNIZER_NATIVE=/tmp/f7-checks/recognizer CLIFFT_GROWTH_RECOGNIZER_NATIVE=/tmp/f7-growth/recognizer CLIFFT_PROXY_NATIVE=/tmp/proxy-build/profile_proxy OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
```

Generate the other optional executables as described in the preceding reports.
For the combined catalog, add `--schedule reverse_checks` to the recognizer
export command. Repeat matched recognizer/circuit timing runs three times.
For sanitizer validation, compile `native.cpp` with `-O1 -g
-fsanitize=address,undefined -fno-omit-frame-pointer`, then run with
`ASAN_OPTIONS=detect_leaks=0` and arguments `1 500`. Generic planning uses
`build-research/profile_structure CIRCUIT 1 0 1` to avoid dense state allocation.
