# Complete noisy five-check factor sequence

The factor contractions now drive a complete native sequence: physical fault
sampling and binding, five coherent gadget measurements, five CSS syndrome
projections, logical-state/frame handoffs, final physical logical-axis readout,
readout errors, and every record/detector/observable output. On the synthetic
37-data-qubit sequence, median time is **1,234 us/attempt versus 15,199 us for
ordinary Clifft**, approximately **12.3x faster**. The smaller controls still
favor ordinary Clifft. The 61-data-qubit sequence completes in **4,401 us**;
its ordinary Clifft plan needs 32 GiB for coefficients alone and was not run.

This closes the end-to-end execution gap left by the
[factor microbenchmark](msc_factor_sampling.md). It does **not** establish a
speedup on an authors' full MSC7 circuit, fault-distance certification, or
best-in-class performance. Circuit synthesis uses Stim-generated CSS geometry,
not an authors' protocol. Full authors' MSC7/high-bond artifacts remain absent.
The reconstructed f7 results and their limitations remain separate.

Raw samples, validation errors, and all timing repeats are in
[msc_factor_sequence_data.json](msc_factor_sequence_data.json). The next
production design is [logical_block_integration.md](logical_block_integration.md).
No production executor, planner, or public API was changed.

## Precisely specified circuit

Use the positive X/Z checks extracted from Stim's generated color-code geometry
at distance labels 3/5/7/9, on 7/19/37/61 compact data wires. `FactorCode`
independently certifies that they define one logical qubit with all-data X/Z
logical axes. Prepare encoded |+>, then apply one logical T using a physical
CNOT parity ladder, a single T, and the inverse ladder. This initialization is
ideal; it is not a fault-tolerant injection protocol.

Repeat **five** times:

1. Independently depolarize every data wire with total probability p.
2. Apply T on every data wire; independently depolarize every wire again.
3. Measure the all-data Y product, with reported-bit error probability p.
4. Independently depolarize every wire; apply T-dagger on every wire;
   independently depolarize every wire again.
5. Measure every specified CSS check, each with reported-bit error probability p.

Finish with an all-data Z-product measurement with the same readout error.
Validation also uses X and Y final readouts. There are no ancillas, hidden
resets, feedback, growth, postselection, or extra inter-round corrections in
this synthetic sequence. The original d3/d5 controls cover those additional
composition features separately.

Report all measurement bits. Declare one diagnostic detector for each CSS
reported-bit difference between consecutive rounds, and include the final
reported measurement in observable 0. These parity monitors are not asserted
to be deterministic in the ideal circuit and do not define a cultivation
acceptance rule. Timings execute every complete attempt, regardless of outputs.

At 37 data wires this means 371 physical T/T-dagger gates, 926 noise sites,
186 reported records, 144 parity monitors and one observable. The elementary
circuit is independently planned/executed by ordinary Clifft with peak active
width 19. The p=0.001 timing circuit is reproduced by
`Sequence(7).circuit()` in `msc_factor_sequence.py`.

## Native representation and composition

The boundary state is a normalized complex pair plus a physical X/Z frame.
For a selected Y-measurement result, the T sandwich has two weighted monomial
terms. Four Pauli-fault layers on each wire bind a **compiled 256-entry local
table per coherent branch**. This retains relative phases, including Y faults;
no commutation analysis or runtime gate decomposition is needed.

The shared `msc_factor_kernel.h` evaluates precomputed syndrome-prefix
contractions. It samples the gadget result and the complete CSS sector while
retaining coherent cross terms. A separate single-copy contraction returns
the two weighted logical amplitudes. Their squared norm gives the selected
round's Born weight; the amplitudes are normalized and the compiled syndrome
duals supply the frame passed into the next round. Nonzero sectors are carried,
not rejected or silently corrected in the physical circuit.

`msc_sequence_native.cpp` also samples the final physical X/Y/Z product using
the logical pair and frame, then applies independent readout flips and writes
all output parities. The reported trajectory probability is conditional on the
sampled physical/readout fault history. Both implementations sample that history
from the same specified channel law; they need not generate the same shots from
the same seed because their RNG engines and consumption order differ.

All plans, vectors, and coefficient scratch are allocated at construction.
Ordinary per-shot work uses fixed arrays and previously compiled tables. Native
allocation instrumentation checks every shot. There is no per-shot compilation,
Clifford tableau evolution, support enumeration, or topology search. The earlier
factor microbenchmark uses the same extracted arithmetic header and retains
its independent full-support/Fourier comparison path.

For this family, compilation once per circuit remains the useful model: faults
and outcomes change bound coefficients and signs, not the contraction topology.
This result does not settle compilation strategy for arbitrary adaptive circuits.

## Correctness coverage

- **512 local matrix checks** cover every combination of four single-wire Pauli
  faults and both coherent branches against elementary 2-by-2 matrices.
- **72 complete native trajectories**, across all four sizes, p=0/0.001/0.02,
  and X/Y/Z final readouts, match Python fixed-history evaluation. All 360
  round handoffs are included in these complete runs. Maximum conditional
  probability relative error is 2.45e-15; maximum final logical-density error
  is 2.23e-16. Physical frames, true/reported bits and every output also agree.
- **54 original elementary-gate Clifft replays**, covering all trajectories
  through 37 data wires, match joint log probabilities within 1.78e-15. The
  replay oracle's explicit dense-width cap was raised from 16 to 20; no
  exponential allocation is allowed beyond that cap. X/Y/Z tails test the
  phase information needed by later logical operations.
- **12 complete coherent-stabilizer replays**, three per size at stress noise,
  include the 61-wire cases; maximum relative error is 1.12e-15. These use the
  original program plus independently bound fault sites and the prior CH/Stim
  gadget reference, not the factor plans or extracted logical handoffs.
- **Six original-gate Aer statevector trajectories** on seven wires cover all
  three final axes. Stim independently evaluates the parity declarations for
  all 72 native histories, including readout flips.
- The existing **59-test MSC regression run passed** after the shared-header
  extraction and native changes. A subsequent five-test sequence run includes
  the added Aer MPS translation check: **60 distinct MSC tests** in total.
- ASAN/UBSAN pass 100 complete attempts each at 37 and 61 wires. Leak detection
  remains disabled because the VM sandbox does not support LeakSanitizer's
  ptrace requirement. Allocation checks pass the entire native timing study.

The MPS comparator translation is checked against exact conditional trajectory
probabilities on the small code. It does not replace the separate original-gate
Aer validation of the logical-block implementation.

## Many-shot comparison

Median of three repeats, pinned to VM CPU 0, p=0.001, terminal Z. Native worker:
10,000 complete attempts per repeat at every size. Ordinary Clifft: 10,000
attempts for 7/19 wires and 1,000 attempts for 37 wires. Setup/compilation and
JSON sample logging are outside both timing loops. Fault sampling, coefficient
execution, all record/output production and output consumption are inside.
Native final amplitude recovery is also inside its loop.

| Data wires | Clifft peak active width | Factor us/attempt | Clifft us/attempt | Factor coefficient scratch |
| --- | ---: | ---: | ---: | ---: |
| 7 | 4 | 29.10 | 1.66 | 1,248 B |
| 19 | 10 | 218.80 | 25.89 | 4,704 B |
| 37 | 19 | 1,234.32 | 15,198.77 | 11,936 B |
| 61 | 31 | 4,400.70 | not allocated | 23,968 B |

Scratch is **not total memory**: immutable plans, fixed term/phase arrays,
logical amplitudes, faults, records, and object overhead are additional. Text
plans are 5,968/30,779/164,026/600,178 bytes respectively, including the local
fault table and single-copy output plan. Clifft's 37-wire coefficient vector
alone is 8 MiB; its 61-wire vector alone is 32 GiB, excluding measurement scratch.
No speedup ratio is inferred for the unexecuted 61-wire dense plan.

The prototype uses C++20, `-O3 -march=native`, assertions and allocation
instrumentation, without fast-math. The existing Clifft Release library uses
native CPU optimization and fast-math, with OpenMP disabled. Regression work
was placed on another VM CPU while the later timing runs completed; these are
single-VM measurements, not a dedicated-host performance guarantee.

## Bounded external comparison and limits

A direct original-gate Qiskit Aer MPS probe used one CPU, zero truncation
threshold, natural generated wire order, an 8 GiB address-space cap and a
60-second wall limit. It did not finish its initial 37-wire attempt before the
wall limit, so a many-shot run was declined. The
[probe record](msc_sequence_mps_probe.json) preserves the command and exit status.
This is a feasibility result, **not** a few-shot benchmark or a numerical
speedup bound. It does not test optimized wire ordering, Clifford-assisted MPS,
or the authors' implementations, and supports no best-in-class claim.

The meaningful outcome is a physics-based, compile-once exact representation
that now beats current Clifft on a fully executed large-residual-state control.
The corpus still lacks an independently sourced full MSC7/high-bond protocol
covering noisy injection, growth and syndrome extraction at this size. The
synthetic geometry and ideal preparation cannot fill that provenance gap.

The next implementation step crosses into the production planner/action
architecture. The [reviewable proposal](logical_block_integration.md) recommends
a narrow opt-in fused block action, retaining the existing dense logical
coefficients at certified boundaries and rejecting unsupported feature mixes
before execution. Implementation is paused for the architectural confirmation
required by `AGENTS.md`; the bounded research work above is complete.

## Reproduce

```bash
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -DCLIFFT_MSC_CHECK_ALLOCATIONS tools/profile/msc_sequence_native.cpp \
  -o /tmp/msc-sequence-native
c++ -std=c++20 -O3 -march=native -I src -I build-research/generated \
  tools/profile/msc_record_probe.cpp build-research/src/clifft/libclifft_core.a \
  -o /tmp/msc-record-probe
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_factor_sequence.py \
  --native /tmp/msc-sequence-native --baseline /tmp/msc-baseline-benchmark \
  --probe /tmp/msc-record-probe --shots 10000 \
  --output /tmp/msc_factor_sequence_data.json
CLIFFT_MSC_SEQUENCE_NATIVE=/tmp/msc-sequence-native \
CLIFFT_MSC_FACTOR_NATIVE=/tmp/msc-factor-native \
CLIFFT_MSC_SAMPLER=/tmp/msc-sampler-native \
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```
