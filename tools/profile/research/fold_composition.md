# Whole-check composition on a 22-active-qubit cultivation reconstruction

Follow-up: [fixed contraction plans](fold_contraction.md) replace the overlap
calculation at ideal code boundaries with a small reusable native kernel,
including d7 core diagnostics. Full noisy-boundary integration remains open.

**Continue this approach.** We now have a complete encoded f5 reconstruction
that reaches 22 active qubits in current Clifft. On the sampled fault histories,
an offline coherent-stabilizer reference needs at most eight terms, reducing to
two at surviving output boundaries. Choosing the injection decomposition in the
logical computational basis halves the term counts compared with an equally
exact Pauli expansion. This is evidence for a different state representation
on a relevant large-state workload, not a demonstrated production speedup.

The full source circuits used by the Reg5 and f7 paper authors remain unavailable
to us. The user plans to ask the authors for them. This reconstruction provides
a useful interim test, explicitly **not a reproduction of their benchmark**.

## What was reconstructed

`fold_cultivation.py` constructs the f3/f5 sequence described in Appendix A of
[Takada, Bartlett, and Williamson](https://arxiv.org/html/2609.16929v1): unitary
T injection, code morphing/growth, regular-code syndrome measurements, two fold
checks at each distance, and verified cat preparation. Injection and cat
circuits were transcribed from Figs. 18 and 19; the d3-to-d5 growth uses the
[pinned Sahay construction](https://github.com/kaavyas99/MSC_foldedH/blob/9378fd228ba83c592875d155136fcbcd16a45680/src/full_clifford_sim/s3_fxns.py).
The fold factors and assignments reuse the previous validated core diagnostic.

The implemented noise is depolarization after each elementary H, T/T_DAG,
CNOT, or native CCZ; reset flips follow preparation, measurement flips precede
readout, and there is no idle noise. The generator preserves the distinct
locations inside T-conjugated CNOTs. All physical measurement records must be
zero. The f5 cat flag is included, not discarded from the model.

Two explicit reconstruction choices prevent claiming exact paper reproduction:
the within-stabilizer CNOT order is north, west, east, south, and the terminal
contract is an additional ideal syndrome round followed by logical X/Y/Z
expectation probes. The paper's exact syndrome CNOT order and final decoding
implementation have not been matched. These choices can affect noisy logical
statistics. The reconstruction does not claim a verified fault distance merely
because its stages are labeled f3/f5.

The f5 instance has 41 data qubits and six reusable ancilla/flag wires, for 47
physical qubits, and contains four logical fold checks. It is not a five-round
memory circuit. One-round memory benchmarks remain excluded from the study.
It is also not the Reg5 schedule in the
[MPS/CAMPS comparison](https://arxiv.org/html/2609.19116v1), which includes
additional rotated-code QEC cycles and a different terminal decode/noise model.
The matching peak active width is a useful complexity signal, not proof of
equivalent workloads or error rates.

## Independent construction and composition checks

Stim verifies every target-code stabilizer and the logical X/Y/Z axes after
injection, morphing, and d3-to-d5 growth, using several independent stabilizer
inputs. It also verifies the prepared cat stabilizers, flag, and decoder. These
checks caught and corrected an initially wrong ordering of the two morphing
layers. Aer verifies the full ideal f3 circuit's final code stabilizers and
logical T-state expectations. Current native Clifft agrees on the ideal f5
logical expectations: `(1/sqrt(2), 1/sqrt(2), 0)`, with no discarded shots.

The composition reference retains the unnormalized conditional output state.
Full f3 fixed-fault histories are compared with a separate Qiskit dense
calculation, including zero-probability outputs, a paired cat fault with
fractional acceptance, and an accepted logical error. Both injection
decompositions must agree with that output state, including its complex phase.
Random small circuits exercise phase-sensitive merging, nonorthogonal overlaps,
and destructive interference. The earlier Aer whole-core tests remain in place.

## How the reference works

`clifford_branches.py` pairs Cirq's phase-preserving CH stabilizer states with
Stim tableaux. Each term carries a coefficient in the exact cyclotomic ring
`Q[exp(i*pi/4)]`. Equal stabilizer states are merged after resolving their
relative global phase through a nonzero amplitude anchor. Only exactly zero
coefficient sums are removed; there is no small-amplitude pruning or MPS
truncation. CH amplitudes and final overlaps still use floating point. Phase
quantization is checked against eighth roots of unity, and the complete output
is independently validated on small cases.

At a recognized core, projecting one cat qubit splits the stabilizer term into
its computational strings. Each string selects the corresponding phase-sensitive
data Clifford operator, including the fixed internal Pauli faults. The actual
noisy preparation, flag measurement, unencoding, and readout remain outside the
core and run through the same reference. This preserves their outcome weights
and phases. Unsupported operations and cat states that need more strings decline.

Pauli measurements project each term separately and coherently add the surviving
terms. Acceptance is the squared norm of their sum, including cross terms.
This is not the probability obtained by sampling the terms as a mixture.
The reference follows the all-zero record branch to compute its exact-form
conditional probability; it is not yet a general record sampler.

## Measurements

We drew 64 physical Pauli-fault histories at p=0.001, using the same histories
for both decompositions. The sample contains zero through three faulty locations.
Ten separate two-fault hook cases flip a cat control after an earlier gate
involving it, then flip it back after a selected CCZ. Both faults occupy legal
noise locations. These targeted cases are diagnostic strata, not weighted
estimates of natural error rates.

| Decomposition and history set | Maximum live terms | Terms at surviving final boundary | Conditional acceptance |
| --- | ---: | ---: | --- |
| Computational injection, natural faults | 8 | 2 | 31 histories have probability 1; 33 have probability 0 |
| Pauli injection, same natural faults | 16 | 4 | Same results |
| Computational injection, ten paired hooks | 4 | 2 | 1/4 for every history |
| Pauli injection, same paired hooks | 8 | 4 | Same results |
| Computational injection, five-fault logical-Z tail | 4 | 2 | 1, with the wrong logical state |

The two decompositions agree in acceptance to 2.3e-14 and surviving logical
expectations to 2.3e-16 on the natural histories. The logical-Z tail gives
`(-1/sqrt(2), -1/sqrt(2), 0)`, confirming that the retained output is not
automatically replaced by the target T state. Its five physical Z faults occur
at data CNOT noise locations late in the final noisy syndrome extraction.

Current native Clifft reaches width 8 for reconstructed f3 and width 22 for f5,
both ideally and with noise in the tested f5 input. Width 22 corresponds to
4,194,304 complex coefficients, or 64 MiB for that array alone in double
precision. Several polynomial-sized stabilizer descriptions offer a substantial
potential storage reduction, but we have not measured a production memory ratio.

Native scalar f5 diagnostics measured about 265 ms/attempt for the ideal input
over eight shots, and 345 ms/attempt for the noisy input over 128 shots, with
75 rejected. These are single-run, unpinned VM measurements, not a comparative
benchmark. The Python reference computes a fixed-history conditional probability
and logical probes instead of sampling the same output contract. Its timings
are therefore intentionally excluded from speedup claims. The small number of
histories cannot estimate rare logical error rates or validate published rates.

The CLI omits conditional logical probes when the computed acceptance is at
most 1e-12, while still recording acceptance and retaining all state terms.
This diagnostic threshold is not an error-rate estimator or state truncation.

## Why the basis choice matters

The Pauli expansion of the injected T gate acts on an X-basis stabilizer state
and introduces logical X-basis components. An alternative exact decomposition
projects the injected qubit into Z-basis components before applying its T phase.
After encoding, these are logical computational-basis components.

Fold-core branch operators are diagonal Clifford phases followed by Pauli
flips. They preserve a Z-type logical operator up to sign. Complete syndrome
projection then brings each surviving stabilizer component back to a fixed
syndrome sector of a one-logical-qubit code. Within that sector, a specified
logical-Z eigenvalue determines a unique stabilizer state up to phase: only
two such states are needed. This explains the observed merging and suggests
an eligibility condition based on the preserved logical operator and complete
syndrome boundaries, rather than on a small physical T count.

To turn this into a general bound, the planner would need to certify logical-Z
transport through growth and the effective syndrome sector after noisy
extraction, for every supported fault pattern. We have not implemented that
certificate or asserted a universal eight-term bound. More logical qubits,
incomplete syndrome extraction, non-Pauli noise, and different checks can remove
the restriction.

## Recommendation and remaining work

The leading candidate is now **whole-check Clifford branches with a logical
basis chosen to encourage merging**, potentially reduced further to a small
logical state plus a Clifford error frame at certified boundaries. This has
stronger evidence on a large residual than the earlier product-state approach.
CAMPS-inspired frame changes remain relevant, especially where these boundaries
cannot be recognized, but a general exact MPS backend is not the first prototype.

The next engineering question is whether a circuit can compile once into
reusable gadget/overlap formulas whose parameters are fault and record bits.
Per-shot compilation of the ordinary physical gate sequence remains unattractive;
the new result changes the representation, not that earlier conclusion.

The reference's per-history tableau evolution and canonicalization explicitly
remain offline research. Moving those operations into Clifft's hot execution
would violate the existing architectural invariant. Production work needs a
concrete design for compiler-precomputed dependencies and phase/overlap updates,
or approval of an architectural change. An unsupported circuit can cheaply
decline before execution and use ordinary Clifft; converting a live branch sum
back to its residual representation still needs a cost model.

Before a backend decision: obtain the authors' exact Reg5/f7 artifacts; validate
the relevant noisy output contract; test more fault strata and full f7 growth;
and measure amortized end-to-end throughput with the same postselection,
logical outputs, and error model. No best-in-class claim is supported yet.

## Reproduction

Use an isolated environment with NumPy, Stim 1.15.0, Qiskit 2.3.0,
Qiskit Aer 0.17.2, and Cirq Core 1.6.1. Cirq is an optional offline reference
dependency, not a Clifft runtime dependency. Tests for that reference explicitly
skip when Cirq is absent; the complete study was tested with it installed.

```sh
python tools/profile/fold_cultivation.py --output /tmp/clifft-fold-cultivation
python tools/profile/clifford_branches.py --distance 5 --histories 64 --probability 0.001 --output /tmp/branches-computational.json
python tools/profile/clifford_branches.py --distance 5 --histories 64 --probability 0.001 --injection-basis pauli --output /tmp/branches-pauli.json
python tools/profile/clifford_branches.py --distance 5 --paired-hooks --output /tmp/hooks-computational.json
python tools/profile/clifford_branches.py --distance 5 --paired-hooks --injection-basis pauli --output /tmp/hooks-pauli.json
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
build-research/profile_structure /tmp/clifft-fold-cultivation/reconstructed_f5_p0.stim 8 22 1
build-research/profile_structure /tmp/clifft-fold-cultivation/reconstructed_f5_p0.001.stim 128 22 1
```

`fold_composition_data.json` records circuit hashes, native diagnostics,
per-history counts/probabilities, stage maxima, and representative traces.
