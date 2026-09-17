# Full original-MSC composition check

The two-term color-code gadget rewrite now composes through both complete
original MSC benchmark controls in an offline fixed-history evaluator.
The evaluator preserves injection, syndrome extraction, growth, physical noise
locations, measurement and reset outcomes, feedforward, all detector parities,
and the declared logical observable. This closes the local-gadget-only
validation gap for these controls. It is not a new native logical-block sampler.

## Representation and output contract

`msc_protocol.py` binds each physical noise location, including idle noise,
two-qubit Pauli channels, reset errors, and measurement readout flips. Each
test fixes a legal physical fault history. The quantum trajectory additionally
specifies true measurement outcomes and the hidden outcomes used to unravel
reset channels. Its probability is conditional on that physical history;
fault-selection probabilities are not included.

Clifford intervals are followed with the existing independent CH/tableau
reference. The lone injection T is expanded exactly. Each certified gadget
applies the two Clifford monomials derived from the actual circuit, retaining
relative phases. This reference deliberately performs tableau evolution and
allocates. It belongs to offline validation, not the production execution path.

The structural upper bound is eight coherent stabilizer terms for complete d3
and sixteen for complete d5: one injection doubles the term count, as does
each of two or three gadget instruments. Fixed Pauli faults and specified
Clifford measurements do not increase that count. Exact merging reduces it
further; at most eight terms occurred in the tested histories. This is a bound
on coherent stabilizer terms, not a claim that every intermediate state fits
two logical amplitudes plus a Pauli frame.

Reported records equal true outcomes XOR readout flips. Classical feedforward
uses reported records. A root fault after its measurement but before reset can
change the hidden reset outcome while being erased from the later physical
state. These distinctions are implemented explicitly and have dedicated tests.
Rejected trajectories retain their outputs; no all-zero-record assumption or
early detector postselection masks a mismatch.

The result contains raw declared detector and observable parities. Stim's
ordinary detector sampler instead reports flips relative to a noiseless
reference. Independent parity checks therefore use its record-to-detector
converter with `skip_reference_sample=True`. A deterministic noisy-feedback
test checks this distinction against Stim's physical measurement sampler.

## Complete-trajectory validation

| Control | Physical qubits | Noise sites including readout | Visible records | Detectors | Quantum events including resets | Feedforward operations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSC d3 | 15 | 518 | 21 | 20 | 48 | 0 |
| MSC d5 | 42 | 3,564 | 112 | 107 | 230 | 17 |

The reproducible study in `msc_protocol_data.json` contains 44 d3 and 51 d5
histories. Each control includes 24 natural fault samples at p=0.001, 12 stress
samples with noise probabilities multiplied by ten, and targeted histories.
The targeted cases include an ideal circuit, reset-erased X/Y/Z faults, root
readout flips, a spectator fault inside a gadget, a two-qubit YZ fault, and a
legal logical-Z tail. d5 additionally tests readout flips at all seven distinct
record sources used by its 17 feedforward gates.

- **d3:** Qiskit Aer statevector execution follows every original elementary
  gate. It generates each true trajectory and supplies all 48 conditional Born
  probabilities. All 2,112 probabilities match, with maximum error 1.34e-15.
  Its independently assembled records, detector parities and observable agree.
- **d5:** ordinary Clifft follows every original elementary T and Clifford
  gate, without the gadget rewrite. Explicit measurement-plus-feedback resets
  expose the hidden reset outcomes. The native probe samples and replays each
  trajectory. All 51 joint log probabilities agree within 4.27e-14. For ten
  targeted histories, replay of every circuit prefix additionally checks all
  230 conditional probabilities: 2,300 comparisons, maximum error 4.39e-15.
- **Both controls:** Stim independently converts reported measurement records
  through the original parity declarations. All 6,636 visible record values,
  6,337 detector values and 95 logical-observable values agree with the
  evaluator and its reference checks.

The legal logical-Z tails produce observable one while every detector remains
zero in both controls. These are deliberately selected high-weight fault
histories, not logical-error-rate estimates. They demonstrate that the output
contract retains accepted logical failures. Both rejected and accepted natural
histories, nonzero records, and outcome-dependent frame changes are included.

Eight new regression tests also cover exact normalization after marginalizing
hidden reset outcomes in a small circuit, malformed dependencies, unsupported
classical operations inside a gadget, and physical-noise-site validation.

A full d5 Aer MPS reference was explored but interrupted when it did not finish
promptly. No MPS result or performance conclusion is used. The full d5 quantum
reference is Clifft, not an external simulator; the earlier independent
38-wire gadget column checks remain relevant. The d3 Aer comparison and the
Stim parity checks provide external coverage, but do not amount to
external full-d5 state validation. Tested histories are not exhaustive over
all possible faults or output records.

## Reproduction

Build the bounded elementary-gate probe against the existing scalar library:

```
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -I src -I build-research/generated tools/profile/msc_record_probe.cpp \
  build-research/src/clifft/libclifft_core.a -o /tmp/msc-record-probe
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
  MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p test_msc_protocol.py
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_protocol.py \
  --clifft-probe /tmp/msc-record-probe \
  --output tools/profile/research/msc_protocol_data.json
```

The raw file retains physical site choices, every true outcome including reset
outcomes, every visible record and detector, observables, term counts, seeds,
and probability discrepancies. The probe limits active width to sixteen before
constructing an executor. It recompiles prefixes only to obtain independent
validation probabilities; this is not a per-shot compilation proposal or a
throughput benchmark.

## Next step

Derive static signed code-boundary maps for these actual MSC syndrome and
growth intervals, including the tested nonzero sectors and readout-dependent
feedforward. Use this complete evaluator as the regression oracle for reducing
them to two logical amplitudes plus a physical frame, with small monomial sums
between boundaries. Precompute the needed fault maps and overlap contractions
before considering native integration. If an interval needs a larger coherent
state, measure that requirement explicitly instead of silently projecting into
the ideal code space.

The existing production architecture and fold adapter are unchanged; the latter
still declines these original MSC inputs. There is no new native throughput
claim. d3/d5 remain coverage controls, the reconstructed f7 remains the
large-state performance target, and an authors' complete MSC7 artifact remains
unavailable. One-round memory and few-shot startup are outside this study.
