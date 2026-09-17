# Static signed boundaries in the original MSC controls

The actual MSC circuits have certified boundaries at which the full conditional
trajectory fits one encoded qubit, with all other qubits in known single-qubit
Pauli eigenstates. Output syndrome signs and a correcting physical Pauli frame
now come from fixed record/fault parity maps. Evaluating these maps requires no
tableau evolution, flow solving, or dependency discovery.

This establishes where a two-amplitude representation is safe. It does not yet
compute its amplitudes, the probability of the selected measurement branch,
or the complete boundary contraction. There is no new native throughput result.

## Circuit-derived certificates

The compiler takes each actual Clifford interval between the injection/check
regions and the next cultivation gadget. Stim solves signed stabilizer flows
from identity at the interval input to the output code generators, mediated by
the supplied measurement record. These flows prove output code membership for
arbitrary incoming states, on every reachable conditional branch.

| Input and interval | Source lines | Data checks | All independent output checks | Visible records in interval |
| --- | --- | ---: | ---: | ---: |
| d3 after injection | 59-129 | 6 on 7 data qubits | 14 on 15 physical qubits | 7 |
| d5 after injection | 86-156 | 6 on 7 data qubits | 41 on 42 physical qubits | 7 |
| d5 growth and syndrome extraction | 200-434 | 18 on 19 data qubits | 41 on 42 physical qubits | 65 |

The additional checks fix every non-data wire in a single-qubit X or Z
eigenstate. They are essential: code membership alone would not rule out
entanglement between the logical qubit and unaccounted ancillas. The complete
independent rank is physical width minus one, leaving exactly one logical
degree of freedom in each fixed, pure trajectory.

Some spectator information precedes the interval. Before d5's first gadget,
27 future wires have only idled since their initial |0> preparation. Before
its second gadget, qubit 5 carries the result of an earlier measurement.
The compiler separately certifies each such wire's preparation/measurement
anchor and absence of subsequent quantum interactions, and retains faults
and readout signs from that entire interval of idling. Assuming those wires
were freshly reset at the boundary would be incorrect.

The d3-to-d5 interval additionally has six input-stabilizer-to-record flows
and both logical-axis flows. In the all-data X/Z convention, its ideal logical
X flow needs no record parity and logical Z uses local records 15, 17, 35, 49
(zero-based). Physical faults add the separately compiled sign corrections.
These certificates establish the measured input syndrome and the signed
logical transport needed for the eventual amplitude contraction.

The seven-wire code basis is mapped from the original d3 control; the final
nineteen-wire basis is read from the actual d5 terminal checks. This is a
bounded compiler for the current protocol shapes, not a general code detector
or production circuit-eligibility API. It checks the supplied gates and does
not select a catalog entry for the syndrome schedule.

## Static fault and record maps

During construction, symbolic X/Z frame components propagate through the
Clifford gates. A measurement contributes its frame-induced bit flip and its
independent readout error. Feedforward contributes the difference between the
reported and ideal control bits. Resets erase the corresponding frame.

For each signed flow, the compiled output sign is

```
constant XOR parity(reported_records & record_mask)
         XOR parity(physical_fault_bits & fault_mask).
```

Evaluation only encodes the selected physical Pauli faults and applies these
fixed masks. Precomputed syndrome duals then give a data Pauli correction.
The duals restore every positive code check and commute with the chosen
logical X and Z, so the frame convention does not silently rotate the logical
axes. No correction is physically applied to the reference trajectory.

The three plans refer to 212, 914 and 1,912 Pauli-component fault slots,
respectively. These are sign-map coordinates, not independent physical noise
sites; X/Y/Z choices and two-qubit channels impose their original correlations.
Python construction data is not a native memory-use measurement.

## Validation

The full study reuses all 95 complete trajectories from `msc_protocol_data.json`
and inspects their unmodified coherent CH/tableau states immediately before
the gadgets. All 146 boundary observations match the compiled signs, including
55 nonzero data-syndrome sectors. Every coherent term separately has the
predicted eigenvalues: 9,596 exact term/stabilizer checks. The computed Pauli
frame restores all data-check signs in each case.

Independent signed-flow checks insert physical Pauli basis faults into the
original Clifford intervals and implement readout errors as inverted
measurement targets. Stim verifies 177,460 signed flows across 4,982 histories
(including one ideal history per interval). The growth cases check input
syndrome and logical transport as well as output membership. Because the
compiler's frame/sign dependence is linear, these basis checks exercise every
allowed fault direction in those intervals; the complete-history checks
add mixed faults and earlier spectator dependencies.

Seven new tests cover these boundaries, both logical-axis frame conventions,
evaluation with Stim and flow solving disabled, a negative preparation sign,
an inherited spectator readout flip, changed commuting CNOT order, and rejection
of a broken syndrome circuit or an unfixed entangled spectator. The combined
MSC-focused suite has 22 tests. The observer added to the existing evaluator
only inspects states at source boundaries; it never projects them to make a
certificate appear to hold.

Reproduction:

```
MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python tools/profile/study_msc_boundaries.py \
  --output tools/profile/research/msc_boundaries_data.json
CLIFFT_MSC_RECORD_PROBE=/tmp/msc-record-probe \
  MPLCONFIGDIR=/tmp/clifft-mpl OPENBLAS_NUM_THREADS=1 \
  /tmp/clifft-fold-env/bin/python -m unittest discover \
  -s tools/profile -p 'test_msc_*.py'
```

The raw report records the signed output rows, spectator dependencies, logical
flow records, validation counts, and each observed syndrome/frame. The prior
protocol report records how the native elementary-gate probe is built.

## What remains

Use the measured-input-syndrome flows and logical transport to derive the
weighted two-amplitude update, including record reachability constraints and
normalization factors. Contract the monomial terms only at these certified
boundaries. The final cultivation and terminal logical measurement can leave
several terms between projections; this study does not invent an additional
code boundary there.

Then compare the complete static evaluator with the existing full-history
reference before implementing or timing a native path. The original f7
reconstruction remains the large-state target; d3/d5 are coverage controls.
The existing fold adapter and production executor are unchanged.
