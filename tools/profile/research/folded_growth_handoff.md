# Bounded f5-to-f7 handoff feasibility experiment

Date: 2026-09-20. The requested short experiment found a viable boundary
transfer for the exact growth and pre-check segment of the reconstructed
f7 protocol. This is an offline compiler/reference prototype, not an
integrated replacement for the native sampler's dense prefix.

Follow-up: the [native integration](folded_native_growth.md) now connects
this transfer to complete f7 attempts and measures the resulting speed and
memory improvements. The text below records the initial feasibility result.

## Certificate

`audit_folded_growth_handoff.py` verifies exact signed Pauli identities in
both directions: the 40 d5 CSS generators and 44 new-qubit Z preparations
map under growth to the full 84-generator d7 stabilizer group. Logical X,
Y and Z map to the corresponding d5 logical axes times known stabilizers.
Their sign relations are consistent with a logical Pauli frame update.

This is stronger than checking growth on the positive code space alone:
it covers arbitrary input syndrome sectors. It also supplies the reason
that sampling the d5 syndrome early need not introduce an extra physical
measurement. For a fixed Pauli fault realization, the actual d7 pre-check
outcomes determine that syndrome. Their complete projector already resolves
the d5 sectors. Faults change signs and records, not the underlying measured
operator span. The physical growth and extraction operations remain intact.

## Transfer prototype and checks

The prototype compiles a linear Boolean map from 40 input syndrome signs
and 14 classical ancilla bits to 212 chronological measurement/reset
records, 84 output syndrome signs, three logical-axis sign corrections
and 14 output ancilla bits. It precomputes record and boundary effects for
all 6,170 Pauli channels at 694 physical noise sites. Applying the compiled
map uses only parity calculations and XORs, with no tableau evolution.
The script checks that its bridge operations exactly match the corresponding
segment in the complete f7 circuit.

Thirty-six cases match independent Stim evolution of the physical bridge:
four random signed-code/ancilla inputs for each logical X/Y/Z eigenstate
at each of p=0, 0.001 and 0.03. They include up to 31 simultaneous faults.
Every measurement and hidden reset outcome is deterministic after the input
sector is fixed; records, final stabilizers, ancillas and logical signs all
agree. These are bridge tests, not end-to-end cultivation-history tests.

The transfer arrays would occupy 249,344 bytes (243.5 KiB) as packed 64-bit
words, excluding fault metadata and containers. This is a calculated native
layout, not the Python object's measured footprint. Compilation and the
36 reference cases finish in about two seconds after caching probe masks.
The [raw results](folded_growth_handoff.json) record the actual timing and
bridge hash. Formatting, lint, type and hygiene checks pass.

## Remaining integration

The native f5 kernel must expose its intermediate sector and logical state
before the standalone f5 post-check/final-evaluation path. Its computational
coset offset must be reconciled with the transfer's physical logical-Pauli
convention; a logical-X component of that offset changes the coefficient
labeling. The bridge then needs native packed tables, physical noise binding,
global record indexing and a certified input adapter for the f7 kernel.

Those changes should allow the small pre-f5 Clifft prefix to replace the
current width-22 prefix, targeting its 64 MiB coefficient allocation. No
end-to-end memory or speed improvement is claimed yet. The bounded experiment
stops here: it removes the main mathematical uncertainty without expanding
into the multi-stage native integration and its required history audit.
