# Noncomputational States

Pauli noise acts within a qubit's two-dimensional computational subspace. Real
hardware can instead drive the state out of that subspace through *leakage*, or
lose the physical carrier from its site entirely through *loss*. Neither
process is a Pauli channel, so the site no longer holds an ordinary
qubit state.

Clifft models these processes with a hybrid quantum-classical trajectory
model. Within the computational subspace, the state keeps its full coherent
dynamics and entanglement. A leaked or lost site instead has a definite,
classically tracked occupation on each trajectory. Transitions between the two
are stochastic quantum jumps, including their back-action on the computational
state.

This page explains the model and how it composes with Clifft's
[factored-state simulation](overview.md). The
[Leakage and Loss guide](../guide/leakage-and-loss.md) shows the Python API.

## The effective five-level site

Each circuit wire denotes a fixed physical *site*. For modeling leakage and
loss, Clifft uses the effective per-site space

$$
\mathcal H_{\mathrm{site}}
=
\mathcal H_C \oplus \mathcal H_N,
\qquad
\mathcal H_C = \operatorname{span}\{\lvert g\rangle,\lvert e\rangle\}
\cong \operatorname{span}\{\lvert 0\rangle,\lvert 1\rangle\},
$$

with

$$
\mathcal H_N = \operatorname{span}\{
\lvert \mathrm{leak\_g}\rangle,
\lvert \mathrm{leak\_e}\rangle,
\lvert \mathrm{lost}\rangle
\}.
$$

The table below lists these levels and their categories:

| index | name | category | meaning |
|---|---|---|---|
| 0 | `g` | computational | $\lvert g\rangle$, identified with logical $\lvert 0\rangle$ |
| 1 | `e` | computational | $\lvert e\rangle$, identified with logical $\lvert 1\rangle$ |
| 2 | `leak_g` | leaked | carrier present, outside the qubit subspace |
| 3 | `leak_e` | leaked | a second leaked level |
| 4 | `lost` | lost | carrier absent from the site |

Under the model's incoherent-jump assumption, there is no coherence between
$\mathcal H_C$ and $\mathcal H_N$, or between different noncomputational
levels. The labels `g` and `e` name the computational basis levels used as
matrix indices. They do not imply that a computational site occupies a
definite level: its state may be any superposition or entangled state in
$\mathcal H_C$.

Each trajectory therefore combines Clifft's ordinary factored quantum state
over the computational sites with a classical status ledger over all sites.
The ledger records one computational status rather than separate `g` and `e`
occupations. Its other entries are `leak_g`, `leak_e`, and `lost`, each of
which names one definite noncomputational level. The model's initial
distribution is sampled independently for each site. An outcome of `g` or `e`
prepares the corresponding computational basis state; any other outcome sets
that definite noncomputational status.

## Transitions and measurement classification

A transition matrix $T[\mathrm{to}][\mathrm{from}]$ attaches stochastic jumps
to circuit positions. An entry gives the probability of a jump from one
source level to one destination level. Each source column must sum to at most
one. For any source $s$, the total jump and no-jump probabilities are

$$
p_{\mathrm{jump}}(s) = \sum_{\ell} T[\ell][s],
\qquad
p_{\mathrm{no\ jump}}(s) = 1 - p_{\mathrm{jump}}(s).
$$

Every nonzero matrix entry is a distinct jump. A diagonal entry $T[s][s]$
still projects onto its source level; it is not part of the no-jump branch.
For a computational source, the column defines a quantum instrument whose
outcomes update the live state. For a noncomputational source, it defines an
ordinary classical transition from an already definite level.

A measurement classifier $P[\mathrm{symbol}][\mathrm{level}]$ defines the
recorded result for each level. It has two record symbols, 0 and 1, and may
have a third herald symbol. If the site occupies `leak_g` when measured, for
example, the `leak_g` column gives the probabilities of reporting 0,
reporting 1, or emitting a herald.

For `M` and `MR` on a computational site, the quantum measurement first
resolves `g` or `e`; the corresponding classifier column can then model
computational-basis readout confusion. Computational `MX`, `MY`, `MRX`, and
`MRY` measurements use their ordinary quantum results without applying the
`g` or `e` columns. Once a site is noncomputational, the classifier supplies
the result regardless of measurement basis. The optional herald probability
may be nonzero only for noncomputational levels.

## Jump back-action

A jump from a computational source changes the quantum state, even if its
destination is also computational. For a source $s \in \{g,e\}$ and
destination $\ell$, the transition-matrix entry corresponds to the jump
operator

$$
K_{\ell \leftarrow s}
=
\sqrt{T[\ell][s]}\,\lvert \ell\rangle\langle s\rvert.
$$

Applying this operator resolves the source against the live coherent state
and projects it onto `g` or `e`. A computational destination prepares the
corresponding basis state and leaves the site in the coherent simulation. A
noncomputational destination removes the site from that simulation and records
its definite level in the status ledger. This hidden collapse writes no
visible measurement record.

For an entangled site, the same collapse updates its partners. Each trajectory
keeps the partner state conditioned on the selected jump outcome; averaging
over trajectories recovers the correct reduced-state statistics. A Bell-pair
partner, for example, is maximally mixed in the ensemble after
source-independent loss of the other half.

When no jump occurs, the state is also updated. If the total jump rates from
`g` and `e` are $p_g$ and $p_e$, the no-jump outcome applies

$$
K_{\mathrm{stay}}
=
\sqrt{1-p_g}\,\lvert g\rangle\langle g\rvert
+
\sqrt{1-p_e}\,\lvert e\rangle\langle e\rvert.
$$

The state is renormalized after conditioning on this outcome. Equal rates
make $K_{\mathrm{stay}}$ proportional to the identity; unequal rates change
the surviving coherent state.

A jump from a noncomputational source is a classical status change. A
destination of `g` or `e` returns the site to the coherent simulation,
prepared at that computational basis level.

## After a site becomes noncomputational

Once a site has a definite noncomputational level, it no longer participates
in coherent evolution. Under the current policy, most operations that touch it
cannot act, a single-site measurement samples its result from the classifier
without regard to measurement basis, and a reset or later transition may
restore it to the computational subspace. The
[Leakage and Loss guide](../guide/leakage-and-loss.md#what-happens-on-a-leaked-or-lost-site)
defines the exact behavior for supported circuit operations.

The visible binary result occupies the same record slot as the original
measurement. Later `rec` references, detectors, observables, and classical
feedback all consume that substituted bit. When the classifier emits its third
symbol, a separate herald marks the slot and the binary record receives a
uniformly drawn placeholder. The herald identifies the readout, not the time
or location of the underlying jump, so it is not an exact spacetime erasure
flag.

## How this composes with Clifft

Ordinary Clifft compiles a circuit once and reuses the program for many shots.
The compiler absorbs deterministic Clifford evolution into an offline frame,
plans active stabilizer coordinates, and prepares fixed actions for the
symbolic-coordinate executor.

With noncomputational transitions, the sampled history can change which later
operations act, which measurements use the classifier, and when a site
returns to the computational subspace. Those choices differ between shots, so
one program compiled before sampling cannot describe every trajectory.

`noncomp.sample` instead alternates execution with compilation when the
sampled history requires it. The executor directly handles outcomes whose state
update leaves the remaining program valid. When an outcome invalidates the
compiled remainder -- for example, when a computational site becomes
noncomputational -- the executor stops at that transition and returns control to the
trajectory driver. The driver samples any destination the executor did not already
select, records the outcome, updates the status ledger, rewrites the original
circuit, and compiles a continuation. The continuation preserves the prefix
already executed, changes the remaining operations to match the new
trajectory, and resumes after the transition. Jumps whose source is already
noncomputational are sampled as classical status changes while this
continuation is constructed.

The prepared transition action holds only the source-dependent total
jump rates, the separate weights for `g` and `e` destinations, and one
combined weight for all noncomputational destinations. This is enough for the
executor to evaluate the live computational state using active coordinates. The
trajectory driver retains the original five-level matrix so that it can
resolve the combined noncomputational outcome outside the executor.

Each continuation still uses Clifft's normal compiler and symbolic-coordinate
architecture. Clifford operations, coordinate planning, and symbolic
dependencies are resolved before execution resumes.

## Active-width cost

With the default exact damping policy, the simulation is exact for this hybrid
quantum-classical model. Most transition positions do not increase the
[active width](overview.md#symbolic-clifford-coordinates). A source
already in the active state uses its existing array axis. A definite dormant
source is determined entirely from the Clifford and Pauli frames.

The exceptional case is a coherent dormant site with source-dependent total
jump rates, $p_g \neq p_e$. The $K_{\mathrm{stay}}$ operator above is then not
proportional to the identity, and is non-Clifford. Exact simulation must
promote the site from dormant to active.

If this occurs frequently, it can increase the peak active width $k$, the
dominant exponent in Clifft's runtime. Users can instead choose
`damping="neglect"`, which omits the no-jump back-action. This is exact when
$p_g = p_e$, when the operator is proportional to the identity. Otherwise it
changes the conditioned no-jump state by order $\lvert p_g-p_e\rvert$ at each
transition position. Under this policy, these transition positions do not
increase $k$.
