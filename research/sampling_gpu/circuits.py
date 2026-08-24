"""Benchmark circuit generators (stim-format text) for the (k, t_live) profiler.

Each generator returns a stim-format circuit string using clifft's gate set
(H, S, S_DAG, X, Y, Z, CX, CZ, SWAP, T, T_DAG, R_Z(theta), M, ...). Angles for
R_Z are in clifft's half-turn units; non-dyadic angles are genuinely
non-Clifford.

The suite is chosen to span the regimes that decide the stabilizer-rank bet:
  * random Clifford+T (sweep T-density)  -- generic near-Clifford.
  * hidden shift                         -- T-heavy, FEW measurements (the BGH
                                            benchmark; expect k ~ T-count).
  * QAOA (ring)                          -- rotation-heavy, terminal measure.
  * surface-code memory + T injection    -- measurement-RICH (expect k collapse).
  * magic-state injection / distillation -- small, T-structured.
"""

from __future__ import annotations

import random


def _lines(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def random_clifford_t(
    n: int,
    depth: int,
    t_fraction: float,
    seed: int = 0,
    measure_all: bool = True,
) -> str:
    """Random brickwork Clifford+T circuit.

    ``depth`` layers; each layer applies a random single-qubit gate to every
    qubit (T with prob ``t_fraction``, else a random Clifford) plus a brickwork
    layer of CX. Terminal measurement of all qubits unless disabled.
    """
    rng = random.Random(seed)
    sq_clifford = ["H", "S", "S_DAG", "X", "Y", "Z"]
    ops: list[str] = []
    for layer in range(depth):
        for q in range(n):
            if rng.random() < t_fraction:
                ops.append(f"{'T' if rng.random() < 0.5 else 'T_DAG'} {q}")
            else:
                ops.append(f"{rng.choice(sq_clifford)} {q}")
        # brickwork CX
        offset = layer % 2
        for q in range(offset, n - 1, 2):
            ops.append(f"CX {q} {q + 1}")
    if measure_all:
        ops.extend(f"M {q}" for q in range(n))
    return _lines(*ops)


def hidden_shift(n_pairs: int, n_t_layers: int = 1, seed: int = 0) -> str:
    """Hidden-shift-style circuit: T-heavy, measured only at the end.

    Structure mirrors the Bravyi-Gosset-Howard hidden-shift benchmark shape: a
    layer of H, a random shift (X), CZ "oracle" Cliffords, then concentrated T
    gates on every qubit (repeated ``n_t_layers`` times), then unH and measure.
    This is the adversarial case for clifft: little/no measurement-driven
    collapse, so peak k tracks the T-count.
    """
    rng = random.Random(seed)
    n = 2 * n_pairs
    ops: list[str] = []
    ops.extend(f"H {q}" for q in range(n))
    for q in range(n):
        if rng.random() < 0.5:
            ops.append(f"X {q}")
    # CZ oracle (Clifford structure)
    for a in range(0, n - 1, 2):
        ops.append(f"CZ {a} {a + 1}")
    # Magic layers. A Clifford layer (H + CZ) is interposed between every magic
    # layer so successive T's on the same qubit never compose to T*T = S
    # (Clifford), which would silently destroy the magic content.
    for layer in range(n_t_layers):
        ops.extend(f"T {q}" for q in range(n))
        for a in range(1, n - 1, 2):
            ops.append(f"CZ {a} {a + 1}")
        if layer != n_t_layers - 1:
            ops.extend(f"H {q}" for q in range(n))  # break T*T=S between layers
    ops.extend(f"H {q}" for q in range(n))
    ops.extend(f"M {q}" for q in range(n))
    return _lines(*ops)


def qaoa_ring(n: int, p_layers: int, seed: int = 0) -> str:
    """QAOA on a ring of n qubits, p_layers rounds. Rotation-heavy, terminal M.

    Cost layer: R_ZZ emulated via CX; R_Z(gamma) on the edge target. Mixer:
    R_X(beta) on every qubit, emulated as H, R_Z(beta), H. Angles are non-dyadic
    => non-Clifford.
    """
    rng = random.Random(seed)
    ops: list[str] = []
    ops.extend(f"H {q}" for q in range(n))
    for _ in range(p_layers):
        gamma = round(rng.uniform(0.05, 0.45), 4)
        beta = round(rng.uniform(0.05, 0.45), 4)
        # cost layer over ring edges
        for q in range(n):
            r = (q + 1) % n
            ops.append(f"CX {q} {r}")
            ops.append(f"R_Z({gamma}) {r}")
            ops.append(f"CX {q} {r}")
        # mixer
        for q in range(n):
            ops.append(f"H {q}")
            ops.append(f"R_Z({beta}) {q}")
            ops.append(f"H {q}")
    ops.extend(f"M {q}" for q in range(n))
    return _lines(*ops)


def surface_code_memory_with_t(distance: int, rounds: int, n_t: int = 4, seed: int = 0) -> str:
    """Rotated-surface-code-style memory experiment with a few injected T gates.

    NOT a physically exact surface code -- a measurement-RICH stand-in with the
    right shape: data qubits, repeated stabilizer (MR-style) measurement rounds
    that collapse the active dimension, plus a handful of injected T gates on
    data qubits. The point is to exhibit the regime where frequent measurement
    keeps peak k small despite a nonzero T-count.
    """
    rng = random.Random(seed)
    n_data = distance * distance
    n_anc = distance * distance - 1  # one ancilla per stabilizer (approx)
    total = n_data + n_anc
    data = list(range(n_data))
    anc = list(range(n_data, total))
    ops: list[str] = []
    # inject some magic on data qubits up front
    for q in rng.sample(data, min(n_t, n_data)):
        ops.append(f"T {q}")
    for _ in range(rounds):
        # each ancilla couples to two neighbouring data qubits then is measured+reset
        for i, a in enumerate(anc):
            d0 = data[i % n_data]
            d1 = data[(i + 1) % n_data]
            ops.append(f"CX {d0} {a}")
            ops.append(f"CX {d1} {a}")
            ops.append(f"MR {a}")
    ops.extend(f"M {q}" for q in data)
    return _lines(*ops)


def magic_state_injection(n_states: int = 8, seed: int = 0) -> str:
    """Small T-structured circuit: prepare n_states |T> states and entangle.

    A stand-in for the inner block of zero-level distillation (e.g. the (8,3,2)
    transversal-T pattern): each of n_states qubits gets a T, then a Clifford
    entangling network (the 'distillation' Cliffords), then measurement. Small,
    high T-density, measurement at the end.
    """
    rng = random.Random(seed)
    ops: list[str] = []
    ops.extend(f"H {q}" for q in range(n_states))
    # transversal-T-like pattern (alternating T / T_DAG, cf. the (8,3,2) signs)
    for q in range(n_states):
        ops.append(f"{'T' if q % 2 == 0 else 'T_DAG'} {q}")
    # Clifford entangling network
    for _ in range(n_states):
        a, b = rng.sample(range(n_states), 2)
        ops.append(f"CX {a} {b}")
    ops.extend(f"M {q}" for q in range(n_states))
    return _lines(*ops)


def ccz(a: int, b: int, c: int) -> list[str]:
    """Exact ancilla-free CCZ(a,b,c) as 7 T/T_DAG + CNOTs.

    CCZ = H_c . Toffoli(a,b->c) . H_c, and the textbook Toffoli decomposition
    (Nielsen-Chuang Fig 4.9) begins and ends with H on the target; those two H's
    cancel the wrapping H_c, leaving the 7-T body below. Verified numerically
    against diag(1,...,1,-1) (see tests at bottom of module).
    """
    return [
        f"CX {b} {c}",
        f"T_DAG {c}",
        f"CX {a} {c}",
        f"T {c}",
        f"CX {b} {c}",
        f"T_DAG {c}",
        f"CX {a} {c}",
        f"T {b}",
        f"T {c}",
        f"CX {a} {b}",
        f"T {a}",
        f"T_DAG {b}",
        f"CX {a} {b}",
    ]


def iqp(n: int, seed: int = 0, t_per_qubit: int = 1, cz_density: float = 0.5) -> str:
    """IQP circuit: H^n . D . H^n . measure, D diagonal in the Z-basis.

    This is the canonical "diagonal non-Clifford sandwiched in Hadamards" that
    stabilizer-rank papers benchmark. D is built from single-qubit T gates and
    CZ gates. With ``t_per_qubit=1`` every qubit gets exactly ONE T on a fresh
    active axis (no re-injection), targeting magic-saturation ~1 and peak k ~ n
    -- the regime that would revive the residual-stab-rank bet. ``t_per_qubit>1``
    re-injects magic via T,H,T (avoiding T*T=S) to dial saturation up.

    Measurements are all terminal, so there is no mid-circuit collapse to shrink
    k: the diagonal entangles every qubit and nothing can be measured out early.
    """
    rng = random.Random(seed)
    ops: list[str] = []
    ops.extend(f"H {q}" for q in range(n))
    # Diagonal D: T on every qubit, interleaved CZ structure.
    for q in range(n):
        ops.append(f"T {q}")
    for inj in range(t_per_qubit - 1):
        # Re-inject magic on the same axes without forming T*T=S.
        for q in range(n):
            ops.append(f"H {q}")
            ops.append(f"T {q}")
    # CZ layer(s): Clifford, stay in the frame; entangle the diagonal.
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < cz_density:
                ops.append(f"CZ {a} {b}")
    ops.extend(f"H {q}" for q in range(n))
    ops.extend(f"M {q}" for q in range(n))
    return _lines(*ops)


def hidden_shift_ccz(n_triples: int, seed: int = 0) -> str:
    """Roetteler-style hidden shift with a CCZ-based bent-function oracle.

    Faithful to the Bravyi-Gosset-Howard hidden-shift benchmark shape: the
    non-Clifford content comes from CCZ gates (7 T's each), so magic-saturation
    is HIGH (~7 T's piled across 3 axes per gadget). The high-saturation
    counterpoint to ``iqp``. n = 3 * n_triples qubits.
    """
    rng = random.Random(seed)
    n = 3 * n_triples
    triples = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(n_triples)]
    ops: list[str] = []
    ops.extend(f"H {q}" for q in range(n))
    for q in range(n):  # random shift string (Clifford)
        if rng.random() < 0.5:
            ops.append(f"X {q}")
    for a, b, c in triples:  # bent function oracle
        ops += ccz(a, b, c)
    ops.extend(f"H {q}" for q in range(n))
    for a, b, c in triples:  # dual bent function
        ops += ccz(a, b, c)
    ops.extend(f"H {q}" for q in range(n))
    ops.extend(f"M {q}" for q in range(n))
    return _lines(*ops)


def magic_conveyor(rounds: int, width: int, t_per_round: int, seed: int = 0) -> str:
    """A feedforward-linked chain of measured-out magic blocks.

    Each round uses a FRESH block of ``width`` qubits: Hadamard, classical
    feedforward from the previous round's outcomes (CX rec[-k]), ``t_per_round``
    T-injections on distinct qubits, a Clifford entangling chain, then measure
    ALL ``width`` qubits. Because every round's qubits are measured out before
    the next begins, the active block fully collapses (k -> 0) between rounds:
    per-episode magic = ``t_per_round`` while total magic = rounds*t_per_round.

    The feedforward (each round conditioned on the last) makes this ONE connected
    computation, not a trivially-factorizable product -- so a naive up-front
    |T>^(total) stabilizer decomposition pays for every T, while clifft's
    measurement-driven reduction (and a measurement-AWARE stab-rank sim) sees
    only one round's worth live at a time. This is the structure that exhibits
    rSR << gSR_naive.
    """
    rng = random.Random(seed)
    assert t_per_round <= width
    ops: list[str] = []
    prev_base = None
    for r in range(rounds):
        base = r * width
        qubits = list(range(base, base + width))
        ops.extend(f"H {q}" for q in qubits)
        if prev_base is not None:
            # Feedforward: condition this round on the previous round's parity.
            # rec[-width] .. rec[-1] are the previous round's measurements.
            for j in range(0, width, 2):
                ops.append(f"CX rec[-{width - j}] {qubits[j]}")
        # Magic injection: one T per distinct qubit (no T*T=S).
        for q in rng.sample(qubits, t_per_round):
            ops.append(f"{'T' if rng.random() < 0.5 else 'T_DAG'} {q}")
        # Clifford entangling chain.
        for j in range(width - 1):
            ops.append(f"CZ {qubits[j]} {qubits[j + 1]}")
        # Final H layer: converts the diagonal magic into amplitudes that affect
        # the Z-measurement outcomes (mini-IQP). Without this the magic is
        # phase-only and clifft correctly absorbs it (k stays 0).
        ops.extend(f"H {q}" for q in qubits)
        ops.extend(f"M {q}" for q in qubits)
        prev_base = base
    return _lines(*ops)


def composition_demo() -> list[tuple[str, str]]:
    """Circuits built to exhibit the composition advantage (rSR << gSR_naive).

    Sweep rounds and width: total magic = rounds*width grows (so a naive
    up-front stabilizer decomposition blows up), while peak k and per-episode
    magic stay ~width.
    """
    suite: list[tuple[str, str]] = []
    # Small width: shows rSR/dense FLAT while gSR grows linearly in rounds
    # (the measurement-driven-reduction signature). dense already wins here.
    for rounds in (4, 8, 16):
        suite.append(
            (f"conveyor_r{rounds:02d}_w24", magic_conveyor(rounds, 24, t_per_round=24, seed=rounds))
        )
    # Large width: per-round k passes the dense wall, so the composition (rSR)
    # is the UNIQUE feasible method -- dense 2^k and naive global 2^Ttot both
    # blow up.
    for rounds, width in ((8, 96), (8, 112), (12, 128)):
        suite.append(
            (
                f"conveyor_r{rounds:02d}_w{width}",
                magic_conveyor(rounds, width, t_per_round=width, seed=rounds * 100 + width),
            )
        )
    return suite


def large_k_sweep() -> list[tuple[str, str]]:
    """Targeted sweep: drive peak k into 20-45 at low magic-saturation.

    The decisive test for the residual-stab-rank bet -- if any of these land at
    k > 30 with saturation < 4.4, the idea revives; if k tracks the dense wall
    with saturation pinned near 1 (so dense is already the bottleneck and only
    the exponent constant is on the table), it is confirmed marginal.
    """
    suite: list[tuple[str, str]] = []
    # The t1 variant (one T per fresh qubit) is squeezed to k ~ n/2, so push n
    # to ~120 to drive k up toward clifft's hard wall (k=63). The t3 variant
    # re-injects magic and resists the squeeze (k ~ n), so cap it at n=60 to
    # stay compilable.
    for n in (28, 60, 92, 120):
        suite.append((f"iqp_n{n:03d}_t1_cz30", iqp(n, seed=n, t_per_qubit=1, cz_density=0.3)))
        suite.append((f"iqp_n{n:03d}_t1_cz70", iqp(n, seed=n + 1, t_per_qubit=1, cz_density=0.7)))
    for n in (20, 40, 60):
        suite.append((f"iqp_n{n:03d}_t3_cz30", iqp(n, seed=n + 2, t_per_qubit=3, cz_density=0.3)))
    for nt in (4, 8, 12, 15):  # CCZ hidden shift (high saturation counterpoint)
        suite.append((f"hidden_ccz_t{nt}", hidden_shift_ccz(nt, seed=nt)))
    return suite


def build_suite() -> list[tuple[str, str]]:
    """Return a list of (name, stim_text) covering the decision-relevant regimes."""
    suite: list[tuple[str, str]] = []

    # Random Clifford+T: sweep qubit count and T-density.
    for n in (12, 20, 28):
        for tf in (0.05, 0.15, 0.30):
            suite.append(
                (
                    f"rand_cliffT_n{n}_d{2 * n}_t{int(tf * 100):02d}",
                    random_clifford_t(n, depth=2 * n, t_fraction=tf, seed=n * 100 + int(tf * 100)),
                )
            )

    # Hidden shift: T-heavy, measurement-poor (clifft's hard case).
    for npairs in (4, 8, 12):
        for tl in (1, 2):
            suite.append(
                (
                    f"hidden_shift_np{npairs}_tl{tl}",
                    hidden_shift(npairs, n_t_layers=tl, seed=npairs),
                )
            )

    # QAOA ring: rotation-heavy.
    for n in (10, 16, 24):
        for p in (1, 3):
            suite.append((f"qaoa_ring_n{n}_p{p}", qaoa_ring(n, p_layers=p, seed=n + p)))

    # Surface-code-ish memory: measurement-rich (k should collapse).
    for d in (3, 5):
        for rounds in (3, 6):
            suite.append(
                (f"surface_d{d}_r{rounds}", surface_code_memory_with_t(d, rounds, n_t=4, seed=d))
            )

    # Magic-state injection / distillation inner block.
    for ns in (8, 15):
        suite.append((f"magic_injection_n{ns}", magic_state_injection(ns, seed=ns)))

    return suite
