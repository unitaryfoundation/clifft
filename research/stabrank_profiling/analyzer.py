"""Static (k, t_live) profiler for clifft's compiled bytecode.

Purpose
-------
clifft stores the active block |phi>_A as a DENSE statevector of dimension
2^k, where k = active dimension (``Program.peak_rank`` is the peak). Its hard
wall is 2^k. We are evaluating whether replacing/augmenting that dense block
with a low-rank STABILIZER-RANK decomposition would help. Stabilizer rank of
the active block scales with the number of *live magic injections* t_live, not
with k:

    dense cost        ~ 2^k
    stab-rank (exact) ~ 2^(0.5  * t)     (Bravyi-Gosset 2016 strong-sim rank)
    stab-rank (apx)   ~ 2^(0.228 * t)    (stabilizer extent, sampling)

The research bet is: does k grow large while t_live stays moderate? If yes, a
residual stabilizer-rank decomposition extends clifft's reach. If t_live ~ k
everywhere, the win is only the 0.5/0.228 factor on the exponent.

Why a static walk is exact for k
--------------------------------
The active block is a STACK. In the SVM kernels:
  * OP_EXPAND / OP_EXPAND_T / OP_EXPAND_T_DAG / OP_EXPAND_ROT push axis k
    (assert ``v == active_k``) and do ``active_k++`` unconditionally.
  * OP_MEAS_ACTIVE_DIAGONAL / OP_MEAS_ACTIVE_INTERFERE / OP_SWAP_MEAS_INTERFERE
    (and their _FORCED variants) pop axis k-1 (assert ``v == active_k - 1``)
    and do ``active_k--`` unconditionally.
So the entire active_k trajectory is determined by the opcode SEQUENCE, with no
dependence on measurement outcomes. (This is also why clifft can size the 2^k
allocation at compile time.) We replay that stack offline over the bytecode.

Metric tiers
------------
Tier 1 (EXACT, no provenance ambiguity):
  * peak_active_k          -- peak of the active_k trajectory (== Program.peak_rank).
  * n_nonclifford_total    -- total magic-injecting ops (the global T-count that
                              drives whole-circuit stabilizer rank).
  * n_expand_clifford      -- plain OP_EXPAND (dimension growth that is FREE under
                              stabilizer rank: pure Clifford superposition).
  * n_expand_magic         -- OP_EXPAND_T-family (dimension growth that carries magic).

Tier 2 (HEURISTIC -- provenance stack -> peak_live_magic_axes):
  A parallel boolean stack marks which live active axes carry magic. An axis is
  "magic" if it was pushed by a magic expansion, or had a magic array gate
  (OP_ARRAY_T / _T_DAG / _ROT, and -- conservatively -- a fused OP_ARRAY_U2 /
  OP_ARRAY_U4) applied to it. Popped on active measurement. This COUNTS SEEDED
  magic axes; it ignores magic spreading via entangling array gates, so it is a
  proxy for "how much of k is genuinely non-stabilizer", not a strict bound on
  the true stabilizer rank. Read it as: if even this is << k, Clifford-inflation
  of the active dimension is real and a residual stab-rank decomposition helps.

The fused-unitary ambiguity (OP_ARRAY_U2 / U4 may be Clifford OR non-Clifford
runs produced by the fusion passes) is surfaced explicitly via ``n_fused_u2`` /
``n_fused_u4`` and the ``fused_treated_as_magic`` flag, so the numbers stay honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

# Opcodes that push a new active axis (active_k++).
_EXPAND_CLIFFORD = {"OP_EXPAND"}
_EXPAND_MAGIC = {"OP_EXPAND_T", "OP_EXPAND_T_DAG", "OP_EXPAND_ROT"}
_EXPAND_ALL = _EXPAND_CLIFFORD | _EXPAND_MAGIC

# Opcodes that pop the top active axis (active_k--).
_POP = {
    "OP_MEAS_ACTIVE_DIAGONAL",
    "OP_MEAS_ACTIVE_INTERFERE",
    "OP_SWAP_MEAS_INTERFERE",
    "OP_MEAS_ACTIVE_DIAGONAL_FORCED",
    "OP_MEAS_ACTIVE_INTERFERE_FORCED",
    "OP_SWAP_MEAS_INTERFERE_FORCED",
}

# Magic injected onto an EXISTING active axis (no k change), targeting axis_1.
_ARRAY_MAGIC = {"OP_ARRAY_T", "OP_ARRAY_T_DAG", "OP_ARRAY_ROT"}

# Fused single/two-qubit unitaries: provenance unknown (Clifford or not).
_FUSED_U2 = {"OP_ARRAY_U2"}
_FUSED_U4 = {"OP_ARRAY_U4"}

# Stabilizer-rank exponent constants (log2 cost per magic state).
ALPHA_STRONG = 0.5  # Bravyi-Gosset 2016 strong-simulation rank exponent.
ALPHA_SAMPLE = 0.22845  # stabilizer extent  = -2*log2(cos(pi/8)).


@dataclass
class ProfileResult:
    name: str
    num_qubits: int
    program_peak_rank: int  # Program.peak_rank, the AOT allocation exponent.

    # Tier 1 (exact).
    peak_active_k: int = 0
    n_nonclifford_total: int = 0
    n_expand_clifford: int = 0
    n_expand_magic: int = 0
    n_array_magic: int = 0
    n_fused_u2: int = 0
    n_fused_u4: int = 0
    n_instructions: int = 0

    # Tier 2 (heuristic).
    peak_live_magic_axes: int = 0
    # m_block: peak number of T-injections folded into a single active episode
    # (reset when the active block fully collapses, k -> 0). An honest UPPER
    # bound on the magic content of the live block -- it overcounts T's that
    # were measured out mid-episode, so it makes the residual estimate
    # conservative (pessimistic) rather than rosy. This is the quantity that
    # decides compressibility: a k-dim block saturated by m_block >> k T's has
    # stabilizer rank approaching the full 2^k (no win); a block with
    # m_block ~ k behaves like k fresh magic states (extent applies, big win).
    peak_t_in_episode: int = 0
    fused_treated_as_magic: bool = True

    # Full trajectory (sampled) for histogramming.
    k_trajectory: list[int] = field(default_factory=list)
    tlive_trajectory: list[int] = field(default_factory=list)

    @property
    def log2_dense(self) -> float:
        """clifft today: dense 2^k active block."""
        return float(self.peak_active_k)

    @property
    def log2_global_stabrank_sample(self) -> float:
        """Whole-circuit stabilizer rank (sampling): pays for every T."""
        return ALPHA_SAMPLE * self.n_nonclifford_total

    @property
    def log2_global_stabrank_strong(self) -> float:
        return ALPHA_STRONG * self.n_nonclifford_total

    @property
    def log2_residual_stabrank_sample(self) -> float:
        """Residual stab-rank cost on clifft's active block (sampling).

        Capped at k: the block lives in 2^k dimensions, so its stabilizer rank
        can never exceed 2^k regardless of how many T's were folded in. The
        extent estimate 0.228*m_block only beats dense when the block is
        magic-SPARSE (0.228*m_block < k); a saturated block (m_block >> k) has
        rank ~2^k and stab-rank buys nothing.
        """
        return min(float(self.peak_active_k), ALPHA_SAMPLE * self.peak_t_in_episode)

    @property
    def log2_residual_stabrank_strong(self) -> float:
        return min(float(self.peak_active_k), ALPHA_STRONG * self.peak_t_in_episode)

    @property
    def residual_beats_dense(self) -> bool:
        """True when a residual stab-rank decomposition is cheaper than dense.

        Equivalent to 0.228*m_block < k: the active block is magic-sparse enough
        that the extent decomposition is below the dense dimension ceiling.
        """
        return self.peak_active_k > 0 and self.log2_residual_stabrank_sample < self.log2_dense

    @property
    def magic_saturation(self) -> float:
        """m_block / k: how saturated the active block is with magic.

        <= 4.4 (= 1/0.228) => residual stab-rank (sampling) beats dense.
        >> 4.4 => block is magic-saturated, rank ~2^k, no win over dense.
        """
        if self.peak_active_k == 0:
            return 0.0
        return self.peak_t_in_episode / self.peak_active_k

    @property
    def clifford_inflation(self) -> float:
        """Fraction of peak active dimension that is NOT seeded magic.

        ~1.0 => part of the active dimension is pure Clifford superposition that
        stabilizer rank handles for free. Empirically ~0 in clifft: plain
        OP_EXPAND is rare because Clifford superposition stays in the frame, so
        the active dimension is built almost entirely from magic expansions.
        """
        if self.peak_active_k == 0:
            return 0.0
        return 1.0 - self.peak_live_magic_axes / self.peak_active_k

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            log2_dense=self.log2_dense,
            log2_global_stabrank_sample=self.log2_global_stabrank_sample,
            log2_global_stabrank_strong=self.log2_global_stabrank_strong,
            log2_residual_stabrank_sample=self.log2_residual_stabrank_sample,
            log2_residual_stabrank_strong=self.log2_residual_stabrank_strong,
            clifford_inflation=self.clifford_inflation,
            magic_saturation=self.magic_saturation,
            residual_beats_dense=self.residual_beats_dense,
        )
        # Trajectories are bulky; keep them out of the summary dict.
        d.pop("k_trajectory", None)
        d.pop("tlive_trajectory", None)
        return d


def profile_program(
    program,
    name: str = "",
    fused_treated_as_magic: bool = True,
    keep_trajectory: bool = False,
) -> ProfileResult:
    """Walk a compiled ``clifft.Program`` and reconstruct the (k, t_live) profile.

    Args:
        program: a ``clifft.Program`` (CompiledModule), iterable over Instruction.
        name: label for the result.
        fused_treated_as_magic: if True (default, conservative), OP_ARRAY_U2/U4
            and fused expansions are assumed to inject magic. Set False to get a
            lower bound on magic that ignores fused unitaries.
        keep_trajectory: if True, record the full per-step k and t_live arrays.
    """
    res = ProfileResult(
        name=name,
        num_qubits=int(program.num_qubits),
        program_peak_rank=int(program.peak_rank),
        fused_treated_as_magic=fused_treated_as_magic,
    )

    active_k = 0
    magic_stack: list[bool] = []  # parallel to the active-axis stack
    t_in_episode = 0  # T-injections folded in since the block last had k==0

    def tlive() -> int:
        return sum(magic_stack)

    def note_magic_injection() -> None:
        nonlocal t_in_episode
        res.n_nonclifford_total += 1
        t_in_episode += 1
        res.peak_t_in_episode = max(res.peak_t_in_episode, t_in_episode)

    for ins in program:
        op = ins.opcode.name
        res.n_instructions += 1

        if op in _EXPAND_ALL:
            is_magic = op in _EXPAND_MAGIC
            if op == "OP_EXPAND_ROT" and not fused_treated_as_magic:
                is_magic = False
            magic_stack.append(is_magic)
            active_k += 1
            if op in _EXPAND_CLIFFORD:
                res.n_expand_clifford += 1
            else:
                res.n_expand_magic += 1
                note_magic_injection()

        elif op in _POP:
            if magic_stack:  # defensive; the stack should be non-empty here
                magic_stack.pop()
            active_k = max(0, active_k - 1)
            if active_k == 0:
                t_in_episode = 0  # block fully collapsed; new episode

        elif op in _ARRAY_MAGIC:
            res.n_array_magic += 1
            note_magic_injection()
            v = ins.axis_1
            if 0 <= v < len(magic_stack):
                magic_stack[v] = True

        elif op in _FUSED_U2:
            res.n_fused_u2 += 1
            if fused_treated_as_magic:
                note_magic_injection()
                v = ins.axis_1
                if 0 <= v < len(magic_stack):
                    magic_stack[v] = True

        elif op in _FUSED_U4:
            res.n_fused_u4 += 1
            if fused_treated_as_magic:
                note_magic_injection()
                for v in (ins.axis_1, ins.axis_2):
                    if 0 <= v < len(magic_stack):
                        magic_stack[v] = True

        # all other opcodes (frame ops, array cliffords, dormant meas, noise,
        # detectors, observables, pauli, exp_val) leave (k, magic_stack) unchanged.

        res.peak_active_k = max(res.peak_active_k, active_k)
        res.peak_live_magic_axes = max(res.peak_live_magic_axes, tlive())
        if keep_trajectory:
            res.k_trajectory.append(active_k)
            res.tlive_trajectory.append(tlive())

    return res


def profile_stim(
    stim_text: str,
    name: str = "",
    *,
    optimize: bool = True,
    fused_treated_as_magic: bool = True,
    keep_trajectory: bool = False,
):
    """Compile a stim-format circuit string and profile it.

    ``optimize=True`` runs clifft's default HIR + bytecode passes (the realistic
    case -- this is what the simulator actually executes, including the squeeze
    pass that reduces peak rank). ``optimize=False`` profiles the naive lowering.
    """
    import clifft

    if optimize:
        program = clifft.compile(stim_text)
    else:
        program = clifft.compile(stim_text, hir_passes=None, bytecode_passes=None)
    return profile_program(
        program,
        name=name,
        fused_treated_as_magic=fused_treated_as_magic,
        keep_trajectory=keep_trajectory,
    )
