"""AOT Compiler + VM for Clifford+T simulation.

Phase 1: Compiler walks circuit with stim.TableauSimulator (unitary only),
         rewinding operators to the Heisenberg frame via inverse tableau.
         Emits flat bytecode instructions.

Phase 2: VM executes bytecode per-shot using a flat complex128 array
         and bitwise math. No matrices, no Cliffords, no dynamic resizing.
"""

import math
import numpy as np
import stim
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── Bytecode opcodes ─────────────────────────────────────────────

class OpCode(Enum):
    # T-gate fast path
    BRANCH = auto()          # New independent shift dimension (T/T†)
    COLLIDE = auto()         # Shift already in span (T/T†)
    SCALAR_PHASE = auto()    # beta=0, just a phase per entry (T/T†)
    # Generic LCU variants
    BRANCH_LCU = auto()      # New dimension (generic 2-term LCU)
    COLLIDE_LCU = auto()     # In-span butterfly (generic 2-term LCU)
    SCALAR_PHASE_LCU = auto()  # Diagonal phase (generic 2-term LCU)
    # Measurement variants
    MEASURE_MERGE = auto()   # x_M in span(V), rank decreases
    MEASURE_FILTER = auto()  # x_M = 0, filter by z parity
    INDEX_CNOT = auto()      # Basis rotation before filter
    MEASURE_INDEPENDENT = auto()  # Doesn't touch v
    AG_PIVOT = auto()        # Out-of-span measurement: random outcome + tableau update
    # Noise
    # (NOISE removed: noise is scheduled separately, not a bytecode instruction)
    # Detectors
    DETECTOR = auto()        # Classical XOR over measurement record indices


@dataclass
class Instruction:
    op: OpCode
    mapped_gamma: int = 0   # bit j set iff V[j] . P.Z_bits = 1 mod 2
    sign: complex = 1        # complex sign from rewound Pauli (includes Y phases)
    bit_index: int = 0       # which bit in the v-index
    x_mask: int = 0          # bitmask for collide/merge
    is_dagger: bool = False  # T vs T-dagger
    control_bit: int = 0     # for INDEX_CNOT
    target_bit: int = 0      # for INDEX_CNOT
    # Sign tracker fields (de-interleaved: bit i = virtual qubit i)
    destab_mask: int = 0     # bitmask over destabilizer sign bits
    stab_mask: int = 0       # bitmask over stabilizer sign bits
    base_phase_idx: int = 0  # index into PHASES = [1, 1j, -1, -1j]
    # AG pivot fields (de-interleaved)
    ag_destab_cols: list = field(default_factory=list)  # n ints: new destab j = XOR of old gens with this mask
    ag_stab_cols: list = field(default_factory=list)    # n ints: new stab j = XOR of old gens with this mask
    ag_stab_slot: int = 0        # which stabilizer slot receives the measurement outcome
    ag_ref_outcome: int = 0      # stim's reference outcome (0 or 1)
    # Noise fields (de-interleaved)
    noise_prob: float = 0.0          # probability of Pauli error
    noise_destab_mask: int = 0       # XOR mask for destab_signs
    noise_stab_mask: int = 0         # XOR mask for stab_signs
    # Deterministic measurement flag
    deterministic: bool = False  # if True, outcome is determined by sign tracker
    # LCU fields (used by BRANCH_LCU, COLLIDE_LCU, SCALAR_PHASE_LCU)
    lcu_weight: complex = 0.0    # relative weight w_m for LCU branches
    lcu_c_dom: complex = 1.0     # dominant term coefficient
    # Detector fields
    detector_targets: list = field(default_factory=list)  # measurement record indices to XOR

    def __repr__(self):
        fields = {}
        if self.op in (OpCode.BRANCH, OpCode.COLLIDE, OpCode.SCALAR_PHASE,
                       OpCode.BRANCH_LCU, OpCode.COLLIDE_LCU, OpCode.SCALAR_PHASE_LCU):
            fields['mapped_gamma'] = bin(self.mapped_gamma)
            fields['sign'] = self.sign
            fields['is_dagger'] = self.is_dagger
            if self.op == OpCode.BRANCH:
                fields['bit_index'] = self.bit_index
            if self.op == OpCode.COLLIDE:
                fields['x_mask'] = bin(self.x_mask)
        elif self.op == OpCode.MEASURE_MERGE:
            fields['x_mask'] = bin(self.x_mask)
            fields['bit_index'] = self.bit_index
        elif self.op == OpCode.MEASURE_FILTER:
            fields['mapped_gamma'] = bin(self.mapped_gamma)
            fields['bit_index'] = self.bit_index
        elif self.op == OpCode.INDEX_CNOT:
            fields['control_bit'] = self.control_bit
            fields['target_bit'] = self.target_bit
        fstr = ', '.join(f'{k}={v}' for k, v in fields.items())
        return f'{self.op.name}({fstr})'


# ── GF(2) linear algebra ─────────────────────────────────────────

def _vec_to_int(v: np.ndarray) -> int:
    """Convert binary vector to integer (v[0] is MSB... no, v[0] is bit 0)."""
    result = 0
    for i, b in enumerate(v):
        if b % 2:
            result |= (1 << i)
    return result


def _int_to_vec(x: int, n: int) -> np.ndarray:
    """Convert integer to binary vector of length n."""
    return np.array([(x >> i) & 1 for i in range(n)], dtype=int)


class GF2Basis:
    """Tracks a set of GF(2) basis vectors with span-check and decomposition.

    Maintains vectors in reduced row echelon form for O(n^2) operations.
    Each basis vector has an associated 'bit_index' for the VM array.
    """

    def __init__(self, vec_len: int):
        self.vec_len = vec_len
        # Parallel arrays: basis[i] is the vector, bit_indices[i] is its VM bit
        self.basis: list[np.ndarray] = []
        self.bit_indices: list[int] = []
        self._next_bit = 0
        self._free_bits: list[int] = []  # recycled bit indices

    @property
    def rank(self) -> int:
        return len(self.basis)

    def _pivot_col(self, vec: np.ndarray) -> int:
        """Find the first nonzero column of a vector, or -1."""
        for i in range(self.vec_len):
            if vec[i] % 2 == 1:
                return i
        return -1

    def decompose(self, vec: np.ndarray) -> tuple[bool, int]:
        """Try to express vec as a GF(2) linear combination of basis vectors.

        Returns (in_span, x_mask) where x_mask has bit j set iff basis[j]
        participates in the linear combination.
        """
        remainder = vec.copy() % 2
        x_mask = 0
        for j, b in enumerate(self.basis):
            pivot = self._pivot_col(b)
            if pivot >= 0 and remainder[pivot] % 2 == 1:
                remainder = (remainder + b) % 2
                x_mask |= (1 << self.bit_indices[j])
        in_span = np.all(remainder % 2 == 0)
        return in_span, x_mask

    def add(self, vec: np.ndarray) -> int:
        """Add a new independent vector. Returns its assigned bit index.

        Uses row echelon form (NOT reduced). Existing basis vectors are
        never modified, preserving the meaning of already-emitted bit indices.
        """
        # Reduce the new vector against existing basis
        v = vec.copy() % 2
        for b in self.basis:
            pivot = self._pivot_col(b)
            if pivot >= 0 and v[pivot] % 2 == 1:
                v = (v + b) % 2
        assert self._pivot_col(v) >= 0, "Vector is not independent!"
        if self._free_bits:
            bit_idx = self._free_bits.pop()
        else:
            bit_idx = self._next_bit
            self._next_bit += 1
        # Do NOT reduce existing basis vectors against new one
        self.basis.append(v)
        self.bit_indices.append(bit_idx)
        self._on_rank_change()
        return bit_idx

    def _on_rank_change(self):
        """Hook for external tracking. Overridden by compiler."""
        pass

    def remove_by_bit(self, bit_idx: int):
        """Remove the basis vector with the given bit index.

        All bit indices above the removed one are shifted down by 1
        to keep the VM array compact.
        """
        found = False
        for j in range(len(self.basis)):
            if self.bit_indices[j] == bit_idx:
                self.basis.pop(j)
                self.bit_indices.pop(j)
                found = True
                break
        if not found:
            raise ValueError(f"Bit index {bit_idx} not found in basis")
        # Shift all higher bit indices down
        for j in range(len(self.bit_indices)):
            if self.bit_indices[j] > bit_idx:
                self.bit_indices[j] -= 1
        # Also shift in _free_bits
        self._free_bits = [b - 1 if b > bit_idx else b
                          for b in self._free_bits if b != bit_idx]
        # Adjust _next_bit
        self._next_bit -= 1

    def compute_mapped_gamma(self, z_bits: np.ndarray) -> int:
        """Compute mask: bit at bit_indices[j] set iff V[j] . z_bits = 1 mod 2."""
        mask = 0
        for j, b in enumerate(self.basis):
            if np.dot(b, z_bits) % 2 == 1:
                mask |= (1 << self.bit_indices[j])
        return mask

    def find_anticommuting(self, z_bits: np.ndarray) -> Optional[int]:
        """Find a basis vector index j where V[j] . z_bits = 1 mod 2.
        Returns the bit_index, or None."""
        for j, b in enumerate(self.basis):
            if np.dot(b, z_bits) % 2 == 1:
                return j  # return the position in self.basis
        return None


# ── GF(2) matrix utilities ───────────────────────────────────────

def _pauli_to_2n_vec(ps, n: int) -> np.ndarray:
    """Pack a stim PauliString into a 2n-bit vector [x0..xn-1, z0..zn-1]."""
    x = np.array([1 if ps[i] in (1, 2) else 0 for i in range(n)], dtype=int)
    z = np.array([1 if ps[i] in (2, 3) else 0 for i in range(n)], dtype=int)
    return np.concatenate([x, z])


def _gf2_inv(M: np.ndarray) -> np.ndarray:
    """Compute the inverse of a square GF(2) matrix via Gauss-Jordan."""
    n = M.shape[0]
    A = np.hstack([M % 2, np.eye(n, dtype=int)])
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if A[row, col] % 2 == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2)")
        A[[col, pivot]] = A[[pivot, col]]
        for row in range(n):
            if row != col and A[row, col] % 2 == 1:
                A[row] = (A[row] + A[col]) % 2
    return A[:, n:] % 2


def _build_generator_matrix(tab, n: int) -> np.ndarray:
    """Build the 2n x 2n generator matrix from a forward tableau.

    Columns: [destab_0, destab_1, ..., stab_0, stab_1, ...]
    Rows: 2n-bit vectors [x0..xn-1, z0..zn-1]
    """
    cols = []
    for i in range(n):
        cols.append(_pauli_to_2n_vec(tab.x_output(i), n))
    for i in range(n):
        cols.append(_pauli_to_2n_vec(tab.z_output(i), n))
    return np.array(cols).T


def _compute_ag_transform(fwd_before, fwd_after, n: int) -> tuple[list[int], list[int]]:
    """Compute the GF(2) transform: new_gen_j = XOR of old_gen_k where T[k,j]=1.

    Returns (destab_cols, stab_cols) where:
    - destab_cols[i]: bitmask of old generators composing new destabilizer i
    - stab_cols[i]: bitmask of old generators composing new stabilizer i
    Each bitmask has bit k set for old destab k (k < n) or old stab k-n (k >= n),
    packed as: destab k at bit k, stab k at bit (k + n).
    """
    old_mat = _build_generator_matrix(fwd_before, n)
    new_mat = _build_generator_matrix(fwd_after, n)
    old_inv = _gf2_inv(old_mat)
    transform = (old_inv @ new_mat) % 2  # 2n x 2n

    # Pack columns into de-interleaved bitmasks.
    # Column layout: [destab_0..destab_{n-1}, stab_0..stab_{n-1}]
    # Row k < n -> old destab k -> bit k
    # Row k >= n -> old stab (k-n) -> bit (k-n) + n
    def pack_col(j):
        mask = 0
        for k in range(2 * n):
            if transform[k, j] % 2 == 1:
                if k < n:
                    mask |= (1 << k)       # old destab k
                else:
                    mask |= (1 << (k - n + n))  # old stab (k-n) at bit n+(k-n) = k
        return mask

    destab_cols = [pack_col(j) for j in range(n)]
    stab_cols = [pack_col(j + n) for j in range(n)]
    return destab_cols, stab_cols



def _sign_to_phase_idx(sign: complex) -> int:
    """Encode a Clifford phase as index into PHASES = [1, 1j, -1, -1j]."""
    PHASES_LOCAL = [1, 1j, -1, -1j]
    best_idx, best_dist = 0, float('inf')
    for idx, p in enumerate(PHASES_LOCAL):
        d = abs(sign - p)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    assert best_dist < 1e-10, f"Sign {sign} is not a Clifford phase"
    return best_idx

# ── AOT Compiler ─────────────────────────────────────────────────

class AOTCompiler:
    """Compiles a Clifford+T circuit into flat bytecode.

    The compiler maintains a stim.TableauSimulator as a unitary proxy.
    It NEVER calls measure/reset on the sim — the frame stays purely unitary.
    Resets use SSA: map logical qubit to a fresh virtual qubit.
    """

    def __init__(self, n_qubits: int):
        self.n_logical = n_qubits
        self.n_virtual = n_qubits
        self.sim = stim.TableauSimulator()
        self.sim.set_num_qubits(self.n_virtual)
        # SSA: logical qubit → virtual qubit
        self.qubit_map: dict[int, int] = {i: i for i in range(n_qubits)}
        self.V = GF2Basis(self.n_virtual)
        self.bytecode: list[Instruction] = []
        self._free_virtual_qubits: list[int] = []  # recycled virtual qubit indices
        self.peak_rank: int = 0  # max rank of V during compilation
        self.noise_schedule: dict[int, list[tuple[float, int, int]]] = {}  # pc -> [(prob, destab_mask, stab_mask)]
        # Wire up peak rank tracking
        compiler_ref = self
        def _track_rank():
            compiler_ref.peak_rank = max(compiler_ref.peak_rank, len(compiler_ref.V.basis))
        self.V._on_rank_change = _track_rank

    def _alloc_virtual(self) -> int:
        """Allocate a virtual qubit, recycling if possible."""
        if self._free_virtual_qubits:
            idx = self._free_virtual_qubits.pop()
            # Reset the qubit in stim to |0>
            self.sim.do(stim.CircuitInstruction('R', [idx]))
            return idx
        # No free slots: grow
        idx = self.n_virtual
        self.n_virtual += 1
        self.sim.set_num_qubits(self.n_virtual)
        # Grow all basis vectors in V to match new size
        for j in range(len(self.V.basis)):
            self.V.basis[j] = np.append(self.V.basis[j], 0)
        self.V.vec_len = self.n_virtual
        return idx

    def _free_virtual(self, vq: int):
        """Return a virtual qubit to the free pool."""
        self._free_virtual_qubits.append(vq)

    def _vq(self, logical: int) -> int:
        return self.qubit_map[logical]

    def _extract_rewound_z(self, logical_q: int):
        """Extract P = inv_tableau(Z_q).

        Returns (x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx) where:
        - x_bits, z_bits: binary vectors over virtual qubits
        - sign: complex sign (PauliString sign × i^{Y count})
        - destab_mask: bit i set iff destabilizer i participates (x_bits[i]=1)
        - stab_mask: bit i set iff stabilizer i participates (z_bits[i]=1)
        - base_phase_idx: index into PHASES=[1, 1j, -1, -1j]
        """
        vq = self._vq(logical_q)
        inv = self.sim.current_inverse_tableau()
        z_out = inv.z_output(vq)
        nv = self.n_virtual
        x_bits = np.array([1 if z_out[i] in (1, 2) else 0 for i in range(nv)], dtype=int)
        z_bits = np.array([1 if z_out[i] in (2, 3) else 0 for i in range(nv)], dtype=int)
        n_y = sum(1 for i in range(nv) if z_out[i] == 2)
        sign = z_out.sign * (1j ** n_y)

        # De-interleaved: bit i = virtual qubit i
        destab_mask = 0
        stab_mask = 0
        for i in range(nv):
            if x_bits[i]:
                destab_mask |= (1 << i)
            if z_bits[i]:
                stab_mask |= (1 << i)

        base_phase_idx = _sign_to_phase_idx(sign)
        return x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx

    # ── Clifford gates: update sim only, emit nothing ──

    def h(self, q: int):
        self.sim.h(self._vq(q))

    def s(self, q: int):
        self.sim.s(self._vq(q))

    def s_dag(self, q: int):
        self.sim.s_dag(self._vq(q))

    def cx(self, c: int, t: int):
        self.sim.cx(self._vq(c), self._vq(t))

    def cz(self, a: int, b: int):
        self.sim.cz(self._vq(a), self._vq(b))

    def x(self, q: int):
        self.sim.x(self._vq(q))

    def y(self, q: int):
        self.sim.y(self._vq(q))

    def z(self, q: int):
        self.sim.z(self._vq(q))

    # ── Pauli noise ──

    def _compute_anticommute_masks(self, pauli_error: list[int]) -> tuple[int, int]:
        """Compute de-interleaved anticommute masks for a Pauli error.

        pauli_error[i] is 0=I, 1=X, 2=Y, 3=Z for virtual qubit i.
        Returns (destab_mask, stab_mask) where bit i is set iff the error
        anticommutes with destabilizer i / stabilizer i respectively.
        """
        nv = self.n_virtual
        inv = self.sim.current_inverse_tableau()
        destab_mask = 0
        stab_mask = 0

        e_x = np.array([1 if pauli_error[i] in (1, 2) else 0 for i in range(nv)], dtype=int)
        e_z = np.array([1 if pauli_error[i] in (2, 3) else 0 for i in range(nv)], dtype=int)

        fwd = inv.inverse()
        for i in range(nv):
            d = fwd.x_output(i)
            d_x = np.array([1 if d[k] in (1, 2) else 0 for k in range(nv)], dtype=int)
            d_z = np.array([1 if d[k] in (2, 3) else 0 for k in range(nv)], dtype=int)
            if (np.dot(d_x, e_z) + np.dot(d_z, e_x)) % 2:
                destab_mask |= (1 << i)

            s = fwd.z_output(i)
            s_x = np.array([1 if s[k] in (1, 2) else 0 for k in range(nv)], dtype=int)
            s_z = np.array([1 if s[k] in (2, 3) else 0 for k in range(nv)], dtype=int)
            if (np.dot(s_x, e_z) + np.dot(s_z, e_x)) % 2:
                stab_mask |= (1 << i)

        return destab_mask, stab_mask

    def pauli_noise(self, qubit: int, px: float, py: float, pz: float):
        """Compile depolarizing-style Pauli noise on a qubit.

        Emits up to 3 NOISE instructions (one per X, Y, Z error).
        Each flips the appropriate generator signs with the given probability.
        """
        vq = self._vq(qubit)
        nv = self.n_virtual

        for prob, pauli_code in [(px, 1), (py, 2), (pz, 3)]:
            if prob <= 0:
                continue
            error = [0] * nv
            error[vq] = pauli_code
            dm, sm = self._compute_anticommute_masks(error)
            pc = len(self.bytecode)  # noise fires before next instruction
            if pc not in self.noise_schedule:
                self.noise_schedule[pc] = []
            self.noise_schedule[pc].append((prob, dm, sm))

    def depolarize(self, qubit: int, p: float):
        """Compile single-qubit depolarizing noise: each of X,Y,Z with prob p/3."""
        self.pauli_noise(qubit, p/3, p/3, p/3)

    # ── Pauli rewinding for arbitrary operators ──

    def _extract_rewound_pauli(self, logical_q: int, pauli: str):
        """Extract rewound Pauli P_q through the inverse tableau.

        pauli: 'I', 'X', 'Y', or 'Z'
        Returns (x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx).
        For 'I', returns all-zero with sign=1.
        """
        if pauli == 'I':
            nv = self.n_virtual
            return (np.zeros(nv, dtype=int), np.zeros(nv, dtype=int),
                    1, 0, 0, 0)
        if pauli == 'Z':
            return self._extract_rewound_z(logical_q)

        vq = self._vq(logical_q)
        inv = self.sim.current_inverse_tableau()
        nv = self.n_virtual

        if pauli == 'X':
            ps = inv.x_output(vq)
        elif pauli == 'Y':
            ps_x = inv.x_output(vq)
            ps_z = inv.z_output(vq)
            ps = ps_x * ps_z
            ps = stim.PauliString('+i') * ps
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")

        x_bits = np.array([1 if ps[i] in (1, 2) else 0 for i in range(nv)], dtype=int)
        z_bits = np.array([1 if ps[i] in (2, 3) else 0 for i in range(nv)], dtype=int)
        n_y = sum(1 for i in range(nv) if ps[i] == 2)
        sign = ps.sign * (1j ** n_y)

        destab_mask = 0
        stab_mask = 0
        for i in range(nv):
            if x_bits[i]:
                destab_mask |= (1 << i)
            if z_bits[i]:
                stab_mask |= (1 << i)

        base_phase_idx = _sign_to_phase_idx(sign)
        return x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx

    # ── T / T† gates ──

    def t(self, q: int, dagger: bool = False):
        """Compile a T or T† gate on logical qubit q."""
        x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx = self._extract_rewound_z(q)
        beta = x_bits

        if not np.any(beta % 2):
            z_filt = self.V.compute_mapped_gamma(z_bits)
            self.bytecode.append(Instruction(
                op=OpCode.SCALAR_PHASE,
                mapped_gamma=z_filt,
                sign=sign,
                is_dagger=dagger,
                destab_mask=destab_mask,
                stab_mask=stab_mask,
                base_phase_idx=base_phase_idx,
            ))
        else:
            in_span, x_mask = self.V.decompose(beta)
            if not in_span:
                partial_mask = x_mask
                bit_idx = self.V.add(beta)
                full_mask = partial_mask | (1 << bit_idx)
                z_filt = self.V.compute_mapped_gamma(z_bits)
                self.bytecode.append(Instruction(
                    op=OpCode.BRANCH,
                    mapped_gamma=z_filt,
                    sign=sign,
                    bit_index=bit_idx,
                    x_mask=full_mask,
                    is_dagger=dagger,
                    destab_mask=destab_mask,
                    stab_mask=stab_mask,
                    base_phase_idx=base_phase_idx,
                ))
            else:
                z_filt = self.V.compute_mapped_gamma(z_bits)
                self.bytecode.append(Instruction(
                    op=OpCode.COLLIDE,
                    mapped_gamma=z_filt,
                    sign=sign,
                    x_mask=x_mask,
                    is_dagger=dagger,
                    destab_mask=destab_mask,
                    stab_mask=stab_mask,
                    base_phase_idx=base_phase_idx,
                ))

    def t_dag(self, q: int):
        self.t(q, dagger=True)

    # ── General LCU ──

    # Pauli multiplication table: (P_a, P_b) -> (phase, P_result)
    # where P_a * P_b = phase * P_result
    _PAULI_MULT = {
        ('I', 'I'): (1, 'I'), ('I', 'X'): (1, 'X'), ('I', 'Y'): (1, 'Y'), ('I', 'Z'): (1, 'Z'),
        ('X', 'I'): (1, 'X'), ('X', 'X'): (1, 'I'), ('X', 'Y'): (1j, 'Z'), ('X', 'Z'): (-1j, 'Y'),
        ('Y', 'I'): (1, 'Y'), ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1, 'I'), ('Y', 'Z'): (1j, 'X'),
        ('Z', 'I'): (1, 'Z'), ('Z', 'X'): (1j, 'Y'), ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1, 'I'),
    }

    def apply_lcu(self, qubit: int, terms: list[tuple[complex, str]]):
        """Apply a generic Linear Combination of Unitaries on a single qubit.

        terms: list of (coefficient, pauli_label) where pauli_label in {'I','X','Y','Z'}.
        The operation is U = sum(c_m * P_m).

        Uses Dominant Term Factoring:
        1. Identify dominant term (largest |c_m|)
        2. Absorb P_dom into the stim simulator
        3. Compute relative operators R_m = P_dom * P_m for remaining terms
        4. Emit bytecode for each relative branch
        """
        if not terms:
            return
        assert len(terms) <= 2, (
            "Python POC restricted to 2-term LCUs to avoid sequential "
            "cross-terms. K>2 requires MULTI_BRANCH accumulation."
        )

        # Find dominant term
        dom_idx = max(range(len(terms)), key=lambda i: abs(terms[i][0]))
        c_dom, p_dom = terms[dom_idx]

        # Absorb P_dom into the simulator (geometrically becomes identity path)
        vq = self._vq(qubit)
        if p_dom == 'X':
            self.sim.x(vq)
        elif p_dom == 'Y':
            self.sim.y(vq)
        elif p_dom == 'Z':
            self.sim.z(vq)
        # 'I' -> nothing to absorb

        # Process each non-dominant term
        for m, (c_m, p_m) in enumerate(terms):
            if m == dom_idx:
                continue

            # Relative operator: R_m = P_m * P_dom (P_dom is Hermitian/self-inverse)
            # Order matters: if P_dom and P_m anticommute, P_dom*P_m = -P_m*P_dom
            rel_phase, rel_pauli = self._PAULI_MULT[(p_m, p_dom)]
            # Relative weight: w_m = (c_m / c_dom) * rel_phase
            w_m = (c_m / c_dom) * rel_phase

            if rel_pauli == 'I':
                # Scalar phase on all entries (no spatial shift)
                # This shouldn't normally happen (two terms with same Pauli)
                # but handle it: just scale all entries
                self.bytecode.append(Instruction(
                    op=OpCode.SCALAR_PHASE_LCU,
                    mapped_gamma=0,
                    sign=1,
                    is_dagger=False,
                    destab_mask=0,
                    stab_mask=0,
                    base_phase_idx=0,
                    # Store the LCU weight in the instruction for VM
                    lcu_weight=w_m,
                    lcu_c_dom=c_dom,
                ))
                continue

            # Extract rewound relative Pauli through the CURRENT tableau
            # (after P_dom has been absorbed)
            x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx = \
                self._extract_rewound_pauli(qubit, rel_pauli)
            beta = x_bits

            z_filt = self.V.compute_mapped_gamma(z_bits)

            if not np.any(beta % 2):
                # beta=0: relative Pauli is a stabilizer product. Scalar phase.
                self.bytecode.append(Instruction(
                    op=OpCode.SCALAR_PHASE_LCU,
                    mapped_gamma=z_filt,
                    sign=sign,
                    is_dagger=False,
                    destab_mask=destab_mask,
                    stab_mask=stab_mask,
                    base_phase_idx=base_phase_idx,
                    lcu_weight=w_m,
                    lcu_c_dom=c_dom,
                ))
            else:
                in_span, x_mask = self.V.decompose(beta)
                if not in_span:
                    bit_idx = self.V.add(beta)
                    full_mask = x_mask | (1 << bit_idx)
                    z_filt = self.V.compute_mapped_gamma(z_bits)
                    self.bytecode.append(Instruction(
                        op=OpCode.BRANCH_LCU,
                        mapped_gamma=z_filt,
                        sign=sign,
                        bit_index=bit_idx,
                        x_mask=full_mask,
                        is_dagger=False,
                        destab_mask=destab_mask,
                    stab_mask=stab_mask,
                        base_phase_idx=base_phase_idx,
                        lcu_weight=w_m,
                        lcu_c_dom=c_dom,
                    ))
                else:
                    self.bytecode.append(Instruction(
                        op=OpCode.COLLIDE_LCU,
                        mapped_gamma=z_filt,
                        sign=sign,
                        x_mask=x_mask,
                        is_dagger=False,
                        destab_mask=destab_mask,
                    stab_mask=stab_mask,
                        base_phase_idx=base_phase_idx,
                        lcu_weight=w_m,
                        lcu_c_dom=c_dom,
                    ))

    # ── Measurements ──

    def measure(self, q: int):
        """Compile a Z-basis measurement.

        Three cases:
        1. x_M in span(V): MEASURE_MERGE (butterfly + shrink)
        2. x_M != 0, not in span(V): AG_PIVOT (random outcome + tableau update)
        3. x_M == 0: either MEASURE_FILTER or MEASURE_INDEPENDENT (deterministic)
        """
        x_bits, z_bits, sign, destab_mask, stab_mask, base_phase_idx = self._extract_rewound_z(q)

        if np.any(x_bits % 2):
            in_span, x_mask = self.V.decompose(x_bits)
            if in_span:
                self._emit_merge(x_mask, sign, destab_mask, stab_mask, base_phase_idx)
            else:
                self._emit_ag_pivot(q)
        else:
            z_filt = self.V.compute_mapped_gamma(z_bits)
            if z_filt != 0:
                self._emit_filter(z_bits, sign, destab_mask, stab_mask, base_phase_idx)
            else:
                # DETERMINISTIC: observable is a pure stabilizer product.
                # Outcome determined by reference sign + sign tracker.
                self.bytecode.append(Instruction(
                    op=OpCode.MEASURE_INDEPENDENT,
                    sign=sign,
                    destab_mask=destab_mask,
                    stab_mask=stab_mask,
                    base_phase_idx=base_phase_idx,
                    deterministic=True,
                ))

    def _emit_ag_pivot(self, q: int):
        """Emit AG_PIVOT: perform Aaronson-Gottesman pivot on stim simulator.

        1. Record forward tableau before measurement
        2. Let stim perform the measurement (AG pivot)
        3. Compute the GF(2) transform matrix (old gens -> new gens)
        4. Find which stabilizer slot received the measurement outcome
        5. Emit AG_PIVOT instruction for the VM
        """
        vq = self._vq(q)
        nv = self.n_virtual

        # Record forward tableau before measurement
        inv_before = self.sim.current_inverse_tableau()
        fwd_before = inv_before.inverse()

        # Let stim perform the AG pivot
        self.sim.do(stim.CircuitInstruction('M', [vq]))

        # Record forward tableau after measurement
        inv_after = self.sim.current_inverse_tableau()
        fwd_after = inv_after.inverse()

        # Compute the GF(2) transform: which old generators compose each new one
        ag_destab_cols, ag_stab_cols = _compute_ag_transform(fwd_before, fwd_after, nv)

        # Find which stabilizer slot was modified by the measurement.
        # After AG pivot, one stabilizer becomes Z_vq (or -Z_vq).
        # Find which stabilizer slot has Z_vq.
        ag_stab_slot = -1
        for i in range(nv):
            s = fwd_after.z_output(i)
            # Check if this is ±Z_vq (only Z bit at position vq set)
            is_z_vq = True
            for k in range(nv):
                expected = 3 if k == vq else 0  # Z at vq, I elsewhere
                if s[k] != expected:
                    is_z_vq = False
                    break
            if is_z_vq:
                ag_stab_slot = i
                break

        if ag_stab_slot == -1:
            # Stim may use a different form; just find the stabilizer that changed
            for i in range(nv):
                s_before = _pauli_to_2n_vec(fwd_before.z_output(i), nv)
                s_after = _pauli_to_2n_vec(fwd_after.z_output(i), nv)
                if not np.array_equal(s_before, s_after):
                    ag_stab_slot = i
                    break

        assert ag_stab_slot >= 0, "Could not find modified stabilizer slot"

        # Record stim's reference outcome. The VM will sample its own
        # outcome; if it differs from the reference, the sign bit must flip.
        stim_outcome = int(self.sim.current_measurement_record()[-1])

        self.bytecode.append(Instruction(
            op=OpCode.AG_PIVOT,
            ag_destab_cols=ag_destab_cols,
            ag_stab_cols=ag_stab_cols,
            ag_stab_slot=ag_stab_slot,
            ag_ref_outcome=stim_outcome,
        ))

    def _emit_merge(self, x_mask: int, sign: complex = 1,
                    destab_mask: int = 0, stab_mask: int = 0,
                    base_phase_idx: int = 0):
        """Emit MEASURE_MERGE + shrink the basis."""
        bit_idx = x_mask.bit_length() - 1
        self.bytecode.append(Instruction(
            op=OpCode.MEASURE_MERGE,
            x_mask=x_mask,
            bit_index=bit_idx,
            sign=sign,
            destab_mask=destab_mask,
            stab_mask=stab_mask,
            base_phase_idx=base_phase_idx,
        ))
        self.V.remove_by_bit(bit_idx)

    def _emit_filter(self, z_bits: np.ndarray, sign: complex,
                     destab_mask: int = 0, stab_mask: int = 0,
                     base_phase_idx: int = 0):
        """Emit INDEX_CNOTs + MEASURE_FILTER to shrink by one bit."""
        # Find a basis vector that anti-commutes with z_bits
        j_pos = self.V.find_anticommuting(z_bits)
        assert j_pos is not None
        target_bit = self.V.bit_indices[j_pos]

        # Rotate basis: for every OTHER vector that also anti-commutes,
        # XOR it with the target vector (basis rotation = INDEX_CNOT)
        for k in range(len(self.V.basis)):
            if k == j_pos:
                continue
            if np.dot(self.V.basis[k], z_bits) % 2 == 1:
                ctrl_bit = self.V.bit_indices[k]
                self.V.basis[k] = (self.V.basis[k] + self.V.basis[j_pos]) % 2
                self.bytecode.append(Instruction(
                    op=OpCode.INDEX_CNOT,
                    control_bit=ctrl_bit,
                    target_bit=target_bit,
                ))

        # Now only V[j_pos] anti-commutes with z. Emit filter.
        z_filt = self.V.compute_mapped_gamma(z_bits)
        # After rotation, z_filt should have exactly 1 bit set = target_bit
        self.bytecode.append(Instruction(
            op=OpCode.MEASURE_FILTER,
            mapped_gamma=z_filt,
            bit_index=target_bit,
            sign=sign,
            destab_mask=destab_mask,
            stab_mask=stab_mask,
            base_phase_idx=base_phase_idx,
        ))
        self.V.remove_by_bit(target_bit)

    def measure_reset(self, q: int):
        """Compile measure + reset via SSA."""
        old_vq = self._vq(q)
        self.measure(q)
        # Free the old virtual qubit and allocate a new one
        self._free_virtual(old_vq)
        new_vq = self._alloc_virtual()
        self.qubit_map[q] = new_vq

    def reset(self, q: int):
        """Compile bare reset via SSA (measure then reassign)."""
        old_vq = self._vq(q)
        self.measure(q)
        self._free_virtual(old_vq)
        new_vq = self._alloc_virtual()
        self.qubit_map[q] = new_vq

    def detector(self, measurement_indices: list[int]):
        """Emit a detector: asserts XOR of specified measurement outcomes is 0.

        In the absence of noise, the detector should never fire.
        If it fires, it indicates a physical error occurred.
        """
        self.bytecode.append(Instruction(
            op=OpCode.DETECTOR,
            detector_targets=list(measurement_indices),
        ))

    def compile_ops(self, ops: list[tuple[str, list]]) -> list[Instruction]:
        """Compile a list of (gate_name, args) operations.

        Supported ops:
        - ('H', [q]), ('S', [q]), ('S_DAG', [q]), ('X', [q]), ('Y', [q]), ('Z', [q])
        - ('T', [q]), ('T_DAG', [q])
        - ('CX', [c, t]), ('CZ', [a, b])
        - ('M', [q]), ('MR', [q]), ('R', [q])
        - ('DEPOLARIZE', [q, prob])
        - ('PAULI_NOISE', [q, px, py, pz])
        """
        gate_map = {
            'H': self.h, 'S': self.s, 'S_DAG': self.s_dag,
            'X': self.x, 'Y': self.y, 'Z': self.z,
            'T': lambda q: self.t(q), 'T_DAG': lambda q: self.t_dag(q),
        }
        for gate_name, args in ops:
            if gate_name in gate_map:
                gate_map[gate_name](args[0])
            elif gate_name == 'CX':
                self.cx(*args)
            elif gate_name == 'CZ':
                self.cz(*args)
            elif gate_name == 'M':
                self.measure(*args)
            elif gate_name == 'MR':
                self.measure_reset(*args)
            elif gate_name == 'R':
                self.reset(*args)
            elif gate_name == 'DEPOLARIZE':
                self.depolarize(args[0], args[1])
            elif gate_name == 'PAULI_NOISE':
                self.pauli_noise(args[0], args[1], args[2], args[3])
            elif gate_name == 'DETECTOR':
                self.detector(args)
            elif gate_name == 'LCU':
                # args = [qubit, [(coeff, pauli), ...]]
                self.apply_lcu(args[0], args[1])
            else:
                raise ValueError(f"Unknown gate: {gate_name}")
        return self.bytecode


# ── AOT Runtime VM ─────────────────────────────────────────────

def _popcount(x: int) -> int:
    return bin(x).count('1')


# Phase lookup table for sign tracker: PHASES[idx] = {1, 1j, -1, -1j}
PHASES = [1, 1j, -1, -1j]


class AOTRuntimeVM:
    """Executes compiled bytecode.

    State:
    - v: flat complex128 array of size 2^rank, v[0] = 1.0 initially
    - destab_signs / stab_signs: n-bit integers tracking generator sign parities

    The v array is always a dense 2^k hypercube. Measurements structurally
    shrink it via _shrink_array, so simple range(len(v)) loops suffice.
    """

    def __init__(self, rng=None):
        self.v = np.array([1.0 + 0j], dtype=complex)
        self.active_bits = 0  # number of active bit dimensions
        self.rng = rng or np.random.default_rng()
        self.measurements: list[int] = []  # recorded outcomes
        self.destab_signs: int = 0  # n-bit destabilizer sign tracker
        self.stab_signs: int = 0   # n-bit stabilizer sign tracker
        self.detections: list[int] = []  # detector outcomes (0 = no error, 1 = error)

    def execute(self, bytecode: list[Instruction],
                noise_schedule: dict[int, list[tuple[float, int, int]]] = None):
        """Execute all instructions with optional noise schedule."""
        if noise_schedule is None:
            noise_schedule = {}
        for pc, inst in enumerate(bytecode):
            # Apply noise events scheduled at this PC
            if pc in noise_schedule:
                for prob, dm, sm in noise_schedule[pc]:
                    if self.rng.random() < prob:
                        self.destab_signs ^= dm
                        self.stab_signs ^= sm
            if inst.op in (OpCode.BRANCH, OpCode.BRANCH_LCU):
                self._exec_branch(inst)
            elif inst.op in (OpCode.COLLIDE, OpCode.COLLIDE_LCU):
                self._exec_collide(inst)
            elif inst.op in (OpCode.SCALAR_PHASE, OpCode.SCALAR_PHASE_LCU):
                self._exec_scalar_phase(inst)
            elif inst.op == OpCode.MEASURE_MERGE:
                self._exec_measure_merge(inst)
            elif inst.op == OpCode.MEASURE_FILTER:
                self._exec_measure_filter(inst)
            elif inst.op == OpCode.INDEX_CNOT:
                self._exec_index_cnot(inst)
            elif inst.op == OpCode.MEASURE_INDEPENDENT:
                self._exec_measure_independent(inst)
            elif inst.op == OpCode.AG_PIVOT:
                self._exec_ag_pivot(inst)
            elif inst.op == OpCode.DETECTOR:
                self._exec_detector(inst)

    def _exec_ag_pivot(self, inst: Instruction):
        """AG Pivot: random measurement with de-interleaved sign update."""
        outcome = int(self.rng.integers(0, 2))
        self.measurements.append(outcome)

        # Combine old signs into a single 2n-bit word for the transform.
        # Layout: old destab bits [0..n-1], old stab bits [n..2n-1]
        n = len(inst.ag_destab_cols)
        old_combined = self.destab_signs | (self.stab_signs << n)

        # Apply GF(2) transform to get new signs
        new_destab = 0
        for j, col_mask in enumerate(inst.ag_destab_cols):
            bit = _popcount(old_combined & col_mask) % 2
            new_destab |= (bit << j)

        new_stab = 0
        for j, col_mask in enumerate(inst.ag_stab_cols):
            bit = _popcount(old_combined & col_mask) % 2
            new_stab |= (bit << j)

        # Set the measured stabilizer's relative sign
        slot = inst.ag_stab_slot
        if outcome != inst.ag_ref_outcome:
            new_stab |= (1 << slot)
        else:
            new_stab &= ~(1 << slot)

        self.destab_signs = new_destab
        self.stab_signs = new_stab

    def _exec_detector(self, inst: Instruction):
        """Detector: compute XOR of targeted measurement outcomes.

        Result 0 = no error detected; 1 = error detected.
        """
        parity = 0
        for idx in inst.detector_targets:
            parity ^= self.measurements[idx]
        self.detections.append(parity)


    def _resolve_sign(self, inst: Instruction) -> complex:
        """Compute the physical sign at runtime using the de-interleaved sign tracker.

        frame_parity = popcount(destab_signs & destab_mask) XOR popcount(stab_signs & stab_mask) mod 2
        physical_sign = PHASES[(base_phase_idx + 2 * frame_parity) % 4]
        """
        frame_parity = (_popcount(self.destab_signs & inst.destab_mask)
                        + _popcount(self.stab_signs & inst.stab_mask)) % 2
        return PHASES[(inst.base_phase_idx + 2 * frame_parity) % 4]

    def _exec_branch(self, inst: Instruction):
        """New dimension: array doubles. Handles both T-gate and general LCU."""
        phys_sign = self._resolve_sign(inst)

        k = inst.bit_index
        x_mask = inst.x_mask
        old_size = len(self.v)
        new_v = np.zeros(old_size * 2, dtype=complex)

        # Embed old v into lower half
        for i in range(old_size):
            new_v[i] = self.v[i]

        result = np.zeros(old_size * 2, dtype=complex)

        if inst.op in (OpCode.BRANCH_LCU, OpCode.COLLIDE_LCU, OpCode.SCALAR_PHASE_LCU):
            # General LCU: base branch scaled by c_dom, relative branch by w_m
            c_dom = inst.lcu_c_dom
            w_m = inst.lcu_weight
            for i in range(old_size * 2):
                j = i ^ x_mask
                xi_j_parity = _popcount(j & inst.mapped_gamma) % 2
                xi_j = phys_sign * ((-1) ** xi_j_parity)
                result[i] = c_dom * new_v[i] + c_dom * w_m * xi_j * new_v[j]
        else:
            # T-gate Tangent Trick: divide by cos(π/8), base branch = 1.0
            tan_val = math.tan(math.pi / 8)
            phase_sign = 1j if inst.is_dagger else -1j
            for i in range(old_size * 2):
                j = i ^ x_mask
                xi_j_parity = _popcount(j & inst.mapped_gamma) % 2
                xi_j = phys_sign * ((-1) ** xi_j_parity)
                result[i] = new_v[i] + phase_sign * tan_val * xi_j * new_v[j]

        self.v = result
        self.active_bits += 1

    def _exec_collide(self, inst: Instruction):
        """In-span shift: in-place butterfly. Handles both T-gate and LCU."""
        phys_sign = self._resolve_sign(inst)

        x_mask = inst.x_mask
        size = len(self.v)
        new_v = np.zeros(size, dtype=complex)

        if inst.op in (OpCode.BRANCH_LCU, OpCode.COLLIDE_LCU, OpCode.SCALAR_PHASE_LCU):
            c_dom = inst.lcu_c_dom
            w_m = inst.lcu_weight
            for i in range(size):
                j = i ^ x_mask
                xi_j_parity = _popcount(j & inst.mapped_gamma) % 2
                xi_j = phys_sign * ((-1) ** xi_j_parity)
                new_v[i] = c_dom * self.v[i] + c_dom * w_m * xi_j * self.v[j]
        else:
            # T-gate Tangent Trick
            tan_val = math.tan(math.pi / 8)
            phase_sign = 1j if inst.is_dagger else -1j
            for i in range(size):
                j = i ^ x_mask
                xi_j_parity = _popcount(j & inst.mapped_gamma) % 2
                xi_j = phys_sign * ((-1) ** xi_j_parity)
                new_v[i] = self.v[i] + phase_sign * tan_val * xi_j * self.v[j]

        self.v = new_v

    def _exec_scalar_phase(self, inst: Instruction):
        """Diagonal phase: no spatial shift. Handles both T-gate and LCU."""
        phys_sign = self._resolve_sign(inst)

        if inst.op in (OpCode.BRANCH_LCU, OpCode.COLLIDE_LCU, OpCode.SCALAR_PHASE_LCU):
            c_dom = inst.lcu_c_dom
            w_m = inst.lcu_weight
            for i in range(len(self.v)):
                xi_parity = _popcount(i & inst.mapped_gamma) % 2
                xi = phys_sign * ((-1) ** xi_parity)
                self.v[i] *= (c_dom + c_dom * w_m * xi)
        else:
            # T-gate Tangent Trick
            tan_val = math.tan(math.pi / 8)
            phase_sign = 1j if inst.is_dagger else -1j
            for i in range(len(self.v)):
                xi_parity = _popcount(i & inst.mapped_gamma) % 2
                xi = phys_sign * ((-1) ** xi_parity)
                self.v[i] *= (1.0 + phase_sign * tan_val * xi)

    def _exec_measure_merge(self, inst: Instruction):
        """Measurement that reduces rank. Butterfly merge + shrink."""
        x_mask = inst.x_mask
        bit_idx = inst.bit_index

        size = len(self.v)
        # Compute projections for both outcomes
        proj = [np.zeros(size, dtype=complex), np.zeros(size, dtype=complex)]
        for i in range(size):
            j = i ^ x_mask
            proj[0][i] = (self.v[i] + self.v[j]) / math.sqrt(2)
            proj[1][i] = (self.v[i] - self.v[j]) / math.sqrt(2)

        # Sample proportional to squared norms
        p0 = sum(abs(c) ** 2 for c in proj[0])
        p1 = sum(abs(c) ** 2 for c in proj[1])
        total = p0 + p1
        if total < 1e-30:
            outcome = self.rng.integers(0, 2)
        else:
            outcome = 0 if self.rng.random() < p0 / total else 1
        self.measurements.append(int(outcome))

        # Renormalize chosen projection
        chosen = proj[outcome]
        norm = math.sqrt(sum(abs(c) ** 2 for c in chosen))
        if norm > 1e-15:
            chosen /= norm

        # Shrink: remove dimension bit_idx
        self.v = self._shrink_array(chosen, bit_idx)
        self.active_bits -= 1

    def _exec_measure_filter(self, inst: Instruction):
        """Measurement that filters by z-parity. Zero out + shrink."""
        bit_idx = inst.bit_index
        z_filt = inst.mapped_gamma

        # Compute probabilities
        prob0 = 0.0
        prob1 = 0.0
        for i in range(len(self.v)):
            p = _popcount(i & z_filt) % 2
            if p == 0:
                prob0 += abs(self.v[i]) ** 2
            else:
                prob1 += abs(self.v[i]) ** 2

        total = prob0 + prob1
        if total < 1e-30:
            outcome = self.rng.integers(0, 2)
        else:
            outcome = 0 if self.rng.random() < prob0 / total else 1
        self.measurements.append(int(outcome))

        # Zero out branches that don't match outcome
        for i in range(len(self.v)):
            p = _popcount(i & z_filt) % 2
            if p != outcome:
                self.v[i] = 0.0

        # Renormalize
        norm = math.sqrt(sum(abs(self.v[i]) ** 2 for i in range(len(self.v))))
        if norm > 1e-15:
            self.v /= norm

        # Shrink: remove dimension bit_idx
        self.v = self._shrink_array(self.v, bit_idx)
        self.active_bits -= 1

    def _exec_index_cnot(self, inst: Instruction):
        """Basis rotation: swap entries to implement index-space CNOT."""
        ctrl = inst.control_bit
        tgt = inst.target_bit
        size = len(self.v)
        for i in range(size):
            if (i >> ctrl) & 1 and not ((i >> tgt) & 1):
                j = i ^ (1 << tgt)
                self.v[i], self.v[j] = self.v[j], self.v[i]

    def _exec_measure_independent(self, inst: Instruction):
        """Measurement that doesn't touch v.

        Two sub-cases:
        - deterministic: outcome determined by reference sign + sign tracker
        - random: x_M outside span(V), outcome uniformly random
        """
        if inst.deterministic:
            # Compute physical sign from sign tracker
            frame_parity = (_popcount(self.destab_signs & inst.destab_mask)
                            + _popcount(self.stab_signs & inst.stab_mask)) % 2
            # base_phase_idx: 0 -> +1, 2 -> -1 (eigenvalue 0 or 1)
            base_eigenvalue = 1 if inst.base_phase_idx == 2 else 0
            outcome = base_eigenvalue ^ frame_parity
        else:
            # Truly random: no interaction with v-array
            outcome = int(self.rng.integers(0, 2))
        self.measurements.append(int(outcome))

    def _shrink_array(self, v: np.ndarray, bit_idx: int) -> np.ndarray:
        """Remove dimension bit_idx from the array.

        Keep entries where bit bit_idx is 0 (or whichever half survived).
        Re-index by removing the bit_idx-th bit position.
        """
        old_size = len(v)
        new_size = old_size // 2
        new_v = np.zeros(new_size, dtype=complex)

        # Find which half has nonzero entries
        sum_0 = sum(abs(v[i]) ** 2 for i in range(old_size) if not ((i >> bit_idx) & 1))
        sum_1 = sum(abs(v[i]) ** 2 for i in range(old_size) if (i >> bit_idx) & 1)
        keep_bit = 0 if sum_0 >= sum_1 else 1

        for i in range(old_size):
            if ((i >> bit_idx) & 1) != keep_bit:
                continue
            # Remove bit_idx from index
            lo = i & ((1 << bit_idx) - 1)
            hi = (i >> (bit_idx + 1)) << bit_idx
            new_idx = lo | hi
            new_v[new_idx] = v[i]

        return new_v


# ── Statevector expansion (for verification) ───────────────────

def aot_to_statevector(
    n_logical: int,
    compiler: AOTCompiler,
    vm: AOTRuntimeVM,
) -> np.ndarray:
    """Expand the AOT VM state into a 2^n_logical statevector.

    Uses the compiler's final stim inverse tableau to reconstruct the
    basis states |b_alpha> and sum v[alpha] * |b_alpha>.

    The basis is: |b_alpha> = product of destabilizers^alpha applied to
    the stabilizer state |psi_S>.

    When noise is present, the physical generators differ from the reference
    generators by sign flips tracked in vm.destab_signs and vm.stab_signs.
    """
    nv = compiler.n_virtual
    dim = 1 << n_logical
    inv_tab = compiler.sim.current_inverse_tableau()
    fwd_tab = inv_tab.inverse()

    # Build stabilizer projector.
    # Physical stabilizer i = (-1)^{stab_signs bit i} * S_i^ref
    dim_v = 1 << nv
    psi_s = _find_stabilizer_state(fwd_tab, nv, vm.stab_signs)

    # Build destabilizer matrices.
    # Physical destabilizer i = (-1)^{destab_signs bit i} * D_i^ref
    destab_mats = []
    for i in range(nv):
        mat = _pauli_to_matrix(fwd_tab.x_output(i), nv)
        if (vm.destab_signs >> i) & 1:
            mat = -mat
        destab_mats.append(mat)

    # Map basis vectors in V to virtual-qubit destabilizer products
    # The VM v-array is indexed by abstract bits assigned by the compiler.
    # We need to know which bit_index corresponds to which basis vector.
    # After compilation, some bits may have been freed by measurements.
    # The active bits are tracked in V.bit_indices.
    active_bits = compiler.V.bit_indices.copy()
    active_basis = [b.copy() for b in compiler.V.basis]

    sv = np.zeros(dim_v, dtype=complex)
    for idx in range(len(vm.v)):
        if abs(vm.v[idx]) < 1e-15:
            continue
        # idx encodes which basis vectors are active
        # Compute the full n_virtual-bit alpha
        alpha_vec = np.zeros(nv, dtype=int)
        for j, bit in enumerate(active_bits):
            if (idx >> bit) & 1:
                alpha_vec = (alpha_vec + active_basis[j]) % 2

        # Apply destabilizer product
        state = psi_s.copy()
        for i in range(nv):
            if alpha_vec[i]:
                state = destab_mats[i] @ state

        sv += vm.v[idx] * state

    # Trace out virtual qubits beyond n_logical
    # (those are the SSA fresh qubits, always in |0>)
    if nv > n_logical:
        sv_logical = _partial_trace_to_logical(sv, n_logical, nv, compiler.qubit_map)
    else:
        sv_logical = sv

    # Normalize
    norm = np.linalg.norm(sv_logical)
    if norm > 1e-15:
        sv_logical /= norm
    return sv_logical


def _pauli_to_matrix(ps: stim.PauliString, n: int) -> np.ndarray:
    """Convert stim PauliString to 2^n x 2^n matrix."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {0: I2, 1: X, 2: Y, 3: Z}

    mat = np.array([[1.0 + 0j]])
    for i in range(n):
        mat = np.kron(mat, pauli_map[ps[i]])
    if ps.sign.real < 0:
        mat = -mat
    if ps.sign.imag != 0:
        mat = mat * ps.sign / abs(ps.sign)
    return mat


def _find_stabilizer_state(fwd_tab: stim.Tableau, n: int,
                           stab_signs: int = 0) -> np.ndarray:
    """Find the +1 eigenstate of all physical stabilizers.

    Physical stabilizer i = (-1)^{stab_signs bit i} * S_i^ref.
    De-interleaved layout: bit i = virtual qubit i.
    """
    dim = 1 << n
    proj = np.eye(dim, dtype=complex)
    for i in range(n):
        stab_mat = _pauli_to_matrix(fwd_tab.z_output(i), n)
        sign_bit = (stab_signs >> i) & 1
        phys_sign = (-1) ** sign_bit
        proj = 0.5 * (np.eye(dim) + phys_sign * stab_mat) @ proj

    for seed_idx in range(dim):
        seed = np.zeros(dim, dtype=complex)
        seed[seed_idx] = 1.0
        state = proj @ seed
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            return state / norm
    raise RuntimeError("Failed to find stabilizer state")


def _partial_trace_to_logical(
    sv: np.ndarray, n_logical: int, n_virtual: int,
    qubit_map: dict[int, int]
) -> np.ndarray:
    """Extract the n_logical-qubit statevector from the n_virtual-qubit one.

    The logical qubits are mapped to specific virtual qubit positions.
    The remaining virtual qubits should be in |0> (SSA fresh qubits).
    We select the computational basis states where all non-logical qubits are 0.
    """
    dim_logical = 1 << n_logical
    sv_out = np.zeros(dim_logical, dtype=complex)

    # Which virtual qubits are the logical ones?
    logical_to_virtual = [qubit_map[i] for i in range(n_logical)]

    dim_virtual = 1 << n_virtual
    for idx_v in range(dim_virtual):
        # Check: all non-logical virtual qubits must be 0
        non_logical_ok = True
        for vq in range(n_virtual):
            if vq not in logical_to_virtual:
                if (idx_v >> vq) & 1:
                    non_logical_ok = False
                    break
        if not non_logical_ok:
            continue

        # Extract logical bits
        idx_l = 0
        for lq in range(n_logical):
            vq = logical_to_virtual[lq]
            if (idx_v >> vq) & 1:
                idx_l |= (1 << lq)

        sv_out[idx_l] += sv[idx_v]

    return sv_out
