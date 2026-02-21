# Stim C++ Source Reference for clifft AOT Design Review

This document was prepared for an LLM reviewer to assess how the clifft AOT
compiler (described in `aot_design_llm.md`, implemented in `aot_compiler.py`)
uses stim's internals, and to evaluate design decisions for the C++ VM.

Stim source: https://github.com/quantumlib/Stim (cloned at commit HEAD of main)

---

## 1. How the AOT Compiler Uses Stim Today (Python Prototype)

The AOT compiler (`aot_compiler.py`) uses stim's Python API as a "unitary proxy":

```python
self.sim = stim.TableauSimulator()  # sole stim object
```

**Clifford gate application** — compiler calls `sim.h()`, `sim.s()`, `sim.cx()`,
etc. These modify the simulator's internal inverse tableau. The VM emits nothing
for Cliffords — they're fully absorbed at compile time.

**Heisenberg rewinding** — to extract the current-frame Pauli for a gate/measurement:
```python
inv = self.sim.current_inverse_tableau()  # returns copy of inv_state (free)
z_out = inv.z_output(vq)                  # direct row lookup on inverse tab
x_out = inv.x_output(vq)                  # direct row lookup on inverse tab
```
The compiler reads rows of the *inverse* tableau to get rewound observables.

**Forward tableau extraction** — needed for building GF(2) generator matrices:
```python
fwd = inv.inverse()                       # EXPENSIVE: transposes + sign fixup
d = fwd.x_output(i)                       # destabilizer i
s = fwd.z_output(i)                       # stabilizer i
```

**AG pivot (measurement collapse):**
```python
self.sim.do(stim.CircuitInstruction('M', [vq]))  # stim does full AG pivot
# Then compare inv_before vs inv_after to extract the GF(2) transform
```

**Dominant-term Pauli absorption** — `sim.x(vq)`, `sim.y(vq)`, `sim.z(vq)` to
apply the dominant Pauli of an LCU decomposition into the tableau.

---

## 2. Key Design Questions for the C++ VM

### Q1: Forward vs Inverse Tableau — Which Do We Want?

The compiler needs BOTH, for different operations:

| Operation | Needs | Why |
|-----------|-------|-----|
| Heisenberg rewind (Z_q, X_q) | **Inverse** rows | `inv.z_output(q)` = "what physical Pauli maps to Z_q?" |
| Extract stabilizers/destabilizers | **Forward** rows | `fwd.z_output(i)` = stabilizer i, `fwd.x_output(i)` = destabilizer i |
| AG pivot comparison | **Forward** before/after | To compute the GF(2) sign transform matrix |
| Gate application | **Inverse** (prepend) | Stim builds the inverse incrementally via prepend |

Stim's `TableauSimulator` natively stores `inv_state` (the inverse tableau).
`current_inverse_tableau()` returns a zero-computation copy. Getting the forward
tableau requires calling `.inverse()` which is O(n²) — it transposes four n×n
bit tables and fixes signs via round-trip evaluation.

For the C++ compiler, we should store the inverse tableau natively (matching stim)
and compute forward on demand only for generator extraction and AG pivot diffs.

### Q2: Scaling Beyond 64 Qubits — Bitword Architecture

Our current VM uses `uint64_t tableau_signs` — a single 64-bit integer for the
2n-bit sign tracker (max 32 qubits). For >32 qubits, we need wider storage.

Stim solves this with a `bitword<W>` template hierarchy:
- `bitword<64>`: wraps `uint64_t` (portable fallback)
- `bitword<128>`: wraps `__m128i` (SSE2)
- `bitword<256>`: wraps `__m256i` (AVX2)

All expose identical interfaces (`^=`, `&=`, `popcount`, etc.). The entire
codebase is templated on `W`, chosen at compile time.

**Impact on VM opcodes and state:**

| VM Field | Current | Scaled |
|----------|---------|--------|
| `tableau_signs` | `uint64_t` | `simd_bits<W>` (2n bits, SIMD-aligned) |
| `gen_mask` | `uint64_t` | `simd_bits<W>` per instruction |
| `anticommute_mask` | `uint64_t` | `simd_bits<W>` per instruction |
| `mapped_gamma` | `uint32_t` | `uint32_t` (unchanged — indexes over V rank, not qubit count) |
| `x_mask` | `uint32_t` | `uint32_t` (unchanged — indexes over V rank, not qubit count) |
| `ag_update_cols` | list of ints | `simd_bit_table<W>` (2n × 2n GF(2) transform) |

Critically: `mapped_gamma`, `x_mask`, `active_indices`, and `v[]` index over
the **stabilizer rank** (dimension of V), NOT over qubit count. These stay as
plain integers/arrays regardless of qubit count. Only the sign tracker and
per-generator masks scale with n.

---

## 3. Preliminary Analysis (LLM-assisted)

The following analysis was produced by LLM sub-agents that read and summarized
the stim source files. It is included as context for the reviewer.

### 3.1 Tableau Internals

**What `Tableau` stores (forward form):**
- `xs[i]` = the Pauli string C·X_i·C† (forward X conjugation)
- `zs[i]` = the Pauli string C·Z_i·C† (forward Z conjugation)
- Each half (`TableauHalf`) contains two n×n bit tables (`xt`, `zt`) plus an
  n-bit sign vector

**What `TableauSimulator` stores:**
- A single field `inv_state` of type `Tableau<W>` — this IS the inverse tableau
- Gates are applied by **prepending the inverse gate** onto `inv_state`
  - Self-inverse gates (H, CNOT): `inv_state.prepend_H_XZ(q)`
  - Non-self-inverse (S): `inv_state.prepend_SQRT_Z_DAG(q)` (note: DAG)
- Prepending is efficient because it operates along contiguous memory (columns)

**Forward vs inverse output methods on `Tableau`:**
- `xs[i]`, `zs[i]` → direct row lookup (O(1))
- `inverse_x_output(i)`, `inverse_z_output(i)` → computed by reading columns
  with x↔z quadrant swaps, then fixing sign via round-trip (O(n))
- `.inverse()` → full O(n²) transpose + sign fixup

**`current_inverse_tableau()`** simply returns a copy of `inv_state` — no
computation. The inverse is the native stored state.

### 3.2 Memory Architecture

**`simd_bits<W>`** — dynamically-sized, SIMD-aligned bit vector:
```cpp
struct simd_bits<W> {
    size_t num_simd_words;  // count of bitword<W> elements
    bitword<W> *ptr_simd;   // SIMD-aligned allocation
};
```
For 65 qubits with W=256, this allocates one 256-bit word (32 bytes).
All bulk operations (`^=`, `&=`, `popcount`) loop over `num_simd_words`.

**`simd_bit_table<W>`** — 2D bit matrix in a single flat allocation:
```cpp
struct simd_bit_table<W> {
    size_t num_simd_words_major;  // rows (padded to W)
    size_t num_simd_words_minor;  // cols (padded to W)
    simd_bits<W> data;            // single flat buffer
};
```
Row-major layout. `operator[](i)` returns a `simd_bits_range_ref<W>` pointing
into the flat buffer — zero-copy row access.

**`PauliString<W>`** — split bit-plane encoding:
```cpp
struct PauliString<W> {
    size_t num_qubits;
    bool sign;
    simd_bits<W> xs;  // X bits
    simd_bits<W> zs;  // Z bits
};
```
I=00, X=10, Z=01, Y=11. Multiplying two N-qubit Paulis is bulk SIMD XOR/AND.

**`Tableau<W>`** data layout:
```cpp
struct TableauHalf<W> {
    size_t num_qubits;
    simd_bit_table<W> xt;   // N×N X-component bits
    simd_bit_table<W> zt;   // N×N Z-component bits  
    simd_bits<W> signs;     // N sign bits
};
struct Tableau<W> {
    size_t num_qubits;
    TableauHalf<W> xs;  // X input observable maps
    TableauHalf<W> zs;  // Z input observable maps
};
```
Total: 4 N×N bit tables + 2 N-bit sign vectors.

### 3.3 AG Pivot (Measurement Collapse)

Stim's `collapse_qubit_z()` implements the Aaronson-Gottesman pivot:
1. Search stabilizer generators for one that anti-commutes with Z_target
2. Gaussian-eliminate all other anti-commuting generators via prepended CNOTs
3. Apply H to the pivot generator to make it commute
4. Assign random measurement outcome; flip pivot sign if needed

This operates on the **transposed** tableau (via `TableauTransposedRaii`) for
memory efficiency — row operations on the transposed tableau are column
operations on the original, which is what Gaussian elimination needs.

### 3.4 Frame Simulator (Batch Noise Sampling)

Stim's `FrameSimulator` is conceptually the closest existing architecture to
our VM — it tracks Pauli frame deviations across many shots simultaneously:
```cpp
struct FrameSimulator<W> {
    simd_bit_table<W> x_table;  // qubit × batch_size
    simd_bit_table<W> z_table;  // qubit × batch_size
    MeasureRecordBatch<W> m_record;
};
```
Each row is one qubit; each column is one shot. Gates operate on rows.
The key difference: FrameSimulator only tracks Pauli frames (Clifford-only),
while our VM additionally maintains complex coefficient vectors for non-Clifford
branching.

---

## 4. File Inventory

All paths relative to `stim/src/`.

### 4.1 Memory Primitives

| File | Lines | Description |
|------|------:|-------------|
| `stim/mem/bitword.h` | 203 | **Included.** Abstract `bitword<W>` concept: type traits, `aligned_malloc`, `popcount`, transpose. |
| `stim/mem/bitword_64.h` | 147 | **Included.** `bitword<64>` specialization wrapping `uint64_t`. Reference implementation for all bitword ops. |
| `stim/mem/bitword_128_sse.h` | 229 | *Not included.* Same API as bitword_64 but wraps `__m128i` with SSE2 intrinsics. |
| `stim/mem/bitword_256_avx.h` | 261 | *Not included.* Same API as bitword_64 but wraps `__m256i` with AVX2 intrinsics. |
| `stim/mem/simd_bits.h` | 213 | **Included.** `simd_bits<W>`: dynamically-sized SIMD-aligned bit vector. Core building block. |
| `stim/mem/simd_bits.inl` | 346 | **Included.** Implementation of simd_bits — allocation, XOR, AND, popcount, equality, randomize. |
| `stim/mem/simd_bits_range_ref.h` | 343 | *Not included.* Non-owning view into simd_bits. Same API, no allocation. Used for zero-copy row access. |
| `stim/mem/simd_bits_range_ref.inl` | 294 | *Not included.* Implementation of range_ref ops. |
| `stim/mem/simd_bit_table.h` | 193 | **Included.** `simd_bit_table<W>`: 2D bit matrix with flat row-major storage. |
| `stim/mem/simd_bit_table.inl` | 354 | **Included.** Table ops: transpose, resize, row access, random fill. |
| `stim/mem/simd_word.h` | 41 | *Not included.* Tiny type alias (`simd_word<W>` = `bitword<W>`). |

### 4.2 Stabilizer Data Structures

| File | Lines | Description |
|------|------:|-------------|
| `stim/stabilizers/pauli_string.h` | 154 | **Included.** `PauliString<W>`: owning Pauli string with sign + xs/zs bit vectors. |
| `stim/stabilizers/pauli_string_ref.h` | 235 | **Included.** `PauliStringRef<W>`: non-owning ref. Key methods: `inplace_right_mul_returning_log_i_scalar`, `commutes`. |
| `stim/stabilizers/pauli_string_ref.inl` | 1400 | *Not included.* Implementations of PauliString operations — multiplication, commutation check, per-gate Heisenberg updates, circuit propagation, parsing, printing. Bulk is per-gate `do_instruction`/`undo_instruction` dispatch. |
| `stim/stabilizers/tableau.h` | 251 | **Included.** `Tableau<W>` and `TableauHalf<W>`: the core data structure. Forward conjugation map stored as 4 N×N bit tables + 2 sign vectors. |
| `stim/stabilizers/tableau.inl` | 786 | **Included.** Tableau operations: `prepend_*` gate methods, `inverse()`, `x_output`/`z_output`, `inverse_x_output`/`inverse_z_output`, `then()` composition. |
| `stim/stabilizers/tableau_transposed_raii.h` | 65 | **Included.** RAII wrapper that transposes tableau on construction, un-transposes on destruction. Enables efficient column operations. |
| `stim/stabilizers/tableau_transposed_raii.inl` | 157 | **Included.** Transposed-mode gate implementations: `append_ZCX`, `append_H_XZ`, `append_SWAP`, `append_X`, etc. Used during AG pivot. |

### 4.3 Simulators

| File | Lines | Description |
|------|------:|-------------|
| `stim/simulators/tableau_simulator.h` | 330 | **Included.** `TableauSimulator<W>`: interactive stabilizer simulator. Stores `inv_state` (inverse tableau). Methods for gates, measurement, collapse, postselection. |
| `stim/simulators/tableau_simulator.inl` | 1815 | *Not included.* Full implementations: `do_H_XZ`, `do_SQRT_Z`, `collapse_qubit_z` (AG pivot), `do_MZ` (measurement), `peek_observable_expectation`, `postselect_z`, etc. Key excerpt: `collapse_qubit_z` searches for anti-commuting stabilizer, Gaussian-eliminates, applies H, samples outcome. |
| `stim/simulators/frame_simulator.h` | 162 | **Included.** `FrameSimulator<W>`: batch Pauli-frame sampler. Tracks x_table/z_table (qubit × shots). Architecturally analogous to our VM's sign tracker but Clifford-only. |
| `stim/simulators/frame_simulator.inl` | 1115 | *Not included.* Per-gate batch implementations, measurement recording, noise injection, detector evaluation. |

### 4.4 Gate Definitions

| File | Lines | Description |
|------|------:|-------------|
| `stim/gates/gates.h` | 414 | **Included.** Gate metadata: enum `GateType`, `GateFlags`, gate name table, categorization (Clifford, noise, measurement, etc.). |
| `stim/gates/gates.cc` | 365 | *Not included.* Gate property table initialization. |

---

## 5. Concatenated Source Files

### 5.1 Memory Primitives

```cpp
// ===== stim/mem/bitword.h (203 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_MEM_BIT_WORD_H
#define _STIM_MEM_BIT_WORD_H

#include <cstddef>

namespace stim {

/// A `bitword` is a bag of bits that can be operated on in a SIMD-esque fashion
/// by individual CPU instructions.
///
/// This template would not have to exist, except that different architectures and
/// operating systems expose different interfaces between native types like
/// uint64_t and intrinsics like __m256i. For example, in some contexts, __m256i
/// values can be operated on by operators (e.g. you can do `a ^= b`) while in
/// other contexts this does not work. The bitword template implementations define
/// a common set of methods required by stim to function, so that the same code
/// can be compiled to use 256 bit registers or 64 bit registers as appropriate.
template <size_t bit_size>
struct bitword;

template <size_t W>
inline bool operator==(const bitword<W> &self, const bitword<W> &other) {
    return self.to_u64_array() == other.to_u64_array();
}

template <size_t W>
inline bool operator<(const bitword<W> &self, const bitword<W> &other) {
    auto v1 = self.to_u64_array();
    auto v2 = other.to_u64_array();
    for (size_t k = 0; k < v1.size(); k++) {
        if (v1[k] != v2[k]) {
            return v1[k] < v2[k];
        }
    }
    return false;
}

template <size_t W>
inline bool operator!=(const bitword<W> &self, const bitword<W> &other) {
    return !(self == other);
}

template <size_t W>
inline bool operator==(const bitword<W> &self, int other) {
    return self == (bitword<W>)other;
}
template <size_t W>
inline bool operator!=(const bitword<W> &self, int other) {
    return self != (bitword<W>)other;
}
template <size_t W>
inline bool operator==(const bitword<W> &self, uint64_t other) {
    return self == (bitword<W>)other;
}
template <size_t W>
inline bool operator!=(const bitword<W> &self, uint64_t other) {
    return self != (bitword<W>)other;
}
template <size_t W>
inline bool operator==(const bitword<W> &self, int64_t other) {
    return self == (bitword<W>)other;
}
template <size_t W>
inline bool operator!=(const bitword<W> &self, int64_t other) {
    return self != (bitword<W>)other;
}

template <size_t W>
std::ostream &operator<<(std::ostream &out, const bitword<W> &v) {
    out << "bitword<" << W << ">{";
    auto u = v.to_u64_array();
    for (size_t k = 0; k < u.size(); k++) {
        for (size_t b = 0; b < 64; b++) {
            if ((b | k) && (b & 7) == 0) {
                out << ' ';
            }
            out << ".1"[(u[k] >> b) & 1];
        }
    }
    out << '}';
    return out;
}

template <size_t W>
inline bitword<W> operator<<(const bitword<W> &self, uint64_t offset) {
    return self.shifted((int)offset);
}

template <size_t W>
inline bitword<W> operator>>(const bitword<W> &self, uint64_t offset) {
    return self.shifted(-(int)offset);
}

template <size_t W>
inline bitword<W> &operator<<=(bitword<W> &self, uint64_t offset) {
    self = (self << offset);
    return self;
}

template <size_t W>
inline bitword<W> &operator>>=(bitword<W> &self, uint64_t offset) {
    self = (self >> offset);
    return self;
}

template <size_t W>
inline bitword<W> operator<<(const bitword<W> &self, int offset) {
    return self.shifted((int)offset);
}

template <size_t W>
inline bitword<W> operator>>(const bitword<W> &self, int offset) {
    return self.shifted(-(int)offset);
}

template <size_t W>
inline bitword<W> &operator<<=(bitword<W> &self, int offset) {
    self = (self << offset);
    return self;
}

template <size_t W>
inline bitword<W> &operator>>=(bitword<W> &self, int offset) {
    self = (self >> offset);
    return self;
}

template <size_t W>
inline bitword<W> operator&(const bitword<W> &self, int mask) {
    return self & bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator&(const bitword<W> &self, uint64_t mask) {
    return self & bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator&(const bitword<W> &self, int64_t mask) {
    return self & bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator|(const bitword<W> &self, int mask) {
    return self | bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator|(const bitword<W> &self, uint64_t mask) {
    return self | bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator|(const bitword<W> &self, int64_t mask) {
    return self | bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator^(const bitword<W> &self, int mask) {
    return self ^ bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator^(const bitword<W> &self, uint64_t mask) {
    return self ^ bitword<W>(mask);
}
template <size_t W>
inline bitword<W> operator^(const bitword<W> &self, int64_t mask) {
    return self ^ bitword<W>(mask);
}

template <size_t W>
inline bitword<W> andnot(const bitword<W> &inv, const bitword<W> &val) {
    return inv.andnot(val);
}
inline uint64_t andnot(uint64_t inv, uint64_t val) {
    return ~inv & val;
}
inline uint32_t andnot(uint32_t inv, uint32_t val) {
    return ~inv & val;
}
inline uint16_t andnot(uint16_t inv, uint16_t val) {
    return ~inv & val;
}
inline uint8_t andnot(uint8_t inv, uint8_t val) {
    return ~inv & val;
}
inline bool andnot(bool inv, bool val) {
    return !inv && val;
}

}  // namespace stim

#endif

```

```cpp
// ===== stim/mem/bitword_64.h (147 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_MEM_SIMD_WORD_64_STD_H
#define _STIM_MEM_SIMD_WORD_64_STD_H

#include <array>
#include <bit>
#include <sstream>
#include <stdlib.h>

#include "stim/mem/bitword.h"
#include "stim/mem/simd_util.h"

namespace stim {

/// Implements a 64 bit bitword using no architecture-specific instructions, just standard C++.
template <>
struct bitword<64> {
    constexpr static size_t BIT_SIZE = 64;
    constexpr static size_t BIT_POW = 6;

    union {
        uint64_t val;
        uint8_t u8[8];
    };

    static void *aligned_malloc(size_t bytes) {
        return malloc(bytes);
    }
    static void aligned_free(void *ptr) {
        free(ptr);
    }

    inline constexpr bitword() : val{} {
    }
    inline bitword(std::array<uint64_t, 1> val) : val{val[0]} {
    }
    inline constexpr bitword(uint64_t v) : val{v} {
    }
    inline constexpr bitword(int64_t v) : val{(uint64_t)v} {
    }
    inline constexpr bitword(int v) : val{(uint64_t)v} {
    }

    constexpr inline static bitword<64> tile64(uint64_t pattern) {
        return bitword<64>(pattern);
    }

    constexpr inline static bitword<64> tile8(uint8_t pattern) {
        return bitword<64>(tile64_helper(pattern, 8));
    }

    inline std::array<uint64_t, 1> to_u64_array() const {
        return std::array<uint64_t, 1>{val};
    }
    inline operator bool() const {  // NOLINT(hicpp-explicit-conversions)
        return (bool)(val);
    }
    inline operator int() const {  // NOLINT(hicpp-explicit-conversions)
        return (int)val;
    }
    inline operator uint64_t() const {  // NOLINT(hicpp-explicit-conversions)
        return val;
    }
    inline operator int64_t() const {  // NOLINT(hicpp-explicit-conversions)
        return (int64_t)val;
    }

    inline bitword<64> &operator^=(const bitword<64> &other) {
        val ^= other.val;
        return *this;
    }

    inline bitword<64> &operator&=(const bitword<64> &other) {
        val &= other.val;
        return *this;
    }

    inline bitword<64> &operator|=(const bitword<64> &other) {
        val |= other.val;
        return *this;
    }

    inline bitword<64> operator^(const bitword<64> &other) const {
        return bitword<64>(val ^ other.val);
    }

    inline bitword<64> operator&(const bitword<64> &other) const {
        return bitword<64>(val & other.val);
    }

    inline bitword<64> operator|(const bitword<64> &other) const {
        return bitword<64>(val | other.val);
    }

    inline bitword<64> andnot(const bitword<64> &other) const {
        return bitword<64>(~val & other.val);
    }

    inline bitword<64> operator~() const {
        return {~val};
    }

    inline uint16_t popcount() const {
        return std::popcount(val);
    }

    inline std::string str() const {
        std::stringstream out;
        out << *this;
        return out.str();
    }

    inline bitword<64> shifted(int offset) const {
        uint64_t v = val;
        if (offset >= 64 || offset <= -64) {
            v = 0;
        } else if (offset > 0) {
            v <<= offset;
        } else {
            v >>= -offset;
        }
        return bitword<64>{v};
    }

    static void inplace_transpose_square(bitword<64> *data_block, size_t stride) {
        inplace_transpose_64x64((uint64_t *)data_block, stride);
    }
};

}  // namespace stim

#endif

```

```cpp
// ===== stim/mem/simd_bits.h (213 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_MEM_SIMD_BITS_H
#define _STIM_MEM_SIMD_BITS_H

#include <cstdint>
#include <random>

#include "stim/mem/bit_ref.h"
#include "stim/mem/simd_bits_range_ref.h"

namespace stim {

/// Densely packed bits, allocated with alignment and padding enabling SIMD operations.
///
/// Note that, due to the padding, the smallest simd_bits you can have is 256 bits (32 bytes) long.
///
/// For performance, simd_bits does not store the "intended" size of the data, only the padded size. Any intended size
/// has to be tracked separately.
template <size_t W>
struct simd_bits {
    size_t num_simd_words;
    union {
        // It is fair to say that this is the most dangerous block, or danger-enabling block, in the entire codebase.
        // C++ is very particular when it comes to touching the same memory as if it had multiple different types.
        // If you know how to make something *for sure work as a flexibly-accessible bag of bits*, please fix this.
        // In the meantime, always build with `-fno-strict-aliasing` and a short ritual prayer to the compiler gods.
        uint8_t *u8;
        uint64_t *u64;
        bitword<W> *ptr_simd;
    };

    /// Constructs a zero-initialized simd_bits with at least the given number of bits.
    explicit simd_bits(size_t min_bits);
    /// Frees allocated bits.
    ~simd_bits();
    /// Copy constructor.
    simd_bits(const simd_bits &other);
    /// Copy constructor from range reference.
    simd_bits(const simd_bits_range_ref<W> other);
    /// Move constructor.
    simd_bits(simd_bits &&other) noexcept;

    /// Copy assignment.
    simd_bits &operator=(const simd_bits &other);
    /// Copy assignment from range reference.
    simd_bits &operator=(const simd_bits_range_ref<W> other);
    /// Move assignment.
    simd_bits &operator=(simd_bits &&other) noexcept;
    // Xor assignment.
    simd_bits &operator^=(const simd_bits_range_ref<W> other);
    // Mask assignment.
    simd_bits &operator&=(const simd_bits_range_ref<W> other);
    simd_bits &operator|=(const simd_bits_range_ref<W> other);
    // Addition assigment
    simd_bits &operator+=(const simd_bits_range_ref<W> other);
    simd_bits &operator-=(const simd_bits_range_ref<W> other);
    // right shift assignment
    simd_bits &operator>>=(int offset);
    // left shift assignment
    simd_bits &operator<<=(int offset);
    // Swap assignment.
    simd_bits &swap_with(simd_bits_range_ref<W> other);

    // Equality.
    bool operator==(const simd_bits_range_ref<W> &other) const;
    bool operator==(const simd_bits<W> &other) const;
    // Inequality.
    bool operator!=(const simd_bits_range_ref<W> &other) const;
    bool operator!=(const simd_bits<W> &other) const;
    /// Determines whether or not any of the bits in the simd_bits are non-zero.
    bool not_zero() const;

    // Arbitrary ordering.
    bool operator<(const simd_bits_range_ref<W> other) const;

    void destructive_resize(size_t new_min_bits);
    void preserving_resize(size_t new_min_bits);

    /// Returns a reference to the bit at offset k.
    bit_ref operator[](size_t k);
    /// Returns a const reference to the bit at offset k.
    const bit_ref operator[](size_t k) const;
    /// Returns a reference to the bits in this simd_bits.
    operator simd_bits_range_ref<W>();
    /// Returns a const reference to the bits in this simd_bits.
    operator const simd_bits_range_ref<W>() const;
    /// Returns a reference to a sub-range of the bits in this simd_bits.
    inline simd_bits_range_ref<W> word_range_ref(size_t word_offset, size_t sub_num_simd_words) {
        return simd_bits_range_ref<W>(ptr_simd + word_offset, sub_num_simd_words);
    }
    /// Returns a reference to a sub-range of the bits at the start of this simd_bits.
    inline simd_bits_range_ref<W> prefix_ref(size_t unpadded_bit_length) {
        return simd_bits_range_ref<W>(ptr_simd, min_bits_to_num_simd_words<W>(unpadded_bit_length));
    }
    /// Returns a const reference to a sub-range of the bits in this simd_bits.
    inline const simd_bits_range_ref<W> word_range_ref(size_t word_offset, size_t sub_num_simd_words) const {
        return simd_bits_range_ref<W>(ptr_simd + word_offset, sub_num_simd_words);
    }

    /// Returns the number of bits that are 1 in the bit range.
    size_t popcnt() const;
    /// Returns the power-of-two-ness of the number, or SIZE_MAX if the number has no 1s.
    size_t countr_zero() const;

    /// Inverts all bits in the range.
    void invert_bits();
    /// Sets all bits in the range to zero.
    void clear();
    /// Randomizes the contents of this simd_bits using the given random number generator, up to the given bit position.
    void randomize(size_t num_bits, std::mt19937_64 &rng);
    /// Returns a simd_bits with at least the given number of bits, with bits up to the given number of bits randomized.
    /// Padding bits beyond the minimum number of bits are not randomized.
    static simd_bits<W> random(size_t min_bits, std::mt19937_64 &rng);

    /// Returns whether or not the two ranges have set bits in common.
    bool intersects(const simd_bits_range_ref<W> other) const;
    /// Returns whether or not all bits that are set in `other` are also set in this range.
    bool is_subset_of_or_equal_to(const simd_bits_range_ref<W> other) const;

    /// Writes bits from another location.
    /// Bits not part of the write are unchanged.
    void truncated_overwrite_from(simd_bits_range_ref<W> other, size_t num_bits);
    /// Sets all bits at the given position and beyond it to 0.
    void clear_bits_past(size_t num_kept_bits);

    /// Returns a description of the contents of the simd_bits.
    std::string str() const;

    /// Number of 64 bit words in the range.
    inline size_t num_u64_padded() const {
        return num_simd_words * (sizeof(bitword<W>) / sizeof(uint64_t));
    }
    /// Number of 32 bit words in the range.
    inline size_t num_u32_padded() const {
        return num_simd_words * (sizeof(bitword<W>) / sizeof(uint32_t));
    }
    /// Number of 16 bit words in the range.
    inline size_t num_u16_padded() const {
        return num_simd_words * (sizeof(bitword<W>) / sizeof(uint16_t));
    }
    /// Number of 8 bit words in the range.
    inline size_t num_u8_padded() const {
        return num_simd_words * (sizeof(bitword<W>) / sizeof(uint8_t));
    }
    /// Number of bits in the range.
    inline size_t num_bits_padded() const {
        return num_simd_words * W;
    }

    uint64_t as_u64() const;

    template <typename CALLBACK>
    void for_each_set_bit(CALLBACK callback) const {
        size_t n = num_u64_padded();
        for (size_t w = 0; w < n; w++) {
            uint64_t v = u64[w];
            while (v) {
                size_t q = w * 64 + std::countr_zero(v);
                v &= v - 1;
                callback(q);
            }
        }
    }
};

template <size_t W>
simd_bits<W> operator^(const simd_bits_range_ref<W> a, const simd_bits_range_ref<W> b);
template <size_t W>
simd_bits<W> operator|(const simd_bits_range_ref<W> a, const simd_bits_range_ref<W> b);
template <size_t W>
simd_bits<W> operator&(const simd_bits_range_ref<W> a, const simd_bits_range_ref<W> b);
template <size_t W>
simd_bits<W> operator^(const simd_bits<W> a, const simd_bits_range_ref<W> b);
template <size_t W>
simd_bits<W> operator|(const simd_bits<W> a, const simd_bits_range_ref<W> b);
template <size_t W>
simd_bits<W> operator&(const simd_bits<W> a, const simd_bits_range_ref<W> b);
template <size_t W>
simd_bits<W> operator^(const simd_bits_range_ref<W> a, const simd_bits<W> b);
template <size_t W>
simd_bits<W> operator|(const simd_bits_range_ref<W> a, const simd_bits<W> b);
template <size_t W>
simd_bits<W> operator&(const simd_bits_range_ref<W> a, const simd_bits<W> b);
template <size_t W>
simd_bits<W> operator^(const simd_bits<W> a, const simd_bits<W> b);
template <size_t W>
simd_bits<W> operator|(const simd_bits<W> a, const simd_bits<W> b);
template <size_t W>
simd_bits<W> operator&(const simd_bits<W> a, const simd_bits<W> b);

template <size_t W>
std::ostream &operator<<(std::ostream &out, const simd_bits<W> m);

}  // namespace stim

#include "stim/mem/simd_bits.inl"

#endif

```

```cpp
// ===== stim/mem/simd_bits.inl (346 lines) =====
// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <sstream>

#include "stim/mem/simd_util.h"

namespace stim {

template <size_t W>
uint64_t *malloc_aligned_padded_zeroed(size_t min_bits) {
    size_t num_u8 = min_bits_to_num_bits_padded<W>(min_bits) >> 3;
    void *result = bitword<W>::aligned_malloc(num_u8);
    memset(result, 0, num_u8);
    return (uint64_t *)result;
}

template <size_t W>
simd_bits<W>::simd_bits(size_t min_bits)
    : num_simd_words(min_bits_to_num_simd_words<W>(min_bits)), u64(malloc_aligned_padded_zeroed<W>(min_bits)) {
}

template <size_t W>
simd_bits<W>::simd_bits(const simd_bits<W> &other)
    : num_simd_words(other.num_simd_words), u64(malloc_aligned_padded_zeroed<W>(other.num_bits_padded())) {
    memcpy(u8, other.u8, num_u8_padded());
}

template <size_t W>
simd_bits<W>::simd_bits(const simd_bits_range_ref<W> other)
    : num_simd_words(other.num_simd_words), u64(malloc_aligned_padded_zeroed<W>(other.num_bits_padded())) {
    memcpy(u8, other.u8, num_u8_padded());
}

template <size_t W>
simd_bits<W>::simd_bits(simd_bits<W> &&other) noexcept : num_simd_words(other.num_simd_words), u64(other.u64) {
    other.u64 = nullptr;
    other.num_simd_words = 0;
}

template <size_t W>
simd_bits<W>::~simd_bits() {
    if (u64 != nullptr) {
        bitword<W>::aligned_free(u64);
        u64 = nullptr;
        num_simd_words = 0;
    }
}

template <size_t W>
void simd_bits<W>::clear() {
    simd_bits_range_ref<W>(*this).clear();
}

template <size_t W>
void simd_bits<W>::invert_bits() {
    simd_bits_range_ref<W>(*this).invert_bits();
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator=(simd_bits<W> &&other) noexcept {
    (*this).~simd_bits();
    new (this) simd_bits(std::move(other));
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator=(const simd_bits<W> &other) {
    *this = simd_bits_range_ref<W>(other);
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator=(const simd_bits_range_ref<W> other) {
    // Avoid re-allocating if already the same size.
    if (num_simd_words == other.num_simd_words) {
        simd_bits_range_ref<W>(*this) = other;
        return *this;
    }

    (*this).~simd_bits();
    new (this) simd_bits(other);
    return *this;
}

template <size_t W>
bool simd_bits<W>::operator==(const simd_bits_range_ref<W> &other) const {
    return simd_bits_range_ref<W>(*this) == other;
}

template <size_t W>
bool simd_bits<W>::operator==(const simd_bits<W> &other) const {
    return simd_bits_range_ref<W>(*this) == simd_bits_range_ref<W>(other);
}

template <size_t W>
bool simd_bits<W>::operator!=(const simd_bits_range_ref<W> &other) const {
    return !(*this == other);
}

template <size_t W>
bool simd_bits<W>::operator!=(const simd_bits<W> &other) const {
    return !(*this == other);
}

template <size_t W>
simd_bits<W> simd_bits<W>::random(size_t min_bits, std::mt19937_64 &rng) {
    simd_bits<W> result(min_bits);
    result.randomize(min_bits, rng);
    return result;
}

template <size_t W>
void simd_bits<W>::randomize(size_t num_bits, std::mt19937_64 &rng) {
    simd_bits_range_ref<W>(*this).randomize(num_bits, rng);
}

template <size_t W>
void simd_bits<W>::truncated_overwrite_from(simd_bits_range_ref<W> other, size_t num_bits) {
    simd_bits_range_ref<W>(*this).truncated_overwrite_from(other, num_bits);
}

template <size_t W>
void simd_bits<W>::clear_bits_past(size_t num_kept_bits) {
    simd_bits_range_ref<W>(*this).clear_bits_past(num_kept_bits);
}

template <size_t W>
bit_ref simd_bits<W>::operator[](size_t k) {
    return bit_ref(u64, k);
}

template <size_t W>
const bit_ref simd_bits<W>::operator[](size_t k) const {
    return bit_ref(u64, k);
}

template <size_t W>
simd_bits<W>::operator simd_bits_range_ref<W>() {
    return simd_bits_range_ref<W>(ptr_simd, num_simd_words);
}

template <size_t W>
simd_bits<W>::operator const simd_bits_range_ref<W>() const {
    return simd_bits_range_ref<W>(ptr_simd, num_simd_words);
}

template <size_t W>
simd_bits<W> operator^(const simd_bits_range_ref<W> a, const simd_bits_range_ref<W> b) {
    assert(a.num_simd_words == b.num_simd_words);
    simd_bits<W> result(a.num_bits_padded());
    ((simd_bits_range_ref<W>)result).for_each_word(a, b, [](bitword<W> &w0, bitword<W> &w1, bitword<W> &w2) {
        w0 = w1 ^ w2;
    });
    return result;
}
template <size_t W>
simd_bits<W> operator|(const simd_bits_range_ref<W> a, const simd_bits_range_ref<W> b) {
    assert(a.num_simd_words == b.num_simd_words);
    simd_bits<W> result(a.num_bits_padded());
    ((simd_bits_range_ref<W>)result).for_each_word(a, b, [](bitword<W> &w0, bitword<W> &w1, bitword<W> &w2) {
        w0 = w1 | w2;
    });
    return result;
}
template <size_t W>
simd_bits<W> operator&(const simd_bits_range_ref<W> a, const simd_bits_range_ref<W> b) {
    assert(a.num_simd_words == b.num_simd_words);
    simd_bits<W> result(a.num_bits_padded());
    ((simd_bits_range_ref<W>)result).for_each_word(a, b, [](bitword<W> &w0, bitword<W> &w1, bitword<W> &w2) {
        w0 = w1 & w2;
    });
    return result;
}
template <size_t W>
simd_bits<W> operator^(const simd_bits<W> a, const simd_bits_range_ref<W> b) {
    return (const simd_bits_range_ref<W>)a ^ (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator^(const simd_bits_range_ref<W> a, const simd_bits<W> b) {
    return (const simd_bits_range_ref<W>)a ^ (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator^(const simd_bits<W> a, const simd_bits<W> b) {
    return (const simd_bits_range_ref<W>)a ^ (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator|(const simd_bits<W> a, const simd_bits_range_ref<W> b) {
    return (const simd_bits_range_ref<W>)a | (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator|(const simd_bits_range_ref<W> a, const simd_bits<W> b) {
    return (const simd_bits_range_ref<W>)a | (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator|(const simd_bits<W> a, const simd_bits<W> b) {
    return (const simd_bits_range_ref<W>)a | (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator&(const simd_bits<W> a, const simd_bits_range_ref<W> b) {
    return (const simd_bits_range_ref<W>)a & (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator&(const simd_bits_range_ref<W> a, const simd_bits<W> b) {
    return (const simd_bits_range_ref<W>)a & (const simd_bits_range_ref<W>)b;
}
template <size_t W>
simd_bits<W> operator&(const simd_bits<W> a, const simd_bits<W> b) {
    return (const simd_bits_range_ref<W>)a & (const simd_bits_range_ref<W>)b;
}

template <size_t W>
bool simd_bits<W>::operator<(const simd_bits_range_ref<W> other) const {
    if (num_simd_words != other.num_simd_words) {
        return num_simd_words < other.num_simd_words;
    }
    for (size_t k = 0; k < num_simd_words; k++) {
        if (ptr_simd[k] != other.ptr_simd[k]) {
            return ptr_simd[k] < other.ptr_simd[k];
        }
    }
    return false;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator^=(const simd_bits_range_ref<W> other) {
    simd_bits_range_ref<W>(*this) ^= other;
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator&=(const simd_bits_range_ref<W> other) {
    simd_bits_range_ref<W>(*this) &= other;
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator|=(const simd_bits_range_ref<W> other) {
    simd_bits_range_ref<W>(*this) |= other;
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator+=(const simd_bits_range_ref<W> other) {
    simd_bits_range_ref<W>(*this) += other;
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator-=(const simd_bits_range_ref<W> other) {
    simd_bits_range_ref<W>(*this) -= other;
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator>>=(int offset) {
    simd_bits_range_ref<W>(*this) >>= offset;
    return *this;
}

template <size_t W>
simd_bits<W> &simd_bits<W>::operator<<=(int offset) {
    simd_bits_range_ref<W>(*this) <<= offset;
    return *this;
}

template <size_t W>
bool simd_bits<W>::not_zero() const {
    return simd_bits_range_ref<W>(*this).not_zero();
}

template <size_t W>
bool simd_bits<W>::intersects(const simd_bits_range_ref<W> other) const {
    return simd_bits_range_ref<W>(*this).intersects(other);
}

template <size_t W>
bool simd_bits<W>::is_subset_of_or_equal_to(const simd_bits_range_ref<W> other) const {
    return simd_bits_range_ref<W>(*this).is_subset_of_or_equal_to(other);
}

template <size_t W>
std::string simd_bits<W>::str() const {
    return simd_bits_range_ref<W>(*this).str();
}

template <size_t W>
simd_bits<W> &simd_bits<W>::swap_with(simd_bits_range_ref<W> other) {
    simd_bits_range_ref<W>(*this).swap_with(other);
    return *this;
}

template <size_t W>
void simd_bits<W>::destructive_resize(size_t new_min_bits) {
    if (min_bits_to_num_bits_padded<W>(new_min_bits) == num_bits_padded()) {
        return;
    }
    *this = std::move(simd_bits<W>(new_min_bits));
}

template <size_t W>
void simd_bits<W>::preserving_resize(size_t new_min_bits) {
    if (min_bits_to_num_bits_padded<W>(new_min_bits) == num_bits_padded()) {
        return;
    }
    simd_bits<W> new_storage(new_min_bits);
    memcpy(new_storage.ptr_simd, ptr_simd, sizeof(bitword<W>) * std::min(num_simd_words, new_storage.num_simd_words));
    *this = std::move(new_storage);
}

template <size_t W>
size_t simd_bits<W>::popcnt() const {
    return simd_bits_range_ref<W>(*this).popcnt();
}

template <size_t W>
uint64_t simd_bits<W>::as_u64() const {
    return simd_bits_range_ref<W>(*this).as_u64();
}

template <size_t W>
size_t simd_bits<W>::countr_zero() const {
    return simd_bits_range_ref<W>(*this).countr_zero();
}

template <size_t W>
std::ostream &operator<<(std::ostream &out, const simd_bits<W> m) {
    return out << simd_bits_range_ref<W>(m);
}

}  // namespace stim

```

```cpp
// ===== stim/mem/simd_bit_table.h (193 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_MEM_SIMD_BIT_TABLE_H
#define _STIM_MEM_SIMD_BIT_TABLE_H

#include "stim/mem/simd_bits.h"

namespace stim {

/// A 2d array of bit-packed booleans, padded and aligned to make simd operations more efficient.
///
/// The table contents are indexed by a major axis (not contiguous in memory) then a minor axis (contiguous in memory).
///
/// Note that, due to the padding, the smallest table you can have is 256x256 bits (8 KiB).
/// Technically the padding of the major axis is not necessary, but it's included so that transposing preserves size.
///
/// Similar to simd_bits, simd_bit_table has no notion of the "intended" size of data, only the padded size. The
/// intended size has to be stored separately.
template <size_t W>
struct simd_bit_table {
    size_t num_simd_words_major;
    size_t num_simd_words_minor;
    simd_bits<W> data;

    /// Creates zero initialized table.
    simd_bit_table(size_t min_bits_major, size_t min_bits_minor);
    /// Creates a randomly initialized table.
    static simd_bit_table random(
        size_t num_randomized_major_bits, size_t num_randomized_minor_bits, std::mt19937_64 &rng);
    /// Creates a square table with 1s down the diagonal.
    static simd_bit_table identity(size_t n);
    /// Concatenates tables together to form a larger table.
    static simd_bit_table from_quadrants(
        size_t n,
        const simd_bit_table &upper_left,
        const simd_bit_table &upper_right,
        const simd_bit_table &lower_left,
        const simd_bit_table &lower_right);
    /// Parses a bit table from some text.
    ///
    /// Args:
    ///     text: A paragraph of characters specifying the contents of a bit table.
    ///         Each line is a row (a major index) of the table.
    ///         Each position within a line is a column (a minor index) of the table.
    ///         A '1' at character C of line L (counting from 0) indicates out[L][C] will be set.
    ///         A '0', '.', or '_' indicates out[L][C] will not be set.
    ///         Leading newlines and spaces at the beginning of the text are ignored.
    ///         Leading spaces at the beginning of a line are ignored.
    ///         Other characters result in errors.
    ///
    /// Returns:
    ///     A simd_bit_table with cell contents corresponding to the text.
    static simd_bit_table from_text(const char *text, size_t min_rows = 0, size_t min_cols = 0);

    /// Resizes the table. Doesn't clear to zero. Does nothing if already the target size.
    void destructive_resize(size_t new_min_bits_major, size_t new_min_bits_minor);

    /// Copies the table into another table.
    ///
    /// It's safe for the other table to have a different size.
    /// When the other table has a different size, only the data at locations common to both
    /// tables are copied over.
    void copy_into_different_size_table(simd_bit_table<W> &other) const;

    /// Resizes the table, keeping any data common to the old and new size and otherwise zeroing data.
    void resize(size_t new_min_bits_major, size_t new_min_bits_minor);

    /// Equality.
    bool operator==(const simd_bit_table &other) const;
    /// Inequality.
    bool operator!=(const simd_bit_table &other) const;

    /// Returns a reference to a row (column) of the table, when using row (column) major indexing.
    inline simd_bits_range_ref<W> operator[](size_t major_index) {
        return data.word_range_ref(major_index * num_simd_words_minor, num_simd_words_minor);
    }
    /// Returns a const reference to a row (column) of the table, when using row (column) major indexing.
    inline const simd_bits_range_ref<W> operator[](size_t major_index) const {
        return data.word_range_ref(major_index * num_simd_words_minor, num_simd_words_minor);
    }
    /// operator[] lets us extract a specified bit as (*this)[major_index][minor_index].
    /// We can decompose these indicies as
    /// major_index = (major_index_high << bitword<W>::BIT_POW) + major_index_low
    /// minor_index = (minor_index_high << bitword<W>::BIT_POW) + minor_index_low
    /// As minor_index_low ranges from 0 to W-1, (*this)[major_index][minor_index] ranges over the
    /// bits of an aligned SIMD word. get_index_of_bitword returns the index in data.ptr_simd of
    /// the corresponding simd word.
    inline size_t get_index_of_bitword(size_t major_index_high, size_t major_index_low, size_t minor_index_high) const {
        size_t major_index = (major_index_high << bitword<W>::BIT_POW) + major_index_low;
        return major_index * num_simd_words_minor + minor_index_high;
    }

    /// Square matrix multiplication (assumes row major indexing). n is the diameter of the matrix.
    simd_bit_table square_mat_mul(const simd_bit_table &rhs, size_t n) const;
    /// Square matrix inverse, assuming input is lower triangular. n is the diameter of the matrix.
    simd_bit_table inverse_assuming_lower_triangular(size_t n) const;
    /// Transposes the table inplace.
    void do_square_transpose();
    /// Transposes the table out of place into a target location.
    void transpose_into(simd_bit_table &out) const;
    /// Transposes the table out of place.
    simd_bit_table transposed() const;
    /// Returns a subset of the table.
    simd_bit_table slice_maj(size_t maj_start_bit, size_t maj_stop_bit) const;

    /// Returns a copy of a column of the table.
    simd_bits<W> read_across_majors_at_minor_index(size_t major_start, size_t major_stop, size_t minor_index) const;

    /// Concatenates the contents of the two tables, along the major axis.
    simd_bit_table<W> concat_major(const simd_bit_table<W> &second, size_t n_first, size_t n_second) const;
    /// Overwrites a range of the table with a range from another table with the same minor size.
    void overwrite_major_range_with(
        size_t dst_major_start, const simd_bit_table<W> &src, size_t src_major_start, size_t num_major_indices) const;

    /// Sets all bits in the table to zero.
    void clear();

    /// Number of 64 bit words in a column (row) assuming row (column) major indexing.
    inline size_t num_major_u64_padded() const {
        return num_simd_words_major * (sizeof(bitword<W>) / sizeof(uint64_t));
    }
    /// Number of 32 bit words in a column (row) assuming row (column) major indexing.
    inline size_t num_major_u32_padded() const {
        return num_simd_words_major * (sizeof(bitword<W>) / sizeof(uint32_t));
    }
    /// Number of 16 bit words in a column (row) assuming row (column) major indexing.
    inline size_t num_major_u16_padded() const {
        return num_simd_words_major * (sizeof(bitword<W>) / sizeof(uint16_t));
    }
    /// Number of 8 bit words in a column (row) assuming row (column) major indexing.
    inline size_t num_major_u8_padded() const {
        return num_simd_words_major * (sizeof(bitword<W>) / sizeof(uint8_t));
    }
    /// Number of bits in a column (row) assuming row (column) major indexing.
    inline size_t num_major_bits_padded() const {
        return num_simd_words_major * W;
    }

    /// Number of 64 bit words in a row (column) assuming row (column) major indexing.
    inline size_t num_minor_u64_padded() const {
        return num_simd_words_minor * (sizeof(bitword<W>) / sizeof(uint64_t));
    }
    /// Number of 32 bit words in a row (column) assuming row (column) major indexing.
    inline size_t num_minor_u32_padded() const {
        return num_simd_words_minor * (sizeof(bitword<W>) / sizeof(uint32_t));
    }
    /// Number of 16 bit words in a row (column) assuming row (column) major indexing.
    inline size_t num_minor_u16_padded() const {
        return num_simd_words_minor * (sizeof(bitword<W>) / sizeof(uint16_t));
    }
    /// Number of 8 bit words in a row (column) assuming row (column) major indexing.
    inline size_t num_minor_u8_padded() const {
        return num_simd_words_minor * (sizeof(bitword<W>) / sizeof(uint8_t));
    }
    /// Number of bits in a row (column) assuming row (column) major indexing.
    inline size_t num_minor_bits_padded() const {
        return num_simd_words_minor * W;
    }

    /// Returns a padded description of the table's contents.
    std::string str() const;
    /// Returns a truncated square description of the table's contents.
    std::string str(size_t n) const;
    /// Returns a truncated rectangle description of the table's contents.
    std::string str(size_t rows, size_t cols) const;
};

template <size_t W>
std::ostream &operator<<(std::ostream &out, const simd_bit_table<W> &v);

constexpr uint8_t lg(size_t k) {
    return k <= 1 ? 0 : lg(k >> 1) + 1;
}

}  // namespace stim

#include "stim/mem/simd_bit_table.inl"

#endif

```

```cpp
// ===== stim/mem/simd_bit_table.inl (354 lines) =====
// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>

namespace stim {

template <size_t W>
simd_bit_table<W>::simd_bit_table(size_t min_bits_major, size_t min_bits_minor)
    : num_simd_words_major(min_bits_to_num_simd_words<W>(min_bits_major)),
      num_simd_words_minor(min_bits_to_num_simd_words<W>(min_bits_minor)),
      data(min_bits_to_num_bits_padded<W>(min_bits_minor) * min_bits_to_num_bits_padded<W>(min_bits_major)) {
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::identity(size_t n) {
    simd_bit_table<W> result(n, n);
    for (size_t k = 0; k < n; k++) {
        result[k][k] = true;
    }
    return result;
}

template <size_t W>
void simd_bit_table<W>::clear() {
    data.clear();
}

template <size_t W>
bool simd_bit_table<W>::operator==(const simd_bit_table<W> &other) const {
    return num_simd_words_minor == other.num_simd_words_minor && num_simd_words_major == other.num_simd_words_major &&
           data == other.data;
}

template <size_t W>
bool simd_bit_table<W>::operator!=(const simd_bit_table<W> &other) const {
    return !(*this == other);
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::square_mat_mul(const simd_bit_table<W> &rhs, size_t n) const {
    assert(num_major_bits_padded() >= n && num_minor_bits_padded() >= n);
    assert(rhs.num_major_bits_padded() >= n && rhs.num_minor_bits_padded() >= n);

    auto tmp = rhs.transposed();

    simd_bit_table<W> result(n, n);
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            bitword<W> acc{};
            (*this)[row].for_each_word(tmp[col], [&](bitword<W> &w1, bitword<W> &w2) {
                acc ^= w1 & w2;
            });
            result[row][col] = acc.popcount() & 1;
        }
    }

    return result;
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::inverse_assuming_lower_triangular(size_t n) const {
    assert(num_major_bits_padded() >= n && num_minor_bits_padded() >= n);

    simd_bit_table<W> result = simd_bit_table<W>::identity(n);
    simd_bits<W> copy_row(num_minor_bits_padded());
    for (size_t target = 0; target < n; target++) {
        copy_row = (*this)[target];
        for (size_t pivot = 0; pivot < target; pivot++) {
            if (copy_row[pivot]) {
                copy_row ^= (*this)[pivot];
                result[target] ^= result[pivot];
            }
        }
    }
    return result;
}

template <size_t W>
void exchange_low_indices(simd_bit_table<W> &table) {
    for (size_t maj_high = 0; maj_high < table.num_simd_words_major; maj_high++) {
        for (size_t min_high = 0; min_high < table.num_simd_words_minor; min_high++) {
            size_t block_start = table.get_index_of_bitword(maj_high, 0, min_high);
            bitword<W>::inplace_transpose_square(table.data.ptr_simd + block_start, table.num_simd_words_minor);
        }
    }
}

template <size_t W>
void simd_bit_table<W>::destructive_resize(size_t new_min_bits_major, size_t new_min_bits_minor) {
    num_simd_words_minor = min_bits_to_num_simd_words<W>(new_min_bits_minor);
    num_simd_words_major = min_bits_to_num_simd_words<W>(new_min_bits_major);
    data.destructive_resize(num_simd_words_minor * num_simd_words_major * W * W);
}

template <size_t W>
void simd_bit_table<W>::copy_into_different_size_table(simd_bit_table<W> &other) const {
    size_t ni = num_simd_words_minor;
    size_t na = num_simd_words_major;
    size_t mi = other.num_simd_words_minor;
    size_t ma = other.num_simd_words_major;
    size_t num_min_bytes = std::min(ni, mi) * (W / 8);
    size_t num_maj = std::min(na, ma) * W;

    if (ni == mi) {
        memcpy(other.data.ptr_simd, data.ptr_simd, num_min_bytes * num_maj);
    } else {
        for (size_t maj = 0; maj < num_maj; maj++) {
            memcpy(other[maj].ptr_simd, (*this)[maj].ptr_simd, num_min_bytes);
        }
    }
}

template <size_t W>
void simd_bit_table<W>::resize(size_t new_min_bits_major, size_t new_min_bits_minor) {
    auto new_num_simd_words_minor = min_bits_to_num_simd_words<W>(new_min_bits_minor);
    auto new_num_simd_words_major = min_bits_to_num_simd_words<W>(new_min_bits_major);
    if (new_num_simd_words_major == num_simd_words_major && new_num_simd_words_minor == num_simd_words_minor) {
        return;
    }
    auto new_table = simd_bit_table<W>(new_min_bits_major, new_min_bits_minor);
    copy_into_different_size_table(new_table);
    *this = std::move(new_table);
}

template <size_t W>
void simd_bit_table<W>::do_square_transpose() {
    assert(num_simd_words_minor == num_simd_words_major);

    // Current address tensor indices: [...min_low ...min_high ...maj_low ...maj_high]

    exchange_low_indices(*this);

    // Current address tensor indices: [...maj_low ...min_high ...min_low ...maj_high]

    // Permute data such that high address bits of majors and minors are exchanged.
    for (size_t maj_high = 0; maj_high < num_simd_words_major; maj_high++) {
        for (size_t min_high = maj_high + 1; min_high < num_simd_words_minor; min_high++) {
            for (size_t maj_low = 0; maj_low < W; maj_low++) {
                std::swap(
                    data.ptr_simd[get_index_of_bitword(maj_high, maj_low, min_high)],
                    data.ptr_simd[get_index_of_bitword(min_high, maj_low, maj_high)]);
            }
        }
    }
    // Current address tensor indices: [...maj_low ...maj_high ...min_low ...min_high]
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::transposed() const {
    simd_bit_table<W> result(num_minor_bits_padded(), num_major_bits_padded());
    transpose_into(result);
    return result;
}

template <size_t W>
simd_bits<W> simd_bit_table<W>::read_across_majors_at_minor_index(
    size_t major_start, size_t major_stop, size_t minor_index) const {
    assert(major_stop >= major_start);
    assert(major_stop <= num_major_bits_padded());
    assert(minor_index < num_minor_bits_padded());
    simd_bits<W> result(major_stop - major_start);
    for (size_t maj = major_start; maj < major_stop; maj++) {
        result[maj - major_start] = (*this)[maj][minor_index];
    }
    return result;
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::slice_maj(size_t maj_start_bit, size_t maj_stop_bit) const {
    simd_bit_table<W> result(maj_stop_bit - maj_start_bit, num_minor_bits_padded());
    for (size_t k = maj_start_bit; k < maj_stop_bit; k++) {
        result[k - maj_start_bit] = (*this)[k];
    }
    return result;
}

template <size_t W>
void simd_bit_table<W>::transpose_into(simd_bit_table<W> &out) const {
    assert(out.num_simd_words_minor == num_simd_words_major);
    assert(out.num_simd_words_major == num_simd_words_minor);

    for (size_t maj_high = 0; maj_high < num_simd_words_major; maj_high++) {
        for (size_t min_high = 0; min_high < num_simd_words_minor; min_high++) {
            for (size_t maj_low = 0; maj_low < W; maj_low++) {
                size_t src_index = get_index_of_bitword(maj_high, maj_low, min_high);
                size_t dst_index = out.get_index_of_bitword(min_high, maj_low, maj_high);
                out.data.ptr_simd[dst_index] = data.ptr_simd[src_index];
            }
        }
    }

    exchange_low_indices(out);
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::from_quadrants(
    size_t n,
    const simd_bit_table<W> &upper_left,
    const simd_bit_table<W> &upper_right,
    const simd_bit_table<W> &lower_left,
    const simd_bit_table<W> &lower_right) {
    assert(upper_left.num_minor_bits_padded() >= n && upper_left.num_major_bits_padded() >= n);
    assert(upper_right.num_minor_bits_padded() >= n && upper_right.num_major_bits_padded() >= n);
    assert(lower_left.num_minor_bits_padded() >= n && lower_left.num_major_bits_padded() >= n);
    assert(lower_right.num_minor_bits_padded() >= n && lower_right.num_major_bits_padded() >= n);

    simd_bit_table<W> result(n << 1, n << 1);
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < n; col++) {
            result[row][col] = upper_left[row][col];
            result[row][col + n] = upper_right[row][col];
            result[row + n][col] = lower_left[row][col];
            result[row + n][col + n] = lower_right[row][col];
        }
    }
    return result;
}

template <size_t W>
std::string simd_bit_table<W>::str(size_t rows, size_t cols) const {
    std::stringstream out;
    for (size_t row = 0; row < rows; row++) {
        if (row) {
            out << "\n";
        }
        for (size_t col = 0; col < cols; col++) {
            out << ".1"[(*this)[row][col]];
        }
    }
    return out.str();
}

template <size_t W>
std::string simd_bit_table<W>::str(size_t n) const {
    return str(n, n);
}

template <size_t W>
std::string simd_bit_table<W>::str() const {
    return str(num_major_bits_padded(), num_minor_bits_padded());
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::concat_major(
    const simd_bit_table<W> &second, size_t n_first, size_t n_second) const {
    if (num_major_bits_padded() < n_first || second.num_major_bits_padded() < n_second ||
        num_minor_bits_padded() != second.num_minor_bits_padded()) {
        throw std::invalid_argument("Size mismatch");
    }
    simd_bit_table<W> result(n_first + n_second, num_minor_bits_padded());
    auto n1 = n_first * num_minor_u8_padded();
    auto n2 = n_second * num_minor_u8_padded();
    memcpy(result.data.u8, data.u8, n1);
    memcpy(result.data.u8 + n1, second.data.u8, n2);
    return result;
}

template <size_t W>
void simd_bit_table<W>::overwrite_major_range_with(
    size_t dst_major_start, const simd_bit_table<W> &src, size_t src_major_start, size_t num_major_indices) const {
    assert(src.num_minor_bits_padded() == num_minor_bits_padded());
    memcpy(
        data.u8 + dst_major_start * num_minor_u8_padded(),
        src.data.u8 + src_major_start * src.num_minor_u8_padded(),
        num_major_indices * num_minor_u8_padded());
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::from_text(const char *text, size_t min_rows, size_t min_cols) {
    std::vector<std::vector<bool>> lines;
    lines.push_back({});

    // Skip indentation.
    while (*text == '\n' || *text == ' ') {
        text++;
    }

    for (const char *c = text; *c;) {
        if (*c == '\n') {
            lines.push_back({});
            c++;
            // Skip indentation.
            while (*c == ' ') {
                c++;
            }
        } else if (*c == '0' || *c == '.' || *c == '_') {
            lines.back().push_back(false);
            c++;
        } else if (*c == '1') {
            lines.back().push_back(true);
            c++;
        } else {
            throw std::invalid_argument(
                "Expected indented characters from \"10._\\n\". Got '" + std::string(1, *c) + "'.");
        }
    }

    // Remove trailing newline.
    if (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }

    size_t num_cols = min_cols;
    for (const auto &v : lines) {
        num_cols = std::max(v.size(), num_cols);
    }
    size_t num_rows = std::max(min_rows, lines.size());
    simd_bit_table<W> out(num_rows, num_cols);
    for (size_t row = 0; row < lines.size(); row++) {
        for (size_t col = 0; col < lines[row].size(); col++) {
            out[row][col] = lines[row][col];
        }
    }

    return out;
}

template <size_t W>
simd_bit_table<W> simd_bit_table<W>::random(
    size_t num_randomized_major_bits, size_t num_randomized_minor_bits, std::mt19937_64 &rng) {
    simd_bit_table<W> result(num_randomized_major_bits, num_randomized_minor_bits);
    for (size_t maj = 0; maj < num_randomized_major_bits; maj++) {
        result[maj].randomize(num_randomized_minor_bits, rng);
    }
    return result;
}

template <size_t W>
std::ostream &operator<<(std::ostream &out, const stim::simd_bit_table<W> &v) {
    for (size_t k = 0; k < v.num_major_bits_padded(); k++) {
        if (k) {
            out << '\n';
        }
        out << v[k];
    }
    return out;
}

}  // namespace stim

```

### 5.2 Stabilizer Data Structures

```cpp
// ===== stim/stabilizers/pauli_string.h (154 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_STABILIZERS_PAULI_STRING_H
#define _STIM_STABILIZERS_PAULI_STRING_H

#include <functional>
#include <iostream>

#include "stim/mem/simd_bits.h"
#include "stim/stabilizers/pauli_string_ref.h"

namespace stim {

/// Converts from the xz encoding
///
///     0b00: I
///     0b01: X
///     0b10: Z
///     0b11: Y
///
/// To the xyz encoding
///
///     0: I
///     1: X
///     2: Y
///     3: Z
inline uint8_t pauli_xz_to_xyz(bool x, bool z) {
    return (uint8_t)(x ^ z) | ((uint8_t)z << 1);
}

/// Converts from the xyz encoding
///
///     0: I
///     1: X
///     2: Y
///     3: Z
///
/// To the xz encoding
///
///     0b00: I
///     0b01: X
///     0b10: Z
///     0b11: Y
inline uint8_t pauli_xyz_to_xz(uint8_t xyz) {
    xyz ^= xyz >> 1;
    return xyz;
}

/// A Pauli string is a product of Pauli operations (I, X, Y, Z) to apply to various qubits.
///
/// In most cases, methods will take a PauliStringRef instead of a PauliString. This is because PauliStringRef can
/// have contents referring into densely packed table row data (or to a PauliString or to other sources). Basically,
/// PauliString is for the special somewhat-unusual case where you want to create data to back a PauliStringRef instead
/// of simply passing existing data along. It's a convenience class.
///
/// The template parameter, W, represents the SIMD width.
template <size_t W>
struct PauliString {
    /// The length of the Pauli string.
    size_t num_qubits;
    /// Whether or not the Pauli string is negated. True means -1, False means +1. Imaginary phase is not permitted.
    bool sign;
    /// The Paulis in the Pauli string, densely bit packed in a fashion enabling the use vectorized instructions.
    /// Paulis are xz-encoded (P=xz: I=00, X=10, Y=11, Z=01) pairwise across the two bit vectors.
    simd_bits<W> xs, zs;

    /// Standard constructors.
    explicit PauliString(size_t num_qubits);
    PauliString(const PauliStringRef<W> &other);  // NOLINT(google-explicit-constructor)
    PauliString(const PauliString<W> &other);
    PauliString(PauliString<W> &&other) noexcept;
    PauliString &operator=(const PauliStringRef<W> &other);
    PauliString &operator=(const PauliString<W> &other);
    PauliString &operator=(PauliString<W> &&other);

    /// Parse constructor.
    explicit PauliString(std::string_view text);
    /// Factory method for creating a PauliString whose Pauli entries are returned by a function.
    static PauliString<W> from_func(bool sign, size_t num_qubits, const std::function<char(size_t)> &func);
    /// Factory method for creating a PauliString by parsing a string (e.g. "-XIIYZ").
    static PauliString<W> from_str(std::string_view text);
    /// Factory method for creating a PauliString with uniformly random sign and Pauli entries.
    static PauliString<W> random(size_t num_qubits, std::mt19937_64 &rng);

    /// Equality.
    bool operator==(const PauliStringRef<W> &other) const;
    bool operator==(const PauliString<W> &other) const;
    /// Inequality.
    bool operator!=(const PauliStringRef<W> &other) const;
    bool operator!=(const PauliString<W> &other) const;
    bool operator<(const PauliStringRef<W> &other) const;
    bool operator<(const PauliString<W> &other) const;

    /// Implicit conversion to a reference.
    operator PauliStringRef<W>();
    /// Implicit conversion to a const reference.
    operator const PauliStringRef<W>() const;
    /// Explicit conversion to a reference.
    PauliStringRef<W> ref();
    /// Explicit conversion to a const reference.
    const PauliStringRef<W> ref() const;

    /// Returns a python-style slice of the Paulis in the Pauli string.
    PauliString<W> py_get_slice(int64_t start, int64_t step, int64_t slice_length) const;
    /// Returns a Pauli from the pauli string, allowing python-style negative indices, using IXYZ encoding.
    uint8_t py_get_item(int64_t index) const;

    /// Returns a string describing the given Pauli string, with one character per qubit.
    std::string str() const;

    /// Grows the pauli string to be at least as large as the given number
    /// of qubits.
    ///
    /// Requires:
    ///     resize_pad_factor >= 1
    ///
    /// Args:
    ///     min_num_qubits: A minimum number of qubits that will be needed.
    ///     resize_pad_factor: When resizing, memory will be overallocated
    ///          so that the pauli string can be expanded to at least this
    ///          many times the number of requested qubits. Use this to
    ///          avoid quadratic overheads from constant slight expansions.
    void ensure_num_qubits(size_t min_num_qubits, double resize_pad_factor);

    void mul_pauli_term(GateTarget t, bool *imag, bool right_mul);
    void left_mul_pauli(GateTarget t, bool *imag);
    void right_mul_pauli(GateTarget t, bool *imag);
};

/// Writes a string describing the given Pauli string to an output stream.
///
/// The template parameter, W, represents the SIMD width.
template <size_t W>
std::ostream &operator<<(std::ostream &out, const PauliString<W> &ps);

}  // namespace stim

#include "stim/stabilizers/pauli_string.inl"

#endif

```

```cpp
// ===== stim/stabilizers/pauli_string_ref.h (235 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_STABILIZERS_PAULI_STRING_REF_H
#define _STIM_STABILIZERS_PAULI_STRING_REF_H

#include <iostream>

#include "stim/mem/bit_ref.h"
#include "stim/mem/simd_bits_range_ref.h"
#include "stim/mem/span_ref.h"

namespace stim {

template <size_t W>
struct PauliString;
struct Circuit;
template <size_t W>
struct Tableau;
struct CircuitInstruction;

/// A Pauli string is a product of Pauli operations (I, X, Y, Z) to apply to various qubits.
///
/// A PauliStringRef is a Pauli string whose contents are backed by referenced memory, instead of memory owned by the
/// class instance. For example, the memory may be a row from the densely packed bits of a stabilizer tableau. This
/// avoids unnecessary copying, and allows for conveniently applying operations inplace on existing data.
///
/// The template parameter, W, represents the SIMD width.
template <size_t W>
struct PauliStringRef {
    /// The length of the Pauli string.
    size_t num_qubits;
    /// Whether or not the Pauli string is negated. True means -1, False means +1. Imaginary phase is not permitted.
    bit_ref sign;
    /// The Paulis in the Pauli string, densely bit packed in a fashion enabling the use of vectorized instructions.
    /// Paulis are xz-encoded (P=xz: I=00, X=10, Y=11, Z=01) pairwise across the two bit vectors.
    simd_bits_range_ref<W> xs, zs;

    /// Constructs a PauliStringRef pointing at the given sign, x, and z data.
    ///
    /// Requires:
    ///     xs.num_bits_padded() == zs.num_bits_padded()
    ///     xs.num_simd_words == ceil(num_qubits / W)
    PauliStringRef(size_t num_qubits, bit_ref sign, simd_bits_range_ref<W> xs, simd_bits_range_ref<W> zs);

    /// Equality.
    bool operator==(const PauliStringRef<W> &other) const;
    /// Inequality.
    bool operator!=(const PauliStringRef<W> &other) const;
    bool operator<(const PauliStringRef<W> &other) const;

    /// Overwrite assignment.
    PauliStringRef<W> &operator=(const PauliStringRef<W> &other);
    /// Swap assignment.
    void swap_with(PauliStringRef<W> other);

    /// Multiplies a commuting Pauli string into this one.
    ///
    /// If the two Pauli strings may anticommute, use `inplace_right_mul_returning_log_i_scalar` instead.
    ///
    /// ASSERTS:
    ///     The given Pauli strings have the same size.
    ///     The given Pauli strings commute.
    PauliStringRef<W> &operator*=(const PauliStringRef<W> &commuting_rhs);

    // A more general version  of `*this *= rhs` which works for anti-commuting Paulis.
    //
    // Instead of updating the sign of `*this`, the base i logarithm of a scalar factor that still needs to be included
    // into the result is returned. For example, when multiplying XZ to get iY, the left hand side would become `Y`
    // and the returned value would be `1` (meaning a factor of `i**1 = i` is missing from the `Y`).
    //
    // Returns:
    //     The logarithm, base i, of a scalar byproduct from the multiplication.
    //     0 if the scalar byproduct is 1.
    //     1 if the scalar byproduct is i.
    //     2 if the scalar byproduct is -1.
    //     3 if the scalar byproduct is -i.
    //
    // ASSERTS:
    //     The given Pauli strings have the same size.
    uint8_t inplace_right_mul_returning_log_i_scalar(const PauliStringRef<W> &rhs) noexcept;

    /// Overwrites the entire given Pauli string's contents with a subset of Paulis from this Pauli string.
    /// Does not affect the sign of the given Pauli string.
    ///
    /// Args:
    ///     out: The Pauli string to overwrite.
    ///     in_indices: For each qubit position in the output Pauli string, which qubit positions is read from in this
    ///         Pauli string.
    void gather_into(PauliStringRef<W> out, SpanRef<const size_t> in_indices) const;

    /// Overwrites part of the given Pauli string with the contents of this Pauli string.
    /// Also multiplies this Pauli string's sign into the given Pauli string's sign.
    ///
    /// Args:
    ///     out: The Pauli string to partially overwrite.
    ///     out_indices: For each qubit position in this Pauli string, which qubit position is overwritten in the output
    ///         Pauli string.
    void scatter_into(PauliStringRef<W> out, SpanRef<const size_t> out_indices) const;

    /// Determines if this Pauli string commutes with the given Pauli string.
    bool commutes(const PauliStringRef<W> &other) const noexcept;

    /// Returns a string describing the given Pauli string, with one character per qubit.
    std::string str() const;
    /// Returns a string describing the given Pauli string, indexing the Paulis so that identities can be omitted.
    std::string sparse_str() const;

    /// Applies the given tableau to the pauli string, at the given targets.
    ///
    /// Args:
    ///     tableau: The Clifford operation to apply.
    ///     targets: The qubits to target. Broadcasting is supported. The length of the span must be a multiple of the
    ///         tableau's size.
    ///     inverse: When true, applies the inverse of the tableau instead of the tableau.
    void do_tableau(const Tableau<W> &tableau, SpanRef<const size_t> targets, bool inverse);
    void do_circuit(const Circuit &circuit);
    void undo_circuit(const Circuit &circuit);
    void do_instruction(const CircuitInstruction &inst);
    void undo_instruction(const CircuitInstruction &inst);

    PauliString<W> after(const Circuit &circuit) const;
    PauliString<W> after(const Tableau<W> &tableau, SpanRef<const size_t> indices) const;
    PauliString<W> after(const CircuitInstruction &operation) const;
    PauliString<W> before(const Circuit &circuit) const;
    PauliString<W> before(const Tableau<W> &tableau, SpanRef<const size_t> indices) const;
    PauliString<W> before(const CircuitInstruction &operation) const;

    size_t weight() const;
    bool has_no_pauli_terms() const;
    bool intersects(PauliStringRef<W> other) const;

    template <typename CALLBACK>
    void for_each_active_pauli(CALLBACK callback) const {
        size_t n = xs.num_u64_padded();
        for (size_t w = 0; w < n; w++) {
            uint64_t v = xs.u64[w] | zs.u64[w];
            while (v) {
                size_t q = w * 64 + std::countr_zero(v);
                v &= v - 1;
                callback(q);
            }
        }
    }

   private:
    void check_avoids_MPP(const CircuitInstruction &inst);
    void check_avoids_reset(const CircuitInstruction &inst);
    void check_avoids_measurement(const CircuitInstruction &inst);
    void undo_reset_xyz(const CircuitInstruction &inst);

    void do_single_cx(const CircuitInstruction &inst, uint32_t c, uint32_t t);
    void do_single_cy(const CircuitInstruction &inst, uint32_t c, uint32_t t);
    void do_single_cz(const CircuitInstruction &inst, uint32_t c, uint32_t t);

    void do_H_XZ(const CircuitInstruction &inst);
    void do_H_YZ(const CircuitInstruction &inst);
    void do_H_XY(const CircuitInstruction &inst);
    void do_H_NXY(const CircuitInstruction &inst);
    void do_H_NXZ(const CircuitInstruction &inst);
    void do_H_NYZ(const CircuitInstruction &inst);
    void do_C_XYZ(const CircuitInstruction &inst);
    void do_C_NXYZ(const CircuitInstruction &inst);
    void do_C_XNYZ(const CircuitInstruction &inst);
    void do_C_XYNZ(const CircuitInstruction &inst);
    void do_C_ZYX(const CircuitInstruction &inst);
    void do_C_NZYX(const CircuitInstruction &inst);
    void do_C_ZNYX(const CircuitInstruction &inst);
    void do_C_ZYNX(const CircuitInstruction &inst);
    void do_SQRT_X(const CircuitInstruction &inst);
    void do_SQRT_Y(const CircuitInstruction &inst);
    void do_SQRT_Z(const CircuitInstruction &inst);
    void do_SQRT_X_DAG(const CircuitInstruction &inst);
    void do_SQRT_Y_DAG(const CircuitInstruction &inst);
    void do_SQRT_Z_DAG(const CircuitInstruction &inst);
    void do_SQRT_XX(const CircuitInstruction &inst);
    void do_SQRT_XX_DAG(const CircuitInstruction &inst);
    void do_SQRT_YY(const CircuitInstruction &inst);
    void do_SQRT_YY_DAG(const CircuitInstruction &inst);
    void do_SQRT_ZZ(const CircuitInstruction &inst);
    void do_SQRT_ZZ_DAG(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_ZCX(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_ZCY(const CircuitInstruction &inst);
    void do_ZCZ(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_SWAP(const CircuitInstruction &inst);
    void do_X(const CircuitInstruction &inst);
    void do_Y(const CircuitInstruction &inst);
    void do_Z(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_ISWAP(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_ISWAP_DAG(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_CXSWAP(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_CZSWAP(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_SWAPCX(const CircuitInstruction &inst);
    void do_XCX(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_XCY(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_XCZ(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_YCX(const CircuitInstruction &inst);
    void do_YCY(const CircuitInstruction &inst);
    template <bool reverse_order>
    void do_YCZ(const CircuitInstruction &inst);
};

/// Writes a string describing the given Pauli string to an output stream.
template <size_t W>
std::ostream &operator<<(std::ostream &out, const PauliStringRef<W> &ps);

}  // namespace stim

#include "stim/stabilizers/pauli_string_ref.inl"

#endif

```

```cpp
// ===== stim/stabilizers/tableau.h (251 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_STABILIZERS_TABLEAU_H
#define _STIM_STABILIZERS_TABLEAU_H

#include <complex>
#include <iostream>
#include <unordered_map>

#include "stim/mem/simd_bit_table.h"
#include "stim/mem/simd_util.h"
#include "stim/mem/span_ref.h"
#include "stim/stabilizers/pauli_string.h"

namespace stim {

template <size_t W>
struct TableauHalf {
    size_t num_qubits;
    simd_bit_table<W> xt;
    simd_bit_table<W> zt;
    simd_bits<W> signs;
    PauliStringRef<W> operator[](size_t input_qubit);
    const PauliStringRef<W> operator[](size_t input_qubit) const;
    TableauHalf(size_t num_qubits);
};

/// A Tableau is a stabilizer tableau representation of a Clifford operation.
/// It stores, for each X and Z observable for each qubit, what is produced when
/// conjugating that observable by the operation. In other words, it explains how
/// to transform "input side" Pauli products into "output side" Pauli products.
///
/// The memory layout used by this class is column major, meaning iterating over
/// the output observable is iterating along the grain of memory. This makes
/// prepending operations cheap. To append operations, use TableauTransposedRaii.
///
/// The template parameter, W, represents the SIMD width.
template <size_t W>
struct Tableau {
    size_t num_qubits;
    TableauHalf<W> xs;
    TableauHalf<W> zs;

    explicit Tableau(size_t num_qubits);
    bool operator==(const Tableau &other) const;
    bool operator!=(const Tableau &other) const;

    PauliString<W> eval_y_obs(size_t qubit) const;

    std::string str() const;

    /// Grows the size of the tableau (or leaves it the same) by adding
    /// new rows and columns with identity elements along the diagonal.
    ///
    /// Requires:
    ///     new_num_qubits >= this.num_qubits
    ///     resize_pad_factor >= 1
    ///
    /// Args:
    ///     new_num_qubits: The new number of qubits the tableau represents.
    ///     resize_pad_factor: When resizing, memory will be overallocated
    ///          so that the tableau can be expanded to at least this many
    ///          times the number of requested qubits. Use this to avoid
    ///          quadratic overheads from constant slight expansions.
    void expand(size_t new_num_qubits, double resize_pad_factor);

    /// Creates a Tableau representing the identity operation.
    static Tableau<W> identity(size_t num_qubits);
    /// Creates a Tableau from a PauliString via conjugation
    static Tableau<W> from_pauli_string(const PauliString<W> &pauli_string);
    /// Creates a Tableau representing a randomly sampled Clifford operation from a uniform distribution.
    static Tableau<W> random(size_t num_qubits, std::mt19937_64 &rng);
    /// Returns the inverse Tableau.
    ///
    /// Args:
    ///     skip_signs: Instead of computing the signs, just set them all to positive.
    Tableau<W> inverse(bool skip_signs = false) const;
    /// Returns the Tableau raised to an integer power (using repeated squaring).
    Tableau<W> raised_to(int64_t exponent) const;

    std::vector<std::complex<float>> to_flat_unitary_matrix(bool little_endian) const;
    bool satisfies_invariants() const;

    /// If a Tableau fixes each pauli upto sign, then it is conjugation by a pauli
    bool is_pauli_product() const;

    /// If tableau is conjugation by a pauli, then return that pauli. Else throw exception.
    PauliString<W> to_pauli_string() const;

    /// Creates a Tableau representing a single qubit gate.
    ///
    /// All observables specified using the string format accepted by `PauliString::from_str`.
    /// For example: "-X" or "+Y".
    ///
    /// Args:
    ///    x: The output-side observable that the input-side X observable gets mapped to.
    ///    z: The output-side observable that the input-side Y observable gets mapped to.
    static Tableau<W> gate1(const char *x, const char *z);

    /// Creates a Tableau representing a two qubit gate.
    ///
    /// All observables specified using the string format accepted by `PauliString::from_str`.
    /// For example: "-IX" or "+YZ".
    ///
    /// Args:
    ///    x1: The output-side observable that the input-side XI observable gets mapped to.
    ///    z1: The output-side observable that the input-side YI observable gets mapped to.
    ///    x2: The output-side observable that the input-side IX observable gets mapped to.
    ///    z2: The output-side observable that the input-side IY observable gets mapped to.
    static Tableau<W> gate2(const char *x1, const char *z1, const char *x2, const char *z2);

    /// Returns the result of applying the tableau to the given Pauli string.
    ///
    /// Args:
    ///     p: The input-side Pauli string.
    ///
    /// Returns:
    ///     The output-side Pauli string.
    ///     Algebraically: $c p c^{-1}$ where $c$ is the tableau's Clifford operation.
    PauliString<W> operator()(const PauliStringRef<W> &p) const;

    /// Returns the result of applying the tableau to `gathered_input.scatter(scattered_indices)`.
    PauliString<W> scatter_eval(
        const PauliStringRef<W> &gathered_input, const std::vector<size_t> &scattered_indices) const;

    /// Returns a tableau equivalent to the composition of two tableaus of the same size.
    Tableau<W> then(const Tableau<W> &second) const;

    /// Applies the Tableau inplace to a subset of a Pauli string.
    void apply_within(PauliStringRef<W> &target, SpanRef<const size_t> target_qubits) const;

    /// Appends a smaller operation into this tableau's operation.
    ///
    /// The new value T' of this tableau will equal the composition T o P = PT where T is the old
    /// value of this tableau and P is the operation to append.
    ///
    /// Args:
    ///     operation: The smaller operation to append into this tableau.
    ///     target_qubits: The qubits being acted on by `operation`.
    void inplace_scatter_append(const Tableau<W> &operation, const std::vector<size_t> &target_qubits);

    /// Prepends a smaller operation into this tableau's operation.
    ///
    /// The new value T' of this tableau will equal the composition P o T = TP where T is the old
    /// value of this tableau and P is the operation to append.
    ///
    /// Args:
    ///     operation: The smaller operation to prepend into this tableau.
    ///     target_qubits: The qubits being acted on by `operation`.
    void inplace_scatter_prepend(const Tableau<W> &operation, const std::vector<size_t> &target_qubits);

    /// Applies a transpose to the X2X, X2Z, Z2X, and Z2Z bit tables within the tableau.
    void do_transpose_quadrants();

    /// Returns the direct sum of two tableaus.
    Tableau<W> operator+(const Tableau<W> &second) const;
    /// Appends the other tableau onto this one, resulting in the direct sum.
    Tableau<W> &operator+=(const Tableau<W> &second);

    /// === Specialized vectorized methods for prepending operations onto the tableau === ///
    void prepend_SWAP(size_t q1, size_t q2);
    void prepend_X(size_t q);
    void prepend_Y(size_t q);
    void prepend_Z(size_t q);
    void prepend_H_XZ(size_t q);
    void prepend_H_YZ(size_t q);
    void prepend_H_XY(size_t q);
    void prepend_H_NXY(size_t q);
    void prepend_H_NXZ(size_t q);
    void prepend_H_NYZ(size_t q);
    void prepend_C_XYZ(size_t q);
    void prepend_C_NXYZ(size_t q);
    void prepend_C_XNYZ(size_t q);
    void prepend_C_XYNZ(size_t q);
    void prepend_C_ZYX(size_t q);
    void prepend_C_NZYX(size_t q);
    void prepend_C_ZNYX(size_t q);
    void prepend_C_ZYNX(size_t q);
    void prepend_SQRT_X(size_t q);
    void prepend_SQRT_X_DAG(size_t q);
    void prepend_SQRT_Y(size_t q);
    void prepend_SQRT_Y_DAG(size_t q);
    void prepend_SQRT_Z(size_t q);
    void prepend_SQRT_Z_DAG(size_t q);
    void prepend_SQRT_XX(size_t q1, size_t q2);
    void prepend_SQRT_XX_DAG(size_t q1, size_t q2);
    void prepend_SQRT_YY(size_t q1, size_t q2);
    void prepend_SQRT_YY_DAG(size_t q1, size_t q2);
    void prepend_SQRT_ZZ(size_t q1, size_t q2);
    void prepend_SQRT_ZZ_DAG(size_t q1, size_t q2);
    void prepend_ZCX(size_t control, size_t target);
    void prepend_ZCY(size_t control, size_t target);
    void prepend_ZCZ(size_t control, size_t target);
    void prepend_ISWAP(size_t q1, size_t q2);
    void prepend_ISWAP_DAG(size_t q1, size_t q2);
    void prepend_XCX(size_t control, size_t target);
    void prepend_XCY(size_t control, size_t target);
    void prepend_XCZ(size_t control, size_t target);
    void prepend_YCX(size_t control, size_t target);
    void prepend_YCY(size_t control, size_t target);
    void prepend_YCZ(size_t control, size_t target);
    void prepend_pauli_product(const PauliStringRef<W> &op);

    /// Builds the Y output by using Y = iXZ.
    PauliString<W> y_output(size_t input_index) const;

    /// Constant-time version of tableau.xs[input_index][output_index].
    uint8_t x_output_pauli_xyz(size_t input_index, size_t output_index) const;
    /// Constant-time version of tableau.y_output(input_index)[output_index].
    uint8_t y_output_pauli_xyz(size_t input_index, size_t output_index) const;
    /// Constant-time version of tableau.zs[input_index][output_index].
    uint8_t z_output_pauli_xyz(size_t input_index, size_t output_index) const;
    /// Constant-time version of tableau.inverse().xs[input_index][output_index].
    uint8_t inverse_x_output_pauli_xyz(size_t input_index, size_t output_index) const;
    /// Constant-time version of tableau.inverse().y_output(input_index)[output_index].
    uint8_t inverse_y_output_pauli_xyz(size_t input_index, size_t output_index) const;
    /// Constant-time version of tableau.inverse().zs[input_index][output_index].
    uint8_t inverse_z_output_pauli_xyz(size_t input_index, size_t output_index) const;
    /// Faster version of tableau.inverse().xs[input_index].
    PauliString<W> inverse_x_output(size_t input_index, bool skip_sign = false) const;
    /// Faster version of tableau.inverse().y_output(input_index).
    PauliString<W> inverse_y_output(size_t input_index, bool skip_sign = false) const;
    /// Faster version of tableau.inverse().zs[input_index].
    PauliString<W> inverse_z_output(size_t input_index, bool skip_sign = false) const;

    std::vector<PauliString<W>> stabilizers(bool canonical) const;
};

template <size_t W>
std::ostream &operator<<(std::ostream &out, const Tableau<W> &ps);

}  // namespace stim

#include "stim/stabilizers/tableau.inl"
#include "stim/stabilizers/tableau_specialized_prepend.inl"

#endif

```

```cpp
// ===== stim/stabilizers/tableau.inl (786 lines) =====
// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <random>

#include "stim/gates/gates.h"
#include "stim/simulators/vector_simulator.h"
#include "stim/stabilizers/pauli_string.h"
#include "stim/stabilizers/tableau.h"

namespace stim {

template <size_t W>
void Tableau<W>::expand(size_t new_num_qubits, double resize_pad_factor) {
    // If the new qubits fit inside the padding, just extend into it.
    assert(new_num_qubits >= num_qubits);
    assert(resize_pad_factor >= 1);
    if (new_num_qubits <= xs.xt.num_major_bits_padded()) {
        size_t old_num_qubits = num_qubits;
        num_qubits = new_num_qubits;
        xs.num_qubits = new_num_qubits;
        zs.num_qubits = new_num_qubits;
        // Initialize identity elements along the diagonal.
        for (size_t k = old_num_qubits; k < new_num_qubits; k++) {
            xs[k].xs[k] = true;
            zs[k].zs[k] = true;
        }
        return;
    }

    // Move state to temporary storage then re-allocate to make room for additional qubits.
    size_t old_num_simd_words = xs.xt.num_simd_words_major;
    size_t old_num_qubits = num_qubits;
    Tableau<W> old_state = std::move(*this);
    *this = Tableau<W>((size_t)(new_num_qubits * resize_pad_factor));
    this->num_qubits = new_num_qubits;
    this->xs.num_qubits = new_num_qubits;
    this->zs.num_qubits = new_num_qubits;

    // Copy stored state back into new larger space.
    auto partial_copy = [=](simd_bits_range_ref<W> dst, simd_bits_range_ref<W> src) {
        dst.word_range_ref(0, old_num_simd_words) = src;
    };
    partial_copy(xs.signs, old_state.xs.signs);
    partial_copy(zs.signs, old_state.zs.signs);
    for (size_t k = 0; k < old_num_qubits; k++) {
        partial_copy(xs[k].xs, old_state.xs[k].xs);
        partial_copy(xs[k].zs, old_state.xs[k].zs);
        partial_copy(zs[k].xs, old_state.zs[k].xs);
        partial_copy(zs[k].zs, old_state.zs[k].zs);
    }
}

template <size_t W>
PauliStringRef<W> TableauHalf<W>::operator[](size_t input_qubit) {
    size_t nw = (num_qubits + W - 1) / W;
    return PauliStringRef<W>(
        num_qubits, signs[input_qubit], xt[input_qubit].word_range_ref(0, nw), zt[input_qubit].word_range_ref(0, nw));
}

template <size_t W>
const PauliStringRef<W> TableauHalf<W>::operator[](size_t input_qubit) const {
    size_t nw = (num_qubits + W - 1) / W;
    return PauliStringRef<W>(
        num_qubits, signs[input_qubit], xt[input_qubit].word_range_ref(0, nw), zt[input_qubit].word_range_ref(0, nw));
}

template <size_t W>
PauliString<W> Tableau<W>::eval_y_obs(size_t qubit) const {
    PauliString<W> result(xs[qubit]);
    uint8_t log_i = result.ref().inplace_right_mul_returning_log_i_scalar(zs[qubit]);
    log_i++;
    assert((log_i & 1) == 0);
    if (log_i & 2) {
        result.sign ^= true;
    }
    return result;
}

template <size_t W>
Tableau<W>::Tableau(size_t num_qubits) : num_qubits(num_qubits), xs(num_qubits), zs(num_qubits) {
    for (size_t q = 0; q < num_qubits; q++) {
        xs.xt[q][q] = true;
        zs.zt[q][q] = true;
    }
}

template <size_t W>
TableauHalf<W>::TableauHalf(size_t num_qubits)
    : num_qubits(num_qubits), xt(num_qubits, num_qubits), zt(num_qubits, num_qubits), signs(num_qubits) {
}

template <size_t W>
Tableau<W> Tableau<W>::identity(size_t num_qubits) {
    return Tableau<W>(num_qubits);
}

template <size_t W>
Tableau<W> Tableau<W>::from_pauli_string(const PauliString<W> &pauli_string) {
    Tableau<W> tableau = identity(pauli_string.num_qubits);
    tableau.xs.signs = pauli_string.zs;
    tableau.zs.signs = pauli_string.xs;
    return tableau;
}

template <size_t W>
Tableau<W> Tableau<W>::gate1(const char *x, const char *z) {
    Tableau<W> result(1);
    result.xs[0] = PauliString<W>::from_str(x);
    result.zs[0] = PauliString<W>::from_str(z);
    assert((bool)result.zs[0].sign == (z[0] == '-'));
    return result;
}

template <size_t W>
Tableau<W> Tableau<W>::gate2(const char *x1, const char *z1, const char *x2, const char *z2) {
    Tableau<W> result(2);
    result.xs[0] = PauliString<W>::from_str(x1);
    result.zs[0] = PauliString<W>::from_str(z1);
    result.xs[1] = PauliString<W>::from_str(x2);
    result.zs[1] = PauliString<W>::from_str(z2);
    return result;
}

template <size_t W>
std::ostream &operator<<(std::ostream &out, const Tableau<W> &t) {
    out << "+-";
    for (size_t k = 0; k < t.num_qubits; k++) {
        out << 'x';
        out << 'z';
        out << '-';
    }
    out << "\n|";
    for (size_t k = 0; k < t.num_qubits; k++) {
        out << ' ';
        out << "+-"[t.xs[k].sign];
        out << "+-"[t.zs[k].sign];
    }
    for (size_t q = 0; q < t.num_qubits; q++) {
        out << "\n|";
        for (size_t k = 0; k < t.num_qubits; k++) {
            out << ' ';
            auto x = t.xs[k];
            auto z = t.zs[k];
            out << "_XZY"[x.xs[q] + 2 * x.zs[q]];
            out << "_XZY"[z.xs[q] + 2 * z.zs[q]];
        }
    }
    return out;
}

template <size_t W>
std::string Tableau<W>::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

template <size_t W>
void Tableau<W>::inplace_scatter_append(const Tableau<W> &operation, const std::vector<size_t> &target_qubits) {
    assert(operation.num_qubits == target_qubits.size());
    if (&operation == this) {
        Tableau<W> independent_copy(operation);
        inplace_scatter_append(independent_copy, target_qubits);
        return;
    }
    for (size_t q = 0; q < num_qubits; q++) {
        auto x = xs[q];
        auto z = zs[q];
        operation.apply_within(x, target_qubits);
        operation.apply_within(z, target_qubits);
    }
}

template <size_t W>
bool truncated_bits_equals(size_t nw, const simd_bits_range_ref<W> &t1, const simd_bits_range_ref<W> &t2) {
    return t1.word_range_ref(0, nw) == t2.word_range_ref(0, nw);
}

template <size_t W>
bool truncated_tableau_equals(size_t n, const simd_bit_table<W> &t1, const simd_bit_table<W> &t2) {
    size_t nw = (n + W - 1) / W;
    for (size_t k = 0; k < n; k++) {
        if (!truncated_bits_equals(nw, t1[k], t2[k])) {
            return false;
        }
    }
    return true;
}

template <size_t W>
bool Tableau<W>::operator==(const Tableau<W> &other) const {
    size_t nw = (num_qubits + W - 1) / W;
    return num_qubits == other.num_qubits && truncated_tableau_equals(num_qubits, xs.xt, other.xs.xt) &&
           truncated_tableau_equals(num_qubits, xs.zt, other.xs.zt) &&
           truncated_tableau_equals(num_qubits, zs.xt, other.zs.xt) &&
           truncated_tableau_equals(num_qubits, zs.zt, other.zs.zt) &&
           xs.signs.word_range_ref(0, nw) == other.xs.signs.word_range_ref(0, nw) &&
           zs.signs.word_range_ref(0, nw) == other.zs.signs.word_range_ref(0, nw);
}

template <size_t W>
bool Tableau<W>::operator!=(const Tableau<W> &other) const {
    return !(*this == other);
}

template <size_t W>
void Tableau<W>::inplace_scatter_prepend(const Tableau<W> &operation, const std::vector<size_t> &target_qubits) {
    assert(operation.num_qubits == target_qubits.size());
    if (&operation == this) {
        Tableau<W> independent_copy(operation);
        inplace_scatter_prepend(independent_copy, target_qubits);
        return;
    }

    std::vector<PauliString<W>> new_x;
    std::vector<PauliString<W>> new_z;
    new_x.reserve(operation.num_qubits);
    new_z.reserve(operation.num_qubits);
    for (size_t q = 0; q < operation.num_qubits; q++) {
        new_x.push_back(scatter_eval(operation.xs[q], target_qubits));
        new_z.push_back(scatter_eval(operation.zs[q], target_qubits));
    }
    for (size_t q = 0; q < operation.num_qubits; q++) {
        xs[target_qubits[q]] = new_x[q];
        zs[target_qubits[q]] = new_z[q];
    }
}

template <size_t W>
PauliString<W> Tableau<W>::scatter_eval(
    const PauliStringRef<W> &gathered_input, const std::vector<size_t> &scattered_indices) const {
    assert(gathered_input.num_qubits == scattered_indices.size());
    auto result = PauliString<W>(num_qubits);
    result.sign = gathered_input.sign;
    for (size_t k_gathered = 0; k_gathered < gathered_input.num_qubits; k_gathered++) {
        size_t k_scattered = scattered_indices[k_gathered];
        bool x = gathered_input.xs[k_gathered];
        bool z = gathered_input.zs[k_gathered];
        if (x) {
            if (z) {
                // Multiply by Y using Y = i*X*Z.
                uint8_t log_i = 1;
                log_i += result.ref().inplace_right_mul_returning_log_i_scalar(xs[k_scattered]);
                log_i += result.ref().inplace_right_mul_returning_log_i_scalar(zs[k_scattered]);
                assert((log_i & 1) == 0);
                result.sign ^= (log_i & 2) != 0;
            } else {
                result.ref() *= xs[k_scattered];
            }
        } else if (z) {
            result.ref() *= zs[k_scattered];
        }
    }
    return result;
}

template <size_t W>
PauliString<W> Tableau<W>::operator()(const PauliStringRef<W> &p) const {
    if (p.num_qubits != num_qubits) {
        throw std::out_of_range("pauli_string.num_qubits != tableau.num_qubits");
    }
    std::vector<size_t> indices;
    for (size_t k = 0; k < p.num_qubits; k++) {
        indices.push_back(k);
    }
    return scatter_eval(p, indices);
}

template <size_t W>
void Tableau<W>::apply_within(PauliStringRef<W> &target, SpanRef<const size_t> target_qubits) const {
    assert(num_qubits == target_qubits.size());
    auto inp = PauliString<W>(num_qubits);
    target.gather_into(inp, target_qubits);
    auto out = (*this)(inp);
    out.ref().scatter_into(target, target_qubits);
}

/// Samples a vector of bits and a permutation from a skewed distribution.
///
/// Reference:
///     "Hadamard-free circuits expose the structure of the Clifford group"
///     Sergey Bravyi, Dmitri Maslov
///     https://arxiv.org/abs/2003.09412
inline std::pair<std::vector<bool>, std::vector<size_t>> sample_qmallows(size_t n, std::mt19937_64 &gen) {
    auto uni = std::uniform_real_distribution<double>(0, 1);

    std::vector<bool> hada;
    std::vector<size_t> permutation;
    std::vector<size_t> remaining_indices;
    for (size_t k = 0; k < n; k++) {
        remaining_indices.push_back(k);
    }
    for (size_t i = 0; i < n; i++) {
        auto m = remaining_indices.size();
        auto u = uni(gen);
        auto eps = pow(4, -(int)m);
        auto k = (size_t)-ceil(log2(u + (1 - u) * eps));
        hada.push_back(k < m);
        if (k >= m) {
            k = 2 * m - k - 1;
        }
        permutation.push_back(remaining_indices[k]);
        remaining_indices.erase(remaining_indices.begin() + k);
    }
    return {hada, permutation};
}

/// Samples a random valid stabilizer tableau.
///
/// Reference:
///     "Hadamard-free circuits expose the structure of the Clifford group"
///     Sergey Bravyi, Dmitri Maslov
///     https://arxiv.org/abs/2003.09412
template <size_t W>
simd_bit_table<W> random_stabilizer_tableau_raw(size_t n, std::mt19937_64 &rng) {
    auto hs_pair = sample_qmallows(n, rng);
    const auto &hada = hs_pair.first;
    const auto &perm = hs_pair.second;

    simd_bit_table<W> symmetric(n, n);
    for (size_t row = 0; row < n; row++) {
        symmetric[row].randomize(row + 1, rng);
        for (size_t col = 0; col < row; col++) {
            symmetric[col][row] = symmetric[row][col];
        }
    }

    simd_bit_table<W> symmetric_m(n, n);
    for (size_t row = 0; row < n; row++) {
        symmetric_m[row].randomize(row + 1, rng);
        symmetric_m[row][row] &= hada[row];
        for (size_t col = 0; col < row; col++) {
            bool b = hada[row] && hada[col];
            b |= hada[row] > hada[col] && perm[row] < perm[col];
            b |= hada[row] < hada[col] && perm[row] > perm[col];
            symmetric_m[row][col] &= b;
            symmetric_m[col][row] = symmetric_m[row][col];
        }
    }

    auto lower = simd_bit_table<W>::identity(n);
    for (size_t row = 0; row < n; row++) {
        lower[row].randomize(row, rng);
    }

    auto lower_m = simd_bit_table<W>::identity(n);
    for (size_t row = 0; row < n; row++) {
        lower_m[row].randomize(row, rng);
        for (size_t col = 0; col < row; col++) {
            bool b = hada[row] < hada[col];
            b |= hada[row] && hada[col] && perm[row] > perm[col];
            b |= !hada[row] && !hada[col] && perm[row] < perm[col];
            lower_m[row][col] &= b;
        }
    }

    auto prod = symmetric.square_mat_mul(lower, n);
    auto prod_m = symmetric_m.square_mat_mul(lower_m, n);

    auto inv = lower.inverse_assuming_lower_triangular(n);
    auto inv_m = lower_m.inverse_assuming_lower_triangular(n);
    inv.do_square_transpose();
    inv_m.do_square_transpose();

    auto fused = simd_bit_table<W>::from_quadrants(n, lower, simd_bit_table<W>(n, n), prod, inv);
    auto fused_m = simd_bit_table<W>::from_quadrants(n, lower_m, simd_bit_table<W>(n, n), prod_m, inv_m);

    simd_bit_table<W> u(2 * n, 2 * n);

    // Apply permutation.
    for (size_t row = 0; row < n; row++) {
        u[row] = fused[perm[row]];
        u[row + n] = fused[perm[row] + n];
    }
    // Apply Hadamards.
    for (size_t row = 0; row < n; row++) {
        if (hada[row]) {
            u[row].swap_with(u[row + n]);
        }
    }

    return fused_m.square_mat_mul(u, 2 * n);
}

template <size_t W>
Tableau<W> Tableau<W>::random(size_t num_qubits, std::mt19937_64 &rng) {
    auto raw = random_stabilizer_tableau_raw<W>(num_qubits, rng);
    Tableau result(num_qubits);
    for (size_t row = 0; row < num_qubits; row++) {
        for (size_t col = 0; col < num_qubits; col++) {
            result.xs[row].xs[col] = raw[row][col];
            result.xs[row].zs[col] = raw[row][col + num_qubits];
            result.zs[row].xs[col] = raw[row + num_qubits][col];
            result.zs[row].zs[col] = raw[row + num_qubits][col + num_qubits];
        }
    }
    result.xs.signs.randomize(num_qubits, rng);
    result.zs.signs.randomize(num_qubits, rng);
    return result;
}

template <size_t W>
bool Tableau<W>::satisfies_invariants() const {
    for (size_t q1 = 0; q1 < num_qubits; q1++) {
        auto x1 = xs[q1];
        auto z1 = zs[q1];
        if (x1.commutes(z1)) {
            return false;
        }
        for (size_t q2 = q1 + 1; q2 < num_qubits; q2++) {
            auto x2 = xs[q2];
            auto z2 = zs[q2];
            if (!x1.commutes(x2) || !x1.commutes(z2) || !z1.commutes(x2) || !z1.commutes(z2)) {
                return false;
            }
        }
    }
    return true;
}

template <size_t W>
bool Tableau<W>::is_pauli_product() const {
    size_t pop_count = xs.xt.data.popcnt() + xs.zt.data.popcnt() + zs.xt.data.popcnt() + zs.zt.data.popcnt();

    if (pop_count != 2 * num_qubits) {
        return false;
    }

    for (size_t q = 0; q < num_qubits; q++) {
        if (xs.xt[q][q] == false)
            return false;
    }

    for (size_t q = 0; q < num_qubits; q++) {
        if (zs.zt[q][q] == false)
            return false;
    }

    return true;
}

template <size_t W>
PauliString<W> Tableau<W>::to_pauli_string() const {
    if (!is_pauli_product()) {
        throw std::invalid_argument("The Tableau isn't equivalent to a Pauli product.");
    }

    PauliString<W> pauli_string(num_qubits);
    pauli_string.xs = zs.signs;
    pauli_string.zs = xs.signs;
    return pauli_string;
}

template <size_t W>
Tableau<W> Tableau<W>::inverse(bool skip_signs) const {
    Tableau<W> result(xs.xt.num_major_bits_padded());
    result.num_qubits = num_qubits;
    result.xs.num_qubits = num_qubits;
    result.zs.num_qubits = num_qubits;

    // Transpose data with xx zz swap tweak.
    result.xs.xt.data = zs.zt.data;
    result.xs.zt.data = xs.zt.data;
    result.zs.xt.data = zs.xt.data;
    result.zs.zt.data = xs.xt.data;
    result.do_transpose_quadrants();

    // Fix signs by checking for consistent round trips.
    if (!skip_signs) {
        PauliString<W> singleton(num_qubits);
        for (size_t k = 0; k < num_qubits; k++) {
            singleton.xs[k] = true;
            bool x_round_trip_sign = (*this)(result(singleton)).sign;
            singleton.xs[k] = false;
            singleton.zs[k] = true;
            bool z_round_trip_sign = (*this)(result(singleton)).sign;
            singleton.zs[k] = false;

            result.xs[k].sign ^= x_round_trip_sign;
            result.zs[k].sign ^= z_round_trip_sign;
        }
    }

    return result;
}

template <size_t W>
void Tableau<W>::do_transpose_quadrants() {
    xs.xt.do_square_transpose();
    xs.zt.do_square_transpose();
    zs.xt.do_square_transpose();
    zs.zt.do_square_transpose();
}

template <size_t W>
Tableau<W> Tableau<W>::then(const Tableau<W> &second) const {
    assert(num_qubits == second.num_qubits);
    Tableau<W> result(num_qubits);
    for (size_t q = 0; q < num_qubits; q++) {
        result.xs[q] = second(xs[q]);
        result.zs[q] = second(zs[q]);
    }
    return result;
}

template <size_t W>
Tableau<W> Tableau<W>::raised_to(int64_t exponent) const {
    Tableau<W> result(num_qubits);
    if (exponent) {
        Tableau<W> square = *this;

        if (exponent < 0) {
            square = square.inverse();
            exponent *= -1;
        }

        while (true) {
            if (exponent & 1) {
                result = result.then(square);
            }
            exponent >>= 1;
            if (exponent == 0) {
                break;
            }
            square = square.then(square);
        }
    }
    return result;
}

template <size_t W>
Tableau<W> Tableau<W>::operator+(const Tableau<W> &second) const {
    Tableau<W> copy = *this;
    copy += second;
    return copy;
}

template <size_t W>
Tableau<W> &Tableau<W>::operator+=(const Tableau<W> &second) {
    size_t n = num_qubits;
    expand(n + second.num_qubits, 1.1);
    for (size_t i = 0; i < second.num_qubits; i++) {
        xs.signs[n + i] = second.xs.signs[i];
        zs.signs[n + i] = second.zs.signs[i];
        for (size_t j = 0; j < second.num_qubits; j++) {
            xs.xt[n + i][n + j] = second.xs.xt[i][j];
            xs.zt[n + i][n + j] = second.xs.zt[i][j];
            zs.xt[n + i][n + j] = second.zs.xt[i][j];
            zs.zt[n + i][n + j] = second.zs.zt[i][j];
        }
    }
    return *this;
}

template <size_t W>
uint8_t Tableau<W>::x_output_pauli_xyz(size_t input_index, size_t output_index) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    if (output_index >= num_qubits) {
        throw std::invalid_argument("output_index >= len(tableau)");
    }
    PauliStringRef<W> x = xs[input_index];
    return pauli_xz_to_xyz(x.xs[output_index], x.zs[output_index]);
}

template <size_t W>
uint8_t Tableau<W>::y_output_pauli_xyz(size_t input_index, size_t output_index) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    if (output_index >= num_qubits) {
        throw std::invalid_argument("output_index >= len(tableau)");
    }
    PauliStringRef<W> x = xs[input_index];
    PauliStringRef<W> z = zs[input_index];
    return pauli_xz_to_xyz(x.xs[output_index] ^ z.xs[output_index], x.zs[output_index] ^ z.zs[output_index]);
}

template <size_t W>
uint8_t Tableau<W>::z_output_pauli_xyz(size_t input_index, size_t output_index) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    if (output_index >= num_qubits) {
        throw std::invalid_argument("output_index >= len(tableau)");
    }
    PauliStringRef<W> z = zs[input_index];
    return pauli_xz_to_xyz(z.xs[output_index], z.zs[output_index]);
}

template <size_t W>
uint8_t Tableau<W>::inverse_x_output_pauli_xyz(size_t input_index, size_t output_index) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    if (output_index >= num_qubits) {
        throw std::invalid_argument("output_index >= len(tableau)");
    }
    return pauli_xz_to_xyz(zs[output_index].zs[input_index], xs[output_index].zs[input_index]);
}

template <size_t W>
uint8_t Tableau<W>::inverse_y_output_pauli_xyz(size_t input_index, size_t output_index) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    if (output_index >= num_qubits) {
        throw std::invalid_argument("output_index >= len(tableau)");
    }
    PauliStringRef<W> x = xs[output_index];
    PauliStringRef<W> z = zs[output_index];
    return pauli_xz_to_xyz(z.zs[input_index] ^ z.xs[input_index], x.zs[input_index] ^ x.xs[input_index]);
}

template <size_t W>
uint8_t Tableau<W>::inverse_z_output_pauli_xyz(size_t input_index, size_t output_index) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    if (output_index >= num_qubits) {
        throw std::invalid_argument("output_index >= len(tableau)");
    }
    return pauli_xz_to_xyz(zs[output_index].xs[input_index], xs[output_index].xs[input_index]);
}

template <size_t W>
PauliString<W> Tableau<W>::inverse_x_output(size_t input_index, bool skip_sign) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    PauliString<W> result(num_qubits);
    for (size_t k = 0; k < num_qubits; k++) {
        result.xs[k] = zs[k].zs[input_index];
        result.zs[k] = xs[k].zs[input_index];
    }
    if (!skip_sign) {
        result.sign = (*this)(result).sign;
    }
    return result;
}

template <size_t W>
PauliString<W> Tableau<W>::inverse_y_output(size_t input_index, bool skip_sign) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    PauliString<W> result(num_qubits);
    for (size_t k = 0; k < num_qubits; k++) {
        result.xs[k] = zs[k].zs[input_index] ^ zs[k].xs[input_index];
        result.zs[k] = xs[k].zs[input_index] ^ xs[k].xs[input_index];
    }
    if (!skip_sign) {
        result.sign = (*this)(result).sign;
    }
    return result;
}

template <size_t W>
PauliString<W> Tableau<W>::inverse_z_output(size_t input_index, bool skip_sign) const {
    if (input_index >= num_qubits) {
        throw std::invalid_argument("input_index >= len(tableau)");
    }
    PauliString<W> result(num_qubits);
    for (size_t k = 0; k < num_qubits; k++) {
        result.xs[k] = zs[k].xs[input_index];
        result.zs[k] = xs[k].xs[input_index];
    }
    if (!skip_sign) {
        result.sign = (*this)(result).sign;
    }
    return result;
}

template <size_t W>
std::vector<std::complex<float>> Tableau<W>::to_flat_unitary_matrix(bool little_endian) const {
    std::vector<PauliString<W>> pauli_strings;
    size_t nw = xs[0].xs.num_simd_words;

    // Add X transformation stabilizers.
    for (size_t k = 0; k < num_qubits; k++) {
        PauliString<W> p(num_qubits * 2);
        p.xs.word_range_ref(0, nw) = xs[k].xs;
        p.zs.word_range_ref(0, nw) = xs[k].zs;
        p.sign = xs[k].sign;
        p.xs[num_qubits + k] ^= true;
        pauli_strings.push_back(p);
    }

    // Add Z transformation stabilizers.
    for (size_t k = 0; k < num_qubits; k++) {
        PauliString<W> p(num_qubits * 2);
        p.xs.word_range_ref(0, nw) = zs[k].xs;
        p.zs.word_range_ref(0, nw) = zs[k].zs;
        p.sign = zs[k].sign;
        p.zs[num_qubits + k] ^= true;
        pauli_strings.push_back(p);
    }

    for (auto &p : pauli_strings) {
        for (size_t q = 0; q < num_qubits - q - 1; q++) {
            size_t q2 = num_qubits - q - 1;
            if (!little_endian) {
                p.xs[q].swap_with(p.xs[q2]);
                p.zs[q].swap_with(p.zs[q2]);
                p.xs[q + num_qubits].swap_with(p.xs[q2 + num_qubits]);
                p.zs[q + num_qubits].swap_with(p.zs[q2 + num_qubits]);
            }
        }
        for (size_t q = 0; q < num_qubits; q++) {
            p.xs[q].swap_with(p.xs[q + num_qubits]);
            p.zs[q].swap_with(p.zs[q + num_qubits]);
        }
    }

    // Turn it into a vector.
    std::vector<PauliStringRef<W>> refs;
    for (const auto &e : pauli_strings) {
        refs.push_back(e.ref());
    }

    return VectorSimulator::state_vector_from_stabilizers<W>(refs, 1 << num_qubits);
}

template <size_t W>
PauliString<W> Tableau<W>::y_output(size_t input_index) const {
    uint8_t log_i = 1;
    PauliString<W> result = xs[input_index];
    log_i += result.ref().inplace_right_mul_returning_log_i_scalar(zs[input_index]);
    assert((log_i & 1) == 0);
    result.sign ^= (log_i & 2) != 0;
    return result;
}

template <size_t W>
std::vector<PauliString<W>> Tableau<W>::stabilizers(bool canonical) const {
    std::vector<PauliString<W>> stabilizers;
    for (size_t k = 0; k < num_qubits; k++) {
        stabilizers.push_back(zs[k]);
    }

    if (canonical) {
        size_t min_pivot = 0;
        for (size_t q = 0; q < num_qubits; q++) {
            for (size_t b = 0; b < 2; b++) {
                size_t pivot = min_pivot;
                while (pivot < num_qubits && !(b ? stabilizers[pivot].zs : stabilizers[pivot].xs)[q]) {
                    pivot++;
                }
                if (pivot == num_qubits) {
                    continue;
                }
                for (size_t s = 0; s < num_qubits; s++) {
                    if (s != pivot && (b ? stabilizers[s].zs : stabilizers[s].xs)[q]) {
                        stabilizers[s].ref() *= stabilizers[pivot];
                    }
                }
                if (min_pivot != pivot) {
                    std::swap(stabilizers[min_pivot], stabilizers[pivot]);
                }
                min_pivot += 1;
            }
        }
    }

    return stabilizers;
}

}  // namespace stim

```

```cpp
// ===== stim/stabilizers/tableau_transposed_raii.h (65 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_STABILIZERS_TABLEAU_TRANSPOSED_RAII_H
#define _STIM_STABILIZERS_TABLEAU_TRANSPOSED_RAII_H

#include <iostream>
#include <unordered_map>

#include "stim/mem/simd_bit_table.h"
#include "stim/mem/simd_util.h"
#include "stim/stabilizers/pauli_string.h"
#include "stim/stabilizers/tableau.h"

namespace stim {

/// When this class is constructed, it transposes the tableau given to it.
/// The transpose is undone on deconstruction.
///
/// This is useful when appending operations to the tableau, since otherwise
/// the append would be working against the grain of memory.
///
/// The template parameter, W, represents the SIMD width.
template <size_t W>
struct TableauTransposedRaii {
    Tableau<W> &tableau;

    explicit TableauTransposedRaii(Tableau<W> &tableau);
    ~TableauTransposedRaii();

    TableauTransposedRaii() = delete;
    TableauTransposedRaii(const TableauTransposedRaii &) = delete;
    TableauTransposedRaii(TableauTransposedRaii &&) = delete;

    PauliString<W> unsigned_x_input(size_t q) const;

    void append_H_XZ(size_t q);
    void append_H_XY(size_t q);
    void append_H_YZ(size_t q);
    void append_S(size_t q);
    void append_ZCX(size_t control, size_t target);
    void append_ZCY(size_t control, size_t target);
    void append_ZCZ(size_t control, size_t target);
    void append_X(size_t q);
    void append_SWAP(size_t q1, size_t q2);
};

}  // namespace stim

#include "stim/stabilizers/tableau_transposed_raii.inl"

#endif

```

```cpp
// ===== stim/stabilizers/tableau_transposed_raii.inl (157 lines) =====
// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <map>

#include "stim/stabilizers/pauli_string.h"
#include "stim/stabilizers/tableau_transposed_raii.h"

namespace stim {

template <size_t W>
TableauTransposedRaii<W>::TableauTransposedRaii(Tableau<W> &tableau) : tableau(tableau) {
    tableau.do_transpose_quadrants();
}

template <size_t W>
TableauTransposedRaii<W>::~TableauTransposedRaii() {
    tableau.do_transpose_quadrants();
}

/// Iterates over the Paulis in a row of the tableau.
///
/// Args:
///     trans: The transposed tableau (where rows are contiguous in memory and so operations can be done efficiently).
///     q: The row to iterate over.
///     body: A function taking X, Z, and SIGN words.
///         The X and Z words are chunks of xz-encoded Paulis from the row.
///         The SIGN word is the corresponding chunk of sign bits from the sign row.
template <size_t W, typename FUNC>
inline void for_each_trans_obs(TableauTransposedRaii<W> &trans, size_t q, FUNC body) {
    for (size_t k = 0; k < 2; k++) {
        TableauHalf<W> &h = k == 0 ? trans.tableau.xs : trans.tableau.zs;
        PauliStringRef<W> p = h[q];
        p.xs.for_each_word(p.zs, h.signs, body);
    }
}

template <size_t W, typename FUNC>
inline void for_each_trans_obs(TableauTransposedRaii<W> &trans, size_t q1, size_t q2, FUNC body) {
    for (size_t k = 0; k < 2; k++) {
        TableauHalf<W> &h = k == 0 ? trans.tableau.xs : trans.tableau.zs;
        PauliStringRef<W> p1 = h[q1];
        PauliStringRef<W> p2 = h[q2];
        p1.xs.for_each_word(p1.zs, p2.xs, p2.zs, h.signs, body);
    }
}

template <size_t W>
void TableauTransposedRaii<W>::append_ZCX(size_t control, size_t target) {
    for_each_trans_obs<W>(
        *this,
        control,
        target,
        [](simd_word<W> &cx, simd_word<W> &cz, simd_word<W> &tx, simd_word<W> &tz, simd_word<W> &s) {
            s ^= (cz ^ tx).andnot(cx & tz);
            cz ^= tz;
            tx ^= cx;
        });
}

template <size_t W>
void TableauTransposedRaii<W>::append_ZCY(size_t control, size_t target) {
    for_each_trans_obs<W>(
        *this,
        control,
        target,
        [](simd_word<W> &cx, simd_word<W> &cz, simd_word<W> &tx, simd_word<W> &tz, simd_word<W> &s) {
            cz ^= tx;
            s ^= cx & cz & (tx ^ tz);
            cz ^= tz;
            tx ^= cx;
            tz ^= cx;
        });
}

template <size_t W>
void TableauTransposedRaii<W>::append_ZCZ(size_t control, size_t target) {
    for_each_trans_obs<W>(
        *this,
        control,
        target,
        [](simd_word<W> &cx, simd_word<W> &cz, simd_word<W> &tx, simd_word<W> &tz, simd_word<W> &s) {
            s ^= cx & tx & (cz ^ tz);
            cz ^= tx;
            tz ^= cx;
        });
}

template <size_t W>
void TableauTransposedRaii<W>::append_SWAP(size_t q1, size_t q2) {
    for_each_trans_obs<W>(
        *this, q1, q2, [](simd_word<W> &x1, simd_word<W> &z1, simd_word<W> &x2, simd_word<W> &z2, simd_word<W> &s) {
            std::swap(x1, x2);
            std::swap(z1, z2);
        });
}

template <size_t W>
void TableauTransposedRaii<W>::append_H_XY(size_t target) {
    for_each_trans_obs<W>(*this, target, [](simd_word<W> &x, simd_word<W> &z, simd_word<W> &s) {
        s ^= x.andnot(z);
        z ^= x;
    });
}

template <size_t W>
void TableauTransposedRaii<W>::append_H_YZ(size_t target) {
    for_each_trans_obs<W>(*this, target, [](simd_word<W> &x, simd_word<W> &z, simd_word<W> &s) {
        s ^= z.andnot(x);
        x ^= z;
    });
}

template <size_t W>
void TableauTransposedRaii<W>::append_S(size_t target) {
    for_each_trans_obs<W>(*this, target, [](simd_word<W> &x, simd_word<W> &z, simd_word<W> &s) {
        s ^= x & z;
        z ^= x;
    });
}

template <size_t W>
void TableauTransposedRaii<W>::append_H_XZ(size_t q) {
    for_each_trans_obs<W>(*this, q, [](simd_word<W> &x, simd_word<W> &z, simd_word<W> &s) {
        std::swap(x, z);
        s ^= x & z;
    });
}

template <size_t W>
void TableauTransposedRaii<W>::append_X(size_t target) {
    for_each_trans_obs<W>(*this, target, [](simd_word<W> &x, simd_word<W> &z, simd_word<W> &s) {
        s ^= z;
    });
}

template <size_t W>
PauliString<W> TableauTransposedRaii<W>::unsigned_x_input(size_t q) const {
    PauliString<W> result(tableau.num_qubits);
    result.xs = tableau.zs[q].zs;
    result.zs = tableau.xs[q].zs;
    return result;
}

}  // namespace stim

```

### 5.3 Simulators

```cpp
// ===== stim/simulators/tableau_simulator.h (330 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_SIMULATORS_TABLEAU_SIMULATOR_H
#define _STIM_SIMULATORS_TABLEAU_SIMULATOR_H

#include <cassert>
#include <functional>
#include <iostream>
#include <new>
#include <random>
#include <sstream>

#include "stim/circuit/circuit.h"
#include "stim/io/measure_record.h"
#include "stim/stabilizers/tableau.h"
#include "stim/stabilizers/tableau_transposed_raii.h"

namespace stim {

/// A stabilizer circuit simulator that tracks an inverse stabilizer tableau
/// and allows interactive usage, where gates and measurements are applied
/// on demand.
///
/// The template parameter, W, represents the SIMD width
template <size_t W>
struct TableauSimulator {
    Tableau<W> inv_state;
    std::mt19937_64 rng;
    int8_t sign_bias;
    MeasureRecord measurement_record;
    bool last_correlated_error_occurred;

    /// Args:
    ///     num_qubits: The initial number of qubits in the simulator state.
    ///     rng: The random number generator to use for random operations.
    ///     sign_bias: 0 means collapse randomly, -1 means collapse towards True, +1 means collapse towards False.
    ///     record: Measurement record configuration.
    explicit TableauSimulator(
        std::mt19937_64 &&rng, size_t num_qubits = 0, int8_t sign_bias = 0, MeasureRecord record = MeasureRecord());
    /// Args:
    ///     other: TableauSimulator to copy state from.
    ///     rng: The random number generator to use for random operations.
    TableauSimulator(const TableauSimulator &other, std::mt19937_64 &&rng);

    /// Samples the given circuit in a deterministic fashion.
    ///
    /// Discards all noisy operations, and biases all collapse events towards +Z instead of randomly +Z/-Z.
    static simd_bits<W> reference_sample_circuit(const Circuit &circuit);
    static simd_bits<W> sample_circuit(const Circuit &circuit, std::mt19937_64 &rng, int8_t sign_bias = 0);
    static void sample_stream(FILE *in, FILE *out, SampleFormat format, bool interactive, std::mt19937_64 &rng);

    /// Expands the internal state of the simulator (if needed) to ensure the given qubit exists.
    ///
    /// Failing to ensure the state is large enough for a qubit, before that qubit is acted on for the first time,
    /// results in undefined behavior.
    void ensure_large_enough_for_qubits(size_t num_qubits);

    /// Forces the size of the internal state of the simulator.
    ///
    /// Shrinking the size will result in qubits beyond the size threshold being collapsed and discarded.
    void set_num_qubits(size_t new_num_qubits);

    /// Finds a state vector satisfying the current stabilizer generators, and returns a vector simulator in that state.
    VectorSimulator to_vector_sim() const;

    /// Returns a state vector satisfying the current stabilizer generators.
    std::vector<std::complex<float>> to_state_vector(bool little_endian) const;

    /// Collapses then records an observable.
    ///
    /// Args:
    ///     pauli_string: The observable to measure.
    bool measure_pauli_string(const PauliStringRef<W> pauli_string, double flip_probability);

    /// Determines if a qubit's X observable commutes (vs anti-commutes) with the current stabilizer generators.
    bool is_deterministic_x(size_t target) const;
    /// Determines if a qubit's Y observable commutes (vs anti-commutes) with the current stabilizer generators.
    bool is_deterministic_y(size_t target) const;
    /// Determines if a qubit's Z observable commutes (vs anti-commutes) with the current stabilizer generators.
    bool is_deterministic_z(size_t target) const;

    /// Runs all of the operations in the given circuit.
    ///
    /// Automatically expands the tableau simulator's state, if needed.
    void safe_do_circuit(const Circuit &circuit, uint64_t reps = 1);
    void do_operation_ensure_size(const CircuitInstruction &operation);

    void apply_tableau(const Tableau<W> &tableau, const std::vector<size_t> &targets);

    std::vector<PauliString<W>> canonical_stabilizers() const;

    void do_gate(const CircuitInstruction &inst);

    /// === SPECIALIZED VECTORIZED OPERATION IMPLEMENTATIONS ===
    void do_I(const CircuitInstruction &inst);
    void do_H_XZ(const CircuitInstruction &inst);
    void do_H_YZ(const CircuitInstruction &inst);
    void do_H_XY(const CircuitInstruction &inst);
    void do_H_NXY(const CircuitInstruction &inst);
    void do_H_NXZ(const CircuitInstruction &inst);
    void do_H_NYZ(const CircuitInstruction &inst);
    void do_C_XYZ(const CircuitInstruction &inst);
    void do_C_NXYZ(const CircuitInstruction &inst);
    void do_C_XNYZ(const CircuitInstruction &inst);
    void do_C_XYNZ(const CircuitInstruction &inst);
    void do_C_ZYX(const CircuitInstruction &inst);
    void do_C_NZYX(const CircuitInstruction &inst);
    void do_C_ZNYX(const CircuitInstruction &inst);
    void do_C_ZYNX(const CircuitInstruction &inst);
    void do_SQRT_X(const CircuitInstruction &inst);
    void do_SQRT_Y(const CircuitInstruction &inst);
    void do_SQRT_Z(const CircuitInstruction &inst);
    void do_SQRT_X_DAG(const CircuitInstruction &inst);
    void do_SQRT_Y_DAG(const CircuitInstruction &inst);
    void do_SQRT_Z_DAG(const CircuitInstruction &inst);
    void do_SQRT_XX(const CircuitInstruction &inst);
    void do_SQRT_XX_DAG(const CircuitInstruction &inst);
    void do_SQRT_YY(const CircuitInstruction &inst);
    void do_SQRT_YY_DAG(const CircuitInstruction &inst);
    void do_SQRT_ZZ(const CircuitInstruction &inst);
    void do_SQRT_ZZ_DAG(const CircuitInstruction &inst);
    void do_ZCX(const CircuitInstruction &inst);
    void do_ZCY(const CircuitInstruction &inst);
    void do_ZCZ(const CircuitInstruction &inst);
    void do_SWAP(const CircuitInstruction &inst);
    void do_X(const CircuitInstruction &inst);
    void do_Y(const CircuitInstruction &inst);
    void do_Z(const CircuitInstruction &inst);
    void do_ISWAP(const CircuitInstruction &inst);
    void do_ISWAP_DAG(const CircuitInstruction &inst);
    void do_CXSWAP(const CircuitInstruction &inst);
    void do_CZSWAP(const CircuitInstruction &inst);
    void do_SWAPCX(const CircuitInstruction &inst);
    void do_XCX(const CircuitInstruction &inst);
    void do_XCY(const CircuitInstruction &inst);
    void do_XCZ(const CircuitInstruction &inst);
    void do_YCX(const CircuitInstruction &inst);
    void do_YCY(const CircuitInstruction &inst);
    void do_YCZ(const CircuitInstruction &inst);
    void do_DEPOLARIZE1(const CircuitInstruction &inst);
    void do_DEPOLARIZE2(const CircuitInstruction &inst);
    void do_HERALDED_ERASE(const CircuitInstruction &inst);
    void do_HERALDED_PAULI_CHANNEL_1(const CircuitInstruction &inst);
    void do_X_ERROR(const CircuitInstruction &inst);
    void do_Y_ERROR(const CircuitInstruction &inst);
    void do_Z_ERROR(const CircuitInstruction &inst);
    void do_PAULI_CHANNEL_1(const CircuitInstruction &inst);
    void do_PAULI_CHANNEL_2(const CircuitInstruction &inst);
    void do_CORRELATED_ERROR(const CircuitInstruction &inst);
    void do_ELSE_CORRELATED_ERROR(const CircuitInstruction &inst);
    void do_MPP(const CircuitInstruction &inst);
    void do_SPP(const CircuitInstruction &inst);
    void do_SPP_DAG(const CircuitInstruction &inst);
    void do_MXX(const CircuitInstruction &inst);
    void do_MYY(const CircuitInstruction &inst);
    void do_MZZ(const CircuitInstruction &inst);
    void do_MPAD(const CircuitInstruction &inst);
    void do_MX(const CircuitInstruction &inst);
    void do_MY(const CircuitInstruction &inst);
    void do_MZ(const CircuitInstruction &inst);
    void do_RX(const CircuitInstruction &inst);
    void do_RY(const CircuitInstruction &inst);
    void do_RZ(const CircuitInstruction &inst);
    void do_MRX(const CircuitInstruction &inst);
    void do_MRY(const CircuitInstruction &inst);
    void do_MRZ(const CircuitInstruction &inst);

    /// Returns the single-qubit stabilizer of a target or, if it is entangled, the identity operation.
    PauliString<W> peek_bloch(uint32_t target) const;

    /// Returns the expectation value of measuring the qubit in the X basis.
    int8_t peek_x(uint32_t target) const;
    /// Returns the expectation value of measuring the qubit in the Y basis.
    int8_t peek_y(uint32_t target) const;
    /// Returns the expectation value of measuring the qubit in the Z basis.
    int8_t peek_z(uint32_t target) const;

    /// Projects the system into a desired qubit X observable, or raises an exception if it was impossible.
    void postselect_x(SpanRef<const GateTarget> targets, bool desired_result);
    /// Projects the system into a desired qubit Y observable, or raises an exception if it was impossible.
    void postselect_y(SpanRef<const GateTarget> targets, bool desired_result);
    /// Projects the system into a desired qubit Z observable, or raises an exception if it was impossible.
    void postselect_z(SpanRef<const GateTarget> targets, bool desired_result);

    /// Applies all of the Pauli operations in the given PauliString to the simulator's state.
    void paulis(const PauliString<W> &paulis);

    /// Performs a measurement and returns a kickback that flips between the possible post-measurement states.
    ///
    /// Deterministic measurements have no kickback.
    /// This is represented by setting the kickback to the empty Pauli string.
    std::pair<bool, PauliString<W>> measure_kickback_z(GateTarget target);
    std::pair<bool, PauliString<W>> measure_kickback_y(GateTarget target);
    std::pair<bool, PauliString<W>> measure_kickback_x(GateTarget target);

    bool read_measurement_record(uint32_t encoded_target) const;
    void single_cx(uint32_t c, uint32_t t);
    void single_cy(uint32_t c, uint32_t t);

    /// Forces a qubit to have a collapsed Z observable.
    ///
    /// If the qubit already has a collapsed Z observable, this method has no effect.
    /// Other, the qubit's Z observable anticommutes with the current stabilizers and this method will apply state
    /// transformations that pick out a single stabilizer generator to destroy and replace with the measurement's
    /// stabilizer.
    ///
    /// Args:
    ///     target: The index of the qubit to collapse.
    ///     transposed_raii: A RAII value whose existence certifies the tableau data is currently transposed
    ///         (to make operations efficient).
    ///
    /// Returns:
    ///    SIZE_MAX: Already collapsed.
    ///    Else: The pivot index. The start-of-time qubit whose X flips the measurement.
    size_t collapse_qubit_z(size_t target, TableauTransposedRaii<W> &transposed_raii);

    /// Collapses the given qubits into the X basis.
    ///
    /// Args:
    ///     targets: The qubits to collapse.
    ///     stride: Defaults to 1. Set to 2 to skip over every other target.
    void collapse_x(SpanRef<const GateTarget> targets, size_t stride = 1);

    /// Collapses the given qubits into the Y basis.
    ///
    /// Args:
    ///     targets: The qubits to collapse.
    ///     stride: Defaults to 1. Set to 2 to skip over every other target.
    void collapse_y(SpanRef<const GateTarget> targets, size_t stride = 1);

    /// Collapses the given qubits into the Z basis.
    ///
    /// Args:
    ///     targets: The qubits to collapse.
    ///     stride: Defaults to 1. Set to 2 to skip over every other target.
    void collapse_z(SpanRef<const GateTarget> targets, size_t stride = 1);

    /// Completely isolates a qubit from the other qubits tracked by the simulator, so it can be safely discarded.
    ///
    /// After this runs, it is guaranteed that the inverse tableau maps the target qubit's X and Z observables to
    /// themselves (possibly negated) and that it maps all other qubits to Pauli products not involving the target
    /// qubit.
    void collapse_isolate_qubit_z(size_t target, TableauTransposedRaii<W> &transposed_raii);

    /// Determines the expected value of an observable (which will always be -1, 0, or +1).
    ///
    /// This is a non-physical operation.
    /// It reports information about the quantum state without disturbing it.
    ///
    /// Args:
    ///     observable: The observable to determine the expected value of.
    ///
    /// Returns:
    ///     +1: Observable will be deterministically false when measured.
    ///     -1: Observable will be deterministically true when measured.
    ///     0: Observable will be random when measured.
    int8_t peek_observable_expectation(const PauliString<W> &observable) const;

    /// Forces a desired measurement result, or raises an exception if it was impossible.
    void postselect_observable(PauliStringRef<W> observable, bool desired_result);

   private:
    uint32_t try_isolate_observable_to_qubit_z(PauliStringRef<W> observable, bool undo);
    void do_MXX_disjoint_controls_segment(const CircuitInstruction &inst);
    void do_MYY_disjoint_controls_segment(const CircuitInstruction &inst);
    void do_MZZ_disjoint_controls_segment(const CircuitInstruction &inst);
    void noisify_new_measurements(SpanRef<const double> args, size_t num_targets);
    void noisify_new_measurements(const CircuitInstruction &target_data);
    void postselect_helper(
        SpanRef<const GateTarget> targets,
        bool desired_result,
        GateType basis_change_gate,
        const char *false_name,
        const char *true_name);
};

template <size_t Q, typename RESET_FLAG, typename ELSE_CORR>
void perform_pauli_errors_via_correlated_errors(
    const CircuitInstruction &target_data, RESET_FLAG reset_flag, ELSE_CORR else_corr) {
    double target_p{};
    GateTarget target_t[Q];
    CircuitInstruction data{GateType::E, {&target_p}, {&target_t[0], &target_t[Q]}, ""};
    for (size_t k = 0; k < target_data.targets.size(); k += Q) {
        reset_flag();
        double used_probability = 0;
        for (size_t pauli = 1; pauli < 1 << (2 * Q); pauli++) {
            double p = target_data.args[pauli - 1];
            if (p == 0) {
                continue;
            }
            double remaining = 1 - used_probability;
            double conditional_prob = remaining <= 0 ? 0 : remaining <= p ? 1 : p / remaining;
            used_probability += p;

            for (size_t q = 0; q < Q; q++) {
                target_t[q] = target_data.targets[k + q];
                bool z = (pauli >> (2 * (Q - q - 1))) & 2;
                bool y = (pauli >> (2 * (Q - q - 1))) & 1;
                if (z ^ y) {
                    target_t[q].data |= TARGET_PAULI_X_BIT;
                }
                if (z) {
                    target_t[q].data |= TARGET_PAULI_Z_BIT;
                }
            }
            target_p = conditional_prob;
            else_corr(data);
        }
    }
}

}  // namespace stim

#include "stim/simulators/tableau_simulator.inl"

#endif

```

```cpp
// ===== stim/simulators/frame_simulator.h (162 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_SIMULATORS_FRAME_SIMULATOR_H
#define _STIM_SIMULATORS_FRAME_SIMULATOR_H

#include <random>

#include "stim/circuit/circuit.h"
#include "stim/io/measure_record_batch.h"
#include "stim/mem/simd_bit_table.h"
#include "stim/stabilizers/pauli_string.h"

namespace stim {

enum class FrameSimulatorMode {
    STORE_MEASUREMENTS_TO_MEMORY,  // all measurements stored, detections not stored
    STREAM_MEASUREMENTS_TO_DISK,   // measurements stored up to lookback, detections not stored
    STORE_DETECTIONS_TO_MEMORY,    // measurements stored up to lookback, all detections stored
    STREAM_DETECTIONS_TO_DISK,     // measurements stored up to lookback, detections stored until write
    STORE_EVERYTHING_TO_MEMORY,    // all measurements stored and all detections stored
};

/// A Pauli Frame simulator that computes many samples simultaneously.
///
/// This simulator tracks, for each qubit, whether or not that qubit is bit flipped and/or phase flipped.
/// Instead of reporting qubit measurements, it reports whether a measurement is inverted or not.
/// This requires a set of reference measurements to diff against.
///
/// The template parameter, W, represents the SIMD width
template <size_t W>
struct FrameSimulator {
    size_t num_qubits;                 // Number of qubits being tracked.
    uint64_t num_observables;          // Number of observables being tracked.
    bool keeping_detection_data;       // Whether or not to store dets and obs data.
    size_t batch_size;                 // Number of instances being tracked.
    simd_bit_table<W> x_table;         // x_table[q][k] is whether or not there's an X error on qubit q in instance k.
    simd_bit_table<W> z_table;         // z_table[q][k] is whether or not there's a Z error on qubit q in instance k.
    MeasureRecordBatch<W> m_record;    // The measurement record.
    MeasureRecordBatch<W> det_record;  // Detection event record.
    simd_bit_table<W> obs_record;      // Accumulating observable flip record.
    simd_bits<W> rng_buffer;           // Workspace used when sampling error processes.
    simd_bits<W> tmp_storage;          // Workspace used when sampling compound error processes.
    simd_bits<W> last_correlated_error_occurred;  // correlated error flag for each instance.
    simd_bit_table<W> sweep_table;                // Shot-to-shot configuration data.
    std::mt19937_64 rng;                          // Random number generator used for generating entropy.

    // Determines whether e.g. 50% Z errors are multiplied into the frame when measuring in the Z basis.
    // This is necessary for correct sampling.
    // It should only be disabled when e.g. using the frame simulator to understand how a fixed set of errors will
    // propagate, without interference from other effects.
    bool guarantee_anticommutation_via_frame_randomization = true;

    /// Constructs a FrameSimulator capable of simulating a circuit with the given size stats.
    ///
    /// Args:
    ///     circuit_stats: Sizes that determine how large internal buffers must be. Get
    ///         this from stim::Circuit::compute_stats.
    ///     mode: Describes the intended usage of the simulator, which affects the sizing
    ///         of buffers.
    ///     batch_size: How many shots to simulate simultaneously.
    ///     rng: The random number generator to pull noise from.
    FrameSimulator(CircuitStats circuit_stats, FrameSimulatorMode mode, size_t batch_size, std::mt19937_64 &&rng);
    FrameSimulator() = delete;

    PauliString<W> get_frame(size_t sample_index) const;
    void set_frame(size_t sample_index, const PauliStringRef<W> &new_frame);
    void configure_for(CircuitStats new_circuit_stats, FrameSimulatorMode new_mode, size_t new_batch_size);
    void ensure_safe_to_do_circuit_with_stats(const CircuitStats &stats);

    void safe_do_instruction(const CircuitInstruction &instruction);
    void safe_do_circuit(const Circuit &circuit, uint64_t repetitions = 1);

    void do_circuit(const Circuit &circuit);
    void reset_all();

    void do_gate(const CircuitInstruction &inst);

    void do_MX(const CircuitInstruction &inst);
    void do_MY(const CircuitInstruction &inst);
    void do_MZ(const CircuitInstruction &inst);
    void do_RX(const CircuitInstruction &inst);
    void do_RY(const CircuitInstruction &inst);
    void do_RZ(const CircuitInstruction &inst);
    void do_MRX(const CircuitInstruction &inst);
    void do_MRY(const CircuitInstruction &inst);
    void do_MRZ(const CircuitInstruction &inst);

    void do_DETECTOR(const CircuitInstruction &inst);
    void do_OBSERVABLE_INCLUDE(const CircuitInstruction &inst);

    void do_I(const CircuitInstruction &inst);
    void do_H_XZ(const CircuitInstruction &inst);
    void do_H_XY(const CircuitInstruction &inst);
    void do_H_YZ(const CircuitInstruction &inst);
    void do_C_XYZ(const CircuitInstruction &inst);
    void do_C_ZYX(const CircuitInstruction &inst);
    void do_ZCX(const CircuitInstruction &inst);
    void do_ZCY(const CircuitInstruction &inst);
    void do_ZCZ(const CircuitInstruction &inst);
    void do_XCX(const CircuitInstruction &inst);
    void do_XCY(const CircuitInstruction &inst);
    void do_XCZ(const CircuitInstruction &inst);
    void do_YCX(const CircuitInstruction &inst);
    void do_YCY(const CircuitInstruction &inst);
    void do_YCZ(const CircuitInstruction &inst);
    void do_SWAP(const CircuitInstruction &inst);
    void do_ISWAP(const CircuitInstruction &inst);
    void do_CXSWAP(const CircuitInstruction &inst);
    void do_CZSWAP(const CircuitInstruction &inst);
    void do_SWAPCX(const CircuitInstruction &inst);
    void do_MPP(const CircuitInstruction &inst);
    void do_SPP(const CircuitInstruction &inst);
    void do_SPP_DAG(const CircuitInstruction &inst);
    void do_MXX(const CircuitInstruction &inst);
    void do_MYY(const CircuitInstruction &inst);
    void do_MZZ(const CircuitInstruction &inst);
    void do_MPAD(const CircuitInstruction &inst);

    void do_SQRT_XX(const CircuitInstruction &inst);
    void do_SQRT_YY(const CircuitInstruction &inst);
    void do_SQRT_ZZ(const CircuitInstruction &inst);

    void do_DEPOLARIZE1(const CircuitInstruction &inst);
    void do_DEPOLARIZE2(const CircuitInstruction &inst);
    void do_X_ERROR(const CircuitInstruction &inst);
    void do_Y_ERROR(const CircuitInstruction &inst);
    void do_Z_ERROR(const CircuitInstruction &inst);
    void do_PAULI_CHANNEL_1(const CircuitInstruction &inst);
    void do_PAULI_CHANNEL_2(const CircuitInstruction &inst);
    void do_CORRELATED_ERROR(const CircuitInstruction &inst);
    void do_ELSE_CORRELATED_ERROR(const CircuitInstruction &inst);
    void do_HERALDED_ERASE(const CircuitInstruction &inst);
    void do_HERALDED_PAULI_CHANNEL_1(const CircuitInstruction &inst);

   private:
    void do_MXX_disjoint_controls_segment(const CircuitInstruction &inst);
    void do_MYY_disjoint_controls_segment(const CircuitInstruction &inst);
    void do_MZZ_disjoint_controls_segment(const CircuitInstruction &inst);
    void xor_control_bit_into(uint32_t control, simd_bits_range_ref<W> target);
    void single_cx(uint32_t c, uint32_t t);
    void single_cy(uint32_t c, uint32_t t);
};

}  // namespace stim

#include "stim/simulators/frame_simulator.inl"

#endif

```

### 5.4 Gate Definitions

```cpp
// ===== stim/gates/gates.h (414 lines) =====
/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _STIM_GATES_GATE_DATA_H
#define _STIM_GATES_GATE_DATA_H

#include <array>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "stim/mem/fixed_cap_vector.h"

namespace stim {

template <size_t W>
struct Tableau;

template <size_t W>
struct Flow;

template <size_t W>
struct PauliString;

/// Used for gates' argument count to indicate that a gate takes a variable number of
/// arguments. This is relevant to coordinate data on detectors and qubits, where there may
/// be any number of coordinates.
constexpr uint8_t ARG_COUNT_SYGIL_ANY = uint8_t{0xFF};

/// Used for gates' argument count to indicate that a gate takes 0 parens arguments or 1
/// parens argument. This is relevant to measurement gates, where 0 parens arguments means
/// a noiseless result whereas 1 parens argument is a noisy result.
constexpr uint8_t ARG_COUNT_SYGIL_ZERO_OR_ONE = uint8_t{0xFE};

constexpr inline uint16_t gate_name_to_hash(std::string_view text) {
    // HACK: A collision is considered to be an error.
    // Just do *anything* that makes all the defined gates have different values.

    constexpr uint16_t const1 = 2126;
    constexpr uint16_t const2 = 9883;
    constexpr uint16_t const3 = 8039;
    constexpr uint16_t const4 = 9042;
    constexpr uint16_t const5 = 4916;
    constexpr uint16_t const6 = 4048;
    constexpr uint16_t const7 = 7081;

    size_t n = text.size();
    const char *v = text.data();
    size_t result = n;
    if (n > 0) {
        auto c_first = v[0] | 0x20;
        auto c_last = v[n - 1] | 0x20;
        result ^= c_first * const1;
        result += c_last * const2;
    }
    if (n > 2) {
        auto c1 = v[1] | 0x20;
        auto c2 = v[2] | 0x20;
        result ^= c1 * const3;
        result += c2 * const4;
    }
    if (n > 4) {
        auto c3 = v[3] | 0x20;
        auto c4 = v[4] | 0x20;
        result ^= c3 * const5;
        result += c4 * const6;
    }
    if (n > 5) {
        auto c5 = v[5] | 0x20;
        result ^= c5 * const7;
    }
    return result & 0x1FF;
}

constexpr size_t NUM_DEFINED_GATES = 82;

enum class GateType : uint8_t {
    NOT_A_GATE = 0,
    // Annotations
    DETECTOR,
    OBSERVABLE_INCLUDE,
    TICK,
    QUBIT_COORDS,
    SHIFT_COORDS,
    // Control flow
    REPEAT,
    // Collapsing gates
    MPAD,
    MX,
    MY,
    M,  // alias when parsing: MZ
    MRX,
    MRY,
    MR,  // alias when parsing: MRZ
    RX,
    RY,
    R,  // alias when parsing: RZ
    // Controlled gates
    XCX,
    XCY,
    XCZ,
    YCX,
    YCY,
    YCZ,
    CX,  // alias when parsing: CNOT, ZCX
    CY,  // alias when parsing: ZCY
    CZ,  // alias when parsing: ZCZ
    // Hadamard-like gates
    H,  // alias when parsing: H_XZ
    H_XY,
    H_YZ,
    H_NXY,
    H_NXZ,
    H_NYZ,
    // Noise channels
    DEPOLARIZE1,
    DEPOLARIZE2,
    X_ERROR,
    Y_ERROR,
    Z_ERROR,
    I_ERROR,
    II_ERROR,
    PAULI_CHANNEL_1,
    PAULI_CHANNEL_2,
    E,  // alias when parsing: CORRELATED_ERROR
    ELSE_CORRELATED_ERROR,
    // Heralded noise channels
    HERALDED_ERASE,
    HERALDED_PAULI_CHANNEL_1,
    // Pauli gates
    I,
    X,
    Y,
    Z,
    // Period 3 gates
    C_XYZ,
    C_ZYX,
    C_NXYZ,
    C_XNYZ,
    C_XYNZ,
    C_NZYX,
    C_ZNYX,
    C_ZYNX,
    // Period 4 gates
    SQRT_X,
    SQRT_X_DAG,
    SQRT_Y,
    SQRT_Y_DAG,
    S,      // alias when parsing: SQRT_Z
    S_DAG,  // alias when parsing: SQRT_Z_DAG
    // Parity phasing gates.
    II,
    SQRT_XX,
    SQRT_XX_DAG,
    SQRT_YY,
    SQRT_YY_DAG,
    SQRT_ZZ,
    SQRT_ZZ_DAG,
    // Pauli product gates
    MPP,
    SPP,
    SPP_DAG,
    // Swap gates
    SWAP,
    ISWAP,
    CXSWAP,
    SWAPCX,
    CZSWAP,
    ISWAP_DAG,
    // Pair measurement gates
    MXX,
    MYY,
    MZZ,
};

enum GateFlags : uint16_t {
    // All gates must have at least one flag set.
    NO_GATE_FLAG = 0,

    // Indicates whether unitary and tableau data is available for the gate, so it can be tested more easily.
    GATE_IS_UNITARY = 1 << 0,
    // Determines whether or not the gate is omitted when computing a reference sample.
    GATE_IS_NOISY = 1 << 1,
    // Controls validation of probability arguments like X_ERROR(0.01).
    GATE_ARGS_ARE_DISJOINT_PROBABILITIES = 1 << 2,
    // Indicates whether the gate puts data into the measurement record or not.
    // Also determines whether or not inverted targets (like "!3") are permitted.
    GATE_PRODUCES_RESULTS = 1 << 3,
    // Prevents the same gate on adjacent lines from being combined into one longer invocation.
    GATE_IS_NOT_FUSABLE = 1 << 4,
    // Controls block functionality for instructions like REPEAT.
    GATE_IS_BLOCK = 1 << 5,
    // Controls validation code checking for arguments coming in pairs.
    GATE_TARGETS_PAIRS = 1 << 6,
    // Controls instructions like CORRELATED_ERROR taking Pauli product targets ("X1 Y2 Z3").
    // Note that this enables the Pauli terms but not the combine terms like X1*Y2.
    GATE_TARGETS_PAULI_STRING = 1 << 7,
    // Controls instructions like DETECTOR taking measurement record targets ("rec[-1]").
    // The "ONLY" refers to the fact that this flag switches the default behavior to not allowing qubit targets.
    // Further flags can then override that default.
    GATE_ONLY_TARGETS_MEASUREMENT_RECORD = 1 << 8,
    // Controls instructions like CX operating allowing measurement record targets and sweep bit targets.
    GATE_CAN_TARGET_BITS = 1 << 9,
    // Controls whether the gate takes qubit/record targets.
    GATE_TAKES_NO_TARGETS = 1 << 10,
    // Controls validation of index arguments like OBSERVABLE_INCLUDE(1).
    GATE_ARGS_ARE_UNSIGNED_INTEGERS = 1 << 11,
    // Controls instructions like MPP taking Pauli product combiners ("X1*Y2 Z3").
    GATE_TARGETS_COMBINERS = 1 << 12,
    // Measurements and resets are dissipative operations.
    GATE_IS_RESET = 1 << 13,
    // Annotations like DETECTOR aren't strictly speaking identity operations, but they can be ignored by code that only
    // cares about effects that happen to qubits (as opposed to in the classical control system).
    GATE_HAS_NO_EFFECT_ON_QUBITS = 1 << 14,
    // Whether or not the gate trivially broadcasts over targets.
    GATE_IS_SINGLE_QUBIT_GATE = 1 << 15,
};

struct Gate {
    /// The canonical name of the gate, used when printing it to a circuit file.
    std::string_view name;
    /// The gate's type, such as stim::GateType::X or stim::GateType::MRZ.
    GateType id;
    /// The id of the gate inverse to this one, or at least the closest thing to an inverse.
    /// Set to GateType::NOT_A_GATE for gates with no inverse.
    GateType best_candidate_inverse_id;
    /// The number of parens arguments the gate expects (e.g. X_ERROR takes 1, PAULI_CHANNEL_1 takes 3).
    /// Set to stim::ARG_COUNT_SYGIL_ANY to indicate any number is allowed (e.g. DETECTOR coordinate data).
    uint8_t arg_count;
    /// Bit-packed data describing details of the gate.
    GateFlags flags;

    /// A word describing what sort of gate this is.
    std::string_view category;
    /// Prose summary of what the gate is, how it fits into Stim, and how to use it.
    std::string_view help;
    /// A unitary matrix describing the gate. (Size 0 if the gate is not unitary.)
    FixedCapVector<FixedCapVector<std::complex<float>, 4>, 4> unitary_data;
    /// A shorthand description of the stabilizer flows of the gate.
    /// For single qubit Cliffords, this should be the output stabilizers for X then Z.
    /// For 2 qubit Cliffords, this should be the output stabilizers for X_, Z_, _X, _Z.
    /// For 2 qubit dissipative gates, this should be flows like "X_ -> XX xor rec[-1]".
    FixedCapVector<const char *, 10> flow_data;
    /// Stim circuit file contents of a decomposition into H+S+CX+M+R operations. (nullptr if not decomposable.)
    const char *h_s_cx_m_r_decomposition;

    inline bool operator==(const Gate &other) const {
        return id == other.id;
    }
    inline bool operator!=(const Gate &other) const {
        return id != other.id;
    }

    const Gate &inverse() const;

    template <size_t W>
    Tableau<W> tableau() const {
        if (!(flags & GateFlags::GATE_IS_UNITARY)) {
            throw std::invalid_argument(std::string(name) + " isn't unitary so it doesn't have a tableau.");
        }
        const auto &d = flow_data;
        if (flow_data.size() == 2) {
            return Tableau<W>::gate1(d[0], d[1]);
        }
        if (flow_data.size() == 4) {
            return Tableau<W>::gate2(d[0], d[1], d[2], d[3]);
        }
        throw std::out_of_range(std::string(name) + " doesn't have 1q or 2q tableau data.");
    }

    template <size_t W>
    std::vector<Flow<W>> flows() const {
        if (has_known_unitary_matrix()) {
            auto t = tableau<W>();
            if (flags & GateFlags::GATE_TARGETS_PAIRS) {
                return {
                    Flow<W>{stim::PauliString<W>::from_str("X_"), t.xs[0], {}},
                    Flow<W>{stim::PauliString<W>::from_str("Z_"), t.zs[0], {}},
                    Flow<W>{stim::PauliString<W>::from_str("_X"), t.xs[1], {}},
                    Flow<W>{stim::PauliString<W>::from_str("_Z"), t.zs[1], {}},
                };
            }
            return {
                Flow<W>{stim::PauliString<W>::from_str("X"), t.xs[0], {}},
                Flow<W>{stim::PauliString<W>::from_str("Z"), t.zs[0], {}},
            };
        }
        std::vector<Flow<W>> out;
        for (const auto &c : flow_data) {
            out.push_back(Flow<W>::from_str(c));
        }
        return out;
    }

    std::vector<std::vector<std::complex<float>>> unitary() const;

    bool is_symmetric() const;
    GateType hadamard_conjugated(bool ignoring_sign) const;

    /// Determines if the gate has a specified unitary matrix.
    ///
    /// Some unitary gates, such as SPP, don't have a specified matrix because the
    /// matrix depends crucially on the targets.
    bool has_known_unitary_matrix() const;

    /// Converts a single qubit unitary gate into an euler-angles rotation.
    ///
    /// Returns:
    ///     {theta, phi, lambda} using the same convention as qiskit.
    ///     Each angle is in radians.
    ///     For stabilizer operations, every angle will be a multiple of pi/2.
    ///
    ///     The unitary matrix U of the operation can be recovered (up to global phase)
    ///     by computing U = RotZ(phi) * RotY(theta) * RotZ(lambda).
    std::array<float, 3> to_euler_angles() const;

    /// Converts a single qubit unitary gate into an axis-angle rotation.
    ///
    /// Returns:
    ///     An array {x, y, z, a}.
    ///     <x, y, z> is a unit vector indicating the axis of rotation.
    ///     <a> is the angle of rotation in radians.
    std::array<float, 4> to_axis_angle() const;
};

inline bool _case_insensitive_mismatch(std::string_view text1, std::string_view text2) {
    if (text1.size() != text2.size()) {
        return true;
    }
    bool failed = false;
    for (size_t k = 0; k < text1.size(); k++) {
        failed |= toupper(text1[k]) != text2[k];
    }
    return failed;
}

struct GateDataMapHashEntry {
    GateType id = GateType::NOT_A_GATE;
    std::string_view expected_name;
};

struct GateDataMap {
   private:
    void add_gate(bool &failed, const Gate &data);
    void add_gate_alias(bool &failed, const char *alt_name, const char *canon_name);
    void add_gate_data_annotations(bool &failed);
    void add_gate_data_blocks(bool &failed);
    void add_gate_data_heralded(bool &failed);
    void add_gate_data_collapsing(bool &failed);
    void add_gate_data_controlled(bool &failed);
    void add_gate_data_hada(bool &failed);
    void add_gate_data_noisy(bool &failed);
    void add_gate_data_pauli(bool &failed);
    void add_gate_data_period_3(bool &failed);
    void add_gate_data_period_4(bool &failed);
    void add_gate_data_pp(bool &failed);
    void add_gate_data_swaps(bool &failed);
    void add_gate_data_pair_measure(bool &failed);
    void add_gate_data_pauli_product(bool &failed);

   public:
    std::array<GateDataMapHashEntry, 512> hashed_name_to_gate_type_table;
    std::array<Gate, NUM_DEFINED_GATES> items;
    GateDataMap();

    inline const Gate &operator[](GateType g) const {
        return items[(uint64_t)g];
    }

    inline const Gate &at(GateType g) const {
        if ((uint8_t)g >= items.size()) {
            throw std::out_of_range("Gate index out of range");
        }
        return items[(uint8_t)g];
    }

    inline const Gate &at(std::string_view text) const {
        auto h = gate_name_to_hash(text);
        const auto &entry = hashed_name_to_gate_type_table[h];
        if (_case_insensitive_mismatch(text, entry.expected_name)) {
            throw std::out_of_range("Gate not found: '" + std::string(text) + "'");
        }
        // Canonicalize.
        return (*this)[entry.id];
    }

    inline bool has(std::string_view text) const {
        auto h = gate_name_to_hash(text);
        const auto &entry = hashed_name_to_gate_type_table[h];
        return !_case_insensitive_mismatch(text, entry.expected_name);
    }
};

extern const GateDataMap GATE_DATA;

}  // namespace stim

#endif

```

---

End of reference.
