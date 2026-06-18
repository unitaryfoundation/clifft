// Bit-packed CH-form stabilizer state (C++20), ported from the validated Python
// research/chform_backend/chform.py. Same conventions (arXiv:1808.00128 Sec 4.1,
// qubit 0 = LSB). The F/G/M rows are packed into 64-bit words so the row XORs in
// the gate updates and the Pauli accumulation in `amplitude` are word-level.
//
// This is the per-term hot path of the residual stab-rank backend; porting it to
// native bit-packed code turns the Python prototype's validated 2^{0.228 n} win
// into wall-clock. Standalone: no clifft / stim / network dependency.
#pragma once

#include <array>
#include <bit>
#include <complex>
#include <cstdint>
#include <vector>

namespace chf {

using cd = std::complex<double>;
inline constexpr double SQRT1_2 = 0.7071067811865476;
inline cd ipow(int k) {  // i^k, k mod 4
    static const cd t[4] = {cd(1, 0), cd(0, 1), cd(-1, 0), cd(0, -1)};
    return t[((k % 4) + 4) % 4];
}

struct CHForm {
    int n = 0;
    int W = 0;  // words per row = ceil(n/64)
    std::vector<uint64_t> F, G, M;  // n rows x W words, row p at [p*W .. p*W+W)
    std::vector<int8_t> g;          // gamma in Z_4
    std::vector<uint8_t> v, s;      // Hadamard layer, basis string
    cd w = cd(1, 0);

    explicit CHForm(int n_) : n(n_), W((n_ + 63) / 64), F(n_ * W, 0), G(n_ * W, 0),
                              M(n_ * W, 0), g(n_, 0), v(n_, 0), s(n_, 0), w(1, 0) {
        for (int q = 0; q < n; ++q) { bset(&F[q * W], q); bset(&G[q * W], q); }  // F=G=I
    }

    // -- bit helpers --
    static inline int bget(const uint64_t* r, int j) { return (r[j >> 6] >> (j & 63)) & 1; }
    static inline void bset(uint64_t* r, int j) { r[j >> 6] |= (1ULL << (j & 63)); }
    static inline void btog(uint64_t* r, int j) { r[j >> 6] ^= (1ULL << (j & 63)); }
    inline void rxor(uint64_t* d, const uint64_t* a) const { for (int i = 0; i < W; ++i) d[i] ^= a[i]; }
    inline int rdot(const uint64_t* a, const uint64_t* b) const {  // (a . b) mod 2
        int acc = 0; for (int i = 0; i < W; ++i) acc ^= (std::popcount(a[i] & b[i]) & 1);
        return acc & 1;
    }
    uint64_t* Fr(int p) { return &F[p * W]; }  const uint64_t* Fr(int p) const { return &F[p * W]; }
    uint64_t* Gr(int p) { return &G[p * W]; }  const uint64_t* Gr(int p) const { return &G[p * W]; }
    uint64_t* Mr(int p) { return &M[p * W]; }  const uint64_t* Mr(int p) const { return &M[p * W]; }

    void scale(cd c) { w *= c; }
    double norm2() const { return std::norm(w); }

    // ---- left-multiplication (apply gate to state): L[Gamma], row ops ----
    void Ls(int q) { rxor(Mr(q), Gr(q)); g[q] = (int8_t)(((g[q] - 1) % 4 + 4) % 4); }
    void Lcz(int q, int r) { rxor(Mr(q), Gr(r)); rxor(Mr(r), Gr(q)); }
    void Lcx(int q, int r) {  // control q, target r
        int mft = rdot(Mr(q), Fr(r));
        int gn = (((g[q] + g[r] + 2 * mft) % 4) + 4) % 4;
        rxor(Gr(r), Gr(q)); rxor(Fr(q), Fr(r)); rxor(Mr(q), Mr(r));
        g[q] = (int8_t)gn;
    }
    // ---- right-multiplication (used inside desuperposition): R[Gamma], col ops ----
    void Rs(int q) {
        for (int p = 0; p < n; ++p) if (bget(Fr(p), q)) { btog(Mr(p), q); g[p] = (int8_t)(((g[p] - 1) % 4 + 4) % 4); }
    }
    void Rcz(int q, int r) {
        for (int p = 0; p < n; ++p) {
            int fq = bget(Fr(p), q), fr = bget(Fr(p), r);
            if (fr) btog(Mr(p), q);
            if (fq) btog(Mr(p), r);
            if (fq && fr) g[p] = (int8_t)((g[p] + 2) % 4);
        }
    }
    void Rcx(int q, int r) {  // control q, target r
        for (int p = 0; p < n; ++p) {
            if (bget(Gr(p), r)) btog(Gr(p), q);
            if (bget(Fr(p), q)) btog(Fr(p), r);
            if (bget(Mr(p), r)) btog(Mr(p), q);
        }
    }

    // ---- public Clifford gates ----
    void s_gate(int q) { Ls(q); }
    void s_dag(int q) { Ls(q); Ls(q); Ls(q); }
    void z(int q) { Ls(q); Ls(q); }
    void cz(int q, int r) { Lcz(q, r); }
    void cx(int q, int r) { Lcx(q, r); }
    void h(int q) { apply_h(q); }
    void x(int q) { h(q); z(q); h(q); }
    void y(int q) { z(q); x(q); w *= cd(0, 1); }
    void swap(int a, int b) { cx(a, b); cx(b, a); cx(a, b); }

    // ---- Proposition 4: desuperposition of U_H(|t> + i^delta |u>) ----
    // Mutates F/G/M/g (R[] right-mults by W_C), v, s; returns omega_P4 (|.|=sqrt2).
    cd desuperpose(const std::vector<uint8_t>& t, const std::vector<uint8_t>& u, int delta) {
        std::vector<int> V0, V1;
        for (int i = 0; i < n; ++i) if (t[i] != u[i]) (v[i] ? V1 : V0).push_back(i);
        int q;
        std::vector<std::array<int, 3>> wc;  // {kind(0=CX,1=CZ), a, b}
        if (!V0.empty()) {
            q = V0[0];
            for (int i : V0) if (i != q) wc.push_back({0, q, i});
            for (int i : V1) wc.push_back({1, q, i});
        } else {
            q = V1[0];
            for (int i : V1) if (i != q) wc.push_back({0, i, q});
        }
        std::vector<uint8_t> y(n), zz(n);
        if (t[q] == 1) { y = u; y[q] ^= 1; zz = u; }
        else { y = t; zz = t; zz[q] ^= 1; }
        // single-qubit residual on q: H^{v_q}(|y_q> + i^delta |z_q>) = w_q S^a H^b |c>
        cd ket[2] = {cd(0, 0), cd(0, 0)};
        ket[y[q]] += cd(1, 0);
        ket[zz[q]] += ipow(delta);
        if (v[q]) { cd k0 = SQRT1_2 * (ket[0] + ket[1]), k1 = SQRT1_2 * (ket[0] - ket[1]); ket[0] = k0; ket[1] = k1; }
        int A = 0, B = 0, C = 0; cd omega;
        match_single_qubit(ket, A, B, C, omega);
        for (auto& gt : wc) { if (gt[0] == 0) Rcx(gt[1], gt[2]); else Rcz(gt[1], gt[2]); }
        for (int i = 0; i < A; ++i) Rs(q);
        v[q] = (uint8_t)B;
        s = y; s[q] = (uint8_t)C;
        return omega;
    }

    // find (a in Z4, b,c in {0,1}, omega) with ket == omega * S^a H^b |c>
    static void match_single_qubit(const cd ket[2], int& A, int& B, int& C, cd& omega) {
        const cd H2[2][2] = {{SQRT1_2, SQRT1_2}, {SQRT1_2, -SQRT1_2}};
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c) {
                cd base[2] = {cd(0, 0), cd(0, 0)}; base[c] = cd(1, 0);
                if (b) { cd t0 = H2[0][0] * base[0] + H2[0][1] * base[1], t1 = H2[1][0] * base[0] + H2[1][1] * base[1]; base[0] = t0; base[1] = t1; }
                for (int a = 0; a < 4; ++a) {
                    cd vec[2] = {base[0], base[1] * ipow(a)};
                    cd om = std::conj(vec[0]) * ket[0] + std::conj(vec[1]) * ket[1];
                    if (std::abs(ket[0] - om * vec[0]) < 1e-9 && std::abs(ket[1] - om * vec[1]) < 1e-9) {
                        A = a; B = b; C = c; omega = om; return;
                    }
                }
            }
        // unreachable for a valid single-qubit stabilizer state
    }

    void apply_h(int p) {
        std::vector<uint8_t> t(n), u(n);
        int alpha = 0, beta = 0;
        for (int j = 0; j < n; ++j) {
            int Gp = bget(Gr(p), j), Fp = bget(Fr(p), j), Mp = bget(Mr(p), j);
            int vb = v[j], sb = s[j];
            t[j] = (uint8_t)(sb ^ (Gp & vb));
            u[j] = (uint8_t)(sb ^ (Fp & (1 - vb)) ^ (Mp & vb));
            alpha ^= (Gp & (1 - vb) & sb);
            beta ^= ((Mp & (1 - vb) & sb) ^ ((Fp & vb) & (Mp ^ sb)));
        }
        int gp = ((g[p] % 4) + 4) % 4;
        bool same = true; for (int j = 0; j < n; ++j) if (t[j] != u[j]) { same = false; break; }
        if (same) {
            w *= SQRT1_2 * ((alpha & 1 ? -1.0 : 1.0) + ipow(gp) * (beta & 1 ? -1.0 : 1.0));
            s = t; return;
        }
        int delta = (((gp + 2 * (alpha + beta)) % 4) + 4) % 4;
        cd omega_q = desuperpose(t, u, delta);
        w *= SQRT1_2 * (alpha & 1 ? -1.0 : 1.0) * omega_q;
    }

    // ---- single-qubit Z-basis projection: (I + (-1)^bit Z_q)/2 ----
    // Returns true if nonzero (state updated in place), false if zero support.
    bool project(int q, int bit) {
        // Q = U_H^{-1} U_C^{-1} P U_C U_H, P=(-1)^bit Z_q -> X(G[q]&v) Z(G[q]&~v)
        std::vector<uint8_t> a(n), b(n), tt(n);
        int bs = 0;  // b . s
        for (int j = 0; j < n; ++j) {
            int Gp = bget(Gr(q), j);
            a[j] = (uint8_t)(Gp & v[j]);
            b[j] = (uint8_t)(Gp & (1 - v[j]));
            bs ^= (b[j] & s[j]);
            tt[j] = (uint8_t)(s[j] ^ a[j]);
        }
        int parity = (bit + bs) & 1;
        bool anyA = false; for (int j = 0; j < n; ++j) if (a[j]) { anyA = true; break; }
        if (!anyA) {           // deterministic outcome for this qubit
            if (parity) return false;     // (I+Q)|s> = 0
            return true;                  // full weight, w unchanged ((w/2)*2)
        }
        int delta = (2 * parity) % 4;
        std::vector<uint8_t> sc = s;  // copy current s as first branch string
        cd omega_q = desuperpose(sc, tt, delta);
        w *= 0.5 * omega_q;
        return true;
    }

    // ---- amplitude <x|phi> (Eq. 56), Pauli-product accumulation ----
    cd amplitude(uint64_t x) const {
        std::vector<uint64_t> ux(W, 0);
        int mu = 0;
        for (int p = 0; p < n; ++p) {
            if (!((x >> p) & 1)) continue;
            const uint64_t* Fp = Fr(p); const uint64_t* Mp = Mr(p);
            // mu += g_p + 2 * ((ux ^ Fp) . Mp)
            int acc = 0; for (int i = 0; i < W; ++i) acc ^= (std::popcount((ux[i] ^ Fp[i]) & Mp[i]) & 1);
            mu = (mu + g[p] + 2 * (acc & 1)) & 3;
            for (int i = 0; i < W; ++i) ux[i] ^= Fp[i];
        }
        int vcount = 0;
        for (int j = 0; j < n; ++j) {
            int uj = (int)((ux[j >> 6] >> (j & 63)) & 1);
            if (v[j] == 0) { if (uj != s[j]) return cd(0, 0); }
            else ++vcount;
        }
        cd amp = w * std::pow(SQRT1_2, vcount) * ipow(((mu % 4) + 4) % 4);
        int hsign = 0;
        for (int j = 0; j < n; ++j) if (v[j]) hsign ^= (((ux[j >> 6] >> (j & 63)) & 1) & s[j]);
        return hsign ? -amp : amp;
    }

    std::vector<cd> statevector() const {
        std::vector<cd> out((size_t)1 << n);
        for (uint64_t x = 0; x < (1ULL << n); ++x) out[x] = amplitude(x);
        return out;
    }

    void clifford_1q(int which, int q) {  // 0=H 1=S 2=Sdag 3=X 4=Y 5=Z
        switch (which) { case 0: h(q); break; case 1: s_gate(q); break; case 2: s_dag(q); break;
                         case 3: x(q); break; case 4: y(q); break; case 5: z(q); break; }
    }
};

}  // namespace chf
