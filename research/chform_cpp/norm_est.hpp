// Native (non-materialising) norm estimation in C++ -- arXiv:1808.00128 Sec 4.3,
// Lemmas 2/3/4. Ported from the validated Python research/chform_backend/norm_est.py.
//
//   exp_sum(B)           Lemma 4: Z(B) = sum_x i^{x B x^T} in O(m^3)
//   inner_equatorial     Lemma 3: <phi | phi_A> for an equatorial state phi_A
//   estimate_norm2       Lemma 2: ||sum_a term_a||^2 by averaging 2^n |<phi_A|psi>|^2
//
// Gives output probabilities P(x) = |<x|psi>|^2 / ||psi||^2 with no 2^n vector.
#pragma once

#include "chform.hpp"

#include <random>
#include <vector>

namespace chf {

using imat = std::vector<std::vector<int>>;

// brute-force exponential sum (validation oracle): B symmetric, diag Z4, off Z2
inline cd exp_sum_direct(const imat& B) {
    int m = (int)B.size();
    cd tot(0, 0);
    for (uint64_t xb = 0; xb < (1ULL << m); ++xb) {
        int e = 0;
        for (int a = 0; a < m; ++a) if ((xb >> a) & 1) {
            e += B[a][a];
            for (int b = a + 1; b < m; ++b) if ((xb >> b) & 1) e += 2 * B[a][b];
        }
        tot += ipow(e);
    }
    return tot;
}

// Bit-packed GF2 exponential sum: sum_x (-1)^{x M x^T + L x^T}. M is given as
// packed rows (n rows x W words, bit c of row r = M[r][c]); L binary. In-place,
// zero per-step heap allocation, row updates are word-level XOR -- the ~64x /
// no-alloc win over the scalar `real_exp_sum` (kept below as the oracle).
inline long long real_exp_sum_packed(std::vector<uint64_t>& M, std::vector<int>& L, int n) {
    int W = (n + 63) / 64;
    auto bget = [&](int r, int c) { return (int)((M[(size_t)r * W + (c >> 6)] >> (c & 63)) & 1); };
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    int cur = n;
    long long factor = 1;
    std::vector<uint64_t> m1(W), m2(W);
    while (true) {
        if (cur == 0) return factor;
        int pa = -1, pb = -1;
        for (int a = 0; a < cur && pa < 0; ++a)
            for (int b = a + 1; b < cur; ++b)
                if (bget(idx[a], idx[b]) != bget(idx[b], idx[a])) { pa = a; pb = b; break; }
        if (pa < 0) {  // symmetric -> linear form (diag(M)+L)
            for (int a = 0; a < cur; ++a) { int i = idx[a]; if ((bget(i, i) ^ L[i]) & 1) return 0; }
            return factor * (1LL << cur);
        }
        int i = idx[pa], j = idx[pb];
        int ai = (L[i] ^ bget(i, i)) & 1, aj = (L[j] ^ bget(j, j)) & 1;
        for (int wi = 0; wi < W; ++wi) { m1[wi] = 0; m2[wi] = 0; }
        for (int b = 0; b < cur; ++b) {
            int k = idx[b]; if (k == i || k == j) continue;
            if (bget(i, k) ^ bget(k, i)) m1[k >> 6] |= (1ULL << (k & 63));
            if (bget(j, k) ^ bget(k, j)) m2[k >> 6] |= (1ULL << (k & 63));
        }
        for (int b = 0; b < cur; ++b) {
            int k = idx[b]; if (k == i || k == j) continue;
            int m1k = (int)((m1[k >> 6] >> (k & 63)) & 1);
            int m2k = (int)((m2[k >> 6] >> (k & 63)) & 1);
            if (m1k) { uint64_t* rk = &M[(size_t)k * W]; for (int wi = 0; wi < W; ++wi) rk[wi] ^= m2[wi]; }
            L[k] ^= (aj & m1k) ^ (ai & m2k);
        }
        factor *= 2 * ((ai & aj) ? -1 : 1);
        idx[pb] = idx[cur - 1]; --cur;       // remove the two pivot indices (order-free)
        idx[pa] = idx[cur - 1]; --cur;
    }
}

// scalar reference (validation oracle): same sum, O(n^3) with per-step allocation.
inline long long real_exp_sum(imat M, std::vector<int> L) {
    int n = (int)L.size();
    long long factor = 1;
    while (true) {
        if (n == 0) return factor;
        int pi = -1, pj = -1;
        for (int i = 0; i < n && pi < 0; ++i)
            for (int j = i + 1; j < n; ++j)
                if (M[i][j] != M[j][i]) { pi = i; pj = j; break; }
        if (pi < 0) {  // symmetric -> linear form (diag(M)+L)
            for (int a = 0; a < n; ++a) if ((M[a][a] + L[a]) & 1) return 0;
            return factor * (1LL << n);
        }
        int i = pi, j = pj;
        int Mii = M[i][i], Mjj = M[j][j], Li = L[i], Lj = L[j];
        std::vector<int> oth;
        for (int k = 0; k < n; ++k) if (k != i && k != j) oth.push_back(k);
        int no = (int)oth.size();
        std::vector<int> m1(no), m2(no);
        for (int a = 0; a < no; ++a) {
            m1[a] = (M[i][oth[a]] + M[oth[a]][i]) & 1;
            m2[a] = (M[j][oth[a]] + M[oth[a]][j]) & 1;
        }
        int ai = (Li + Mii) & 1, aj = (Lj + Mjj) & 1;
        imat Mnew(no, std::vector<int>(no));
        std::vector<int> Lnew(no);
        for (int a = 0; a < no; ++a) {
            for (int b = 0; b < no; ++b) Mnew[a][b] = (M[oth[a]][oth[b]] + m1[a] * m2[b]) & 1;
            Lnew[a] = (L[oth[a]] + aj * m1[a] + ai * m2[a]) & 1;
        }
        factor *= 2 * ((ai & aj) ? -1 : 1);
        M = std::move(Mnew); L = std::move(Lnew); n = no;
    }
}

// Lemma 4: Z(B) = sum_x i^{x B x^T}, B symmetric (diag Z4, off Z2), O(m^3).
// Builds the Z2 form (Prop.6) as packed rows and uses the bit-packed sum.
inline cd exp_sum(const imat& B) {
    int n = (int)B.size();
    if (n == 0) return cd(1, 0);
    std::vector<int> Kv(n), Lv(n);
    for (int a = 0; a < n; ++a) { int d = ((B[a][a] % 4) + 4) % 4; Kv[a] = d & 1; Lv[a] = (d >> 1) & 1; }
    int N = n + 1, Wn = (N + 63) / 64;
    std::vector<uint64_t> QM((size_t)N * Wn, 0);
    auto setbit = [&](int r, int c) { QM[(size_t)r * Wn + (c >> 6)] |= (1ULL << (c & 63)); };
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) if ((B[a][b] + Kv[a] * Kv[b]) & 1) setbit(a, b);
        if (Kv[a] & 1) setbit(a, n);
    }
    std::vector<int> QL(N, 0);
    for (int a = 0; a < n; ++a) QL[a] = Lv[a];
    std::vector<uint64_t> QMc = QM; std::vector<int> QLc = QL;
    double re = real_exp_sum_packed(QMc, QLc, N) / 2.0;
    QMc = QM; QLc = QL; QLc[n] ^= 1;
    double im = real_exp_sum_packed(QMc, QLc, N) / 2.0;
    return cd(re, im);
}

inline imat random_equatorial(int n, std::mt19937_64& rng) {
    imat A(n, std::vector<int>(n, 0));
    for (int a = 0; a < n; ++a) {
        A[a][a] = (int)(rng() & 3);
        for (int b = a + 1; b < n; ++b) { int v = (int)(rng() & 1); A[a][b] = v; A[b][a] = v; }
    }
    return A;
}

// Lemma 3: <phi | phi_A>, including phi.w (returns conj(w) * value).
// K = G^T (A+J) G decomposed as G^T Off G (GF2, bit-packed) + G^T Diag G (Z4):
//   K[a][b] mod 2 = col_a^T Off col_b  ^  parity(col_a & col_b & Diag_odd)
//   K[a][a] mod 4 = (sum_{i,j in col_a} Off[i][j])  +  sum_{i in col_a} Diag[i]
//   sK[c]         = sum_{j in col_c} z[j],  z = (A+J)(G s)   -- O(n^2), no full K.
inline cd inner_equatorial(const CHForm& phi, const imat& A) {
    int n = phi.n, W = phi.W;
    auto pcand = [&](const uint64_t* x, const uint64_t* y) {
        int c = 0; for (int w = 0; w < W; ++w) c += std::popcount(x[w] & y[w]); return c; };
    auto par = [&](const uint64_t* x, const uint64_t* y) {
        int c = 0; for (int w = 0; w < W; ++w) c ^= (std::popcount(x[w] & y[w]) & 1); return c & 1; };

    // Off (GF2 symmetric, zero diag) and Diag (Z4) of A + J,  J = M F^T.
    std::vector<uint64_t> Off((size_t)n * W, 0);
    std::vector<int> Diag(n);
    for (int a = 0; a < n; ++a) {
        Diag[a] = ((A[a][a] + ((phi.g[a] % 4) + 4) % 4) % 4 + 4) % 4;
        uint64_t* Oa = &Off[(size_t)a * W];
        for (int b = a + 1; b < n; ++b)
            if ((A[a][b] + phi.rdot(phi.Mr(a), phi.Fr(b))) & 1) {
                Oa[b >> 6] |= 1ULL << (b & 63); Off[(size_t)b * W + (a >> 6)] |= 1ULL << (a & 63);
            }
    }
    // columns of G, plus packed s and Diag-odd masks
    std::vector<uint64_t> Gcol((size_t)n * W, 0), sp(W, 0), dodd(W, 0);
    for (int i = 0; i < n; ++i) {
        const uint64_t* Gi = phi.Gr(i);
        for (int w = 0; w < W; ++w) { uint64_t bb = Gi[w];
            while (bb) { int c = w * 64 + std::countr_zero(bb); Gcol[(size_t)c * W + (i >> 6)] |= 1ULL << (i & 63); bb &= bb - 1; } }
        if (phi.s[i]) sp[i >> 6] |= 1ULL << (i & 63);
        if (Diag[i] & 1) dodd[i >> 6] |= 1ULL << (i & 63);
    }
    auto col = [&](int c) { return &Gcol[(size_t)c * W]; };

    // sK[c] = sum_{j in col_c} z[j],  y = G s,  z = (A+J) y          (O(n^2), mod 4)
    std::vector<long long> y(n), z(n), sK(n, 0);
    for (int i = 0; i < n; ++i) y[i] = pcand(phi.Gr(i), sp.data());
    for (int j = 0; j < n; ++j) {
        long long zz = (long long)Diag[j] * y[j];
        const uint64_t* Oj = &Off[(size_t)j * W];
        for (int w = 0; w < W; ++w) { uint64_t bb = Oj[w]; while (bb) { int i = w * 64 + std::countr_zero(bb); zz += y[i]; bb &= bb - 1; } }
        z[j] = zz;
    }
    for (int j = 0; j < n; ++j) { long long zj = z[j]; if (!zj) continue;
        const uint64_t* Gj = phi.Gr(j);
        for (int w = 0; w < W; ++w) { uint64_t bb = Gj[w]; while (bb) { int c = w * 64 + std::countr_zero(bb); sK[c] += zj; bb &= bb - 1; } } }
    long long sKsT = 0; for (int c = 0; c < n; ++c) if (phi.s[c]) sKsT += sK[c];

    // B matrix from the decomposed K
    std::vector<int> vidx; for (int j = 0; j < n; ++j) if (phi.v[j]) vidx.push_back(j);
    int m = (int)vidx.size();
    std::vector<uint64_t> Ocol((size_t)m * W, 0);  // Off * col(vidx[bi])  (GF2)
    for (int bi = 0; bi < m; ++bi) {
        const uint64_t* cb = col(vidx[bi]); uint64_t* dst = &Ocol[(size_t)bi * W];
        for (int i = 0; i < n; ++i) if (par(&Off[(size_t)i * W], cb)) dst[i >> 6] |= 1ULL << (i & 63);
    }
    imat Bm(m, std::vector<int>(m, 0));
    for (int ai = 0; ai < m; ++ai) {
        int a = vidx[ai]; const uint64_t* ca = col(a);
        long long Kaa = 0;                                   // = sum_{i in col_a} (pcand(Off_i, col_a) + Diag[i])
        for (int w = 0; w < W; ++w) { uint64_t bb = ca[w]; while (bb) { int i = w * 64 + std::countr_zero(bb);
            Kaa += pcand(&Off[(size_t)i * W], ca) + Diag[i]; bb &= bb - 1; } }
        Bm[ai][ai] = (int)(((Kaa + 2 * (phi.s[a] + sK[a])) % 4 + 4) % 4);
        for (int bi = ai + 1; bi < m; ++bi) {
            const uint64_t* cb = col(vidx[bi]);
            int off2 = par(ca, &Ocol[(size_t)bi * W]);
            int diag2 = 0; for (int w = 0; w < W; ++w) diag2 ^= (std::popcount(ca[w] & cb[w] & dodd[w]) & 1);
            int v = (off2 ^ diag2) & 1; Bm[ai][bi] = v; Bm[bi][ai] = v;
        }
    }
    cd Z = exp_sum(Bm);
    int sKsT4 = (int)((sKsT % 4 + 4) % 4);
    int sv = 0; for (int j = 0; j < n; ++j) sv ^= (phi.s[j] & phi.v[j]); sv &= 1;
    cd val = ipow(sKsT4) * (sv ? -1.0 : 1.0) * Z * std::pow(SQRT1_2, n + m);
    return std::conj(phi.w) * val;
}

// Lemma 2: estimate ||sum_a term_a||^2 over `samples` random equatorial A.
inline double estimate_norm2(const std::vector<CHForm>& terms, int n, int samples,
                             std::mt19937_64& rng) {
    double acc = 0.0;
    double two_n = std::pow(2.0, n);
    for (int t = 0; t < samples; ++t) {
        imat A = random_equatorial(n, rng);
        cd inner(0, 0);
        for (const auto& term : terms) inner += inner_equatorial(term, A);
        acc += two_n * std::norm(inner);
    }
    return acc / samples;
}

}  // namespace chf
