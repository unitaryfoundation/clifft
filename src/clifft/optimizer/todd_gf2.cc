#include "clifft/optimizer/todd_gf2.h"

#include "clifft/optimizer/pauli_axis.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace clifft {

namespace {

bool col_equal(const Gf2Matrix& mat, size_t a, size_t b) {
    const uint64_t* ca = mat.col(a);
    const uint64_t* cb = mat.col(b);
    for (uint32_t w = 0; w < mat.num_words; ++w) {
        if (ca[w] != cb[w])
            return false;
    }
    return true;
}

bool col_is_zero(const Gf2Matrix& mat, size_t j) {
    const uint64_t* c = mat.col(j);
    for (uint32_t w = 0; w < mat.num_words; ++w) {
        if (c[w] != 0)
            return false;
    }
    return true;
}

bool bit_at(const uint64_t* words, uint32_t q) {
    return ((words[q / 64] >> (q % 64)) & 1ULL) != 0;
}

void col_xor(const Gf2Matrix& mat, size_t a, size_t b, uint64_t* out) {
    const uint64_t* ca = mat.col(a);
    const uint64_t* cb = mat.col(b);
    for (uint32_t w = 0; w < mat.num_words; ++w) {
        out[w] = ca[w] ^ cb[w];
    }
}

int mod8(int x) {
    x %= 8;
    if (x < 0)
        x += 8;
    return x;
}

// Gaussian elimination: return nullspace basis columns as y vectors (length m).
std::vector<std::vector<uint8_t>> nullspace_basis(const std::vector<std::vector<uint8_t>>& rows) {
    if (rows.empty())
        return {};

    const size_t nrows = rows.size();
    const size_t ncols = rows[0].size();
    std::vector<std::vector<uint8_t>> a = rows;
    std::vector<size_t> pivot_col;
    std::vector<size_t> pivot_row;
    size_t r = 0;

    for (size_t c = 0; c < ncols && r < nrows; ++c) {
        size_t sel = r;
        for (size_t i = r + 1; i < nrows; ++i) {
            if (a[i][c])
                sel = i;
        }
        if (!a[sel][c])
            continue;
        std::swap(a[r], a[sel]);
        pivot_col.push_back(c);
        pivot_row.push_back(r);
        for (size_t i = 0; i < nrows; ++i) {
            if (i != r && a[i][c]) {
                for (size_t j = c; j < ncols; ++j) {
                    a[i][j] ^= a[r][j];
                }
            }
        }
        ++r;
    }

    std::vector<bool> is_pivot(ncols, false);
    for (size_t c : pivot_col)
        is_pivot[c] = true;

    std::vector<std::vector<uint8_t>> basis;
    for (size_t free = 0; free < ncols; ++free) {
        if (is_pivot[free])
            continue;
        std::vector<uint8_t> y(ncols, 0);
        y[free] = 1;
        for (size_t pi = 0; pi < pivot_col.size(); ++pi) {
            size_t pc = pivot_col[pi];
            size_t pr = pivot_row[pi];
            y[pc] = a[pr][free];
        }
        basis.push_back(std::move(y));
    }
    return basis;
}

void build_chi_rows(const Gf2Matrix& mat, const uint64_t* z_words, size_t m,
                    std::vector<std::vector<uint8_t>>& chi_rows) {
    chi_rows.clear();
    chi_rows.reserve(static_cast<size_t>(mat.n) * mat.n * mat.n);

    for (uint32_t alpha = 0; alpha < mat.n; ++alpha) {
        for (uint32_t beta = 0; beta < mat.n; ++beta) {
            for (uint32_t gamma = 0; gamma < mat.n; ++gamma) {
                std::vector<uint8_t> row(m, 0);
                const bool za = bit_at(z_words, alpha);
                const bool zb = bit_at(z_words, beta);
                const bool zg = bit_at(z_words, gamma);

                for (size_t j = 0; j < m; ++j) {
                    const uint64_t* col = mat.col(j);
                    const bool aa = bit_at(col, alpha);
                    const bool ab = bit_at(col, beta);
                    const bool ag = bit_at(col, gamma);
                    int temp = static_cast<int>(za) * static_cast<int>(zb) * static_cast<int>(zg);
                    temp += static_cast<int>(za) * static_cast<int>(zb) * static_cast<int>(ag);
                    temp += static_cast<int>(zb) * static_cast<int>(zg) * static_cast<int>(aa);
                    temp += static_cast<int>(zg) * static_cast<int>(za) * static_cast<int>(ab);
                    temp += static_cast<int>(za) * static_cast<int>(ab) * static_cast<int>(ag);
                    temp += static_cast<int>(zb) * static_cast<int>(ag) * static_cast<int>(aa);
                    temp += static_cast<int>(zg) * static_cast<int>(aa) * static_cast<int>(ab);
                    row[j] = static_cast<uint8_t>(temp & 1);
                }
                chi_rows.push_back(std::move(row));
            }
        }
    }
}

void apply_z_to_axes(std::vector<PauliAxis>& axes, const std::vector<uint8_t>& y,
                     const uint64_t* z_words, uint32_t num_words) {
    PauliAxis z_axis;
    z_axis.resize(num_words);
    for (uint32_t w = 0; w < num_words; ++w) {
        z_axis.x[w] = z_words[w];
    }
    MaskView zx{std::span<const uint64_t>(z_axis.x)};
    MaskView zz{std::span<const uint64_t>(z_axis.z)};
    for (size_t j = 0; j < y.size() && j < axes.size(); ++j) {
        if (y[j]) {
            MaskView ax{std::span<const uint64_t>(axes[j].x)};
            MaskView az{std::span<const uint64_t>(axes[j].z)};
            axes[j].xor_with(zx, zz);
        }
    }
}

bool todd_once(Gf2Matrix& mat, std::vector<int>& coeffs, std::vector<PauliAxis>* axes) {
    const size_t m = mat.num_cols();
    if (m < 2)
        return false;

    std::vector<uint64_t> z_words(mat.num_words, 0);
    std::vector<std::vector<uint8_t>> chi_rows;
    std::vector<uint64_t> anew(mat.cols.size());

    for (size_t j1 = 0; j1 + 1 < m; ++j1) {
        for (size_t j2 = j1 + 1; j2 < m; ++j2) {
            col_xor(mat, j1, j2, z_words.data());

            build_chi_rows(mat, z_words.data(), m, chi_rows);
            auto basis = nullspace_basis(chi_rows);
            if (basis.empty())
                continue;

            for (const auto& y : basis) {
                if (((y[j1] ^ y[j2]) & 1) == 0)
                    continue;

                anew.resize(m * mat.num_words);
                for (size_t j = 0; j < m; ++j) {
                    uint64_t* dst = &anew[j * mat.num_words];
                    const uint64_t* src = mat.col(j);
                    for (uint32_t w = 0; w < mat.num_words; ++w) {
                        dst[w] = src[w] ^ (y[j] ? z_words[w] : 0);
                    }
                }

                mat.cols = std::move(anew);
                if (axes != nullptr) {
                    apply_z_to_axes(*axes, y, z_words.data(), mat.num_words);
                }
                properize(mat, coeffs, axes);
                return true;
            }
        }
    }
    return false;
}

}  // namespace

void Gf2Matrix::resize(uint32_t n_qubits, size_t m) {
    n = n_qubits;
    num_words = (n + 63) / 64;
    cols.assign(m * num_words, 0);
}

void Gf2Matrix::append_col(const uint64_t* words) {
    cols.insert(cols.end(), words, words + num_words);
}

size_t properize(Gf2Matrix& mat, std::vector<int>& coeffs_mod8, std::vector<PauliAxis>* axes) {
    const size_t m = mat.num_cols();
    if (m == 0)
        return 0;

    std::vector<uint8_t> deleted(m, 0);
    for (size_t j1 = 0; j1 < m; ++j1) {
        if (deleted[j1])
            continue;
        for (size_t j2 = j1 + 1; j2 < m; ++j2) {
            if (deleted[j2])
                continue;
            if (col_equal(mat, j1, j2)) {
                coeffs_mod8[j1] = mod8(coeffs_mod8[j1] + coeffs_mod8[j2]);
                deleted[j2] = 1;
            }
        }
    }

    Gf2Matrix compact;
    compact.n = mat.n;
    compact.num_words = mat.num_words;
    std::vector<int> new_coeffs;
    std::vector<PauliAxis> new_axes;
    for (size_t j = 0; j < m; ++j) {
        if (deleted[j])
            continue;
        if (col_is_zero(mat, j))
            continue;
        int c = mod8(coeffs_mod8[j]);
        if (c == 0)
            continue;
        compact.append_col(mat.col(j));
        new_coeffs.push_back(c);
        if (axes != nullptr)
            new_axes.push_back((*axes)[j]);
    }
    mat = std::move(compact);
    coeffs_mod8 = std::move(new_coeffs);
    if (axes != nullptr)
        *axes = std::move(new_axes);
    return mat.num_cols();
}

bool todd_optimize(Gf2Matrix& mat, std::vector<int>& coeffs_mod8, uint32_t max_n, size_t max_m,
                   size_t max_rounds, std::vector<PauliAxis>* axes) {
    if (mat.n == 0 || mat.n > max_n)
        return false;
    if (mat.num_cols() > max_m)
        return false;

    properize(mat, coeffs_mod8, axes);
    size_t rounds = 0;
    while (rounds < max_rounds) {
        size_t before = mat.num_cols();
        if (!todd_once(mat, coeffs_mod8, axes))
            break;
        if (mat.num_cols() >= before)
            break;
        ++rounds;
    }
    return rounds > 0;
}

}  // namespace clifft
