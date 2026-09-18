// Fixed-plan CSS syndrome contraction microbenchmark, not a full MSC executor.
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
static size_t allocations = 0;
void* operator new(size_t n) {
    ++allocations;
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](size_t n) {
    return ::operator new(n);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, size_t) noexcept {
    std::free(p);
}
#endif
using C = std::complex<double>;
constexpr unsigned MAX_WIDTH = 128, MAX_TERMS = 4;
const std::array<C, 8> roots{C{1},  C{M_SQRT1_2, M_SQRT1_2},   C{0, 1},  C{-M_SQRT1_2, M_SQRT1_2},
                             C{-1}, C{-M_SQRT1_2, -M_SQRT1_2}, C{0, -1}, C{M_SQRT1_2, -M_SQRT1_2}};
void require(bool yes) {
    if (!yes)
        throw std::invalid_argument("invalid bounded factor plan");
}
struct Reader {
    std::ifstream stream;
    explicit Reader(const char* path) : stream(path) { require(bool(stream)); }
    template <class T = unsigned>
    T get() {
        T v;
        if (!(stream >> v))
            throw std::invalid_argument("incomplete factor plan");
        return v;
    }
    unsigned count() {
        unsigned n = get();
        require(n <= 2000000);
        return n;
    }
    std::vector<unsigned> ints() {
        std::vector<unsigned> result(count());
        for (auto& v : result)
            v = get();
        return result;
    }
    C complex() {
        auto a = get<double>(), b = get<double>();
        require(std::isfinite(a) && std::isfinite(b));
        return {a, b};
    }
};
struct Leaf {
    unsigned parameter, offset;
    std::vector<unsigned> parity;
};
struct Step {
    unsigned offset, size, inputs;
    std::vector<unsigned> gathers;
};
struct Plan {
    unsigned storage;
    std::vector<Leaf> leaves;
    std::vector<Step> steps;
    std::vector<unsigned> outputs;
    explicit Plan(Reader& r, unsigned parameters) {
        storage = r.count();
        leaves.resize(r.count());
        for (auto& leaf : leaves) {
            leaf.parameter = r.get();
            leaf.offset = r.get();
            leaf.parity = r.ints();
            require(leaf.parameter < parameters && leaf.offset <= storage &&
                    leaf.parity.size() <= storage - leaf.offset);
            for (auto b : leaf.parity)
                require(b < 2);
        }
        steps.resize(r.count());
        for (auto& step : steps) {
            step.offset = r.get();
            step.size = r.count();
            step.inputs = r.count();
            require(step.offset <= storage && step.size <= storage - step.offset && step.size > 0 &&
                    step.inputs > 0);
            require(uint64_t{2} * step.size * step.inputs <= 2000000);
            step.gathers.resize(2 * step.size * step.inputs);
            for (auto& index : step.gathers) {
                index = r.get();
                require(index < step.offset);
            }
        }
        outputs = r.ints();
        for (auto index : outputs)
            require(index < storage);
    }
    C evaluate(const std::array<unsigned, 3 * MAX_WIDTH>& parameters, C* scratch) const noexcept {
        for (const auto& leaf : leaves) {
            C phase = roots[parameters[leaf.parameter] & 7];
            for (unsigned j = 0; j < leaf.parity.size(); ++j)
                scratch[leaf.offset + j] = leaf.parity[j] ? phase : C{1};
        }
        for (const auto& step : steps)
            for (unsigned j = 0; j < step.size; ++j) {
                C a{1}, b{1};
                for (unsigned k = 0; k < step.inputs; ++k) {
                    unsigned offset = 2 * step.size * k + 2 * j;
                    a *= scratch[step.gathers[offset]];
                    b *= scratch[step.gathers[offset + 1]];
                }
                scratch[step.offset + j] = .5 * (a + b);
            }
        C result{1};
        for (auto offset : outputs)
            result *= scratch[offset];
        return result;
    }
};
struct Case {
    unsigned terms;
    std::vector<unsigned> labels;
    std::array<std::array<C, 2>, MAX_TERMS> amplitude{};
    std::array<std::array<std::array<unsigned, MAX_WIDTH>, 2>, MAX_TERMS> phase{};
    std::array<std::array<C, MAX_TERMS>, MAX_TERMS> gram{};
    std::array<std::array<std::array<C, MAX_TERMS>, MAX_TERMS>, 2> coherence{};
    Case(Reader& r, unsigned width, unsigned rank) {
        terms = r.get();
        require(terms > 0 && terms <= MAX_TERMS);
        labels = r.ints();
        require(labels.size() == terms);
        for (auto label : labels)
            require(label < (uint64_t{1} << rank));
        for (unsigned t = 0; t < terms; ++t)
            for (unsigned l = 0; l < 2; ++l) {
                amplitude[t][l] = r.complex();
                for (unsigned q = 0; q < width; ++q) {
                    phase[t][l][q] = r.get();
                    require(phase[t][l][q] < 8);
                }
            }
        for (unsigned a = 0; a < terms; ++a)
            for (unsigned b = 0; b < terms; ++b)
                gram[a][b] = r.complex();
        for (unsigned l = 0; l < 2; ++l)
            for (unsigned a = 0; a < terms; ++a)
                for (unsigned b = 0; b < terms; ++b)
                    coherence[l][a][b] = gram[a][b] * std::conj(amplitude[a][l]) * amplitude[b][l];
    }
};
struct Sample {
    unsigned xs, zs;
    double probability;
};
struct Worker {
    unsigned width, rank;
    std::vector<unsigned> order, supports;
    std::vector<Plan> prefixes;
    std::vector<Case> cases;
    std::vector<C> scratch, tensor;
    std::vector<double> weights;
    std::mt19937_64 rng;
    Worker(Reader& r, bool fourier) {
        width = r.get();
        rank = r.get();
        unsigned count = r.count();
        require(width <= MAX_WIDTH && rank < 32 && rank < width && count > 0);
        order = r.ints();
        supports = r.ints();
        require(order.size() == rank && supports.size() == width);
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        for (unsigned k = 0; k < rank; ++k)
            require(sorted[k] == k);
        for (auto support : supports)
            require(support < (uint64_t{1} << rank));
        unsigned storage = 0;
        for (unsigned m = 0; m <= rank; ++m) {
            prefixes.emplace_back(r, 2 * width + rank);
            storage = std::max(storage, prefixes.back().storage);
        }
        scratch.resize(storage);
        for (unsigned k = 0; k < count; ++k)
            cases.emplace_back(r, width, rank);
        if (fourier) {
            require(rank <= 18);
            tensor.resize(size_t{8} << rank);
            weights.resize(size_t{4} << rank);
        }
    }
    double mass(const Case& data, unsigned label, unsigned count, unsigned syndrome) noexcept {
        std::array<unsigned, 3 * MAX_WIDTH> parameters{};
        C result{};
        for (unsigned k = 0; k < rank; ++k)
            parameters[2 * width + k] = 4 * ((syndrome >> k) & 1);
        for (unsigned a = 0; a < data.terms; ++a)
            if (data.labels[a] == label)
                for (unsigned b = a; b < data.terms; ++b)
                    if (data.labels[b] == label)
                        for (unsigned l = 0; l < 2; ++l) {
                            C coefficient = data.coherence[l][a][b];
                            if (coefficient == C{})
                                continue;
                            for (unsigned q = 0; q < width; ++q) {
                                parameters[q] = (8 - data.phase[a][l][q]) & 7;
                                parameters[width + q] = data.phase[b][l][q];
                            }
                            C value =
                                coefficient * prefixes[count].evaluate(parameters, scratch.data());
                            result += a == b ? value : C{2 * value.real()};
                        }
        assert(std::abs(result.imag()) < 1e-10 && result.real() > -1e-10);
        return std::max(0., result.real());
    }
    std::pair<unsigned, double> draw(const double* weights, unsigned count) noexcept {
        double total = 0;
        for (unsigned k = 0; k < count; ++k) {
            assert(std::isfinite(weights[k]) && weights[k] >= -1e-10);
            total += std::max(0., weights[k]);
        }
        assert(total > 0);
        double threshold = ((rng() >> 11) * 0x1.0p-53) * total, sum = 0;
        for (unsigned k = 0; k < count; ++k) {
            sum += std::max(0., weights[k]);
            if (sum > threshold)
                return {k, weights[k] / total};
        }
        assert(false);
        return {0, 0};
    }
    Sample factor(const Case& data) noexcept {
        std::array<double, MAX_TERMS> weights{};
        for (unsigned a = 0; a < data.terms; ++a) {
            bool seen = false;
            for (unsigned b = 0; b < a; ++b)
                seen |= data.labels[a] == data.labels[b];
            if (!seen)
                weights[a] = mass(data, data.labels[a], 0, 0);
        }
        auto [selected, chance] = draw(weights.data(), data.terms);
        (void)chance;
        double current = weights[selected], total = 0;
        for (auto w : weights)
            total += w;
        unsigned syndrome = 0;
        for (unsigned m = 0; m < rank; ++m) {
            double zero = mass(data, data.labels[selected], m + 1, syndrome);
            assert(zero < current + 1e-10 * total);
            zero = std::min(current, zero);
            std::array<double, 2> conditional{zero, current - zero};
            unsigned bit = draw(conditional.data(), 2).first;
            syndrome |= bit << order[m];
            current = conditional[bit];
        }
        return {syndrome, data.labels[selected], current / total};
    }
    Sample fourier(const Case& data) noexcept {
        unsigned n = 1u << rank;
        const double scale = 1 / std::sqrt(double(n));
        for (unsigned a = 0; a < data.terms; ++a)
            for (unsigned l = 0; l < 2; ++l) {
                C* values = tensor.data() + (2 * a + l) * n;
                for (unsigned u = 0; u < n; ++u) {
                    unsigned phase = 0;
                    for (unsigned q = 0; q < width; ++q)
                        phase += data.phase[a][l][q] * (std::popcount(u & supports[q]) & 1);
                    values[u] = data.amplitude[a][l] * roots[phase & 7] * scale;
                }
                for (unsigned stride = 1; stride < n; stride *= 2)
                    for (unsigned base = 0; base < n; base += 2 * stride)
                        for (unsigned j = 0; j < stride; ++j) {
                            C a = values[base + j], b = values[base + stride + j];
                            values[base + j] = (a + b) * M_SQRT1_2;
                            values[base + stride + j] = (a - b) * M_SQRT1_2;
                        }
            }
        std::fill(weights.begin(), weights.end(), 0);
        for (unsigned a = 0; a < data.terms; ++a) {
            bool seen = false;
            for (unsigned b = 0; b < a; ++b)
                seen |= data.labels[a] == data.labels[b];
            if (seen)
                continue;
            for (unsigned b = 0; b < data.terms; ++b)
                if (data.labels[b] == data.labels[a])
                    for (unsigned c = 0; c < data.terms; ++c)
                        if (data.labels[c] == data.labels[a])
                            for (unsigned j = 0; j < n; ++j)
                                for (unsigned l = 0; l < 2; ++l)
                                    weights[a * n + j] +=
                                        (data.gram[b][c] * std::conj(tensor[(2 * b + l) * n + j]) *
                                         tensor[(2 * c + l) * n + j])
                                            .real();
        }
        auto [selected, probability] = draw(weights.data(), data.terms * n);
        return {selected % n, data.labels[selected / n], probability};
    }
};
int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "usage: msc_factor_native PLAN factor|fourier|sample SHOTS SEED");
        std::string mode(argv[2]);
        require(mode == "factor" || mode == "fourier" || mode == "sample");
        Reader r(argv[1]);
        Worker worker(r, mode == "fourier");
        unsigned shots = std::stoul(argv[3]);
        require(shots > 0);
        worker.rng.seed(std::stoull(argv[4]));
        double checksum = 0;
        auto start = std::chrono::steady_clock::now();
        for (unsigned k = 0; k < shots; ++k) {
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            auto before = allocations;
#endif
            const auto& data = worker.cases[k % worker.cases.size()];
            Sample sample = mode == "fourier" ? worker.fourier(data) : worker.factor(data);
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            assert(allocations == before);
#endif
            checksum += sample.probability + sample.xs + sample.zs;
            if (mode == "sample")
                std::cout << std::setprecision(17) << "{\"case\":" << k % worker.cases.size()
                          << ",\"xs\":" << sample.xs << ",\"zs\":" << sample.zs
                          << ",\"probability\":" << sample.probability << "}\n";
        }
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (mode != "sample")
            std::cout << std::setprecision(17) << "{\"shots\":" << shots
                      << ",\"seconds\":" << seconds
                      << ",\"microseconds_per_sample\":" << seconds * 1e6 / shots
                      << ",\"coefficient_scratch_bytes\":"
                      << worker.scratch.size() * sizeof(C) + worker.tensor.size() * sizeof(C)
                      << ",\"probability_scratch_bytes\":" << worker.weights.size() * sizeof(double)
                      << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
