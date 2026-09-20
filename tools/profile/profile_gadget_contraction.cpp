// Arithmetic-only benchmark of an offline-prepared terminal probability plan.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Complex = std::complex<double>;

struct Leaf {
    size_t offset;
    std::vector<size_t> parity;
};
struct Step {
    size_t offset, size;
    std::vector<std::vector<size_t>> gathers;
};
struct Term {
    Complex coefficient;
    std::vector<Complex> local;
};

int main(int argc, char** argv) {
    if (argc != 3)
        throw std::invalid_argument("usage: profile_gadget_contraction plan.txt repeats");
    std::ifstream input(argv[1]);
    auto integer = [&]() {
        size_t result;
        if (!(input >> result) || result > 10000000)
            throw std::invalid_argument("invalid or oversized plan field");
        return result;
    };
    auto number = [&]() {
        double result;
        if (!(input >> result) || !std::isfinite(result))
            throw std::invalid_argument("invalid numeric parameter");
        return result;
    };
    auto complex = [&]() {
        const auto real = number();
        return Complex(real, number());
    };
    const auto marginal = integer();
    if (marginal > 1)
        throw std::invalid_argument("invalid evaluation mode");
    const auto rank = integer();
    const auto storage = integer();
    if (rank > 60 || !storage)
        throw std::invalid_argument("invalid rank or storage");
    std::vector<Leaf> leaves(integer());
    size_t initialized = 0;
    for (auto& leaf : leaves) {
        leaf.offset = integer();
        leaf.parity.resize(integer());
        if (leaf.offset != initialized || leaf.parity.empty() ||
            leaf.parity.size() > storage - initialized)
            throw std::invalid_argument("invalid leaf storage");
        initialized += leaf.parity.size();
        for (auto& parity : leaf.parity) {
            parity = integer();
            if (parity > 1)
                throw std::invalid_argument("invalid leaf parity");
        }
    }
    std::vector<Step> steps(integer());
    size_t max_product = 0;
    for (auto& step : steps) {
        step.offset = integer();
        step.size = integer();
        if (step.offset != initialized || !step.size || step.size > storage - initialized)
            throw std::invalid_argument("invalid step storage");
        step.gathers.resize(integer());
        for (auto& gather : step.gathers) {
            gather.resize(2 * step.size);
            for (auto& address : gather) {
                address = integer();
                if (address >= initialized)
                    throw std::invalid_argument("gather reads uninitialized storage");
            }
        }
        initialized += step.size;
        max_product = std::max(max_product, 2 * step.size);
    }
    if (initialized != storage)
        throw std::invalid_argument("unused plan storage");
    std::vector<size_t> outputs(integer());
    for (auto& output : outputs) {
        output = integer();
        if (output >= storage)
            throw std::invalid_argument("invalid output address");
    }
    std::vector<std::vector<Term>> cases(integer());
    if (cases.empty())
        throw std::invalid_argument("no input cases");
    for (auto& terms : cases) {
        terms.resize(integer());
        for (auto& term : terms) {
            term.coefficient = complex();
            term.local.resize(2 * leaves.size());
            for (auto& value : term.local)
                value = complex();
        }
    }
    const auto repeats = std::stoull(argv[2]);
    if (!repeats)
        throw std::invalid_argument("repeats must be positive");
    std::vector<Complex> scratch(storage), product(max_product);
    const auto normalization = std::ldexp(1.0, -int(rank));
    auto evaluate = [&](const std::vector<Term>& terms) noexcept {
        Complex amplitude = 0;
        for (const auto& term : terms) {
            for (size_t i = 0; i < leaves.size(); ++i) {
                const auto& leaf = leaves[i];
                for (size_t j = 0; j < leaf.parity.size(); ++j)
                    scratch[leaf.offset + j] = term.local[2 * i + leaf.parity[j]];
            }
            for (const auto& step : steps) {
                std::fill_n(product.begin(), 2 * step.size, Complex(1));
                for (const auto& gather : step.gathers)
                    for (size_t j = 0; j < 2 * step.size; ++j)
                        product[j] *= scratch[gather[j]];
                for (size_t j = 0; j < step.size; ++j)
                    scratch[step.offset + j] = product[j] + product[j + step.size];
            }
            Complex value = normalization;
            for (auto output : outputs)
                value *= scratch[output];
            amplitude += term.coefficient * value;
        }
        return marginal ? amplitude.real() : std::norm(amplitude);
    };
    std::cout << std::setprecision(17) << "{\"probabilities\":[";
    for (size_t i = 0; i < cases.size(); ++i)
        std::cout << (i ? "," : "") << evaluate(cases[i]);
    double checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < repeats; ++repeat)
        for (const auto& terms : cases)
            checksum += evaluate(terms);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "],\"seconds\":" << seconds << ",\"evaluations\":" << repeats * cases.size()
              << ",\"checksum\":" << checksum
              << ",\"workspace_bytes\":" << (storage + max_product) * sizeof(Complex) << "}\n";
}
