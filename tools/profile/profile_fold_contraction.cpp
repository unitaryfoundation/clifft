// Standalone diagnostic for compiler-generated code-boundary overlap plans.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Complex = std::complex<double>;

struct Leaf {
    size_t offset;
    std::array<std::vector<size_t>, 2> multipliers;
};
struct Step {
    size_t offset;
    size_t size;
    std::vector<std::vector<size_t>> gathers;
};
struct Fixture {
    size_t phase;
    std::vector<size_t> parameters;
    std::array<Complex, 2> expected;
};

size_t integer(std::istream& in, size_t maximum) {
    long long value;
    if (!(in >> value) || value < 0 || static_cast<unsigned long long>(value) > maximum) {
        throw std::invalid_argument("invalid plan integer");
    }
    return static_cast<size_t>(value);
}

struct Plan {
    std::vector<Leaf> leaves;
    std::vector<Step> steps;
    std::vector<size_t> outputs;
    std::vector<Fixture> cases;
    std::vector<Complex> values;
    std::array<Complex, 8> roots;
    double normalization;
    size_t gather_entries = 0;

    explicit Plan(std::istream& in) {
        if (integer(in, 1) != 1) {
            throw std::invalid_argument("unsupported plan version");
        }
        size_t gauge = integer(in, 60);
        size_t storage = integer(in, 1 << 20);
        size_t leaf_count = integer(in, 4096);
        size_t step_count = integer(in, 4096);
        size_t output_count = integer(in, 4096);
        size_t case_count = integer(in, 4096);
        if (!storage || !leaf_count || !output_count || !case_count) {
            throw std::invalid_argument("empty plan");
        }
        normalization = std::ldexp(1., -static_cast<int>(gauge));
        double s = std::sqrt(0.5);
        roots = {Complex(1),  Complex(s, s),   Complex(0, 1),  Complex(-s, s),
                 Complex(-1), Complex(-s, -s), Complex(0, -1), Complex(s, -s)};
        size_t next = 0;
        for (size_t j = 0; j < leaf_count; ++j) {
            Leaf leaf;
            leaf.offset = integer(in, storage);
            size_t size = integer(in, storage);
            if (leaf.offset != next || !size || size > storage - next) {
                throw std::invalid_argument("invalid leaf range");
            }
            for (auto& row : leaf.multipliers) {
                for (size_t k = 0; k < size; ++k) {
                    row.push_back(integer(in, 1));
                }
            }
            next += size;
            leaves.push_back(std::move(leaf));
        }
        for (size_t j = 0; j < step_count; ++j) {
            Step step;
            step.offset = integer(in, storage);
            step.size = integer(in, storage);
            size_t count = integer(in, leaf_count + step_count);
            if (step.offset != next || !step.size || step.size > storage - next) {
                throw std::invalid_argument("invalid output range");
            }
            for (size_t k = 0; k < count; ++k) {
                std::vector<size_t> gather;
                for (size_t l = 0; l < 2 * step.size; ++l) {
                    gather.push_back(integer(in, next - 1));
                }
                gather_entries += gather.size();
                if (gather_entries > (1 << 24)) {
                    throw std::invalid_argument("oversized plan");
                }
                step.gathers.push_back(std::move(gather));
            }
            next += step.size;
            steps.push_back(std::move(step));
        }
        if (next != storage) {
            throw std::invalid_argument("unused plan storage");
        }
        for (size_t j = 0; j < output_count; ++j) {
            outputs.push_back(integer(in, storage - 1));
        }
        for (size_t j = 0; j < case_count; ++j) {
            Fixture fixture;
            fixture.phase = integer(in, 7);
            for (size_t k = 0; k < leaf_count; ++k) {
                fixture.parameters.push_back(integer(in, 7));
            }
            for (auto& expected : fixture.expected) {
                double re, im;
                if (!(in >> re >> im) || !std::isfinite(re) || !std::isfinite(im)) {
                    throw std::invalid_argument("invalid expected amplitude");
                }
                expected = Complex(re, im);
            }
            cases.push_back(std::move(fixture));
        }
        std::string extra;
        if (in >> extra) {
            throw std::invalid_argument("trailing plan input");
        }
        values.resize(storage);
    }

    std::array<Complex, 2> execute(const Fixture& fixture) noexcept {
        std::array<Complex, 2> result;
        for (size_t logical = 0; logical < 2; ++logical) {
            for (size_t k = 0; k < leaves.size(); ++k) {
                const auto& leaf = leaves[k];
                for (size_t j = 0; j < leaf.multipliers[logical].size(); ++j) {
                    values[leaf.offset + j] =
                        roots[fixture.parameters[k] * leaf.multipliers[logical][j]];
                }
            }
            for (const auto& step : steps) {
                for (size_t j = 0; j < step.size; ++j) {
                    Complex a = 1, b = 1;
                    for (const auto& gather : step.gathers) {
                        a *= values[gather[2 * j]];
                        b *= values[gather[2 * j + 1]];
                    }
                    values[step.offset + j] = a + b;
                }
            }
            Complex total = roots[fixture.phase] * normalization;
            for (size_t offset : outputs) {
                total *= values[offset];
            }
            result[logical] = total;
        }
        return result;
    }
};
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument("usage: profile_fold_contraction PLAN REPETITIONS");
        }
        std::ifstream file(argv[1]);
        if (!file) {
            throw std::invalid_argument("cannot open plan");
        }
        Plan plan(file);
        std::string repeats_text(argv[2]);
        size_t parsed = 0;
        unsigned long repeats = std::stoul(repeats_text, &parsed);
        if (parsed != repeats_text.size() || !repeats || repeats > 1000000) {
            throw std::invalid_argument("invalid repetitions");
        }
        double error = 0;
        for (const auto& fixture : plan.cases) {
            auto result = plan.execute(fixture);
            for (size_t b = 0; b < 2; ++b) {
                error = std::max(error, std::abs(result[b] - fixture.expected[b]));
            }
        }
        if (error > 2e-12) {
            throw std::runtime_error("native and Python contraction results disagree");
        }
        std::array<double, 5> timings;
        Complex checksum = 0;
        for (double& time : timings) {
            auto start = std::chrono::steady_clock::now();
            for (size_t repetition = 0; repetition < repeats; ++repetition) {
                for (const auto& fixture : plan.cases) {
                    auto result = plan.execute(fixture);
                    checksum += result[0] + result[1];
                }
            }
            auto end = std::chrono::steady_clock::now();
            time = std::chrono::duration<double, std::micro>(end - start).count() /
                   (static_cast<double>(repeats) * plan.cases.size());
        }
        auto sorted = timings;
        std::sort(sorted.begin(), sorted.end());
        std::cout << std::setprecision(17) << "{\"cases\":" << plan.cases.size()
                  << ",\"repetitions\":" << repeats << ",\"max_amplitude_error\":" << error
                  << ",\"coefficient_bytes\":" << plan.values.size() * sizeof(Complex)
                  << ",\"gather_bytes\":" << plan.gather_entries * sizeof(size_t)
                  << ",\"median_us_per_two_input_phase_sums\":" << sorted[2] << ",\"trial_us\":[";
        for (size_t j = 0; j < timings.size(); ++j) {
            std::cout << (j ? "," : "") << timings[j];
        }
        std::cout << "],\"checksum\":[" << checksum.real() << "," << checksum.imag() << "]}\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
