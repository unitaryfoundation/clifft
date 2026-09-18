// Complete synthetic noisy five-check CSS sequence using compiled factor plans.
#include "msc_factor_kernel.h"
using namespace msc_factor;
constexpr unsigned ROUNDS = 5;
using Mask = uint64_t;
unsigned parity(Mask x) noexcept {
    return std::popcount(x) & 1;
}
struct Local {
    unsigned flip, phase, linear;
};
struct CheckRow {
    unsigned x, k;
    Mask cx, cz;
};
struct Monomial {
    Mask flip = 0;
    unsigned phase = 0;
    std::array<unsigned, MAX_WIDTH> linear{};
    double weight = .5;
};
struct Sequence {
    Worker kernel;
    double noise;
    unsigned axis;
    Plan single;
    std::vector<Mask> xchecks, zchecks, coordinates;
    std::vector<CheckRow> checks;
    std::array<std::array<Local, 2>, 256> table{};
    std::array<std::array<unsigned, 4 * MAX_WIDTH>, ROUNDS> faults{};
    std::array<unsigned, ROUNDS * MAX_WIDTH + 1> flips{}, outcomes{}, records{};
    std::array<unsigned, (ROUNDS - 1) * MAX_WIDTH> detectors{};
    std::array<C, 2> logical{};
    Mask frame_x = 0, frame_z = 0, all;
    unsigned stride, count;
    double probability = 1;
    Sequence(Reader& r)
        : kernel(r, false),
          noise(r.get<double>()),
          axis(r.get()),
          single(r, kernel.width + kernel.rank) {
        require(kernel.width < 64 && 2 * kernel.rank + 1 == kernel.width);
        require(noise >= 0 && noise <= 1 && axis < 3);
        all = (Mask{1} << kernel.width) - 1;
        for (auto* rows : {&xchecks, &zchecks, &coordinates}) {
            rows->resize(r.count());
            require(rows->size() == kernel.rank);
            for (auto& v : *rows) {
                v = r.get<Mask>();
                require(v <= all);
            }
        }
        checks.resize(r.count());
        require(checks.size() == 2 * kernel.rank);
        std::array<Mask, 2> seen{};
        for (auto& row : checks) {
            row = {r.get(), r.get(), r.get<Mask>(), r.get<Mask>()};
            require(row.x < 2 && row.k < kernel.rank && row.cx <= all && row.cz <= all);
            require(!(seen[row.x] >> row.k & 1));
            seen[row.x] |= Mask{1} << row.k;
        }
        for (auto& row : table)
            for (auto& v : row) {
                v = {r.get(), r.get(), r.get()};
                require(v.flip < 2 && v.phase < 8 && v.linear < 8);
            }
        kernel.scratch.resize(std::max<size_t>(kernel.scratch.size(), single.storage));
        stride = checks.size() + 1;
        count = ROUNDS * stride + 1;
    }
    unsigned coordinate(Mask bits) const noexcept {
        unsigned result = 0;
        for (unsigned k = 0; k < kernel.rank; ++k)
            result |= parity(bits & coordinates[k]) << k;
        return result;
    }
    Mask expand(unsigned bits) const noexcept {
        Mask result = 0;
        for (unsigned k = 0; k < kernel.rank; ++k)
            if (bits >> k & 1)
                result ^= xchecks[k];
        return result;
    }
    double uniform() noexcept { return (kernel.rng() >> 11) * 0x1.0p-53; }
    void sample_faults() noexcept {
        for (auto& row : faults)
            for (unsigned q = 0; q < 4 * kernel.width; ++q)
                row[q] = uniform() < noise ? 1 + unsigned(3 * uniform()) : 0;
        for (unsigned k = 0; k < count; ++k)
            flips[k] = uniform() < noise;
    }
    std::array<Monomial, 2> payload(unsigned round, unsigned bit) const noexcept {
        std::array<Monomial, 2> terms{};
        for (unsigned b = 0; b < 2; ++b) {
            auto& op = terms[b];
            op.weight = b && bit ? -.5 : .5;
            for (unsigned q = 0; q < kernel.width; ++q) {
                unsigned index = 0;
                for (unsigned layer = 0; layer < 4; ++layer)
                    index |= faults[round][layer * kernel.width + q] << (2 * layer);
                const auto& local = table[index][b];
                op.flip |= Mask{local.flip} << q;
                op.phase = (op.phase + local.phase) & 7;
                op.linear[q] = local.linear;
            }
        }
        return terms;
    }
    Case prepare(const std::array<Monomial, 2>& terms) const noexcept {
        Case data;
        data.terms = 2;
        for (unsigned t = 0; t < 2; ++t) {
            const auto& op = terms[t];
            Mask offset = frame_x ^ op.flip;
            for (unsigned k = 0; k < kernel.rank; ++k)
                data.labels[t] |= parity(offset & zchecks[k]) << k;
            for (unsigned l = 0; l < 2; ++l) {
                unsigned source = l ^ parity(offset);
                Mask delta =
                    frame_x ^ (source ? all : 0) ^ expand(coordinate(offset ^ (source ? all : 0)));
                unsigned constant = op.phase + 4 * parity(frame_z & (delta ^ frame_x));
                for (unsigned q = 0; q < kernel.width; ++q) {
                    unsigned bit = (delta >> q) & 1;
                    constant += op.linear[q] * bit;
                    data.phase[t][l][q] =
                        (op.linear[q] * (bit ? 7 : 1) + 4 * ((frame_z >> q) & 1)) & 7;
                }
                data.amplitude[t][l] = logical[source] * roots[constant & 7];
            }
            for (unsigned b = 0; b < 2; ++b)
                data.gram[t][b] = op.weight * terms[b].weight;
        }
        data.refresh();
        return data;
    }
    double norm(const Case& data) noexcept {
        double result = kernel.mass(data, data.labels[0], 0, 0);
        if (data.labels[1] != data.labels[0])
            result += kernel.mass(data, data.labels[1], 0, 0);
        return result;
    }
    double contract(const Case& data, const std::array<Monomial, 2>& terms, unsigned xs,
                    unsigned zs) noexcept {
        Mask cx = 0, cz = 0;
        for (const auto& row : checks)
            if ((row.x ? xs : zs) >> row.k & 1) {
                cx ^= row.cx;
                cz ^= row.cz;
            }
        Mask logical_rep = all ^ expand(coordinate(all));
        Mask representative = cx ^ expand(coordinate(cx)) ^ (parity(cx) ? logical_rep : 0);
        std::array<C, 2> output{};
        std::array<unsigned, 3 * MAX_WIDTH> parameters{};
        for (unsigned k = 0; k < kernel.rank; ++k)
            parameters[kernel.width + k] = 4 * ((xs >> k) & 1);
        for (unsigned t = 0; t < 2; ++t)
            if (data.labels[t] == zs)
                for (unsigned l = 0; l < 2; ++l) {
                    for (unsigned q = 0; q < kernel.width; ++q)
                        parameters[q] = data.phase[t][l][q];
                    Mask delta = representative ^ (l ? logical_rep : 0) ^ cx;
                    output[l ^ parity(cx)] += terms[t].weight * data.amplitude[t][l] *
                                              single.evaluate(parameters, kernel.scratch.data()) *
                                              (parity(delta & cz) ? -1. : 1.);
                }
        double weight = std::norm(output[0]) + std::norm(output[1]);
        assert(weight > 0 && std::isfinite(weight));
        for (unsigned l = 0; l < 2; ++l)
            logical[l] = output[l] / std::sqrt(weight);
        frame_x = cx;
        frame_z = cz;
        return weight;
    }
    void run() noexcept {
        sample_faults();
        logical = {C{M_SQRT1_2}, roots[1] * M_SQRT1_2};
        frame_x = frame_z = 0;
        probability = 1;
        for (unsigned r = 0; r < ROUNDS; ++r) {
            auto terms = payload(r, 0);
            Case data = prepare(terms);
            double zero = norm(data);
            assert(zero >= 0 && zero <= 1 + 1e-10);
            std::array<double, 2> weights{std::min(1., zero), std::max(0., 1 - zero)};
            unsigned bit = kernel.draw(weights.data(), 2).first;
            if (bit) {
                terms[1].weight = -.5;
                data = prepare(terms);
            }
            auto sample = kernel.factor(data);
            double weight = contract(data, terms, sample.xs, sample.zs);
            assert(std::abs(weight - weights[bit] * sample.probability) < 1e-10);
            probability *= weight;
            outcomes[r * stride] = bit;
            for (unsigned k = 0; k < checks.size(); ++k)
                outcomes[r * stride + k + 1] =
                    ((checks[k].x ? sample.xs : sample.zs) >> checks[k].k) & 1;
        }
        C overlap = std::conj(logical[0]) * logical[1];
        double expectation = axis == 0   ? 2 * overlap.real()
                             : axis == 1 ? 2 * overlap.imag()
                                         : std::norm(logical[0]) - std::norm(logical[1]);
        unsigned sign = axis == 0   ? parity(frame_z)
                        : axis == 1 ? parity(frame_x ^ frame_z) ^ (((kernel.width - 1) / 2) & 1)
                                    : parity(frame_x);
        double zero = .5 * (1 + (sign ? -expectation : expectation));
        assert(zero >= -1e-10 && zero <= 1 + 1e-10);
        zero = std::clamp(zero, 0., 1.);
        std::array<double, 2> weights{zero, 1 - zero};
        auto [bit, chance] = kernel.draw(weights.data(), 2);
        outcomes[count - 1] = bit;
        probability *= chance;
        for (unsigned k = 0; k < count; ++k)
            records[k] = outcomes[k] ^ flips[k];
        for (unsigned r = 1; r < ROUNDS; ++r)
            for (unsigned k = 1; k < stride; ++k)
                detectors[(r - 1) * (stride - 1) + k - 1] =
                    records[r * stride + k] ^ records[(r - 1) * stride + k];
    }
    void print() const {
        auto bits = [](const auto& row, unsigned n) {
            std::cout << '"';
            for (unsigned k = 0; k < n; ++k)
                std::cout << row[k];
            std::cout << '"';
        };
        std::cout << std::setprecision(17) << "{\"probability\":" << probability
                  << ",\"logical\":[[" << logical[0].real() << ',' << logical[0].imag() << "],["
                  << logical[1].real() << ',' << logical[1].imag() << "]],\"frame\":[" << frame_x
                  << ',' << frame_z << "],\"faults\":[";
        for (unsigned r = 0; r < ROUNDS; ++r) {
            if (r)
                std::cout << ',';
            bits(faults[r], 4 * kernel.width);
        }
        std::cout << "],\"flips\":";
        bits(flips, count);
        std::cout << ",\"outcomes\":";
        bits(outcomes, count);
        std::cout << ",\"records\":";
        bits(records, count);
        std::cout << ",\"detectors\":";
        bits(detectors, (ROUNDS - 1) * (stride - 1));
        std::cout << ",\"observables\":[" << records[count - 1] << "]}\n";
    }
};
int main(int argc, char** argv) {
    try {
        require(argc == 5);
        std::string mode(argv[2]);
        require(mode == "sample" || mode == "bench");
        Reader reader(argv[1]);
        Sequence sequence(reader);
        unsigned shots = std::stoul(argv[3]);
        require(shots > 0);
        sequence.kernel.rng.seed(std::stoull(argv[4]));
        uint64_t checksum = 0;
        auto start = std::chrono::steady_clock::now();
        for (unsigned k = 0; k < shots; ++k) {
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            auto before = allocations;
#endif
            sequence.run();
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            assert(before == allocations);
#endif
            for (unsigned j = 0; j < sequence.count; ++j)
                checksum += sequence.records[j];
            for (unsigned j = 0; j < (ROUNDS - 1) * (sequence.stride - 1); ++j)
                checksum += sequence.detectors[j];
            checksum += sequence.records[sequence.count - 1];
            if (mode == "sample")
                sequence.print();
        }
        double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (mode == "bench")
            std::cout << std::setprecision(17) << "{\"shots\":" << shots
                      << ",\"seconds\":" << seconds
                      << ",\"microseconds_per_shot\":" << seconds * 1e6 / shots
                      << ",\"coefficient_scratch_bytes\":"
                      << sequence.kernel.scratch.size() * sizeof(C) << ",\"checksum\":" << checksum
                      << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
