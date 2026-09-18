// Standalone logical-block sampler for compiler-generated MSC research plans.
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
static size_t allocation_count = 0;
void* operator new(size_t bytes) {
    ++allocation_count;
    if (void* p = std::malloc(bytes ? bytes : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](size_t bytes) {
    return ::operator new(bytes);
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
using Pair = std::array<C, 2>;
using Bits = std::vector<uint64_t>;
constexpr unsigned CAP = 512;
const std::array<C, 8> roots{C{1},  C{M_SQRT1_2, M_SQRT1_2},   C{0, 1},  C{-M_SQRT1_2, M_SQRT1_2},
                             C{-1}, C{-M_SQRT1_2, -M_SQRT1_2}, C{0, -1}, C{M_SQRT1_2, -M_SQRT1_2}};
unsigned parity(unsigned x) {
    return std::popcount(x) & 1;
}
unsigned bit(const Bits& b, unsigned k) {
    return (b[k / 64] >> (k % 64)) & 1;
}
void set(Bits& b, unsigned k, unsigned v) {
    b[k / 64] = (b[k / 64] & ~(uint64_t{1} << (k % 64))) | (uint64_t{v} << (k % 64));
}
struct Word {
    unsigned index;
    uint64_t mask;
};
struct Mask {
    std::vector<Word> words;
    unsigned eval(const Bits& bits) const {
        unsigned v = 0;
        for (auto word : words) {
            assert(word.index < bits.size());
            v ^= std::popcount(bits[word.index] & word.mask) & 1;
        }
        return v;
    }
};
struct Row {
    Mask records, faults;
    unsigned constant;
    unsigned eval(const Bits& r, const Bits& f) const {
        return records.eval(r) ^ faults.eval(f) ^ constant;
    }
};
struct Reader {
    std::ifstream in;
    explicit Reader(const char* path) : in(path) {
        if (!in)
            throw std::runtime_error("cannot read plan");
    }
    template <class T = unsigned>
    T get() {
        T value;
        if (!(in >> value))
            throw std::runtime_error("incomplete plan");
        return value;
    }
    unsigned count() {
        auto n = get();
        if (n > 1000000)
            throw std::runtime_error("plan count exceeds bound");
        return n;
    }
    std::vector<unsigned> ints() {
        std::vector<unsigned> v(count());
        for (auto& x : v)
            x = get();
        return v;
    }
    Mask mask() {
        Mask m;
        m.words.resize(count());
        for (auto& w : m.words) {
            w.index = get();
            w.mask = get<uint64_t>();
        }
        return m;
    }
    Row row() {
        Row r;
        r.records = mask();
        r.faults = mask();
        r.constant = get();
        return r;
    }
    std::vector<Row> rows() {
        std::vector<Row> v(count());
        for (auto& r : v)
            r = row();
        return v;
    }
    C complex() {
        double real = get<double>(), imag = get<double>();
        return {real, imag};
    }
    std::vector<std::array<unsigned, 2>> pairs() {
        std::vector<std::array<unsigned, 2>> v(count());
        for (auto& p : v) {
            p[0] = get();
            p[1] = get();
        }
        return v;
    }
};
struct Affine {
    struct Pivot {
        unsigned event;
        Row row;
        Mask rhs;
    };
    std::vector<unsigned> free, events;
    std::vector<Pivot> pivots;
    explicit Affine(Reader& r) {
        free = r.ints();
        events = r.ints();
        pivots.resize(r.count());
        for (auto& p : pivots) {
            p.event = r.get();
            p.row = r.row();
            p.rhs = r.mask();
        }
    }
    void sample(Bits& records, const Bits& faults, const Bits& values, std::mt19937_64& rng) const {
        for (auto k : events)
            set(records, k, 0);
        for (auto k : free)
            set(records, k, rng() & 1);
        for (const auto& p : pivots)
            set(records, p.event, p.row.eval(records, faults) ^ p.rhs.eval(values));
    }
};
struct Polynomial {
    struct Product {
        unsigned c;
        Mask a, b;
    };
    std::vector<Product> products;
    void read(Reader& r) {
        products.resize(r.count());
        for (auto& p : products) {
            p.c = r.get();
            p.a = r.mask();
            p.b = r.mask();
        }
    }
    unsigned eval(const Bits& bits) const {
        unsigned v = 0;
        for (const auto& p : products)
            v += p.c * p.a.eval(bits) * p.b.eval(bits);
        return v & 7;
    }
};
struct Monomial {
    unsigned flips = 0, phase = 0;
    std::array<unsigned, 42> linear{};
    std::array<unsigned, 2> column(unsigned b, unsigned width) const {
        unsigned p = phase;
        for (unsigned q = 0; q < width; ++q)
            p += linear[q] * ((b >> q) & 1);
        return {b ^ flips, p & 7};
    }
};
struct Branch {
    std::vector<Mask> flips;
    Polynomial phase;
    std::vector<Polynomial> linear;
    void read(Reader& r) {
        flips.resize(r.count());
        for (auto& x : flips)
            x = r.mask();
        phase.read(r);
        linear.resize(flips.size());
        for (auto& x : linear)
            x.read(r);
    }
};
// Gadgets have up to 38 wires; their flip mask therefore needs 64 bits.
struct WideMonomial {
    uint64_t flips = 0;
    unsigned phase = 0;
    std::array<unsigned, 42> linear{};
};
WideMonomial bind_wide(const Branch& branch, const Bits& b) {
    WideMonomial op;
    op.phase = branch.phase.eval(b);
    for (unsigned q = 0; q < branch.flips.size(); ++q) {
        op.flips |= uint64_t{branch.flips[q].eval(b)} << q;
        op.linear[q] = branch.linear[q].eval(b);
    }
    return op;
}
struct Term {
    Monomial op;
    C weight{1};
    std::array<Pair, 42> ancillas{};
};
struct Payload {
    unsigned count = 1;
    std::array<Term, 4> terms{};
};
Pair eigen(unsigned axis, unsigned sign) {
    if (axis == 3)
        return {C{double(1 - sign)}, C{double(sign)}};
    return {C{M_SQRT1_2}, (sign ? -M_SQRT1_2 : M_SQRT1_2) * (axis == 2 ? C{0, 1} : C{1})};
}
C dot(const Pair& a, const Pair& b) {
    return std::conj(a[0]) * b[0] + std::conj(a[1]) * b[1];
}
double norm(const Pair& p) {
    return std::norm(p[0]) + std::norm(p[1]);
}
void normalize(Pair& p) {
    double scale = 1 / std::sqrt(norm(p));
    p[0] *= scale;
    p[1] *= scale;
}
struct Step {
    unsigned kind = 0, data = 0, q = 0, x = 0, z = 0, event = 0, outcome_bit = 0;
    int hidden = -1;
    Mask reset_flip;
    std::vector<unsigned> dm;
    std::vector<std::array<unsigned, 2>> am;
    std::array<Branch, 2> branches;
};
struct PayloadPlan {
    unsigned width, ancillas;
    std::vector<Row> rows;
    std::vector<std::array<unsigned, 2>> duals;
    std::vector<std::array<unsigned, 3>> initial;
    std::vector<Step> steps;
    explicit PayloadPlan(Reader& r) {
        width = r.get();
        ancillas = r.get();
        rows = r.rows();
        duals = r.pairs();
        initial.resize(r.count());
        for (auto& i : initial)
            for (auto& x : i)
                x = r.get();
        steps.resize(r.count());
        for (auto& s : steps) {
            s.kind = r.get();
            if (s.kind == 0) {
                s.data = r.get();
                s.q = r.get();
                s.x = r.get();
                s.z = r.get();
            } else if (s.kind == 1) {
                s.q = r.get();
                s.event = r.get();
            } else {
                s.event = r.get();
                s.hidden = r.get<int>();
                s.outcome_bit = r.get();
                s.reset_flip = r.mask();
                s.dm = r.ints();
                s.am = r.pairs();
                for (auto& b : s.branches)
                    b.read(r);
            }
        }
        if (width > 19 || ancillas > 42 || rows.size() > 42)
            throw std::runtime_error("payload exceeds capacity");
    }
    std::array<unsigned, 2> start(Payload& p, const Bits& records, const Bits& faults) const {
        p = Payload{};
        std::array<unsigned, 42> signs{};
        for (unsigned k = 0; k < rows.size(); ++k)
            signs[k] = rows[k].eval(records, faults);
        std::array<unsigned, 2> frame{};
        for (unsigned k = 0; k < duals.size(); ++k)
            if (signs[k]) {
                frame[0] ^= duals[k][0];
                frame[1] ^= duals[k][1];
            }
        for (auto i : initial)
            p.terms[0].ancillas[i[0]] = eigen(i[1], signs[i[2]]);
        return frame;
    }
    void fault(Payload& p, const Step& s, const Bits& faults) const {
        unsigned x = bit(faults, s.x), z = bit(faults, s.z);
        if (!(x | z))
            return;
        for (unsigned k = 0; k < p.count; ++k) {
            auto& t = p.terms[k];
            if (s.data) {
                t.op.phase = (t.op.phase + 4 * z * ((t.op.flips >> s.q) & 1) + 2 * x * z) & 7;
                t.op.linear[s.q] = (t.op.linear[s.q] + 4 * z) & 7;
                t.op.flips ^= x << s.q;
            } else {
                auto& a = t.ancillas[s.q];
                if (z)
                    a[1] = -a[1];
                if (x)
                    std::swap(a[0], a[1]);
                if (x && z) {
                    a[0] *= C{0, 1};
                    a[1] *= C{0, 1};
                }
            }
        }
    }
    void measure(Payload& p, const Step& s, unsigned outcome) const {
        auto target = eigen(1, outcome);
        for (unsigned k = 0; k < p.count; ++k) {
            p.terms[k].weight *= dot(target, p.terms[k].ancillas[s.q]);
            p.terms[k].ancillas[s.q] = target;
        }
    }
    void expand(const Payload& p, Payload& out, const Step& s, Bits& packed,
                unsigned outcome) const {
        assert(p.count <= 2);
        out.count = p.count * 2;
        set(packed, s.outcome_bit, outcome);
        for (unsigned b = 0; b < 2; ++b) {
            auto op = bind_wide(s.branches[b], packed);
            for (unsigned k = 0; k < p.count; ++k) {
                auto& t = out.terms[b * p.count + k];
                t = p.terms[k];
                t.weight *= .5;
                t.op.phase = (t.op.phase + op.phase) & 7;
                for (unsigned q = 0; q < width; ++q) {
                    unsigned linear = op.linear[s.dm[q]], flip = (p.terms[k].op.flips >> q) & 1;
                    t.op.phase = (t.op.phase + linear * flip) & 7;
                    t.op.linear[q] = (t.op.linear[q] + linear * (1 - 2 * flip)) & 7;
                    t.op.flips ^= unsigned((op.flips >> s.dm[q]) & 1) << q;
                }
                for (auto [j, q] : s.am) {
                    t.ancillas[j][1] *= roots[op.linear[q]];
                    if ((op.flips >> q) & 1)
                        std::swap(t.ancillas[j][0], t.ancillas[j][1]);
                }
            }
        }
    }
};
struct Code {
    unsigned width;
    std::vector<unsigned> xchecks, zchecks, coordinates, span, source_coordinates;
    std::vector<std::array<unsigned, 2>> order;
    explicit Code(Reader& r) {
        width = r.get();
        xchecks = r.ints();
        zchecks = r.ints();
        coordinates = r.ints();
        span = r.ints();
        order = r.pairs();
        source_coordinates = r.ints();
        if (width > 19 || span.size() > CAP || source_coordinates.size() != 2 * span.size())
            throw std::runtime_error("code exceeds capacity");
    }
    unsigned coordinate(unsigned b) const {
        unsigned value = 0;
        for (unsigned k = 0; k < coordinates.size(); ++k)
            value |= parity(b & coordinates[k]) << k;
        return value;
    }
    unsigned label(unsigned b) const {
        unsigned value = 0;
        for (unsigned k = 0; k < zchecks.size(); ++k)
            value |= parity(b & zchecks[k]) << k;
        return value;
    }
};
struct CodeScratch {
    std::array<std::array<C, 2 * CAP>, 4> tensor{};
    std::array<unsigned, 4> labels{};
    std::array<std::array<C, 4>, 4> gram{};
    std::array<double, 4 * CAP> weights{};
    void amplitudes(const Code& code, const Pair& logical, const std::array<unsigned, 2>& frame,
                    const Payload& p, unsigned ancillas) {
        const unsigned n = code.span.size(), lx = (1 << code.width) - 1;
        for (unsigned a = 0; a < p.count; ++a) {
            const auto& t = p.terms[a];
            tensor[a].fill(C{});
            const unsigned offset = frame[0] ^ t.op.flips, shift = code.coordinate(offset);
            labels[a] = code.label(offset);
            for (unsigned s = 0; s < 2; ++s)
                for (unsigned j = 0; j < n; ++j) {
                    unsigned b = code.span[j] ^ (s ? lx : 0);
                    auto [out, phase] = t.op.column(b ^ frame[0], code.width);
                    (void)out;
                    phase += 4 * parity(b & frame[1]);
                    unsigned index =
                        (s ^ parity(offset)) * n + (code.source_coordinates[s * n + j] ^ shift);
                    tensor[a][index] = logical[s] * roots[phase & 7] / std::sqrt(double(n));
                }
            for (unsigned b = 0; b < p.count; ++b) {
                C g = std::conj(t.weight) * p.terms[b].weight;
                for (unsigned q = 0; q < ancillas; ++q)
                    g *= dot(t.ancillas[q], p.terms[b].ancillas[q]);
                gram[a][b] = g;
            }
        }
    }
    double norm(const Code& code, const Pair& logical, const std::array<unsigned, 2>& frame,
                const Payload& p, unsigned ancillas) {
        amplitudes(code, logical, frame, p, ancillas);
        C result{};
        for (unsigned a = 0; a < p.count; ++a)
            for (unsigned b = 0; b < p.count; ++b)
                if (labels[a] == labels[b]) {
                    C overlap{};
                    for (unsigned j = 0; j < 2 * code.span.size(); ++j)
                        overlap += std::conj(tensor[a][j]) * tensor[b][j];
                    result += gram[a][b] * overlap;
                }
        return result.real();
    }
    void distribution(const Code& code, const Pair& logical, const std::array<unsigned, 2>& frame,
                      const Payload& p, unsigned ancillas) {
        amplitudes(code, logical, frame, p, ancillas);
        unsigned n = code.span.size();
        for (unsigned a = 0; a < p.count; ++a)
            for (unsigned s = 0; s < 2; ++s)
                for (unsigned stride = 1; stride < n; stride *= 2)
                    for (unsigned base = 0; base < n; base += 2 * stride)
                        for (unsigned j = 0; j < stride; ++j) {
                            auto& x = tensor[a][s * n + base + j];
                            auto& y = tensor[a][s * n + base + j + stride];
                            C u = x, v = y;
                            x = (u + v) * M_SQRT1_2;
                            y = (u - v) * M_SQRT1_2;
                        }
        weights.fill(0);
        for (unsigned a = 0; a < p.count; ++a) {
            bool duplicate = false;
            for (unsigned b = 0; b < a; ++b)
                duplicate |= labels[a] == labels[b];
            if (duplicate)
                continue;
            for (unsigned b = 0; b < p.count; ++b)
                for (unsigned c = 0; c < p.count; ++c)
                    if (labels[a] == labels[b] && labels[a] == labels[c])
                        for (unsigned j = 0; j < n; ++j)
                            weights[a * n + j] +=
                                (gram[b][c] * (std::conj(tensor[b][j]) * tensor[c][j] +
                                               std::conj(tensor[b][n + j]) * tensor[c][n + j]))
                                    .real();
        }
    }
};
std::pair<unsigned, double> draw(const double* weights, unsigned count, std::mt19937_64& rng) {
    double total = 0;
    for (unsigned k = 0; k < count; ++k) {
        assert(std::isfinite(weights[k]) && weights[k] > -1e-10);
        total += std::max(0., weights[k]);
    }
    assert(total > 0);
    double target = ((rng() >> 11) * 0x1.0p-53) * total, sum = 0;
    for (unsigned k = 0; k < count; ++k) {
        sum += std::max(0., weights[k]);
        if (sum > target)
            return {k, weights[k] / total};
    }
    assert(false);
    return {0, 0};
}
struct Growth {
    struct Ancilla {
        unsigned row, q;
        std::array<Pair, 2> bras;
    };
    struct Projector {
        unsigned row;
        std::array<unsigned, 128> targets;
        std::array<C, 128> phases;
    };
    PayloadPlan payload;
    Code code;
    Affine records;
    std::vector<Row> inputs, logical;
    unsigned random_power, y_sign;
    std::vector<unsigned> data_rows, unprojected;
    std::vector<std::array<unsigned, 2>> duals;
    std::vector<Ancilla> ancillas;
    std::array<std::array<C, 128>, 2> decoder;
    std::vector<Projector> projectors;
    explicit Growth(Reader& r) : payload(r), code(r), records(r) {
        inputs = r.rows();
        logical = r.rows();
        random_power = r.get();
        y_sign = r.get();
        data_rows = r.ints();
        duals = r.pairs();
        ancillas.resize(r.count());
        for (auto& a : ancillas) {
            a.row = r.get();
            a.q = r.get();
            for (auto& pair : a.bras)
                for (auto& z : pair)
                    z = r.complex();
        }
        unprojected = r.ints();
        for (auto& row : decoder)
            for (auto& z : row)
                z = r.complex();
        projectors.resize(r.count());
        for (auto& p : projectors) {
            p.row = r.get();
            for (unsigned j = 0; j < 128; ++j) {
                p.targets[j] = r.get();
                p.phases[j] = r.complex();
            }
        }
    }
};
struct GrowthState {
    std::array<std::array<C, 128>, 2> vectors{};
    std::array<std::array<Pair, 42>, 2> ancillas{};
    std::array<C, 2> weights{};
    double norm(unsigned count) const {
        C result{};
        for (unsigned a = 0; a < 2; ++a)
            for (unsigned b = 0; b < 2; ++b) {
                C data{}, scalar = std::conj(weights[a]) * weights[b];
                for (unsigned j = 0; j < 128; ++j)
                    data += std::conj(vectors[a][j]) * vectors[b][j];
                for (unsigned q = 0; q < count; ++q)
                    scalar *= dot(ancillas[a][q], ancillas[b][q]);
                result += scalar * data;
            }
        return result.real();
    }
};
struct Noise {
    double probability;
    std::vector<std::vector<unsigned>> choices;
};
struct Sampler {
    unsigned events, fault_count;
    std::vector<Noise> sites;
    Affine injection;
    std::vector<Row> injection_rows;
    std::array<Pair, 4> initial;
    std::vector<Growth> growth;
    std::vector<PayloadPlan> terminal;
    std::vector<Code> code;
    std::vector<std::array<unsigned, 2>> duals;
    std::vector<unsigned> terminal_events;
    std::vector<std::array<int, 2>> visible;
    std::vector<std::vector<unsigned>> detectors, observables;
    Bits records, faults, packed, values;
    std::vector<int> choices;
    std::vector<unsigned> outputs, detector_bits, observable_bits;
    CodeScratch scratch;
    Pair logical{};
    double probability = 1;
    std::mt19937_64 rng;
    static Affine read_prefix(Reader& r, unsigned& events, unsigned& fault_count,
                              std::vector<Noise>& sites, bool& has_growth) {
        events = r.get();
        fault_count = r.get();
        has_growth = r.get();
        sites.resize(r.count());
        if (events > 256 || fault_count > 32768)
            throw std::runtime_error("plan exceeds bit capacity");
        for (auto& s : sites) {
            s.probability = r.get<double>();
            s.choices.resize(r.count());
            for (auto& c : s.choices)
                c = r.ints();
        }
        return Affine(r);
    }
    explicit Sampler(Reader& r, bool& has_growth)
        : injection(read_prefix(r, events, fault_count, sites, has_growth)) {
        injection_rows = r.rows();
        for (auto& p : initial)
            for (auto& z : p)
                z = r.complex();
        if (has_growth)
            growth.emplace_back(r);
        terminal.emplace_back(r);
        code.emplace_back(r);
        duals = r.pairs();
        terminal_events = r.ints();
        visible.resize(r.count());
        for (auto& p : visible) {
            p[0] = r.get<int>();
            p[1] = r.get<int>();
        }
        detectors.resize(r.count());
        for (auto& d : detectors)
            d = r.ints();
        observables.resize(r.count());
        for (auto& o : observables)
            o = r.ints();
        records.resize((events + 63) / 64);
        faults.resize((fault_count + 63) / 64);
        packed.resize((fault_count + 65) / 64);
        values.resize(4);
        choices.resize(sites.size());
        outputs.resize(visible.size());
        detector_bits.resize(detectors.size());
        observable_bits.resize(observables.size());
    }
    void validate() const {
        auto require = [](bool ok) {
            if (!ok)
                throw std::invalid_argument("invalid compiled sampler layout");
        };
        auto mask = [&](const Mask& m, unsigned count) {
            for (auto word : m.words) {
                require(word.index < (count + 63) / 64);
                if (word.index == count / 64 && count % 64)
                    require((word.mask >> (count % 64)) == 0);
            }
        };
        auto row = [&](const Row& r) {
            mask(r.records, events);
            mask(r.faults, fault_count);
            require(r.constant < 2);
        };
        auto affine = [&](const Affine& a, unsigned count) {
            for (auto k : a.free)
                require(k < events);
            for (auto k : a.events)
                require(k < events);
            for (const auto& p : a.pivots) {
                require(p.event < events);
                row(p.row);
                mask(p.rhs, count);
            }
        };
        auto payload = [&](const PayloadPlan& p) {
            require(p.width == 7 || p.width == 19);
            require(p.duals.size() == p.width - 1);
            for (const auto& r : p.rows)
                row(r);
            for (auto d : p.duals)
                require(d[0] < (1u << p.width) && d[1] < (1u << p.width));
            for (auto i : p.initial)
                require(i[0] < p.ancillas && i[1] > 0 && i[1] < 4 && i[2] < p.rows.size());
            unsigned gadgets = 0;
            for (const auto& step : p.steps) {
                require(step.kind < 3);
                if (step.kind == 0)
                    require(step.data < 2 && step.q < (step.data ? p.width : p.ancillas) &&
                            step.x < fault_count && step.z < fault_count);
                else if (step.kind == 1)
                    require(step.q < p.ancillas && step.event < events);
                else {
                    require(++gadgets <= 2 && step.event < events && step.hidden >= -1 &&
                            step.hidden < int(events) && step.outcome_bit == fault_count + 1);
                    mask(step.reset_flip, fault_count);
                    require(step.dm.size() == p.width);
                    for (const auto& branch : step.branches) {
                        require(branch.flips.size() <= 42);
                        for (auto k : step.dm)
                            require(k < branch.flips.size());
                        for (auto a : step.am)
                            require(a[0] < p.ancillas && a[1] < branch.flips.size());
                        for (const auto& f : branch.flips)
                            mask(f, fault_count + 2);
                        auto poly = [&](const Polynomial& p) {
                            for (const auto& t : p.products) {
                                require(t.c < 8);
                                mask(t.a, fault_count + 2);
                                mask(t.b, fault_count + 2);
                            }
                        };
                        poly(branch.phase);
                        for (const auto& l : branch.linear)
                            poly(l);
                    }
                }
            }
            require(gadgets > 0);
        };
        auto code_layout = [&](const Code& c) {
            require(c.width == 7 || c.width == 19);
            require(c.xchecks.size() + c.zchecks.size() == c.width - 1);
            require(c.span.size() == (1u << c.xchecks.size()) &&
                    c.coordinates.size() == c.xchecks.size());
            require(c.order.size() == c.width - 1);
            for (auto x : c.xchecks)
                require(x < (1u << c.width));
            for (auto z : c.zchecks)
                require(z < (1u << c.width));
            for (auto x : c.span)
                require(x < (1u << c.width));
            for (auto x : c.source_coordinates)
                require(x < c.span.size());
            for (auto x : c.order)
                require(x[0] < 2 && x[1] < (x[0] ? c.xchecks.size() : c.zchecks.size()));
        };
        require(injection_rows.size() == 2);
        affine(injection, injection.pivots.size());
        for (const auto& r : injection_rows)
            row(r);
        for (const auto& s : sites) {
            require(std::isfinite(s.probability) && s.probability >= 0 && s.probability <= 1 &&
                    !s.choices.empty());
            for (const auto& c : s.choices)
                for (auto slot : c)
                    require(slot < fault_count);
        }
        payload(terminal[0]);
        code_layout(code[0]);
        require(terminal[0].width == code[0].width && duals.size() == code[0].order.size() &&
                terminal_events.size() == duals.size());
        for (auto d : duals)
            require(d[0] < (1u << code[0].width) && d[1] < (1u << code[0].width));
        for (auto e : terminal_events)
            require(e < events);
        for (const auto& g : growth) {
            payload(g.payload);
            code_layout(g.code);
            require(g.code.width == 7 && g.payload.width == 7);
            require(g.inputs.size() < 256 && g.logical.size() == 3 &&
                    g.data_rows.size() == g.duals.size() && g.data_rows.size() == 6);
            require(g.random_power < 256 && g.y_sign < 2);
            affine(g.records, g.records.pivots.size());
            for (const auto& r : g.inputs)
                row(r);
            for (const auto& r : g.logical)
                row(r);
            for (auto k : g.data_rows)
                require(k < g.inputs.size());
            for (auto d : g.duals)
                require(d[0] < 128 && d[1] < 128);
            for (auto q : g.unprojected)
                require(q < g.payload.ancillas);
            for (const auto& a : g.ancillas)
                require(a.row < g.inputs.size() && a.q < g.payload.ancillas);
            for (const auto& p : g.projectors) {
                require(p.row < g.inputs.size());
                for (auto j : p.targets)
                    require(j < 128);
            }
        }
        for (auto v : visible)
            require(v[0] >= 0 && v[0] < int(events) && v[1] >= -1 && v[1] < int(fault_count));
        for (const auto& d : detectors)
            for (auto k : d)
                require(k < visible.size());
        for (const auto& o : observables)
            for (auto k : o)
                require(k < visible.size());
    }
    double sample_payload(const PayloadPlan& plan, const Code& basis, Payload& payload,
                          std::array<unsigned, 2>& frame) {
        frame = plan.start(payload, records, faults);
        double chance = 1;
        for (const auto& step : plan.steps) {
            if (step.kind == 0) {
                plan.fault(payload, step, faults);
                continue;
            }
            std::array<Payload, 2> candidates;
            std::array<double, 2> weights{};
            for (unsigned b = 0; b < 2; ++b) {
                if (step.kind == 2)
                    plan.expand(payload, candidates[b], step, packed, b);
                else {
                    candidates[b] = payload;
                    plan.measure(candidates[b], step, b);
                }
                weights[b] = scratch.norm(basis, logical, frame, candidates[b], plan.ancillas);
            }
            auto [b, p] = draw(weights.data(), 2, rng);
            chance *= p;
            payload = candidates[b];
            for (unsigned k = 0; k < payload.count; ++k)
                payload.terms[k].weight /= std::sqrt(weights[b]);
            set(records, step.event, b);
            if (step.kind == 2 && step.hidden >= 0)
                set(records, step.hidden, b ^ step.reset_flip.eval(faults));
        }
        return chance;
    }
    void sample_growth(const Growth& g, Payload& payload, const std::array<unsigned, 2>& frame) {
        assert(payload.count == 2);
        GrowthState state;
        for (unsigned k = 0; k < 2; ++k) {
            state.weights[k] = payload.terms[k].weight;
            state.ancillas[k] = payload.terms[k].ancillas;
            for (unsigned s = 0; s < 2; ++s)
                for (auto raw : g.code.span) {
                    unsigned b = raw ^ (s ? 127 : 0);
                    auto [out, phase] = payload.terms[k].op.column(b ^ frame[0], 7);
                    state.vectors[k][out] +=
                        logical[s] * roots[(phase + 4 * parity(b & frame[1])) & 7] / std::sqrt(8.);
                }
        }
        std::fill(values.begin(), values.end(), 0);
        double chance = 1;
        for (const auto& a : g.ancillas) {
            std::array<GrowthState, 2> candidates{state, state};
            std::array<double, 2> weights{};
            for (unsigned b = 0; b < 2; ++b) {
                for (unsigned k = 0; k < 2; ++k) {
                    auto input = state.ancillas[k][a.q];
                    C scalar = a.bras[b][0] * input[0] + a.bras[b][1] * input[1];
                    candidates[b].ancillas[k][a.q] = {scalar * std::conj(a.bras[b][0]),
                                                      scalar * std::conj(a.bras[b][1])};
                }
                weights[b] = candidates[b].norm(g.payload.ancillas);
            }
            auto [b, p] = draw(weights.data(), 2, rng);
            chance *= p;
            state = candidates[b];
            set(values, a.row, b);
        }
        for (const auto& op : g.projectors) {
            std::array<GrowthState, 2> candidates{state, state};
            std::array<double, 2> weights{};
            for (unsigned b = 0; b < 2; ++b) {
                for (unsigned k = 0; k < 2; ++k)
                    for (unsigned j = 0; j < 128; ++j)
                        candidates[b].vectors[k][op.targets[j]] =
                            .5 * (state.vectors[k][op.targets[j]] +
                                  (b ? -1. : 1.) * op.phases[j] * state.vectors[k][j]);
                weights[b] = candidates[b].norm(g.payload.ancillas);
            }
            auto [b, p] = draw(weights.data(), 2, rng);
            chance *= p;
            state = candidates[b];
            set(values, op.row, b);
        }
        g.records.sample(records, faults, values, rng);
        unsigned cx = 0, cz = 0;
        for (unsigned k = 0; k < g.data_rows.size(); ++k)
            if (bit(values, g.data_rows[k])) {
                cx ^= g.duals[k][0];
                cz ^= g.duals[k][1];
            }
        Pair result{};
        for (unsigned k = 0; k < 2; ++k) {
            const auto& term = payload.terms[k];
            C scalar = term.weight;
            for (const auto& a : g.ancillas) {
                const auto& bra = a.bras[bit(values, a.row)];
                scalar *= bra[0] * term.ancillas[a.q][0] + bra[1] * term.ancillas[a.q][1];
            }
            for (auto q : g.unprojected)
                scalar *= dot(payload.terms[0].ancillas[q], term.ancillas[q]);
            for (unsigned s = 0; s < 2; ++s)
                for (auto raw : g.code.span) {
                    unsigned b = raw ^ (s ? 127 : 0);
                    auto [out, phase] = term.op.column(b ^ frame[0], 7);
                    out ^= cx;
                    phase += 4 * (parity(b & frame[1]) + parity(out & cz));
                    C amplitude = scalar * logical[s] * roots[phase & 7] / std::sqrt(8.);
                    for (unsigned t = 0; t < 2; ++t)
                        result[t] += amplitude * g.decoder[t][out];
                }
        }
        unsigned sx = g.logical[0].eval(records, faults), sy = g.logical[1].eval(records, faults),
                 sz = g.logical[2].eval(records, faults);
        assert(sy == (sx ^ sz ^ g.y_sign));
        if (sx)
            result[1] = -result[1];
        if (sz)
            std::swap(result[0], result[1]);
        assert(std::abs(norm(result) - chance) < 1e-10 * chance);
        logical = result;
        normalize(logical);
        probability *= std::ldexp(chance, -int(g.random_power));
    }
    void shot() {
        std::fill(records.begin(), records.end(), 0);
        std::fill(faults.begin(), faults.end(), 0);
        std::fill(values.begin(), values.end(), 0);
        for (unsigned k = 0; k < sites.size(); ++k) {
            choices[k] = -1;
            if ((rng() >> 11) * 0x1.0p-53 < sites[k].probability) {
                const uint64_t bound = sites[k].choices.size();
                const uint64_t threshold = -bound % bound;
                uint64_t word;
                do {
                    word = rng();
                } while (word < threshold);
                unsigned choice = word % bound;
                choices[k] = choice;
                for (auto slot : sites[k].choices[choice])
                    set(faults, slot, 1);
            }
        }
        std::fill(packed.begin(), packed.end(), 0);
        set(packed, 0, 1);
        for (unsigned k = 0; k < fault_count; ++k)
            if (bit(faults, k))
                set(packed, k + 1, 1);
        injection.sample(records, faults, values, rng);
        unsigned signs = 0;
        for (unsigned k = 0; k < injection_rows.size(); ++k)
            signs |= injection_rows[k].eval(records, faults) << k;
        logical = initial[signs];
        probability = norm(logical);
        normalize(logical);
        Payload payload;
        std::array<unsigned, 2> frame{};
        if (!growth.empty()) {
            const auto& g = growth[0];
            probability *= sample_payload(g.payload, g.code, payload, frame);
            sample_growth(g, payload, frame);
        }
        const auto& plan = terminal[0];
        const auto& basis = code[0];
        probability *= sample_payload(plan, basis, payload, frame);
        scratch.distribution(basis, logical, frame, payload, plan.ancillas);
        unsigned n = basis.span.size();
        auto [selected, chance] = draw(scratch.weights.data(), payload.count * n, rng);
        unsigned xs = selected % n, zs = scratch.labels[selected / n], cx = 0, cz = 0;
        for (unsigned k = 0; k < basis.order.size(); ++k) {
            auto [x, index] = basis.order[k];
            unsigned sign = ((x ? xs : zs) >> index) & 1;
            set(records, terminal_events[k], sign);
            if (sign) {
                cx ^= duals[k][0];
                cz ^= duals[k][1];
            }
        }
        Pair result{};
        for (unsigned k = 0; k < payload.count; ++k) {
            const auto& term = payload.terms[k];
            C scalar = term.weight;
            for (unsigned q = 0; q < plan.ancillas; ++q)
                scalar *= dot(payload.terms[0].ancillas[q], term.ancillas[q]);
            for (unsigned s = 0; s < 2; ++s)
                for (auto raw : basis.span) {
                    unsigned b = raw ^ (s ? (1 << basis.width) - 1 : 0);
                    auto [out, phase] = term.op.column(b ^ frame[0], basis.width);
                    out ^= cx;
                    phase += 4 * (parity(b & frame[1]) + parity(out & cz));
                    if (!basis.label(out))
                        result[parity(out)] += scalar * logical[s] * roots[phase & 7] / double(n);
                }
        }
        assert(std::abs(norm(result) - chance) < 1e-10 * chance);
        logical = result;
        normalize(logical);
        probability *= chance;
        for (unsigned k = 0; k < visible.size(); ++k)
            outputs[k] =
                bit(records, visible[k][0]) ^ (visible[k][1] < 0 ? 0 : bit(faults, visible[k][1]));
        for (unsigned k = 0; k < detectors.size(); ++k) {
            unsigned v = 0;
            for (auto d : detectors[k])
                v ^= outputs[d];
            detector_bits[k] = v;
        }
        for (unsigned k = 0; k < observables.size(); ++k) {
            unsigned v = 0;
            for (auto o : observables[k])
                v ^= outputs[o];
            observable_bits[k] = v;
        }
    }
    void print() const {
        std::cout << "{\"fault_choices\":[";
        bool first = true;
        for (unsigned k = 0; k < choices.size(); ++k)
            if (choices[k] >= 0) {
                std::cout << (first ? "" : ",") << '[' << k << ',' << choices[k] << ']';
                first = false;
            }
        std::cout << "],\"outcomes\":[";
        for (unsigned k = 0; k < events; ++k)
            std::cout << (k ? "," : "") << bit(records, k);
        std::cout << "],\"probability\":" << probability << ",\"logical\":[[" << logical[0].real()
                  << ',' << logical[0].imag() << "],[" << logical[1].real() << ','
                  << logical[1].imag() << "]],\"records\":[";
        for (unsigned k = 0; k < outputs.size(); ++k)
            std::cout << (k ? "," : "") << outputs[k];
        std::cout << "],\"detectors\":[";
        for (unsigned k = 0; k < detector_bits.size(); ++k)
            std::cout << (k ? "," : "") << detector_bits[k];
        std::cout << "],\"observables\":[";
        for (unsigned k = 0; k < observable_bits.size(); ++k)
            std::cout << (k ? "," : "") << observable_bits[k];
        std::cout << "]}\n";
    }
};
int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "usage: msc_sampler_native PLAN SEED SHOTS sample|benchmark");
        Reader reader(argv[1]);
        bool growth = false;
        Sampler sampler(reader, growth);
        sampler.validate();
        sampler.rng.seed(std::stoull(argv[2]));
        unsigned shots = std::stoul(argv[3]);
        std::string mode(argv[4]);
        if (!shots || (mode != "sample" && mode != "benchmark"))
            throw std::invalid_argument("invalid run mode");
        std::cout << std::setprecision(17);
        unsigned accepted = 0;
        double checksum = 0;
        auto start = std::chrono::steady_clock::now();
        for (unsigned shot = 0; shot < shots; ++shot) {
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            size_t before = allocation_count;
#endif
            sampler.shot();
#ifdef CLIFFT_MSC_CHECK_ALLOCATIONS
            assert(allocation_count == before);
#endif
            accepted += std::none_of(sampler.detector_bits.begin(), sampler.detector_bits.end(),
                                     [](auto b) { return b != 0; });
            checksum += sampler.probability + sampler.logical[0].real();
            if (mode == "sample")
                sampler.print();
        }
        auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (mode == "benchmark")
            std::cout << "{\"shots\":" << shots << ",\"accepted\":" << accepted
                      << ",\"seconds\":" << elapsed
                      << ",\"microseconds_per_shot\":" << elapsed * 1e6 / shots
                      << ",\"checksum\":" << checksum << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
