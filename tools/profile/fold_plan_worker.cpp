// Reusable research worker; all loading and certification precede native sampling.
#include "fold_recognition_main.h"

#include <cstring>
#include <memory>

namespace {
using namespace fold_blocks;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

struct Reader {
    std::ifstream input;
    std::vector<std::shared_ptr<void>> owners;
    size_t allocated = 0;

    explicit Reader(const char* path) : input(path, std::ios::binary) {
        char magic[8];
        input.read(magic, 8);
        require(input && std::memcmp(magic, "FLDBLK01", 8) == 0, "invalid plan header");
    }
    uint64_t u() {
        unsigned char bytes[8];
        input.read(reinterpret_cast<char*>(bytes), 8);
        require(bool(input), "truncated plan");
        uint64_t value = 0;
        for (unsigned k = 0; k < 8; ++k)
            value |= uint64_t(bytes[k]) << (8 * k);
        return value;
    }
    unsigned n(unsigned limit = 1000000) {
        auto value = u();
        require(value <= limit, "plan integer exceeds capacity");
        return static_cast<unsigned>(value);
    }
    int fault() { return int(n(8192)) - 1; }
    Mask mask() {
        Mask lo = u();
        return lo | (Mask(u()) << 64);
    }
    template <class T>
    const T* own(T value) {
        auto pointer = std::make_shared<T>(std::move(value));
        auto result = pointer.get();
        owners.push_back(std::move(pointer));
        return result;
    }
    template <class T, class F>
    std::span<const T> seq(F read, unsigned limit = 1000000) {
        unsigned count = n(limit);
        allocated += size_t(count) * sizeof(T);
        require(allocated <= 64 * 1024 * 1024, "plan storage exceeds capacity");
        std::vector<T> values;
        values.reserve(count);
        for (unsigned k = 0; k < count; ++k)
            values.push_back(read());
        return *own(std::move(values));
    }
    std::string_view string() {
        unsigned size = n(4000000);
        allocated += size;
        require(allocated <= 64 * 1024 * 1024, "plan storage exceeds capacity");
        std::string value(size, '\0');
        input.read(value.data(), size);
        require(bool(input), "truncated string");
        return *own(std::move(value));
    }
    std::span<const Mask> masks() {
        return seq<Mask>([&] { return mask(); }, 128);
    }
    Map map() {
        std::vector<Word> words;
        auto rows = seq<Row>(
            [&] {
                unsigned start = static_cast<unsigned>(words.size());
                unsigned count = n(128);
                for (unsigned k = 0; k < count; ++k)
                    words.push_back({n(127), u()});
                return Row{start, count};
            },
            128);
        return {rows, *own(std::move(words))};
    }
    Linear linear() { return {map(), map(), map()}; }
    static void bounded(Mask value, unsigned width) {
        require(width == 128 || (value >> width) == 0, "plan mask exceeds width");
    }
    const Geometry* geometry() {
        Geometry g{n(85),
                   n(85),
                   n(2851),
                   mask(),
                   masks(),
                   seq<Edge>([&] { return Edge{n(84), n(84)}; }, 128),
                   seq<Leaf>([&] { return Leaf{n(2851), n(2851)}; }, 213),
                   seq<Step>([&] { return Step{n(2851), n(2851), n(213), n()}; }),
                   seq<unsigned>([&] { return n(2850); }),
                   seq<unsigned>([&] { return n(1); }, 5702),
                   seq<unsigned>([&] { return n(2850); }, 213)};
        require(g.width > 0 && g.rank <= g.width, "invalid geometry rank");
        bounded(g.logical_z, g.width);
        for (auto m : g.z_masks)
            bounded(m, g.width);
        for (auto e : g.edges)
            require(e.q < g.width && e.r < g.width, "invalid geometry edge");
        require(g.leaves.size() == g.width + g.edges.size() &&
                    g.multipliers.size() == 2 * g.leaf_storage,
                "invalid geometry leaves");
        std::array<bool, 2851> initialized{};
        for (auto l : g.leaves) {
            require(l.size && l.offset + l.size <= g.leaf_storage, "invalid leaf range");
            for (unsigned k = l.offset; k < l.offset + l.size; ++k)
                initialized[k] = true;
        }
        for (auto s : g.steps) {
            require(s.size && s.inputs && s.offset + s.size <= 2851 &&
                        uint64_t(s.gather) + uint64_t(s.inputs) * 2 * s.size <= g.gathers.size(),
                    "invalid contraction range");
            // Outputs must not overwrite an input used by a later element of this step.
            for (unsigned k = s.gather; k < s.gather + s.inputs * 2 * s.size; ++k) {
                auto index = g.gathers[k];
                require(initialized[index] && (index < s.offset || index >= s.offset + s.size),
                        "invalid contraction dependency");
            }
            for (unsigned k = s.offset; k < s.offset + s.size; ++k)
                initialized[k] = true;
        }
        for (auto k : g.outputs)
            require(initialized[k], "uninitialized contraction output");
        return own(g);
    }
    const Fold* fold(unsigned width) {
        Fold f{geometry(),
               n(8),
               mask(),
               linear(),
               map(),
               seq<Action>(
                   [&] { return Action{n(5), n(84), n(84), n(84), n(127), fault(), fault()}; }),
               seq<int>([&] { return int(n(2)) - 1; }, 65536)};
        require(f.geometry->width == width && f.cats && f.preparation.x.rows.size() == f.cats &&
                    f.preparation.z.rows.size() == f.cats && f.records.rows.size() == f.cats &&
                    f.decode.size() == size_t(1) << (2 * f.cats),
                "invalid fold dimensions");
        bounded(f.equal_flag_mask, static_cast<unsigned>(f.preparation.records.rows.size()));
        for (auto a : f.actions) {
            require(a.q < (a.kind == 1 || a.kind >= 4 ? f.cats : width), "invalid action qubit");
            if (a.kind >= 4)
                require(a.r < width, "invalid controlled target");
            if (a.kind == 5)
                require(a.s < width && a.edge < f.geometry->edges.size(), "invalid edge action");
        }
        return own(f);
    }
    const Boundary* boundary(unsigned& width) {
        Boundary b{geometry(),
                   linear(),
                   masks(),
                   masks(),
                   seq<Pair>([&] { return Pair{mask(), mask()}; }, 128),
                   seq<Pair>([&] { return Pair{mask(), mask()}; }, 128)};
        require(b.before->width == width && b.noise.x.rows.size() <= 85 &&
                    b.noise.x.rows.size() == b.noise.z.rows.size() &&
                    b.syndrome.size() == b.duals.size() && b.duals.size() == b.transported.size(),
                "invalid boundary dimensions");
        for (auto m : b.constraints)
            bounded(m, static_cast<unsigned>(b.noise.records.rows.size()));
        for (auto m : b.syndrome)
            bounded(m, static_cast<unsigned>(b.noise.records.rows.size()));
        for (auto pair : b.duals)
            for (auto m : pair)
                bounded(m, width);
        width = static_cast<unsigned>(b.noise.x.rows.size());
        for (auto pair : b.transported)
            for (auto m : pair)
                bounded(m, width);
        return own(b);
    }
    fold_recognition::FamilySource source() {
        auto distance = n(7);
        require(distance == 3 || distance == 5 || distance == 7, "unsupported distance");
        auto body = string();
        auto checks = seq<std::string_view>([&] { return string(); }, 84);
        std::array<std::string_view, 3> axes{string(), string(), string()};
        auto flags = seq<std::array<unsigned, 6>>(
            [&] {
                std::array<unsigned, 6> result;
                for (auto& v : result)
                    v = n();
                return result;
            },
            2);
        auto prefix = linear();
        unsigned width = static_cast<unsigned>(prefix.x.rows.size()), live = 1;
        require(width == 13 && prefix.z.rows.size() == width, "invalid initial code");
        auto stages = seq<Stage>(
            [&] {
                bool is_fold = n(1);
                live = is_fold ? live * 2 : 1;
                require(live <= 4, "too many coherent terms");
                return is_fold ? Stage{fold(width), nullptr} : Stage{nullptr, boundary(width)};
            },
            64);
        auto noise = seq<NoiseSite>(
            [&] {
                NoiseSite s{n(3), n(1), {fault(), fault(), fault()}, {fault(), fault(), fault()}};
                require(s.arity && (!s.flip || s.arity == 1), "invalid noise arity");
                for (unsigned k = 0; k < s.arity; ++k)
                    require(s.x[k] >= 0 && (s.flip || s.z[k] >= 0), "missing noise fault bit");
                return s;
            },
            8192);
        auto data_width = distance * distance + (distance - 1) * (distance - 1);
        require(!stages.empty() && stages.back().boundary && width == data_width,
                "invalid terminal code");
        auto parsed = clifft::parse(body);
        for (auto group : flags)
            for (auto k : group)
                require(k < parsed.num_measurements, "invalid flag record");
        return {
            distance, data_width, body, checks, axes, flags, own(Protocol{prefix, stages, noise})};
    }
    std::span<const Fixture> fixtures() {
        auto result = seq<Fixture>(
            [&] {
                Fixture f;
                for (auto& v : f.history)
                    v = u();
                for (auto& v : f.expected) {
                    v = std::bit_cast<double>(u());
                    require(std::isfinite(v), "nonfinite fixture");
                }
                return f;
            },
            10000);
        require(input.peek() == std::char_traits<char>::eof(), "trailing plan data");
        return result;
    }
};

void parse_json(const char* path) {
    std::ifstream input(path);
    require(bool(input), "cannot read circuit");
    auto circuit = clifft::parse(std::string(std::istreambuf_iterator<char>(input), {}));
    std::cout << std::setprecision(17) << "{\"qubits\":" << circuit.num_qubits << ",\"nodes\":[";
    bool first = true;
    for (const auto& node : circuit.nodes) {
        const auto& traits = clifft::gate_traits(node.gate);
        std::cout << (first ? "" : ",") << "{\"gate\":" << std::quoted(std::string(traits.name))
                  << ",\"noise\":" << traits.noise << ",\"tagged\":" << !node.tag.empty()
                  << ",\"clifford\":" << fold_recognition::plain_clifford(node) << ",\"args\":";
        fold_recognition::json_array<double>(node.args);
        std::cout << ",\"targets\":[";
        for (size_t k = 0; k < node.targets.size(); ++k)
            std::cout << (k ? "," : "") << node.targets[k].bits;
        std::cout << "]}";
        first = false;
    }
    std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--parse") {
            parse_json(argv[2]);
            return 0;
        }
        require(
            argc == 8 || (argc == 3 && std::string_view(argv[1]) == "--check"),
            "usage: worker PLAN CIRCUIT SHOTS SEED KEEP_RECORDS REPEATS REQUEST or --check PLAN");
        bool check = argc == 3;
        Reader reader(argv[check ? 2 : 1]);
        std::array sources{reader.source()};
        auto fixtures = reader.fixtures();
        Executor executor;
        double error = 0;
        for (const auto& f : fixtures) {
            auto result = executor.evaluate(*sources[0].kernel, f.history);
            for (unsigned k = 0; k < 4; ++k) {
                require(std::isfinite(result[k]), "nonfinite native result");
                error = std::max(error, std::abs(result[k] - f.expected[k]));
            }
        }
        require(error <= 2e-12, "native fixture mismatch");
        if (check) {
            std::cout << std::setprecision(17) << "{\"fixtures\":" << fixtures.size()
                      << ",\"max_error\":" << error << "}\n";
            return 0;
        }
        return fold_recognition::run(sources, argc - 1, argv + 1);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
