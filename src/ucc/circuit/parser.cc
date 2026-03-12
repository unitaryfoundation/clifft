#include "ucc/circuit/parser.h"

#include "ucc/util/config.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fast_float/fast_float.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ucc {

namespace {

// Gate name lookup table.
const std::unordered_map<std::string_view, GateType> kGateNames = {
    // Single-qubit Clifford
    {"H", GateType::H},
    {"H_XZ", GateType::H},  // Stim alias
    {"S", GateType::S},
    {"S_DAG", GateType::S_DAG},
    {"SQRT_Z", GateType::S},          // Stim alias
    {"SQRT_Z_DAG", GateType::S_DAG},  // Stim alias
    {"X", GateType::X},
    {"Y", GateType::Y},
    {"Z", GateType::Z},
    // Additional single-qubit Cliffords
    {"SQRT_X", GateType::SQRT_X},
    {"SQRT_X_DAG", GateType::SQRT_X_DAG},
    {"SQRT_Y", GateType::SQRT_Y},
    {"SQRT_Y_DAG", GateType::SQRT_Y_DAG},
    {"H_XY", GateType::H_XY},
    {"H_YZ", GateType::H_YZ},
    {"H_NXY", GateType::H_NXY},
    {"H_NXZ", GateType::H_NXZ},
    {"H_NYZ", GateType::H_NYZ},
    {"C_XYZ", GateType::C_XYZ},
    {"C_ZYX", GateType::C_ZYX},
    {"C_NXYZ", GateType::C_NXYZ},
    {"C_NZYX", GateType::C_NZYX},
    {"C_XNYZ", GateType::C_XNYZ},
    {"C_XYNZ", GateType::C_XYNZ},
    {"C_ZNYX", GateType::C_ZNYX},
    {"C_ZYNX", GateType::C_ZYNX},
    // Non-Clifford
    {"T", GateType::T},
    {"T_DAG", GateType::T_DAG},
    // Parameterized rotations
    // Note: RX, RY, RZ without underscore are Stim aliases for resets,
    // so rotation gates always use the underscore form R_X, R_Y, R_Z.
    {"R_X", GateType::R_X},
    {"R_Y", GateType::R_Y},
    {"R_Z", GateType::R_Z},
    {"U3", GateType::U3},
    {"U", GateType::U3},
    {"R_XX", GateType::R_XX},
    {"RXX", GateType::R_XX},
    {"R_YY", GateType::R_YY},
    {"RYY", GateType::R_YY},
    {"R_ZZ", GateType::R_ZZ},
    {"RZZ", GateType::R_ZZ},
    {"R_PAULI", GateType::R_PAULI},
    // Two-qubit Clifford
    {"CX", GateType::CX},
    {"CNOT", GateType::CX},  // Alias
    {"ZCX", GateType::CX},   // Stim alias
    {"CY", GateType::CY},
    {"ZCY", GateType::CY},  // Stim alias
    {"CZ", GateType::CZ},
    {"ZCZ", GateType::CZ},  // Stim alias
    // Additional two-qubit Cliffords
    {"SWAP", GateType::SWAP},
    {"ISWAP", GateType::ISWAP},
    {"ISWAP_DAG", GateType::ISWAP_DAG},
    {"SQRT_XX", GateType::SQRT_XX},
    {"SQRT_XX_DAG", GateType::SQRT_XX_DAG},
    {"SQRT_YY", GateType::SQRT_YY},
    {"SQRT_YY_DAG", GateType::SQRT_YY_DAG},
    {"SQRT_ZZ", GateType::SQRT_ZZ},
    {"SQRT_ZZ_DAG", GateType::SQRT_ZZ_DAG},
    {"CXSWAP", GateType::CXSWAP},
    {"CZSWAP", GateType::CZSWAP},
    {"SWAPCZ", GateType::CZSWAP},  // Stim alias
    {"SWAPCX", GateType::SWAPCX},
    {"XCX", GateType::XCX},
    {"XCY", GateType::XCY},
    {"XCZ", GateType::XCZ},
    {"YCX", GateType::YCX},
    {"YCY", GateType::YCY},
    {"YCZ", GateType::YCZ},
    // Measurements
    {"M", GateType::M},
    {"MZ", GateType::M},  // Stim alias
    {"MX", GateType::MX},
    {"MY", GateType::MY},
    {"MR", GateType::MR},
    {"MRZ", GateType::MR},  // Stim alias
    {"MRX", GateType::MRX},
    {"MPP", GateType::MPP},
    {"MXX", GateType::MXX},
    {"MYY", GateType::MYY},
    {"MZZ", GateType::MZZ},
    // Resets
    {"R", GateType::R},
    {"RZ", GateType::R},  // Stim alias
    {"RX", GateType::RX},
    {"RY", GateType::RY},
    {"MRY", GateType::MRY},
    // Deterministic padding
    {"MPAD", GateType::MPAD},
    // Identity no-ops
    {"I", GateType::I},
    {"II", GateType::II},
    {"I_ERROR", GateType::I_ERROR},
    {"II_ERROR", GateType::II_ERROR},
    // Noise channels
    {"X_ERROR", GateType::X_ERROR},
    {"Y_ERROR", GateType::Y_ERROR},
    {"Z_ERROR", GateType::Z_ERROR},
    {"DEPOLARIZE1", GateType::DEPOLARIZE1},
    {"DEPOLARIZE2", GateType::DEPOLARIZE2},
    {"PAULI_CHANNEL_1", GateType::PAULI_CHANNEL_1},
    {"PAULI_CHANNEL_2", GateType::PAULI_CHANNEL_2},
    // QEC annotations
    {"DETECTOR", GateType::DETECTOR},
    {"OBSERVABLE_INCLUDE", GateType::OBSERVABLE_INCLUDE},
    // Annotations
    {"TICK", GateType::TICK},
};

// Coordinate annotations to silently discard (no AST nodes emitted).
const std::unordered_set<std::string_view> kDiscardedAnnotations = {
    "QUBIT_COORDS",
    "SHIFT_COORDS",
};

// Trim whitespace from both ends.
std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.remove_suffix(1);
    }
    return s;
}

// Parse an integer from string_view.
bool parse_int(std::string_view s, int& out) {
    auto result = std::from_chars(s.data(), s.data() + s.size(), out);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size();
}

// Parse an unsigned integer from string_view.
bool parse_uint(std::string_view s, uint32_t& out) {
    auto result = std::from_chars(s.data(), s.data() + s.size(), out);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size();
}

// Extract the next whitespace-delimited token from a string_view.
std::string_view next_token(std::string_view& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        s = {};
        return {};
    }
    s.remove_prefix(start);
    auto end = s.find_first_of(" \t\r\n");
    std::string_view tok = s.substr(0, end);
    if (end != std::string_view::npos) {
        s.remove_prefix(end);
    } else {
        s = {};
    }
    return tok;
}

// Check if character is valid for gate names.
bool is_gate_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Parser state.
class Parser {
  public:
    explicit Parser(std::string_view text, size_t max_ops) : text_(text), max_ops_(max_ops) {}

    Circuit parse() {
        Circuit circuit;
        uint32_t line_num = 0;

        // Defensive check for non-ASCII/Unicode characters
        auto non_ascii_it = std::find_if(
            text_.begin(), text_.end(), [](char c) { return static_cast<unsigned char>(c) > 127; });

        if (non_ascii_it != text_.end()) {
            size_t offset = std::distance(text_.begin(), non_ascii_it);
            throw ParseError("Non-ASCII/Unicode character detected at byte offset " +
                                 std::to_string(offset) + ". Only plain ASCII is supported.",
                             0);
        }

        std::string_view remaining = text_;
        parse_block(remaining, line_num, circuit, 0);

        return circuit;
    }

    // Parse a block of text line-by-line into the circuit.
    // `remaining` and `line_num` are advanced as text is consumed.
    void parse_block(std::string_view& remaining, uint32_t& line_num, Circuit& circuit,
                     uint32_t depth) {
        while (!remaining.empty()) {
            line_num++;
            size_t end_of_line = remaining.find('\n');
            std::string_view line;

            if (end_of_line == std::string_view::npos) {
                line = remaining;
                remaining = {};
            } else {
                line = remaining.substr(0, end_of_line);
                remaining.remove_prefix(end_of_line + 1);
            }

            if (!line.empty() && line.back() == '\r') {
                line.remove_suffix(1);
            }

            parse_line(line, line_num, circuit, remaining, depth);

            if (circuit.nodes.size() > max_ops_) {
                throw ParseError("Circuit exceeds maximum unrolled operations limit", line_num);
            }
        }
    }

  private:
    static constexpr uint32_t kMaxRecursionDepth = 100;
    std::string_view text_;
    size_t max_ops_;

    void parse_line(std::string_view line, uint32_t& line_num, Circuit& circuit,
                    std::string_view& remaining, uint32_t depth) {
        // Strip comments.
        auto comment_pos = line.find('#');
        if (comment_pos != std::string_view::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim(line);
        if (line.empty()) {
            return;
        }

        // Stray closing braces without REPEAT are an error.
        if (line == "}") {
            throw ParseError("Unexpected closing brace '}'", line_num);
        }

        // Handle REPEAT N { ... } blocks.
        if (line.starts_with("REPEAT") && (line.size() == 6 || !is_gate_char(line[6]))) {
            parse_repeat(line, line_num, circuit, remaining, depth);
            return;
        }

        size_t name_end = 0;
        while (name_end < line.size() && is_gate_char(line[name_end])) {
            name_end++;
        }

        if (name_end == 0) {
            throw ParseError("Expected gate name", line_num);
        }

        std::string_view gate_name = line.substr(0, name_end);
        std::string_view rest = trim(line.substr(name_end));

        // Parse optional parenthesized arguments (comma-separated floats).
        std::vector<double> args;
        if (!rest.empty() && rest[0] == '(') {
            auto close_paren = rest.find(')');
            if (close_paren == std::string_view::npos) {
                throw ParseError("Unclosed parenthesis", line_num);
            }
            std::string_view args_str = trim(rest.substr(1, close_paren - 1));
            while (!args_str.empty()) {
                auto comma_pos = args_str.find(',');
                std::string_view token = trim(
                    comma_pos == std::string_view::npos ? args_str : args_str.substr(0, comma_pos));
                if (!token.empty()) {
                    double val = 0.0;
                    auto result =
                        fast_float::from_chars(token.data(), token.data() + token.size(), val);
                    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
                        throw ParseError("Invalid gate argument: " + std::string(token), line_num);
                    }
                    args.push_back(val);
                }
                if (comma_pos == std::string_view::npos) {
                    break;
                }
                args_str = trim(args_str.substr(comma_pos + 1));
            }
            rest = trim(rest.substr(close_paren + 1));
        }
        double arg = args.empty() ? 0.0 : args[0];

        // Silently discard coordinate annotations (no AST nodes emitted).
        if (kDiscardedAnnotations.contains(gate_name)) {
            return;
        }

        // Look up gate type.
        auto gate_it = kGateNames.find(gate_name);
        if (gate_it == kGateNames.end()) {
            throw ParseError("Unknown gate: " + std::string(gate_name), line_num);
        }

        GateType gate = gate_it->second;

        // Validate argument counts for multi-probability channels.
        if (gate == GateType::PAULI_CHANNEL_1 && args.size() != 3) {
            throw ParseError("PAULI_CHANNEL_1 requires exactly 3 arguments", line_num);
        }
        if (gate == GateType::PAULI_CHANNEL_2 && args.size() != 15) {
            throw ParseError("PAULI_CHANNEL_2 requires exactly 15 arguments", line_num);
        }

        // Validate argument counts for parameterized rotations.
        if ((gate == GateType::R_X || gate == GateType::R_Y || gate == GateType::R_Z ||
             gate == GateType::R_XX || gate == GateType::R_YY || gate == GateType::R_ZZ ||
             gate == GateType::R_PAULI) &&
            args.size() != 1) {
            throw ParseError(
                std::string(ucc::gate_name(gate)) + " requires exactly 1 argument (alpha)",
                line_num);
        }
        if (gate == GateType::U3 && args.size() != 3) {
            throw ParseError("U3 requires exactly 3 arguments (theta, phi, lambda)", line_num);
        }

        // Parse based on gate type.
        switch (gate) {
            case GateType::MPP:
                parse_mpp(rest, line_num, circuit, arg);
                break;
            case GateType::R_PAULI:
                parse_r_pauli(rest, line_num, circuit, args);
                break;
            case GateType::DETECTOR:
                parse_detector(rest, line_num, circuit);
                break;
            case GateType::OBSERVABLE_INCLUDE:
                parse_observable_include(rest, line_num, circuit, arg);
                break;
            case GateType::TICK:
                if (!rest.empty()) {
                    throw ParseError("TICK takes no targets", line_num);
                }
                circuit.nodes.push_back({GateType::TICK, {}, args, line_num});
                break;
            default:
                parse_standard_gate(gate, rest, line_num, circuit, arg, args);
                break;
        }
    }

    // Parse REPEAT N { ... } block via text-level unrolling.
    void parse_repeat(std::string_view line, uint32_t& line_num, Circuit& circuit,
                      std::string_view& remaining, uint32_t depth) {
        if (depth >= kMaxRecursionDepth) {
            throw ParseError("Max recursion depth exceeded", line_num);
        }
        // Parse the repetition count from the REPEAT line.
        std::string_view after_keyword = trim(line.substr(6));

        // Find the opening brace - it may be on this line.
        auto brace_pos = after_keyword.find('{');
        std::string_view count_str;
        if (brace_pos != std::string_view::npos) {
            count_str = trim(after_keyword.substr(0, brace_pos));
        } else {
            count_str = after_keyword;
        }

        uint32_t repeat_count = 0;
        if (count_str.empty() || !parse_uint(count_str, repeat_count)) {
            throw ParseError("REPEAT requires a positive integer count", line_num);
        }
        if (repeat_count == 0) {
            throw ParseError("REPEAT count must be positive", line_num);
        }

        // Locate the opening brace. It might be on this line or on a subsequent line.
        bool found_open_brace = (brace_pos != std::string_view::npos);
        if (!found_open_brace) {
            // Scan remaining for the opening brace, skipping comments.
            bool scan_comment = false;
            bool found = false;
            size_t scan = 0;
            while (scan < remaining.size()) {
                char c = remaining[scan];
                if (scan_comment) {
                    if (c == '\n') {
                        scan_comment = false;
                        line_num++;
                    }
                } else if (c == '#') {
                    scan_comment = true;
                } else if (c == '{') {
                    found = true;
                    break;
                } else if (c == '\n') {
                    line_num++;
                }
                scan++;
            }
            if (!found) {
                throw ParseError("REPEAT block missing opening brace '{'", line_num);
            }
            remaining.remove_prefix(scan + 1);
        } else {
            // If there's content after the '{' on the same line, allow comments.
            std::string_view after_brace = trim(after_keyword.substr(brace_pos + 1));
            if (!after_brace.empty() && after_brace[0] != '#') {
                // Body must start on the next line after the opening brace.
                throw ParseError("REPEAT body must start on the line after the opening brace",
                                 line_num);
            }
        }

        // Scan remaining for the matching closing brace, tracking brace depth.
        // Skip characters inside comments (# to end of line).
        int brace_depth = 1;
        size_t body_start = 0;
        size_t scan_pos = 0;
        bool in_comment = false;
        while (scan_pos < remaining.size()) {
            char c = remaining[scan_pos];
            if (in_comment) {
                if (c == '\n')
                    in_comment = false;
            } else if (c == '#') {
                in_comment = true;
            } else if (c == '{') {
                brace_depth++;
            } else if (c == '}') {
                brace_depth--;
                if (brace_depth == 0) {
                    break;
                }
            }
            scan_pos++;
        }

        if (brace_depth != 0) {
            throw ParseError("REPEAT block missing closing brace '}'", line_num);
        }

        // Extract the body (text between braces).
        std::string_view body = remaining.substr(body_start, scan_pos);

        // Advance remaining past the closing brace.
        remaining.remove_prefix(scan_pos + 1);

        // Count lines in the body for accurate line numbering during replay.
        uint32_t body_lines = 0;
        for (char c : body) {
            if (c == '\n')
                body_lines++;
        }

        // Skip empty bodies to avoid spinning on e.g. REPEAT 4000000000 {}
        bool body_has_content = false;
        {
            bool in_cmt = false;
            for (char c : body) {
                if (in_cmt) {
                    if (c == '\n')
                        in_cmt = false;
                } else if (c == '#') {
                    in_cmt = true;
                } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                    body_has_content = true;
                    break;
                }
            }
        }

        // Text-level replay: parse the body N times.
        uint32_t base_line = line_num;
        for (uint32_t i = 0; i < repeat_count && body_has_content; i++) {
            // Each iteration re-parses from the body text, but line numbers
            // reflect the original source location for error reporting.
            line_num = base_line;
            std::string_view body_remaining = body;
            parse_block(body_remaining, line_num, circuit, depth + 1);

            // Enforce safety limit after each iteration.
            if (circuit.nodes.size() > max_ops_) {
                throw ParseError("Circuit exceeds maximum unrolled operations limit", line_num);
            }
        }

        // Advance line_num past the closing brace line.
        line_num = base_line + body_lines;
    }

    // Parse a standard gate with qubit targets (possibly with rec references).
    void parse_standard_gate(GateType gate, std::string_view targets_str, uint32_t line_num,
                             Circuit& circuit, double arg, const std::vector<double>& args) {
        // Resets (R, RX) don't accept noise arguments.
        if (is_reset(gate) && arg != 0.0) {
            throw ParseError("Reset gates do not accept arguments", line_num);
        }

        std::vector<Target> targets;
        GateArity arity = gate_arity(gate);

        // Tokenize targets.
        std::string_view remaining = targets_str;
        while (true) {
            std::string_view token = next_token(remaining);
            if (token.empty())
                break;

            if (targets.size() >= kMaxTargetsPerInstruction) {
                throw ParseError(
                    "Too many targets (limit: " + std::to_string(kMaxTargetsPerInstruction) + ")",
                    line_num);
            }

            // Validate rec targets are only used for CX/CZ feedback forms.
            bool is_rec_token = token.starts_with("rec[") || (token.size() > 1 && token[0] == '!' &&
                                                              token.substr(1).starts_with("rec["));
            if (is_rec_token && gate != GateType::CX && gate != GateType::CZ) {
                throw ParseError(
                    "rec targets are only supported as feedback controls for CX/CZ gates",
                    line_num);
            }

            Target target = parse_target(token, line_num, circuit);
            targets.push_back(target);
        }

        if (targets.empty() && arity != GateArity::ANNOTATION) {
            throw ParseError("Gate " + std::string(gate_name(gate)) + " requires targets",
                             line_num);
        }

        // MPAD targets must be classical boolean literals (0 or 1), not rec references.
        if (gate == GateType::MPAD) {
            for (Target t : targets) {
                if (t.is_rec() || t.value() > 1) {
                    throw ParseError("MPAD targets must be 0 or 1", line_num);
                }
            }
        }

        // Expand based on arity.
        // Identity no-ops: validate syntax and update num_qubits, but never emit AST nodes.
        if (is_identity_noop(gate)) {
            if (arity == GateArity::PAIR && targets.size() % 2 != 0) {
                throw ParseError(
                    "Gate " + std::string(gate_name(gate)) + " requires pairs of targets",
                    line_num);
            }
            for (Target t : targets) {
                if (!t.is_rec()) {
                    circuit.num_qubits = std::max(circuit.num_qubits, t.value() + 1);
                }
            }
            return;
        }

        switch (arity) {
            case GateArity::SINGLE:
                // One AstNode per target.
                for (Target t : targets) {
                    // For noisy measurements M(p), MX(p), MY(p): decompose into
                    // clean measurement followed by READOUT_NOISE.
                    bool is_noisy_meas = is_measurement(gate) && arg > 0.0;

                    // Pass args through directly; zero-arg gates get an empty vector.
                    std::vector<double> node_args = args;
                    if (is_noisy_meas && !node_args.empty()) {
                        node_args[0] = 0.0;
                    }

                    AstNode node{gate, {t}, std::move(node_args), line_num};
                    update_circuit_stats(node, circuit);
                    circuit.nodes.push_back(std::move(node));

                    if (is_noisy_meas) {
                        // Emit READOUT_NOISE targeting the just-created measurement.
                        uint32_t meas_idx = circuit.num_measurements - 1;
                        circuit.nodes.push_back(
                            {GateType::READOUT_NOISE, {Target::rec(meas_idx)}, {arg}, line_num});
                    }
                }
                break;

            case GateArity::PAIR:
                // Targets consumed in pairs.
                if (targets.size() % 2 != 0) {
                    throw ParseError(
                        "Gate " + std::string(gate_name(gate)) + " requires pairs of targets",
                        line_num);
                }

                // MXX/MYY/MZZ: desugar into MPP with Pauli-tagged targets.
                if (gate == GateType::MXX || gate == GateType::MYY || gate == GateType::MZZ) {
                    uint32_t pauli_flag = (gate == GateType::MXX)   ? Target::kPauliX
                                          : (gate == GateType::MYY) ? Target::kPauliY
                                                                    : Target::kPauliZ;
                    for (size_t i = 0; i < targets.size(); i += 2) {
                        Target t0 = targets[i];
                        Target t1 = targets[i + 1];
                        // Build Pauli-tagged targets, preserving inversion flags.
                        Target p0 = Target::pauli(t0.value(), pauli_flag);
                        Target p1 = Target::pauli(t1.value(), pauli_flag);
                        if (t0.is_inverted())
                            p0 = p0.inverted();
                        if (t1.is_inverted())
                            p1 = p1.inverted();

                        // Update qubit tracking.
                        circuit.num_qubits = std::max(circuit.num_qubits, t0.value() + 1);
                        circuit.num_qubits = std::max(circuit.num_qubits, t1.value() + 1);

                        bool is_noisy_meas = arg > 0.0;
                        circuit.nodes.push_back({GateType::MPP,
                                                 {p0, p1},
                                                 is_noisy_meas ? std::vector<double>{} : args,
                                                 line_num});
                        circuit.num_measurements++;

                        if (is_noisy_meas) {
                            uint32_t meas_idx = circuit.num_measurements - 1;
                            circuit.nodes.push_back({GateType::READOUT_NOISE,
                                                     {Target::rec(meas_idx)},
                                                     {arg},
                                                     line_num});
                        }
                    }
                    break;
                }

                for (size_t i = 0; i < targets.size(); i += 2) {
                    Target t0 = targets[i];
                    Target t1 = targets[i + 1];

                    // Validate CX/CZ feedback syntax: rec must be first, qubit second.
                    if ((gate == GateType::CX || gate == GateType::CZ) &&
                        (t0.is_rec() || t1.is_rec())) {
                        if (!t0.is_rec() || t1.is_rec()) {
                            throw ParseError("Feedback syntax requires: " +
                                                 std::string(gate_name(gate)) + " rec[-k] qubit",
                                             line_num);
                        }
                    }

                    AstNode node{gate, {t0, t1}, args, line_num};
                    update_circuit_stats(node, circuit);
                    circuit.nodes.push_back(std::move(node));
                }
                break;

            case GateArity::ANNOTATION:
                circuit.nodes.push_back({gate, targets, args, line_num});
                break;

            case GateArity::MULTI:
                // Should be handled by parse_mpp.
                throw ParseError("Internal error: MULTI arity in parse_standard_gate", line_num);
        }
    }

    // Parse a single target token.
    Target parse_target(std::string_view token, uint32_t line_num, Circuit& circuit) {
        bool inverted = false;
        std::string_view s = token;

        // Check for inversion flag.
        if (!s.empty() && s[0] == '!') {
            inverted = true;
            s.remove_prefix(1);
        }

        // Check for rec[-k] reference.
        if (s.starts_with("rec[")) {
            auto close_bracket = s.find(']');
            if (close_bracket == std::string_view::npos) {
                throw ParseError("Unclosed bracket in rec reference", line_num);
            }

            // Ensure ] is at the end (no trailing characters).
            if (close_bracket + 1 != s.size()) {
                throw ParseError("Trailing characters after rec reference: " + std::string(token),
                                 line_num);
            }

            std::string_view offset_str = s.substr(4, close_bracket - 4);
            int offset;
            if (!parse_int(offset_str, offset)) {
                throw ParseError("Invalid rec offset: " + std::string(offset_str), line_num);
            }

            // rec[-k] means k measurements back from current visible count.
            if (offset >= 0) {
                throw ParseError("rec offset must be negative", line_num);
            }

            // Calculate measurement index
            int meas_count = static_cast<int>(circuit.num_measurements);
            int meas_index = meas_count + offset;
            if (meas_index < 0) {
                throw ParseError("rec reference out of bounds: rec[" + std::to_string(offset) + "]",
                                 line_num);
            }

            // Validate that the index fits in 28-bit value mask.
            if (static_cast<uint32_t>(meas_index) >= (1u << 28)) {
                throw ParseError("Measurement index too large (must be < 2^28)", line_num);
            }

            Target result = Target::rec(static_cast<uint32_t>(meas_index));
            return inverted ? result.inverted() : result;
        }

        // Plain qubit index.
        uint32_t qubit;
        if (!parse_uint(s, qubit)) {
            throw ParseError("Invalid target: " + std::string(token), line_num);
        }

        // Validate that the qubit index fits within the 28-bit encoding mask.
        if (qubit >= (1u << 28)) {
            throw ParseError("Qubit index too large (must be < 2^28): " + std::string(token),
                             line_num);
        }

        Target result = Target::qubit(qubit);
        return inverted ? result.inverted() : result;
    }

    // Parse MPP instruction with multiple Pauli products.
    void parse_mpp(std::string_view targets_str, uint32_t line_num, Circuit& circuit, double arg) {
        std::string_view remaining = targets_str;
        uint32_t product_count = 0;

        while (true) {
            std::string_view product_str = next_token(remaining);
            if (product_str.empty())
                break;

            if (product_count >= kMaxTargetsPerInstruction) {
                throw ParseError("Too many MPP products (limit: " +
                                     std::to_string(kMaxTargetsPerInstruction) + ")",
                                 line_num);
            }
            product_count++;

            // Each product is like "X0*Z1*Y2".
            std::vector<Target> pauli_targets;
            std::unordered_set<uint32_t> seen_qubits;

            size_t pos = 0;
            while (pos < product_str.size()) {
                // Handle '*' separator with validation.
                if (product_str[pos] == '*') {
                    // Disallow leading, trailing, or consecutive '*' characters.
                    if (pos == 0 || pos + 1 >= product_str.size() || product_str[pos + 1] == '*') {
                        throw ParseError("Malformed Pauli product in MPP: misplaced '*'", line_num);
                    }
                    pos++;
                    continue;
                }

                // Parse Pauli letter.
                char pauli_char = product_str[pos];
                uint32_t pauli_flag;
                switch (pauli_char) {
                    case 'X':
                        pauli_flag = Target::kPauliX;
                        break;
                    case 'Y':
                        pauli_flag = Target::kPauliY;
                        break;
                    case 'Z':
                        pauli_flag = Target::kPauliZ;
                        break;
                    default:
                        throw ParseError("Invalid Pauli in MPP: " + std::string(1, pauli_char),
                                         line_num);
                }
                pos++;

                // Parse qubit index.
                size_t num_start = pos;
                while (pos < product_str.size() && std::isdigit(product_str[pos])) {
                    pos++;
                }

                if (num_start == pos) {
                    throw ParseError("Expected qubit index after Pauli letter", line_num);
                }

                uint32_t qubit;
                if (!parse_uint(std::string_view(product_str).substr(num_start, pos - num_start),
                                qubit)) {
                    throw ParseError("Invalid qubit index in MPP", line_num);
                }

                // Validate qubit index fits in 28-bit encoding.
                if (qubit >= (1u << 28)) {
                    throw ParseError("Qubit index too large (must be < 2^28)", line_num);
                }

                if (!seen_qubits.insert(qubit).second) {
                    throw ParseError("Duplicate qubit in MPP product", line_num);
                }

                if (pauli_targets.size() >= kMaxTargetsPerInstruction) {
                    throw ParseError("Too many Pauli terms in product (limit: " +
                                         std::to_string(kMaxTargetsPerInstruction) + ")",
                                     line_num);
                }

                pauli_targets.push_back(Target::pauli(qubit, pauli_flag));

                // Update max qubit.
                circuit.num_qubits = std::max(circuit.num_qubits, qubit + 1);
            }

            if (pauli_targets.empty()) {
                throw ParseError("Empty Pauli product in MPP", line_num);
            }

            // Decompose noisy MPP: MPP(p) -> MPP + READOUT_NOISE
            bool is_noisy_meas = arg > 0.0;

            // Emit one AstNode per product.
            // MPP is a visible measurement.
            AstNode node{GateType::MPP, std::move(pauli_targets), {}, line_num};
            circuit.num_measurements++;
            circuit.nodes.push_back(std::move(node));

            if (is_noisy_meas) {
                uint32_t meas_idx = circuit.num_measurements - 1;
                circuit.nodes.push_back(
                    {GateType::READOUT_NOISE, {Target::rec(meas_idx)}, {arg}, line_num});
            }
        }

        if (product_count == 0) {
            throw ParseError("MPP requires at least one Pauli product", line_num);
        }
    }

    // Parse R_PAULI instruction: R_PAULI(alpha) X0*Y1*Z2
    // Exactly one Pauli product with the rotation angle from args[0].
    void parse_r_pauli(std::string_view targets_str, uint32_t line_num, Circuit& circuit,
                       const std::vector<double>& args) {
        std::string_view product_str = next_token(targets_str);
        if (product_str.empty()) {
            throw ParseError("R_PAULI requires a Pauli product (e.g. X0*Y1*Z2)", line_num);
        }

        // Check no extra tokens.
        std::string_view extra = next_token(targets_str);
        if (!extra.empty()) {
            throw ParseError("R_PAULI takes exactly one Pauli product", line_num);
        }

        std::vector<Target> pauli_targets;
        std::unordered_set<uint32_t> seen_qubits;

        size_t pos = 0;
        while (pos < product_str.size()) {
            if (product_str[pos] == '*') {
                if (pos == 0 || pos + 1 >= product_str.size() || product_str[pos + 1] == '*') {
                    throw ParseError("Malformed Pauli product in R_PAULI: misplaced '*'", line_num);
                }
                pos++;
                continue;
            }

            char pauli_char = product_str[pos];
            uint32_t pauli_flag;
            switch (pauli_char) {
                case 'X':
                    pauli_flag = Target::kPauliX;
                    break;
                case 'Y':
                    pauli_flag = Target::kPauliY;
                    break;
                case 'Z':
                    pauli_flag = Target::kPauliZ;
                    break;
                default:
                    throw ParseError("Invalid Pauli in R_PAULI: " + std::string(1, pauli_char),
                                     line_num);
            }
            pos++;

            size_t num_start = pos;
            while (pos < product_str.size() && std::isdigit(product_str[pos])) {
                pos++;
            }

            if (num_start == pos) {
                throw ParseError("Expected qubit index after Pauli letter in R_PAULI", line_num);
            }

            uint32_t qubit;
            if (!parse_uint(std::string_view(product_str).substr(num_start, pos - num_start),
                            qubit)) {
                throw ParseError("Invalid qubit index in R_PAULI", line_num);
            }

            if (qubit >= (1u << 28)) {
                throw ParseError("Qubit index too large (must be < 2^28)", line_num);
            }

            if (!seen_qubits.insert(qubit).second) {
                throw ParseError("Duplicate qubit in R_PAULI product", line_num);
            }

            if (pauli_targets.size() >= kMaxTargetsPerInstruction) {
                throw ParseError("Too many Pauli terms in R_PAULI product", line_num);
            }
            pauli_targets.push_back(Target::pauli(qubit, pauli_flag));
            circuit.num_qubits = std::max(circuit.num_qubits, qubit + 1);
        }

        if (pauli_targets.empty()) {
            throw ParseError("Empty Pauli product in R_PAULI", line_num);
        }

        circuit.nodes.push_back({GateType::R_PAULI, std::move(pauli_targets), args, line_num});
    }

    // Parse DETECTOR with rec[-k] targets.
    void parse_detector(std::string_view targets_str, uint32_t line_num, Circuit& circuit) {
        std::vector<Target> targets;
        std::string_view remaining = targets_str;

        while (true) {
            std::string_view token = next_token(remaining);
            if (token.empty())
                break;

            if (targets.size() >= kMaxTargetsPerInstruction) {
                throw ParseError("Too many DETECTOR targets (limit: " +
                                     std::to_string(kMaxTargetsPerInstruction) + ")",
                                 line_num);
            }

            // DETECTOR only accepts rec[-k] targets.
            if (!token.starts_with("rec[")) {
                throw ParseError("DETECTOR targets must be rec[-k] references", line_num);
            }

            Target target = parse_target(token, line_num, circuit);
            targets.push_back(target);
        }

        circuit.nodes.push_back({GateType::DETECTOR, std::move(targets), {}, line_num});
        circuit.num_detectors++;
    }

    // Parse OBSERVABLE_INCLUDE with observable index and rec[-k] targets.
    void parse_observable_include(std::string_view targets_str, uint32_t line_num, Circuit& circuit,
                                  double arg) {
        std::vector<Target> targets;
        std::string_view remaining = targets_str;

        while (true) {
            std::string_view token = next_token(remaining);
            if (token.empty())
                break;

            if (targets.size() >= kMaxTargetsPerInstruction) {
                throw ParseError("Too many OBSERVABLE_INCLUDE targets (limit: " +
                                     std::to_string(kMaxTargetsPerInstruction) + ")",
                                 line_num);
            }

            // OBSERVABLE_INCLUDE only accepts rec[-k] targets.
            if (!token.starts_with("rec[")) {
                throw ParseError("OBSERVABLE_INCLUDE targets must be rec[-k] references", line_num);
            }

            Target target = parse_target(token, line_num, circuit);
            targets.push_back(target);
        }

        // Observable index is stored in arg. Validate it's a non-negative integer.
        if (arg < 0.0) {
            throw ParseError("OBSERVABLE_INCLUDE index must be non-negative", line_num);
        }
        auto obs_idx = static_cast<uint32_t>(arg);
        if (static_cast<double>(obs_idx) != arg) {
            throw ParseError("OBSERVABLE_INCLUDE index must be an integer", line_num);
        }
        circuit.num_observables = std::max(circuit.num_observables, obs_idx + 1);

        circuit.nodes.push_back(
            {GateType::OBSERVABLE_INCLUDE, std::move(targets), {arg}, line_num});
    }

    // Update circuit statistics after adding a node.
    void update_circuit_stats(const AstNode& node, Circuit& circuit) {
        // MPAD targets are classical boolean literals, not qubit indices.
        if (node.gate != GateType::MPAD) {
            for (Target t : node.targets) {
                if (!t.is_rec()) {
                    uint32_t q = t.value();
                    circuit.num_qubits = std::max(circuit.num_qubits, q + 1);
                }
            }
        }

        // Update measurement count for visible measurements.
        // is_measurement includes M, MX, MY, MR, MRX, MPP.
        // R and RX are resets without visible measurements (is_reset returns true).
        if (is_measurement(node.gate)) {
            circuit.num_measurements++;
        }
    }
};

}  // namespace

GateType parse_gate_name(std::string_view name) {
    auto it = kGateNames.find(name);
    if (it != kGateNames.end()) {
        return it->second;
    }
    return GateType::UNKNOWN;
}

Circuit parse(std::string_view text) {
    return parse(text, kMaxUnrolledOps);
}

Circuit parse(std::string_view text, size_t max_ops) {
    Parser parser(text, max_ops);
    return parser.parse();
}

Circuit parse_file(const std::string& path) {
    return parse_file(path, kMaxUnrolledOps);
}

Circuit parse_file(const std::string& path, size_t max_ops) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    auto size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("Error determining file size: " + path);
    }

    constexpr std::streamsize kMaxFileSize = 1024LL * 1024LL * 1024LL;
    if (size > kMaxFileSize) {
        throw std::runtime_error("Circuit file exceeds 1GB memory limit (" + std::to_string(size) +
                                 " bytes).");
    }
    file.seekg(0, std::ios::beg);

    auto buf_size = static_cast<size_t>(size);
    auto buffer = std::make_unique<char[]>(buf_size);
    if (!file.read(buffer.get(), size)) {
        throw std::runtime_error("Error reading file: " + path);
    }
    return parse(std::string_view(buffer.get(), buf_size), max_ops);
}

}  // namespace ucc
