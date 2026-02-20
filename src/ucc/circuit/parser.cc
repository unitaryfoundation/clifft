#include "ucc/circuit/parser.h"

#include "ucc/util/config.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fast_float/fast_float.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ucc {

namespace {

// Gate name lookup table.
const std::unordered_map<std::string_view, GateType> kGateNames = {
    // Single-qubit Clifford
    {"H", GateType::H},
    {"S", GateType::S},
    {"S_DAG", GateType::S_DAG},
    {"X", GateType::X},
    {"Y", GateType::Y},
    {"Z", GateType::Z},
    // Non-Clifford
    {"T", GateType::T},
    {"T_DAG", GateType::T_DAG},
    // Two-qubit Clifford
    {"CX", GateType::CX},
    {"CNOT", GateType::CX},  // Alias
    {"CY", GateType::CY},
    {"CZ", GateType::CZ},
    // Measurements
    {"M", GateType::M},
    {"MX", GateType::MX},
    {"MY", GateType::MY},
    {"MR", GateType::MR},
    {"MRX", GateType::MRX},
    {"MPP", GateType::MPP},
    // Annotations
    {"TICK", GateType::TICK},
};

// Reset gates that need decomposition.
const std::unordered_map<std::string_view, std::pair<GateType, GateType>> kResetDecomposition = {
    {"R", {GateType::M, GateType::CX}},    // R -> M + CX rec[-1]
    {"RX", {GateType::MX, GateType::CZ}},  // RX -> MX + CZ rec[-1]
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
    explicit Parser(std::string_view text) : text_(text) {}

    Circuit parse() {
        Circuit circuit;
        int line_num = 0;

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

            parse_line(line, line_num, circuit);
        }

        return circuit;
    }

  private:
    std::string_view text_;

    void parse_line(std::string_view line, int line_num, Circuit& circuit) {
        // Strip comments.
        auto comment_pos = line.find('#');
        if (comment_pos != std::string_view::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim(line);
        if (line.empty()) {
            return;
        }

        // Check for REPEAT (unsupported).
        if (line.starts_with("REPEAT")) {
            throw ParseError("REPEAT blocks are not supported in MVP", line_num);
        }

        // Stray closing braces without REPEAT are an error.
        if (line == "}") {
            throw ParseError("Unexpected closing brace '}'", line_num);
        }

        // Extract gate name.
        size_t name_end = 0;
        while (name_end < line.size() && is_gate_char(line[name_end])) {
            name_end++;
        }

        if (name_end == 0) {
            throw ParseError("Expected gate name", line_num);
        }

        std::string_view gate_name = line.substr(0, name_end);
        std::string_view rest = trim(line.substr(name_end));

        // Skip optional parenthesized arguments (noise probabilities, etc.).
        // We don't use them in MVP but need to parse past them.
        double arg = 0.0;
        if (!rest.empty() && rest[0] == '(') {
            auto close_paren = rest.find(')');
            if (close_paren == std::string_view::npos) {
                throw ParseError("Unclosed parenthesis", line_num);
            }
            // Parse argument if present.
            std::string_view arg_str = trim(rest.substr(1, close_paren - 1));
            if (!arg_str.empty()) {
                auto result =
                    fast_float::from_chars(arg_str.data(), arg_str.data() + arg_str.size(), arg);

                if (result.ec != std::errc{} || result.ptr != arg_str.data() + arg_str.size()) {
                    throw ParseError("Invalid gate argument: " + std::string(arg_str), line_num);
                }
            }
            rest = trim(rest.substr(close_paren + 1));
        }

        // Check for reset decomposition.
        auto reset_it = kResetDecomposition.find(gate_name);
        if (reset_it != kResetDecomposition.end()) {
            parse_reset(rest, line_num, circuit, reset_it->second.first, reset_it->second.second,
                        arg);
            return;
        }

        // Look up gate type.
        auto gate_it = kGateNames.find(gate_name);
        if (gate_it == kGateNames.end()) {
            throw ParseError("Unknown gate: " + std::string(gate_name), line_num);
        }

        GateType gate = gate_it->second;

        // Parse based on gate type.
        switch (gate) {
            case GateType::MPP:
                parse_mpp(rest, line_num, circuit, arg);
                break;
            case GateType::TICK:
                if (!rest.empty()) {
                    throw ParseError("TICK takes no targets", line_num);
                }
                circuit.nodes.push_back({GateType::TICK, {}, arg});
                break;
            default:
                parse_standard_gate(gate, rest, line_num, circuit, arg);
                break;
        }
    }

    // Parse a standard gate with qubit targets (possibly with rec references).
    void parse_standard_gate(GateType gate, std::string_view targets_str, int line_num,
                             Circuit& circuit, double arg) {
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

        // Expand based on arity.
        switch (arity) {
            case GateArity::SINGLE:
                // One AstNode per target.
                for (Target t : targets) {
                    AstNode node{gate, {t}, arg};
                    update_circuit_stats(node, circuit);
                    circuit.nodes.push_back(std::move(node));
                }
                break;

            case GateArity::PAIR:
                // Targets consumed in pairs.
                if (targets.size() % 2 != 0) {
                    throw ParseError(
                        "Gate " + std::string(gate_name(gate)) + " requires pairs of targets",
                        line_num);
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

                    AstNode node{gate, {t0, t1}, arg};
                    update_circuit_stats(node, circuit);
                    circuit.nodes.push_back(std::move(node));
                }
                break;

            case GateArity::ANNOTATION:
                circuit.nodes.push_back({gate, targets, arg});
                break;

            case GateArity::MULTI:
                // Should be handled by parse_mpp.
                throw ParseError("Internal error: MULTI arity in parse_standard_gate", line_num);
        }
    }

    // Parse a single target token.
    Target parse_target(std::string_view token, int line_num, Circuit& circuit) {
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

            // rec[-k] means k measurements back from current count.
            // We resolve to absolute index.
            if (offset >= 0) {
                throw ParseError("rec offset must be negative", line_num);
            }

            int abs_index = static_cast<int>(circuit.num_measurements) + offset;
            if (abs_index < 0) {
                throw ParseError("rec reference out of bounds: rec[" + std::to_string(offset) + "]",
                                 line_num);
            }

            // Validate that the resolved index fits in 28-bit value mask.
            if (static_cast<uint32_t>(abs_index) >= (1u << 28)) {
                throw ParseError("Measurement index too large (must be < 2^28)", line_num);
            }

            Target result = Target::rec(static_cast<uint32_t>(abs_index));
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
    void parse_mpp(std::string_view targets_str, int line_num, Circuit& circuit, double arg) {
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

            // Emit one AstNode per product.
            AstNode node{GateType::MPP, std::move(pauli_targets), arg};
            circuit.num_measurements++;
            circuit.nodes.push_back(std::move(node));
        }

        if (product_count == 0) {
            throw ParseError("MPP requires at least one Pauli product", line_num);
        }
    }

    // Parse reset and decompose into measurement + conditional feedback.
    void parse_reset(std::string_view targets_str, int line_num, Circuit& circuit,
                     GateType meas_gate, GateType feedback_gate, double arg) {
        // Reset doesn't use arguments; reject if provided.
        if (arg != 0.0) {
            throw ParseError("Reset gates do not accept arguments", line_num);
        }

        // Parse qubit targets.
        std::string_view remaining = targets_str;
        uint32_t target_count = 0;

        while (true) {
            std::string_view token = next_token(remaining);
            if (token.empty())
                break;

            if (target_count >= kMaxTargetsPerInstruction) {
                throw ParseError("Too many reset targets (limit: " +
                                     std::to_string(kMaxTargetsPerInstruction) + ")",
                                 line_num);
            }
            target_count++;

            uint32_t qubit;
            if (!parse_uint(token, qubit)) {
                throw ParseError("Invalid reset target: " + std::string(token), line_num);
            }

            // Validate qubit index fits in 28-bit encoding.
            if (qubit >= (1u << 28)) {
                throw ParseError("Qubit index too large (must be < 2^28)", line_num);
            }

            // Update max qubit.
            circuit.num_qubits = std::max(circuit.num_qubits, qubit + 1);

            // Emit measurement.
            circuit.nodes.push_back({meas_gate, {Target::qubit(qubit)}, 0.0});
            circuit.num_measurements++;

            // Emit conditional feedback: CX/CZ rec[-1] qubit.
            Target rec_target = Target::rec(circuit.num_measurements - 1);
            circuit.nodes.push_back({feedback_gate, {rec_target, Target::qubit(qubit)}, 0.0});
        }

        if (target_count == 0) {
            throw ParseError("Reset requires at least one qubit target", line_num);
        }
    }

    // Update circuit statistics after adding a node.
    void update_circuit_stats(const AstNode& node, Circuit& circuit) {
        // Update qubit count.
        for (Target t : node.targets) {
            if (!t.is_rec()) {
                uint32_t q = t.value();
                circuit.num_qubits = std::max(circuit.num_qubits, q + 1);
            }
        }

        // Update measurement count.
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
    Parser parser(text);
    return parser.parse();
}

Circuit parse_file(const std::string& path) {
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

    std::string buffer(static_cast<size_t>(size), '\0');
    if (file.read(buffer.data(), size)) {
        return parse(buffer);
    }
    throw std::runtime_error("Error reading file: " + path);
}

}  // namespace ucc
