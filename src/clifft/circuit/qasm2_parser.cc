#include "clifft/circuit/qasm2_parser.h"

#include "clifft/circuit/parser.h"
#include "clifft/util/config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fast_float/fast_float.h>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace clifft {

namespace {

enum class TokenKind : uint8_t {
    End,
    Identifier,
    Number,
    String,
    Symbol,
};

struct Token {
    TokenKind kind = TokenKind::End;
    std::string_view text;
    uint32_t line = 1;
};

class Lexer {
  public:
    explicit Lexer(std::string_view text) : text_(text) {}

    Token next() {
        skip_space_and_comments();
        if (offset_ == text_.size()) {
            return {.kind = TokenKind::End, .line = line_};
        }

        const size_t start = offset_;
        const uint32_t token_line = line_;
        const char first = text_[offset_];

        if (is_identifier_start(first)) {
            ++offset_;
            while (offset_ < text_.size() && is_identifier_continue(text_[offset_])) {
                ++offset_;
            }
            return {TokenKind::Identifier, text_.substr(start, offset_ - start), token_line};
        }

        if (std::isdigit(static_cast<unsigned char>(first)) ||
            (first == '.' && offset_ + 1 < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[offset_ + 1])))) {
            scan_number();
            return {TokenKind::Number, text_.substr(start, offset_ - start), token_line};
        }

        if (first == '"') {
            ++offset_;
            const size_t content_start = offset_;
            while (offset_ < text_.size() && text_[offset_] != '"') {
                if (text_[offset_] == '\n' || text_[offset_] == '\r') {
                    throw ParseError("Unterminated string literal", token_line);
                }
                if (text_[offset_] == '\\') {
                    throw ParseError("Escapes are not supported in include paths", token_line);
                }
                ++offset_;
            }
            if (offset_ == text_.size()) {
                throw ParseError("Unterminated string literal", token_line);
            }
            const std::string_view content = text_.substr(content_start, offset_ - content_start);
            ++offset_;
            return {TokenKind::String, content, token_line};
        }

        ++offset_;
        return {TokenKind::Symbol, text_.substr(start, 1), token_line};
    }

  private:
    static bool is_identifier_start(char c) {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    }

    static bool is_identifier_continue(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    void scan_number() {
        while (offset_ < text_.size() &&
               std::isdigit(static_cast<unsigned char>(text_[offset_]))) {
            ++offset_;
        }
        if (offset_ < text_.size() && text_[offset_] == '.') {
            ++offset_;
            while (offset_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[offset_]))) {
                ++offset_;
            }
        }
        if (offset_ < text_.size() && (text_[offset_] == 'e' || text_[offset_] == 'E')) {
            ++offset_;
            if (offset_ < text_.size() && (text_[offset_] == '+' || text_[offset_] == '-')) {
                ++offset_;
            }
            const size_t exponent_start = offset_;
            while (offset_ < text_.size() &&
                   std::isdigit(static_cast<unsigned char>(text_[offset_]))) {
                ++offset_;
            }
            if (exponent_start == offset_) {
                throw ParseError("Malformed numeric exponent", line_);
            }
        }
    }

    void skip_space_and_comments() {
        while (offset_ < text_.size()) {
            const char current = text_[offset_];
            if (std::isspace(static_cast<unsigned char>(current))) {
                line_ += current == '\n';
                ++offset_;
                continue;
            }
            if (current != '/' || offset_ + 1 == text_.size()) {
                return;
            }
            const char next = text_[offset_ + 1];
            if (next == '/') {
                offset_ += 2;
                while (offset_ < text_.size() && text_[offset_] != '\n') {
                    ++offset_;
                }
                continue;
            }
            if (next != '*') {
                return;
            }

            const uint32_t comment_line = line_;
            offset_ += 2;
            bool closed = false;
            while (offset_ + 1 < text_.size()) {
                line_ += text_[offset_] == '\n';
                if (text_[offset_] == '*' && text_[offset_ + 1] == '/') {
                    offset_ += 2;
                    closed = true;
                    break;
                }
                ++offset_;
            }
            if (!closed) {
                throw ParseError("Unterminated block comment", comment_line);
            }
        }
    }

    std::string_view text_;
    size_t offset_ = 0;
    uint32_t line_ = 1;
};

enum class PhaseRule : uint8_t {
    None,
    QasmU,
};

struct GateSpec {
    GateType gate;
    uint8_t num_args;
    uint8_t num_qubits;
    PhaseRule phase_rule = PhaseRule::None;
    bool identity = false;
    bool requires_qelib1 = true;
};

const GateSpec* find_gate(std::string_view name) {
    // U and CX are built into OpenQASM 2.0. The remaining entries are the
    // directly lowerable unitary portion of qelib1.inc.
    static const std::unordered_map<std::string_view, GateSpec> gates = {
        {"U", {GateType::U3, 3, 1, PhaseRule::QasmU, false, false}},
        {"CX", {GateType::CX, 0, 2, PhaseRule::None, false, false}},
        {"id", {GateType::I, 0, 1, PhaseRule::None, true}},
        {"x", {GateType::X, 0, 1}},
        {"y", {GateType::Y, 0, 1}},
        {"z", {GateType::Z, 0, 1}},
        {"h", {GateType::H, 0, 1}},
        {"s", {GateType::S, 0, 1}},
        {"sdg", {GateType::S_DAG, 0, 1}},
        {"t", {GateType::T, 0, 1}},
        {"tdg", {GateType::T_DAG, 0, 1}},
        {"rx", {GateType::R_X, 1, 1}},
        {"ry", {GateType::R_Y, 1, 1}},
        {"rz", {GateType::R_Z, 1, 1}},
        {"u1", {GateType::U3, 1, 1, PhaseRule::QasmU}},
        {"u2", {GateType::U3, 2, 1, PhaseRule::QasmU}},
        {"u3", {GateType::U3, 3, 1, PhaseRule::QasmU}},
        {"cx", {GateType::CX, 0, 2}},
        {"cy", {GateType::CY, 0, 2}},
        {"cz", {GateType::CZ, 0, 2}},
        {"swap", {GateType::SWAP, 0, 2}},
    };
    const auto it = gates.find(name);
    return it == gates.end() ? nullptr : &it->second;
}

struct Register {
    uint32_t offset;
    uint32_t width;
};

struct QubitArgument {
    uint32_t offset;
    uint32_t width;

    uint32_t at(uint32_t index) const { return offset + (width == 1 ? 0 : index); }
};

class Parser {
  public:
    Parser(std::string_view text, size_t max_ops) : lexer_(text), max_ops_(max_ops) {
        const auto non_ascii = std::find_if(text.begin(), text.end(), [](char c) {
            return static_cast<unsigned char>(c) > 127;
        });
        if (non_ascii != text.end()) {
            throw ParseError("Non-ASCII/Unicode character detected. Only plain ASCII is supported.",
                             0);
        }
        advance();
    }

    Qasm2Import parse() {
        expect_identifier("OPENQASM");
        if (token_.kind != TokenKind::Number || token_.text != "2.0") {
            if (token_.kind == TokenKind::Number && token_.text == "3.0") {
                throw ParseError("OpenQASM 3 is not supported; expected OPENQASM 2.0", token_.line);
            }
            throw ParseError("Expected OpenQASM version 2.0", token_.line);
        }
        advance();
        expect_symbol(';');

        while (token_.kind != TokenKind::End) {
            if (is_identifier("include")) {
                parse_include();
            } else if (is_identifier("qreg")) {
                parse_qreg();
            } else if (is_identifier("barrier")) {
                parse_barrier();
            } else if (is_identifier("creg") || is_identifier("measure") ||
                       is_identifier("reset") || is_identifier("if")) {
                throw ParseError("Non-unitary and classical statements are not supported by the "
                                 "OpenQASM 2 importer",
                                 token_.line);
            } else if (is_identifier("gate") || is_identifier("opaque")) {
                throw ParseError("Custom gate and opaque declarations are not supported", token_.line);
            } else if (token_.kind == TokenKind::Identifier) {
                parse_gate();
            } else {
                throw ParseError("Expected an OpenQASM 2 statement", token_.line);
            }
        }

        return std::move(result_);
    }

  private:
    void advance() { token_ = lexer_.next(); }

    bool is_identifier(std::string_view value) const {
        return token_.kind == TokenKind::Identifier && token_.text == value;
    }

    bool is_symbol(char value) const {
        return token_.kind == TokenKind::Symbol && token_.text.size() == 1 &&
               token_.text.front() == value;
    }

    void expect_identifier(std::string_view value) {
        if (!is_identifier(value)) {
            throw ParseError("Expected '" + std::string(value) + "'", token_.line);
        }
        advance();
    }

    void expect_symbol(char value) {
        if (!is_symbol(value)) {
            throw ParseError("Expected '" + std::string(1, value) + "'", token_.line);
        }
        advance();
    }

    bool consume_symbol(char value) {
        if (!is_symbol(value)) {
            return false;
        }
        advance();
        return true;
    }

    std::string parse_identifier() {
        if (token_.kind != TokenKind::Identifier) {
            throw ParseError("Expected identifier", token_.line);
        }
        std::string result(token_.text);
        advance();
        return result;
    }

    uint32_t parse_uint() {
        if (token_.kind != TokenKind::Number || token_.text.find_first_of(".eE") !=
                                                   std::string_view::npos) {
            throw ParseError("Expected a non-negative integer", token_.line);
        }
        uint64_t value = 0;
        for (const char c : token_.text) {
            value = value * 10 + static_cast<uint64_t>(c - '0');
            if (value > std::numeric_limits<uint32_t>::max()) {
                throw ParseError("Integer exceeds the supported range", token_.line);
            }
        }
        advance();
        return static_cast<uint32_t>(value);
    }

    void parse_include() {
        const uint32_t line = token_.line;
        advance();
        if (token_.kind != TokenKind::String) {
            throw ParseError("Expected include path string", token_.line);
        }
        if (token_.text != "qelib1.inc") {
            throw ParseError("Only the built-in qelib1.inc include is supported", line);
        }
        qelib1_included_ = true;
        advance();
        expect_symbol(';');
    }

    void parse_qreg() {
        const uint32_t line = token_.line;
        advance();
        const std::string name = parse_identifier();
        if (registers_.contains(name)) {
            throw ParseError("Duplicate quantum register '" + name + "'", line);
        }
        expect_symbol('[');
        const uint32_t width = parse_uint();
        if (width == 0) {
            throw ParseError("Quantum register width must be positive", line);
        }
        expect_symbol(']');
        expect_symbol(';');

        const uint64_t next_qubit = static_cast<uint64_t>(result_.circuit.num_qubits) + width;
        if (next_qubit > std::numeric_limits<uint32_t>::max()) {
            throw ParseError("Total quantum register width exceeds the supported range", line);
        }
        registers_.emplace(name, Register{result_.circuit.num_qubits, width});
        result_.circuit.num_qubits = static_cast<uint32_t>(next_qubit);
    }

    QubitArgument parse_qubit_argument() {
        const uint32_t line = token_.line;
        const std::string name = parse_identifier();
        const auto found = registers_.find(name);
        if (found == registers_.end()) {
            throw ParseError("Unknown quantum register '" + name + "'", line);
        }
        const Register reg = found->second;
        if (!consume_symbol('[')) {
            return {reg.offset, reg.width};
        }
        const uint32_t index = parse_uint();
        if (index >= reg.width) {
            throw ParseError("Quantum register index is out of range", line);
        }
        expect_symbol(']');
        return {reg.offset + index, 1};
    }

    std::vector<QubitArgument> parse_qubit_arguments(size_t max_arguments = 3) {
        std::vector<QubitArgument> arguments;
        arguments.push_back(parse_qubit_argument());
        while (consume_symbol(',')) {
            arguments.push_back(parse_qubit_argument());
            if (arguments.size() > max_arguments) {
                throw ParseError("Too many quantum arguments", token_.line);
            }
        }
        return arguments;
    }

    void parse_barrier() {
        advance();
        (void)parse_qubit_arguments(kMaxTargetsPerInstruction);
        expect_symbol(';');
    }

    double parse_expression() { return parse_additive(); }

    double parse_additive() {
        double value = parse_multiplicative();
        while (is_symbol('+') || is_symbol('-')) {
            const bool subtract = is_symbol('-');
            advance();
            const double rhs = parse_multiplicative();
            value = subtract ? value - rhs : value + rhs;
        }
        return value;
    }

    double parse_multiplicative() {
        double value = parse_power();
        while (is_symbol('*') || is_symbol('/')) {
            const bool divide = is_symbol('/');
            advance();
            const double rhs = parse_power();
            value = divide ? value / rhs : value * rhs;
        }
        return value;
    }

    double parse_power() {
        double value = parse_unary();
        if (consume_symbol('^')) {
            value = std::pow(value, parse_power());
        }
        return value;
    }

    double parse_unary() {
        if (consume_symbol('+')) {
            return parse_unary();
        }
        if (consume_symbol('-')) {
            return -parse_unary();
        }
        return parse_primary();
    }

    double parse_primary() {
        if (consume_symbol('(')) {
            const double value = parse_expression();
            expect_symbol(')');
            return value;
        }
        if (token_.kind == TokenKind::Number) {
            double value = 0;
            const auto parsed = fast_float::from_chars(token_.text.data(),
                                                       token_.text.data() + token_.text.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != token_.text.data() + token_.text.size()) {
                throw ParseError("Invalid numeric literal", token_.line);
            }
            advance();
            return value;
        }
        if (is_identifier("pi")) {
            advance();
            return std::numbers::pi;
        }
        if (token_.kind == TokenKind::Identifier) {
            const uint32_t line = token_.line;
            const std::string function = parse_identifier();
            expect_symbol('(');
            const double value = parse_expression();
            expect_symbol(')');
            if (function == "sin")
                return std::sin(value);
            if (function == "cos")
                return std::cos(value);
            if (function == "tan")
                return std::tan(value);
            if (function == "exp")
                return std::exp(value);
            if (function == "ln")
                return std::log(value);
            if (function == "sqrt")
                return std::sqrt(value);
            throw ParseError("Unsupported expression function '" + function + "'", line);
        }
        throw ParseError("Expected a constant angle expression", token_.line);
    }

    std::vector<double> parse_gate_arguments() {
        std::vector<double> arguments;
        if (!consume_symbol('(')) {
            return arguments;
        }
        if (consume_symbol(')')) {
            return arguments;
        }
        arguments.push_back(parse_expression());
        while (consume_symbol(',')) {
            if (arguments.size() == 3) {
                throw ParseError("Too many gate arguments", token_.line);
            }
            arguments.push_back(parse_expression());
        }
        expect_symbol(')');
        for (const double value : arguments) {
            if (!std::isfinite(value)) {
                throw ParseError("Gate arguments must evaluate to finite values", token_.line);
            }
        }
        return arguments;
    }

    static std::vector<double> lower_arguments(std::string_view gate_name,
                                                const std::vector<double>& radians) {
        const double to_half_turns = 1.0 / std::numbers::pi;
        if (gate_name == "u1") {
            return {0.0, 0.0, radians[0] * to_half_turns};
        }
        if (gate_name == "u2") {
            return {0.5, radians[0] * to_half_turns, radians[1] * to_half_turns};
        }
        std::vector<double> result;
        result.reserve(radians.size());
        for (const double value : radians) {
            result.push_back(value * to_half_turns);
        }
        return result;
    }

    static double qasm_u_phase_turns(std::string_view gate_name,
                                     const std::vector<double>& radians) {
        double phi = 0.0;
        double lambda = 0.0;
        if (gate_name == "u1") {
            lambda = radians[0];
        } else if (gate_name == "u2") {
            phi = radians[0];
            lambda = radians[1];
        } else {
            phi = radians[1];
            lambda = radians[2];
        }
        return (phi + lambda) / (2.0 * std::numbers::pi);
    }

    void parse_gate() {
        const uint32_t line = token_.line;
        const std::string gate_name = parse_identifier();
        const GateSpec* spec = find_gate(gate_name);
        if (spec == nullptr) {
            throw ParseError("Unsupported unitary OpenQASM 2 gate '" + gate_name + "'", line);
        }
        if (spec->requires_qelib1 && !qelib1_included_) {
            throw ParseError("Gate '" + gate_name + "' requires include \"qelib1.inc\"", line);
        }

        const std::vector<double> radians = parse_gate_arguments();
        if (radians.size() != spec->num_args) {
            throw ParseError("Gate '" + gate_name + "' expects " +
                                 std::to_string(spec->num_args) + " angle arguments",
                             line);
        }
        const std::vector<QubitArgument> qubits = parse_qubit_arguments();
        if (qubits.size() != spec->num_qubits) {
            throw ParseError("Gate '" + gate_name + "' expects " +
                                 std::to_string(spec->num_qubits) + " quantum arguments",
                             line);
        }
        expect_symbol(';');

        uint32_t broadcast_width = 1;
        for (const QubitArgument& qubit : qubits) {
            broadcast_width = std::max(broadcast_width, qubit.width);
        }
        for (const QubitArgument& qubit : qubits) {
            if (qubit.width != 1 && qubit.width != broadcast_width) {
                throw ParseError("Register operands must have matching widths", line);
            }
        }

        if (spec->phase_rule == PhaseRule::QasmU) {
            result_.global_phase_turns = std::remainder(
                result_.global_phase_turns +
                    static_cast<double>(broadcast_width) * qasm_u_phase_turns(gate_name, radians),
                2.0);
        }
        if (spec->identity) {
            return;
        }
        if (broadcast_width > max_ops_ - std::min(max_ops_, result_.circuit.nodes.size())) {
            throw ParseError("Circuit exceeds maximum lowered operations limit", line);
        }

        const std::vector<double> lowered_args = lower_arguments(gate_name, radians);
        for (uint32_t index = 0; index < broadcast_width; ++index) {
            std::vector<Target> targets;
            targets.reserve(qubits.size());
            std::unordered_set<uint32_t> used;
            for (const QubitArgument& qubit : qubits) {
                const uint32_t target = qubit.at(index);
                if (!used.insert(target).second) {
                    throw ParseError("A gate application cannot use the same qubit twice", line);
                }
                targets.push_back(Target::qubit(target));
            }
            result_.circuit.nodes.push_back(
                {spec->gate, std::move(targets), lowered_args, line, {}});
        }
    }

    Lexer lexer_;
    Token token_;
    size_t max_ops_;
    bool qelib1_included_ = false;
    std::unordered_map<std::string, Register> registers_;
    Qasm2Import result_;
};

}  // namespace

Qasm2Import parse_qasm2(std::string_view text) {
    return parse_qasm2(text, kMaxUnrolledOps);
}

Qasm2Import parse_qasm2(std::string_view text, size_t max_ops) {
    Parser parser(text, max_ops);
    return parser.parse();
}

Qasm2Import parse_qasm2_file(const std::string& path) {
    return parse_qasm2_file(path, kMaxUnrolledOps);
}

Qasm2Import parse_qasm2_file(const std::string& path, size_t max_ops) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    const std::streamoff length = file.tellg();
    if (length < 0) {
        throw std::runtime_error("Error determining file size: " + path);
    }
    constexpr std::streamsize kMaxFileSize = 1024LL * 1024LL * 1024LL;
    if (length > kMaxFileSize) {
        throw std::runtime_error("Circuit file exceeds 1GB memory limit (" +
                                 std::to_string(length) + " bytes).");
    }
    std::string contents(static_cast<size_t>(length), '\0');
    file.seekg(0, std::ios::beg);
    if (!contents.empty() && !file.read(contents.data(), length)) {
        throw std::runtime_error("Error reading file: " + path);
    }
    return parse_qasm2(contents, max_ops);
}

}  // namespace clifft
