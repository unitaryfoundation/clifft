#include "clifft/util/runtime_isa.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace clifft::internal {

namespace {

#if defined(CLIFFT_ENABLE_RUNTIME_DISPATCH)

bool host_supports_avx2_kernel() {
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64))
    // Every AVX2 kernel TU uses -mavx2 -mbmi2 -mfma, so all three
    // features are required even when a particular kernel uses fewer.
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

bool host_supports_avx512_kernel() {
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64))
    // Every AVX-512 kernel TU additionally uses -mavx512f -mavx512dq.
    return host_supports_avx2_kernel() && __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

std::string normalize_force_isa(const char* environment_value) {
    std::string name(environment_value);
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    name.erase(name.begin(), std::find_if(name.begin(), name.end(), not_space));
    name.erase(std::find_if(name.rbegin(), name.rend(), not_space).base(), name.end());
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
}

RuntimeIsa resolve_runtime_isa() {
    if (const char* environment_value = std::getenv("CLIFFT_FORCE_ISA")) {
        const std::string name = normalize_force_isa(environment_value);
        if (name == "avx512") {
            return host_supports_avx512_kernel() ? RuntimeIsa::Avx512 : RuntimeIsa::TrapAvx512;
        }
        if (name == "avx2") {
            return host_supports_avx2_kernel() ? RuntimeIsa::Avx2 : RuntimeIsa::TrapAvx2;
        }
        if (name == "scalar") {
            return RuntimeIsa::Scalar;
        }
        // Treat an empty override as auto-detect so shell callers do not need
        // to distinguish an unset variable from CLIFFT_FORCE_ISA=.
        if (!name.empty()) {
            return RuntimeIsa::TrapUnknown;
        }
    }

    if (host_supports_avx512_kernel()) {
        return RuntimeIsa::Avx512;
    }
    if (host_supports_avx2_kernel()) {
        return RuntimeIsa::Avx2;
    }
#endif
    return RuntimeIsa::Scalar;
}

}  // namespace

RuntimeIsa runtime_isa() {
    static const RuntimeIsa selected = resolve_runtime_isa();
    return selected;
}

const char* runtime_isa_name(RuntimeIsa isa) noexcept {
    switch (isa) {
        case RuntimeIsa::Scalar:
            return "scalar";
        case RuntimeIsa::Avx2:
            return "avx2";
        case RuntimeIsa::Avx512:
            return "avx512";
        case RuntimeIsa::TrapAvx2:
            return "trap:avx2";
        case RuntimeIsa::TrapAvx512:
            return "trap:avx512";
        case RuntimeIsa::TrapUnknown:
            return "trap:unknown";
    }
    return "trap:unknown";
}

void validate_runtime_isa(RuntimeIsa isa) {
    switch (isa) {
        case RuntimeIsa::Scalar:
        case RuntimeIsa::Avx2:
        case RuntimeIsa::Avx512:
            return;
        case RuntimeIsa::TrapAvx2:
            throw std::runtime_error(
                "CLIFFT_FORCE_ISA=avx2 requested but host CPU lacks one or more required "
                "features (avx2, bmi2, fma). Unset CLIFFT_FORCE_ISA to use the auto-detected "
                "fallback, or set it to 'scalar' explicitly.");
        case RuntimeIsa::TrapAvx512:
            throw std::runtime_error(
                "CLIFFT_FORCE_ISA=avx512 requested but host CPU lacks one or more required "
                "features (avx2, bmi2, fma, avx512f, avx512dq). Unset CLIFFT_FORCE_ISA to use "
                "the auto-detected fallback, or set it to 'avx2' or 'scalar' explicitly.");
        case RuntimeIsa::TrapUnknown:
            throw std::runtime_error(
                "CLIFFT_FORCE_ISA is set to an unrecognized value. Accepted values are "
                "'avx512', 'avx2', 'scalar' (case-insensitive). Unset CLIFFT_FORCE_ISA to "
                "use the auto-detected fallback.");
    }
}

}  // namespace clifft::internal
