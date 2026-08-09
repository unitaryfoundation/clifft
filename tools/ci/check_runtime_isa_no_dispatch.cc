#include "clifft/util/runtime_isa.h"

#include <string_view>

int main() {
    using clifft::internal::runtime_isa;
    using clifft::internal::runtime_isa_name;
    using clifft::internal::RuntimeIsa;
    using clifft::internal::validate_runtime_isa;

    const RuntimeIsa selected = runtime_isa();
    if (selected != RuntimeIsa::Scalar ||
        std::string_view(runtime_isa_name(selected)) != "scalar") {
        return 1;
    }
    validate_runtime_isa(selected);
    return 0;
}
