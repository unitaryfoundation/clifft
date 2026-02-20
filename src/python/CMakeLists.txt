# Python bindings using nanobind
# This file is only included when building via scikit-build-core (SKBUILD is set)

# Find Python with the components needed by nanobind
find_package(Python 3.12
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# Find nanobind (installed as a Python package)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

# Build the extension module
nanobind_add_module(
    _ucc_core
    # Target stable ABI for Python 3.12+ (reduces number of wheels needed)
    STABLE_ABI
    # Source files
    bindings.cc
)

# Link against our core library
target_link_libraries(_ucc_core PRIVATE ucc_core)

# Install the extension module into the ucc package directory
# LIBRARY for .so (Linux/macOS), RUNTIME for .pyd (Windows)
install(TARGETS _ucc_core
    LIBRARY DESTINATION ucc
    RUNTIME DESTINATION ucc)

# Install Python package files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ucc/ DESTINATION ucc)
