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

# Generate a tiny Python build-config module so import-time checks can know the
# wheel/source CPU baseline without importing the extension first.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/clifft)
set(CLIFFT_PY_REQUIRES_X86_64_V3 "False")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)" AND
   CLIFFT_CPU_BASELINE STREQUAL "x86-64-v3")
    set(CLIFFT_PY_REQUIRES_X86_64_V3 "True")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/_build_config.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/clifft/_build_config.py
    @ONLY
)

# Build the extension module
nanobind_add_module(
    _clifft_core
    # Target stable ABI for Python 3.12+ (reduces number of wheels needed)
    STABLE_ABI
    # Source files
    bindings.cc
)

# Link against our core library
target_link_libraries(_clifft_core PRIVATE clifft_core)

# Code coverage: propagate gcov linkage to the shared module
if(CLIFFT_COVERAGE)
    target_link_options(_clifft_core PRIVATE --coverage)
endif()

# Install the extension module into the clifft package directory
# LIBRARY for .so (Linux/macOS), RUNTIME for .pyd (Windows)
install(TARGETS _clifft_core
    LIBRARY DESTINATION clifft
    RUNTIME DESTINATION clifft)

# Generate type stubs for the C++ extension at install time so that type
# checkers (mypy, pyright) can see the full API.  INSTALL_TIME runs stubgen
# after the .so is in its final location, which is required for import.
nanobind_add_stub(
    _clifft_core_stub
    MODULE        clifft._clifft_core
    OUTPUT        clifft/_clifft_core.pyi
    MARKER_FILE   clifft/py.typed
    INCLUDE_PRIVATE
    INSTALL_TIME
    DEPENDS       _clifft_core
)

# Install Python package files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/clifft/ DESTINATION clifft)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/clifft/_build_config.py DESTINATION clifft)
