set(CLIFFT_OPENMP "AUTO" CACHE STRING
    "OpenMP intra-shot kernels: OFF, AUTO, or ON")
set_property(CACHE CLIFFT_OPENMP PROPERTY STRINGS OFF AUTO ON)
if(NOT CLIFFT_OPENMP STREQUAL "OFF" AND
   NOT CLIFFT_OPENMP STREQUAL "AUTO" AND
   NOT CLIFFT_OPENMP STREQUAL "ON")
    message(FATAL_ERROR "CLIFFT_OPENMP must be one of: OFF, AUTO, ON")
endif()

set(CLIFFT_OPENMP_ENABLED OFF)
if(NOT CLIFFT_OPENMP STREQUAL "OFF")
    if(EMSCRIPTEN)
        if(CLIFFT_OPENMP STREQUAL "ON")
            message(FATAL_ERROR "CLIFFT_OPENMP=ON is not supported under Emscripten")
        endif()
    else()
        # Apple Clang does not search Homebrew's libomp prefix by default.
        if(APPLE AND NOT DEFINED OpenMP_ROOT)
            execute_process(
                COMMAND brew --prefix libomp
                OUTPUT_VARIABLE _CLIFFT_BREW_LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _CLIFFT_BREW_LIBOMP_RESULT
            )
            if(_CLIFFT_BREW_LIBOMP_RESULT EQUAL 0 AND _CLIFFT_BREW_LIBOMP_PREFIX)
                set(OpenMP_ROOT "${_CLIFFT_BREW_LIBOMP_PREFIX}")
            endif()
        endif()
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(CLIFFT_OPENMP_ENABLED ON)
            if(APPLE AND SKBUILD)
                # dyld coalesces weak C++ symbols across distinct libomp dylibs,
                # mixing incompatible runtime internals with packages such as Aer.
                # The extension's export list keeps a static runtime private.
                set(_CLIFFT_OPENMP_PYTHON_LIBRARIES "")
                foreach(_CLIFFT_OPENMP_LIBRARY IN LISTS OpenMP_CXX_LIBRARIES)
                    if(_CLIFFT_OPENMP_LIBRARY MATCHES "/libomp\\.(dylib|a)$")
                        get_filename_component(_CLIFFT_OPENMP_DIRECTORY
                            "${_CLIFFT_OPENMP_LIBRARY}" DIRECTORY)
                        set(_CLIFFT_OPENMP_LIBRARY "${_CLIFFT_OPENMP_DIRECTORY}/libomp.a")
                        if(NOT EXISTS "${_CLIFFT_OPENMP_LIBRARY}")
                            set(_CLIFFT_OPENMP_MISSING_ARCHIVE
                                "macOS Python OpenMP support requires ${_CLIFFT_OPENMP_LIBRARY}.")
                            if(CLIFFT_OPENMP STREQUAL "ON")
                                message(FATAL_ERROR
                                    "${_CLIFFT_OPENMP_MISSING_ARCHIVE} "
                                    "Install a runtime with libomp.a or configure CLIFFT_OPENMP=OFF.")
                            endif()
                            message(WARNING
                                "${_CLIFFT_OPENMP_MISSING_ARCHIVE} "
                                "Disabling OpenMP; serial and cross-shot sampling remain available.")
                            set(CLIFFT_OPENMP_ENABLED OFF)
                            break()
                        endif()
                    endif()
                    list(APPEND _CLIFFT_OPENMP_PYTHON_LIBRARIES "${_CLIFFT_OPENMP_LIBRARY}")
                endforeach()
                if(CLIFFT_OPENMP_ENABLED)
                    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES
                        "${_CLIFFT_OPENMP_PYTHON_LIBRARIES}")
                endif()
            endif()
        elseif(CLIFFT_OPENMP STREQUAL "ON")
            message(FATAL_ERROR "CLIFFT_OPENMP=ON requested, but OpenMP C++ support was not found")
        endif()
    endif()
endif()
message(STATUS "CLIFFT_OPENMP = ${CLIFFT_OPENMP} (enabled: ${CLIFFT_OPENMP_ENABLED})")
