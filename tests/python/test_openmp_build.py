"""Configuration coverage for optional OpenMP and private macOS linkage."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("mode", "archive", "apple", "python_build", "expected"),
    [
        ("AUTO", True, True, True, "static"),
        ("ON", True, True, True, "static"),
        ("AUTO", False, True, True, "disabled"),
        ("ON", False, True, True, "error"),
        ("OFF", False, True, True, "disabled"),
        ("OFF", True, True, True, "disabled"),
        ("AUTO", False, True, False, "shared"),
        ("ON", False, True, False, "shared"),
        ("ON", False, False, True, "shared"),
    ],
)
def test_openmp_configuration(
    tmp_path: Path, mode: str, archive: bool, apple: bool, python_build: bool, expected: str
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for build configuration tests")
    assert cmake is not None

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    shared = runtime / "libomp.dylib"
    static = runtime / "libomp.a"
    shared.touch()
    if archive:
        static.touch()

    # Model discovery independently of the host toolchain so a dylib-only
    # installation is exercised even on CI machines that have Homebrew's archive.
    (tmp_path / "FindOpenMP.cmake").write_text(
        "set(OpenMP_CXX_FOUND TRUE)\n"
        f'set(OpenMP_CXX_LIBRARIES "{shared.as_posix()}")\n'
        "add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)\n"
        "set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES\n"
        '    "${OpenMP_CXX_LIBRARIES}")\n'
    )
    module = Path(__file__).parents[2] / "cmake" / "ClifftOpenMP.cmake"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(openmp_configuration NONE)\n"
        'list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")\n'
        f"set(APPLE {'TRUE' if apple else 'FALSE'})\n"
        f"set(SKBUILD {'TRUE' if python_build else 'FALSE'})\n"
        'set(OpenMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/runtime")\n'
        f'include("{module.as_posix()}")\n'
        "if(CLIFFT_OPENMP_ENABLED)\n"
        "    get_target_property(runtime_libraries OpenMP::OpenMP_CXX INTERFACE_LINK_LIBRARIES)\n"
        "else()\n"
        '    set(runtime_libraries "disabled")\n'
        "endif()\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/runtime.txt" "${runtime_libraries}")\n'
    )
    build = tmp_path / "build"
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build), f"-DCLIFFT_OPENMP={mode}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = result.stdout + result.stderr
    if expected == "error":
        assert result.returncode != 0
        assert "macOS Python OpenMP support requires" in output
        assert "libomp.a" in output
        return

    assert result.returncode == 0, output
    assert (build / "runtime.txt").read_text() == {
        "static": static.as_posix(),
        "shared": shared.as_posix(),
        "disabled": "disabled",
    }[expected]
    if mode == "AUTO" and expected == "disabled":
        assert "CMake Warning" in output
        assert "Disabling OpenMP" in output
