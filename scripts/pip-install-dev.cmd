@echo off
setlocal
cd /d "%~dp0.."

if not exist .venv\Scripts\python.exe (
  python -m venv .venv
)

set "DEPS_DIR=%CD%\build\cp312-abi3-win_amd64\_deps"
if not exist "%DEPS_DIR%\stim-src\.git" (
  git -c safe.directory=* clone --branch v1.15.0 --depth 1 https://github.com/quantumlib/Stim.git "%DEPS_DIR%\stim-src"
)
if not exist "%DEPS_DIR%\fast_float-src\.git" (
  git -c safe.directory=* clone --branch v8.2.3 --depth 1 https://github.com/fastfloat/fast_float.git "%DEPS_DIR%\fast_float-src"
)

set "SKBUILD_CMAKE_ARGS=-DFETCHCONTENT_SOURCE_DIR_STIM=%DEPS_DIR%/stim-src;-DFETCHCONTENT_SOURCE_DIR_FAST_FLOAT=%DEPS_DIR%/fast_float-src"
.venv\Scripts\python -m pip install -e .
endlocal
