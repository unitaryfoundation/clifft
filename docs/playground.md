# Playground

The Clifft Playground is an interactive browser-based tool for writing, compiling, and simulating quantum circuits. It runs entirely in your browser using WebAssembly -- no installation required.

<div class="grid cards" markdown>

- :material-open-in-new: **[Launch Playground](../playground/){ target="_blank" }**

    Write Stim circuits, see the compiled bytecode, and simulate measurement outcomes — all in your browser.

</div>

## Features

- **Monaco editor** with Stim syntax highlighting
- **Live compilation** showing the RISC bytecode and Heisenberg IR
- **Simulation** with configurable shot count and histogram visualization
- **Active qubit history** chart showing $k$ (active dimension) over time
- **Share links** for saving and sharing circuits via URL
- **Dark/light theme** with system preference detection
- **Guided tour** for first-time users

## How It Works

The playground compiles the Clifft C++ core to WebAssembly using Emscripten. The full AOT compilation pipeline -- parsing, front-end Clifford absorption, optimization, back-end virtual compression, and bytecode emission -- runs client-side in the browser. Simulation executes the same Schrodinger Virtual Machine as the native build.

!!! note "Performance"
    The WebAssembly build is slower than native C++ (roughly 3-5x). For production workloads, use the Python package.
