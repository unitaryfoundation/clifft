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
- **External circuit loading** from GitHub Gists and arbitrary URLs
- **Dark/light theme** with system preference detection
- **Guided tour** for first-time users

## Sharing Circuits

There are three ways to share a circuit via URL:

### Inline (`?code=`)

The **Share** button in the toolbar compresses the circuit with LZ-String and encodes it directly in the URL. This works for small-to-medium circuits. The button turns orange when the URL is long and may not work in all browsers.

### GitHub Gist (`?gist=`)

For larger circuits, create a [GitHub Gist](https://gist.github.com) containing your `.stim` file, then share the playground URL with the gist ID:

```
https://playground.clifft.dev/?gist=abc123def456
```

The playground fetches the first file from the gist via the GitHub API.

### External URL (`?url=`)

You can load a circuit from any CORS-enabled URL (e.g. GitHub raw content):

```
https://playground.clifft.dev/?url=https://raw.githubusercontent.com/org/repo/main/circuit.stim
```

The URL parameter value must be URL-encoded.

## How It Works

The playground compiles the Clifft C++ core to WebAssembly using Emscripten. The full AOT compilation pipeline -- parsing, front-end Clifford absorption, optimization, back-end virtual compression, and bytecode emission -- runs client-side in the browser. Simulation executes the same Schrodinger Virtual Machine as the native build.

!!! note "Circuit Size Limits"
    The browser build supports up to 50,000 instructions and a peak active dimension of $k = 24$ (~256 MB statevector). Near-Clifford circuits with hundreds of qubits but few T gates keep $k$ low and simulate comfortably. For high-rank circuits, use the native Python package.

!!! note "Performance"
    The WebAssembly build is slower than native C++ (roughly 3-5x). For production workloads, use the Python package.
