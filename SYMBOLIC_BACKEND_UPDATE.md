# Clifft symbolic backend update

The current team briefing is a small FAQ-style webpage in
[`symbolic-backend-update/`](symbolic-backend-update/). It compares Clifft's
legacy backend, the symbolic backend through PR #288, SymFT single-shot, and
SymFT batched execution; it also summarizes the architecture, the ablation
findings, and the next steps.

The measured results and detailed caveats remain in
[`SYMBOLIC_BACKEND_RESULTS.md`](SYMBOLIC_BACKEND_RESULTS.md).

To run the briefing locally:

```sh
cd symbolic-backend-update
npm ci
npm run dev
```
