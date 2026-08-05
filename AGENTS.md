# Clifft Agent Instructions

This file records repository-specific constraints that are easy to miss from
the code alone. Use `README.md`, `docs/development/`, and `just --list` for
project orientation and routine build, test, and contribution instructions.

## Source of Truth and Architecture Changes

Use `src/clifft/`, `tests/`, `docs/`, and `README.md` as the source of truth.
Inspect the current implementation and tests before resolving architectural
ambiguity.

If a proposed change contradicts an architectural invariant or the current
architecture cannot support it:

1. Stop instead of implementing a workaround or rewriting architecture-facing
   documentation to justify the change.
2. Explain the discrepancy and propose an explicit architectural change.
3. Wait for human confirmation before implementation.

## Architectural Invariants

- **Legacy SVM instruction ABI:** The legacy VM `Instruction` must remain
  exactly 32 bytes (`static_assert(sizeof(Instruction) == 32)`). New executors
  must use separate instruction or plan types rather than expanding it.
- **Stim is immutable:** Fetch Stim through CMake `FetchContent`. Do not fork,
  vendor, or patch Stim source.
- **Allocation-free hot execution:** The legacy `SchrodingerState` coefficient
  array is allocated once at construction from `peak_rank`. New executors must
  likewise preallocate coefficient, record, symbolic-state, and scratch storage
  before entering hot dispatch or kernels. The sanctioned exception is an
  explicit trap or continuation boundary, where storage may grow but never
  shrink before dispatch resumes. No allocation is allowed inside an ordinary
  dispatch loop or kernel.
- **Deterministic RNG:** Do not use `std::uniform_real_distribution`; its output
  is implementation-defined. Use `(rng() >> 11) * 0x1.0p-53` for `[0, 1)`.
- **No runtime topology planning:** The compiler or planner must precompute
  coordinate changes, Pauli pairings, phase behavior, active-width transitions,
  and symbolic dependencies. The legacy SVM continues to execute localized
  operations. A symbolic-coordinate executor may apply compiler-precomputed
  multi-coordinate active-Pauli actions directly, but runtime code must not
  perform tableau evolution, commutation analysis, localization, or dependency
  discovery.

## Repository-Specific Source and Test Rules

- Keep source files ASCII-only: use `pi`, `|0>`, and `Schrodinger` rather than
  Unicode alternatives.
- Comments should explain why, not restate what the code does.
- Do not put issue numbers, task phases, or planning details in code comments
  or test names.
- Catch2 `TEST_CASE` names must avoid special characters such as `[]`, `()`,
  and `,`.
- Validate Python-facing unitaries against Qiskit Aer and stochastic Clifford
  behavior against Stim when those independent references apply.

## Git and AI-Assisted Contribution Rules

- Never commit directly to `main`; use a feature branch.
- Keep commits atomic and use conventional prefixes such as `feat:`, `fix:`,
  `test:`, and `docs:`.
- Run `uv run --frozen pre-commit run --all-files --show-diff-on-failure`
  before every commit, in addition to the tests appropriate for the change.
- Include an `Assisted-by:` trailer identifying the agent that actually
  assisted. Each agent must use its own name, model, and provider address; do
  not copy a named example literally. Format:

  ```text
  Assisted-by: AGENT_NAME (MODEL_NAME) <PROVIDER_NOREPLY_EMAIL>
  ```
