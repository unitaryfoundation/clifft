# Clifft Agent Instructions

This file records repository-specific constraints and recurring review patterns
that are easy to miss from the code alone. Keep design details in their
canonical docs and use `just --list` for routine build and test commands:

- `README.md`: high-level project overview and quick links
- `docs/guide/`: user-facing task guides (motivate *why* and *when* to use a
  feature before options and limits)
- `docs/theory/`: mathematical and architectural explanations
- `docs/reference/`: API, gate, and optimization-pass reference
- `docs/development/`: contributor workflow (`contributing.md`), test strategy
  (`testing.md`), and hardware backend notes

## Project Planning

Use the [Clifft development project](https://github.com/orgs/unitaryfoundation/projects/35)
and its README for current priorities and the planning workflow.

- When asked to manage planning, read the live project and relevant issue/PR
  discussions. Apply the maintainer's agreed priorities and verify saved changes.
- Keep priority and order in project fields, and goals, evidence, and acceptance
  criteria in issues. Avoid duplicating the backlog in repository documentation.
- Check existing membership and linked PRs before adding items. Keep bugs in
  their owning track with the `bug` label; prioritize them by impact and evidence.

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

- **Stim is immutable:** Fetch Stim through CMake `FetchContent`. Do not fork,
  vendor, or patch Stim source.
- **Allocation-free hot execution:** Executors must preallocate coefficient,
  record, symbolic-state, and scratch storage before entering hot dispatch or
  kernels. The sanctioned exception is an explicit trap, continuation, or online
  activation boundary, where storage may grow but never shrink before dispatch
  resumes. No allocation is allowed inside an ordinary dispatch loop or kernel.
- **Exception-free hot execution:** Validate external and compiler-produced
  inputs before entering hot execution. Use assertions rather than exceptions
  for internal preconditions in per-shot reset, ordinary dispatch, and kernels.
  Throwing validation remains appropriate during construction, planning,
  lowering, and explicit trap, continuation, or online activation boundaries.
- **Deterministic RNG:** Do not use `std::uniform_real_distribution`; its output
  is implementation-defined. Use `(rng() >> 11) * 0x1.0p-53` for `[0, 1)`.
- **No runtime topology planning in compiled executors:** The compiler or planner
  must precompute coordinate changes, Pauli pairings, phase behavior,
  active-width transitions, and symbolic dependencies. The executor may apply
  compiler-precomputed multi-coordinate active-Pauli actions directly, but
  runtime code must not perform tableau evolution, commutation analysis,
  localization, or dependency discovery. The explicitly selected
  noncomputational online backend is a separate mode: it may evolve its
  Clifford frame and active coordinates at runtime. Its ordinary dispatch
  remains allocation-free and exception-free; coefficient storage may grow only
  after suspension at an explicit activation boundary.

## Code Simplicity, Prose, and Test Rules

- **Simplify before finishing:** Before completing a change, review the full diff
  in hindsight. Inline single-caller helpers, collapse temporary adapter structs
  or flags introduced mid-development, and fold tiny pass-through headers or
  single-function bridge files into their natural owning module. Prefer
  extending an existing module over adding new files or layers unless the
  boundary serves a distinct caller or compile-time isolation need (such as
  ISA-specific kernel translation units).
- **Replace, do not shim:** When lifting an earlier limitation or replacing an
  execution or compilation path, delete the superseded validation guard,
  write-only field, legacy path, error test, and stale comment in the same
  change.
- **Plain, physics-grounded prose:** Write comments, docstrings, error messages,
  and docs in direct English grounded in standard quantum-circuit and simulator
  concepts. Avoid internal specification jargon (such as *"executable
  contract"*, *"result contract"*, *"review contract"*, *"supported scientific
  workflow"*, *"fixed-plan samplers"*, *"production worker layout"*, or
  *"instrument materialization"*) and invented terms that clash with standard
  quantum or CS terminology. Define specialized terms at first use. Describe
  what an API, pass, or mode does and when to use it; omit defensive *"what this
  is not"* disclaimers unless preventing a likely user mistake.
- **High-signal, present-tense comments:** Keep source files ASCII-only (use
  `pi`, `|0>`, and `Schrodinger` rather than Unicode alternatives). Give files
  and non-trivial classes or structs a concise overview comment stating what
  they represent or do (with a short example or formula when helpful) before
  member details. Add brief guidepost comments in dense functions and named
  constants (or short rationale comments) for non-obvious thresholds; omit
  line-by-line comments that merely restate the code. Write in the present tense
  so comments and docs stand alone; do not narrate how a bug was fixed, record
  migration history, or include issue/PR numbers, task phases, or planning notes.
- **Right-sized documentation with one canonical home:** Give each topic one
  owner and link rather than duplicate. User guides (`docs/guide/`) exist to
  help users choose and use Clifft features: open sections by motivating *why* a
  user would care in 1-2 plain sentences, keep examples small (folding long
  circuits or dense tables into collapsible blocks), and leave internal
  implementation catalogs out of user guides. Update existing sections instead
  of appending overlapping sections or per-release notes.
- **Test invariants, not prose or snapshots:** Validate behavior, mathematical
  identities, and statistical properties (cross-checking Python-facing unitaries
  against Qiskit Aer and stochastic Clifford behavior against Stim when those
  independent references apply). Catch2 `TEST_CASE` names must avoid special
  characters such as `[]`, `()`, and `,`. Do not write tests that
  substring-match English prose in docs or comments, or that assert deleted
  helpers remain absent. Follow
  [Writing Tests](docs/development/contributing.md#writing-tests).

## Acknowledgments and References

Reserve `docs/acknowledgments.md` (and the `README.md` summary) for substantive
external research or software influences, not minor internal or incidental
items. Each acknowledgment should complete the sentence "Clifft benefited from
this work through..." with a concrete algorithm, implementation, design
influence, validation method, or reused artifact. Distinguish direct use from
inspiration, link the relevant paper, software, or other source, and preserve
citations beside the technical discussion. A paper is not required; the specific
contribution is.

## Git and AI-Assisted Contribution Rules

- Never commit directly to `main`; use a feature branch.
- Keep commits atomic and use conventional prefixes such as `feat:`, `fix:`,
  `test:`, and `docs:`.
- Mark any backward-incompatible public API change, including removals and
  renames, with a `!` after the type (`feat(api)!:`) and a `BREAKING CHANGE:`
  footer describing the migration. Signature, return-type, default, and
  behavioral changes count. The changelog surfaces both markers; without them
  the entry reads as an ordinary feature.
- Run `uv run --frozen --only-group dev pre-commit run --all-files --show-diff-on-failure`
  before every commit, in addition to the tests appropriate for the change.
- Include an `Assisted-by:` trailer identifying the agent that actually
  assisted. Each agent must use its own name, model, and provider address; do
  not copy a named example literally. Format:

  ```text
  Assisted-by: AGENT_NAME (MODEL_NAME) <PROVIDER_NOREPLY_EMAIL>
  ```
